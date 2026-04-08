#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import pytest
import torch
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import flashinfer


# ===========================================================================
# CuTe DSL RMSNorm
# ===========================================================================


class CuteDslRmsNorm:
    def __init__(
        self,
        N: int,
        threads_per_cta: Optional[int] = None,
    ):
        self.N = N  # hidden_size
        self.elems_per_thread = 8
        self.warp_size = 32
        self.threads_per_cta = threads_per_cta or self.heuristic_threads()
        self.warps_per_cta = (self.threads_per_cta + 31) // self.warp_size

    def heuristic_threads(self):
        """Optimized thread configuration matching Triton."""
        elems_per_warp = self.elems_per_thread * self.warp_size

        # Calculate warps needed
        warps_needed = (self.N + elems_per_warp - 1) // elems_per_warp

        # Match Triton's configuration: 4 warps (128 threads) for most cases
        if self.N <= 1024:
            num_warps = min(max(warps_needed, 2), 4)
        elif self.N <= 8192:
            num_warps = 4  # 128 threads - optimal for 4096/8192
        else:
            num_warps = 8  # 256 threads for very large hidden sizes

        # Round to multiple of 2
        num_warps = (num_warps + 1) // 2 * 2

        return num_warps * 32

    @cute.jit
    def __call__(
        self,
        mY,
        mX,
        mWeight,
        eps: cutlass.Float32 = 1e-6,
    ):
        M, _ = mX.shape
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        t_layout = cute.make_layout(self.threads_per_cta)
        v_layout = cute.make_layout(self.elems_per_thread)
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)
        self.kernel(mY, mX, mWeight, tiled_copy, eps).launch(
            grid=[M, 1, 1],
            block=[self.warps_per_cta * self.warp_size, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mWeight: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        eps: cute.Float,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        thr_copy = tiled_copy.get_slice(tidx)

        gY = cute.local_tile(mY, tiler=(1, self.N), coord=(bidx, 0))
        gX = cute.local_tile(mX, tiler=(1, self.N), coord=(bidx, 0))
        gY, gX = gY[0, None], gX[0, None]

        tYgY = thr_copy.partition_S(gY)
        pred = cute.make_rmem_tensor(
            cute.size(tYgY, mode=[1]), cutlass.Boolean,
        )
        for i in range(cute.size(pred)):
            offset = (i * self.threads_per_cta + tidx) * self.elems_per_thread
            pred[i] = offset < self.N

        tXgX = thr_copy.partition_S(gX)
        tWgW = thr_copy.partition_S(mWeight)
        tXrX = cute.make_fragment_like(tXgX)
        tXrX.fill(0)
        tWrW = cute.make_fragment_like(tWgW)

        # Coalesced load with predication
        for i in range(cute.size(tXrX, mode=[1])):
            if pred[i]:
                cute.autovec_copy(tXgX[None, i], tXrX[None, i])
                cute.autovec_copy(tWgW[None, i], tWrW[None, i])

        tYrY = self.apply_rmsnorm(tXrX, tWrW, eps, tidx, pred)

        # Coalesced store with predication
        for i in range(cute.size(tXrX, mode=[1])):
            if pred[i]:
                cute.autovec_copy(tYrY[None, i], tYgY[None, i])

    @cute.jit
    def warp_reduce(self, val, reduce_size=32):
        iters = int(math.log2(reduce_size))
        for i in range(iters):
            val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
        return val

    @cute.jit
    def cta_reduce(self, val, acc, tidx):
        """Optimized CTA reduction with minimal barriers."""
        warp_id = tidx >> 5
        lane_id = tidx & 31

        # Step 1: Write warp results to shared memory
        if lane_id == 0:
            acc[warp_id] = val

        # Only need one sync before reading
        cute.arch.sync_threads()

        # Step 2: First warp reads and reduces all warp sums
        if warp_id == 0:
            # Load all warp sums
            if lane_id < self.warps_per_cta:
                val = acc[lane_id]
            else:
                val = cutlass.Float32(0)

            # Reduce within first warp
            val = self.warp_reduce(val)

            # Store final result
            if lane_id == 0:
                acc[0] = val

        # Final sync to ensure result is written
        cute.arch.sync_threads()

        # All threads read the final result
        return acc[0]

    @cute.jit
    def apply_rmsnorm(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        eps: cute.Float,
        tidx: cutlass.Int32,
        pred: cute.Tensor,
    ):
        """
        y[i] = x[i] / sqrt(sum(x ^ 2) / D + eps) * w[i]
        """
        smem = cutlass.utils.SmemAllocator()
        acc = smem.allocate_tensor(cutlass.Float32, self.warps_per_cta + 1)

        # Compute sum of squares
        val = cute.Float32(0.0)
        for i in range(cute.size(x, mode=[1])):
            if pred[i]:
                for idx in range(cute.size(x[None, i])):
                    x_fp32 = x[None, i][idx].to(cutlass.Float32)
                    val += x_fp32 * x_fp32

        # Warp-level reduction
        val = self.warp_reduce(val)
        # CTA-level reduction
        acc_sq = self.cta_reduce(val, acc, tidx)

        # Compute normalization factor
        factor = cute.rsqrt(acc_sq / self.N + eps)

        # Apply normalization and weight
        tNrN = cute.make_fragment_like(x)
        tNrN.store((x.load() * factor * weight.load()).to(tNrN.element_type))
        return tNrN


_compiled_cache = {}


def _get_compiled(B, H, dtype):
    key = (B, H, dtype)
    if key not in _compiled_cache:
        x_dummy = torch.empty(B, H, device="cuda", dtype=dtype)
        w_dummy = torch.empty(H, device="cuda", dtype=dtype)
        y_dummy = torch.empty(B, H, device="cuda", dtype=dtype)

        _x = from_dlpack(x_dummy, assumed_align=16, enable_tvm_ffi=True)
        _w = from_dlpack(w_dummy, assumed_align=16, enable_tvm_ffi=True)
        _y = from_dlpack(y_dummy, assumed_align=16, enable_tvm_ffi=True)

        cutedsl_rmsnorm_kernel = CuteDslRmsNorm(H)
        compiled_rmsnorm_kernel = cute.compile(
            cutedsl_rmsnorm_kernel, _y, _x, _w,
            options="--generate-line-info --enable-tvm-ffi",
        )
        _compiled_cache[key] = compiled_rmsnorm_kernel
    return _compiled_cache[key]


def cutedsl_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RMSNorm using CuTe DSL.

    input: (batch_size, hidden_size); weight: (hidden_size,).
    out[b, h] = (input[b, h] / RMS(input[b, :])) * weight[h].
    """
    assert input.is_cuda, "input must be CUDA tensor"
    assert weight.is_cuda, "weight must be CUDA tensor"
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "only fp16 / bf16 / fp32 supported"
    assert weight.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "weight must be fp16 / bf16 / fp32"
    assert input.dim() == 2, "input must be 2D (batch_size, hidden_size)"

    B, H = input.shape
    assert H % 8 == 0, "hidden_size must be a multiple of 8"
    assert weight.shape == (H,), "weight must have shape (hidden_size,)"

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
        assert out.is_cuda

    compiled = _get_compiled(B, H, input.dtype)
    compiled(out, input, weight, eps)
    return out


# ===========================================================================
# Triton RMSNorm
# ===========================================================================


@triton.jit
def triton_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    B,  # batch size (rows)
    H,  # hidden size (last dimension)
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_POWER_OF_2: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    x_row_ptr = x_ptr + row_id * stride_xb
    y_row_ptr = y_ptr + row_id * stride_yb

    if IS_POWER_OF_2:
        # Optimized path for power-of-2 hidden sizes
        x_val = tl.load(x_row_ptr + offs * stride_xh)
        w_val = tl.load(w_ptr + offs)

        # Fused computation: convert, square, and reduce in one step
        x_f32 = x_val.to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x_f32 * x_f32, axis=0) / H + eps)

        # Fused normalize and apply weight
        y = (x_f32 * inv_rms * w_val.to(tl.float32)).to(y_ptr.dtype.element_ty)
        tl.store(y_row_ptr + offs * stride_yh, y)
    else:
        mask = offs < H
        x_val = tl.load(x_row_ptr + offs * stride_xh, mask=mask, other=0.0)
        w_val = tl.load(w_ptr + offs, mask=mask, other=0.0)

        x_f32 = x_val.to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x_f32 * x_f32, axis=0) / H + eps)

        y = (x_f32 * inv_rms * w_val.to(tl.float32)).to(y_ptr.dtype.element_ty)
        tl.store(y_row_ptr + offs * stride_yh, y, mask=mask)


def triton_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RMSNorm using Triton.

    input: (batch_size, hidden_size); weight: (hidden_size,).
    out[b, h] = (input[b, h] / RMS(input[b, :])) * weight[h].
    """
    assert input.is_cuda, "input must be CUDA tensor"
    assert weight.is_cuda, "weight must be CUDA tensor"
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "only fp16 / bf16 / fp32 supported"
    assert weight.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "weight must be fp16 / bf16 / fp32"
    assert input.dim() == 2, "input must be 2D (batch_size, hidden_size)"

    B, H = input.shape
    assert weight.shape == (H,), "weight must have shape (hidden_size,)"

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
        assert out.is_cuda

    # Optimized configuration for RMSNorm
    # BLOCK_SIZE must be power of 2 for Triton, and >= H
    BLOCK_SIZE = triton.next_power_of_2(H)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192

    # Check if H is a power of 2 (common case for transformers)
    IS_POWER_OF_2 = (H & (H - 1)) == 0

    # Optimized warp count based on hidden size
    # Smaller warps = less overhead for memory-bound kernels
    if H <= 512:
        num_warps = 2
    elif H <= 2048:
        num_warps = 4
    elif H <= 4096:
        num_warps = 4
    else:
        num_warps = 8

    grid = (B,)

    # Single stage is optimal for RMSNorm
    num_stages = 1

    triton_rmsnorm_kernel[grid](
        input,
        weight,
        out,
        B,
        H,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_POWER_OF_2=IS_POWER_OF_2,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


# ===========================================================================
# Reference (flashinfer)
# ===========================================================================


def flashinfer_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference: flashinfer.norm.rmsnorm."""
    return flashinfer.norm.rmsnorm(input, weight, eps=eps)


# ===========================================================================
# Tests (pytest) — reference is flashinfer
# ===========================================================================


DTYPES = [torch.float16, torch.bfloat16]


def _make_inputs(
    shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn((shape[-1],), device=device, dtype=dtype)
    return x, weight


def _assert_close(out, ref, dtype, tag):
    assert not torch.isnan(out).any(), f"{tag}: output contains NaN"
    assert not torch.isinf(out).any(), f"{tag}: output contains Inf"

    atol = 5e-3 if dtype == torch.float16 else 1e-2
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < atol, (
        f"{tag}: max_diff={max_diff:.6f} exceeds atol={atol}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 16),
        (2, 64),
        (9, 256),
        (17, 768),
        (31, 4096),
        (66, 8192),
    ],
)
def test_cutedsl_rmsnorm(dtype, shape):
    device = torch.device("cuda")
    x, w = _make_inputs(shape, dtype, device)

    out = cutedsl_rmsnorm(x, w)
    ref = flashinfer_rmsnorm(x, w)

    tag = f"CuTeDSL RMSNorm shape={shape} {dtype}"
    _assert_close(out, ref, dtype, tag)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 16),
        (2, 64),
        (9, 256),
        (17, 768),
        (31, 4096),
        (66, 8192),
    ],
)
def test_triton_rmsnorm(dtype, shape):
    device = torch.device("cuda")
    x, w = _make_inputs(shape, dtype, device)

    out = triton_rmsnorm(x, w)
    ref = flashinfer_rmsnorm(x, w)

    tag = f"Triton RMSNorm shape={shape} {dtype}"
    _assert_close(out, ref, dtype, tag)


# ===========================================================================
# Benchmark — triton.testing.perf_report (ref = flashinfer)
# ===========================================================================


def _bench_fn(provider, shape, dtype, device, eps=1e-6):
    torch.manual_seed(0)
    x, w = _make_inputs(shape, dtype, device)

    if provider == "flashinfer":
        def fn(): return flashinfer_rmsnorm(x, w, eps=eps)
    elif provider == "cutedsl":
        def fn(): return cutedsl_rmsnorm(x, w, eps=eps)
    elif provider == "triton":
        def fn(): return triton_rmsnorm(x, w, eps=eps)
    else:
        raise ValueError(provider)

    return fn


_BATCH_VALUES = [2**i for i in range(15)]  # 1, 2, 4, ..., 16384


def _make_bench(plot_name: str, hidden_size: int):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch"],
            x_vals=_BATCH_VALUES,
            x_log=True,
            line_arg="provider",
            line_vals=["flashinfer", "triton", "cutedsl"],
            line_names=["FlashInfer", "Triton", "CuTe DSL"],
            styles=[("red", "--"), ("green", "-."), ("blue", "-")],
            ylabel="Latency (ms)",
            plot_name=plot_name,
            args={
                "hidden_size": hidden_size,
                "dtype": torch.bfloat16,
                "device": torch.device("cuda"),
            },
        )
    )


@_make_bench("rmsnorm_bf16_hidden_4096", hidden_size=4096)
def bench_rmsnorm_hidden_4096(batch, provider, hidden_size, dtype, device):
    shape = (batch, hidden_size)
    fn = _bench_fn(provider, shape, dtype, device)
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


@_make_bench("rmsnorm_bf16_hidden_8192", hidden_size=8192)
def bench_rmsnorm_hidden_8192(batch, provider, hidden_size, dtype, device):
    shape = (batch, hidden_size)
    fn = _bench_fn(provider, shape, dtype, device)
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        print("=== RMSNorm bf16 (hidden=4096) ===")
        bench_rmsnorm_hidden_4096.run(
            print_data=True, show_plots=False, save_path="."
        )

        print("\n=== RMSNorm bf16 (hidden=8192) ===")
        bench_rmsnorm_hidden_8192.run(
            print_data=True, show_plots=False, save_path="."
        )
