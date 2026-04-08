#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

python3 -m pytest -v $WORK_PATH/src/rmsnorm/rmsnorm.py
python3 $WORK_PATH/src/rmsnorm/rmsnorm.py
