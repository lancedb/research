#!/usr/bin/env bash
# Run inside the benchmark directory on a temporary Brev H100.
# Supply only VOYAGE_API_KEY and TYPESAFE_API_KEY in .env on this machine.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
# Brev's H100 image currently has driver 570 (CUDA 12.8). The newest default
# PyPI torch wheel uses CUDA 13, so select a compatible official GPU build.
.venv/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q test_bench.py test_smoke.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Match the published 18-CPU allocation when Linux exposes at least that many.
# Resolve allowed CPUs instead of assuming the host starts its allocation at 0.
CPU_SET=$(.venv/bin/python -c 'import os; cpus=sorted(os.sched_getaffinity(0)); print(",".join(map(str, cpus[:18])))')
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
taskset -c "$CPU_SET" .venv/bin/python smoke.py --require-cuda --run-dir runs/h100-smoke
.venv/bin/python -m pip freeze > runs/h100-smoke/requirements.lock.txt
