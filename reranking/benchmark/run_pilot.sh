#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CPU_SET=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:18])))')
export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18
taskset -c "$CPU_SET" .venv/bin/python pilot.py "$@"
