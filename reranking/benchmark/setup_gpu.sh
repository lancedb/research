#!/usr/bin/env bash
# Run once on a Linux CUDA-12.8-compatible GPU host, before timed benchmark stages.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
python3 -m venv .venv-colbert
for env in .venv .venv-colbert; do
    "$env/bin/python" -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
done
.venv/bin/python -m pip install -r requirements-gpu.lock
.venv-colbert/bin/python -m pip install -r requirements-colbert-gpu.lock
.venv/bin/python -m pip check
.venv-colbert/bin/python -m pip check
.venv/bin/python -m pytest -q test_bench.py test_smoke.py test_run_state.py test_pipeline.py
if [[ -f test_full.py ]]; then
    .venv/bin/python -m pytest -q test_full.py
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
.venv-colbert/bin/python -c 'import torch; from pylate import models; assert torch.cuda.is_available()'
