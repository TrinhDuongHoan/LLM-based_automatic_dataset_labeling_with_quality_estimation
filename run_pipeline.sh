#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.cache/matplotlib}"

mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" scripts/prepare_data.py --source data/newsCorpora.csv
"${PYTHON_BIN}" scripts/generate_pseudo_labels.py
"${PYTHON_BIN}" scripts/train_models.py
"${PYTHON_BIN}" scripts/evaluate.py
"${PYTHON_BIN}" scripts/visualize.py
