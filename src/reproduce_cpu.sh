#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" src/run_tmb_transductive_cpu_baseline.py
"$PYTHON_BIN" src/analyze_tmb_results.py
"$PYTHON_BIN" src/bootstrap_tmb_comparison.py
"$PYTHON_BIN" src/plot_tmb_figures.py
printf '%s\n' "CPU baseline, analysis, and figure regeneration completed."
