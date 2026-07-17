#!/usr/bin/env bash
# Run the full dots3 nvfp4 GEMM benchmark (6 dense + 2 group categories -> 8 CSVs in ../result/).
set -e
cd "$(dirname "$0")"
PY=/opt/uv/bin/python
$PY bench_dense.py
$PY bench_group.py
echo "All done. CSVs in $(cd ../result && pwd)"
