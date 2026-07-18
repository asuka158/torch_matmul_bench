#!/bin/bash
# Full dots3 mxfp8 GEMM benchmark: 6 dense + 2 group categories -> ../result/*.csv
set -e
cd "$(dirname "$0")"
PY=/opt/uv/bin/python
$PY bench_dense.py
$PY bench_group.py
