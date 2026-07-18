"""Load the vendored copies of DeepGEMM's pure-python testing helpers
(bench_kineto, count_bytes) from ./dg_testing.

Vendored (copied verbatim) from DeepGEMM deep_gemm/testing/{bench,numeric}.py:
importing the deep_gemm package itself fails on this image (its compiled _C.so
was built against a different torch ABI than torch 2.13), and the bench must not
depend on the state of any other checkout anyway. The helpers are torch-only
python with no other DeepGEMM dependencies."""
import importlib.util
import os

_DG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dg_testing')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench_kineto = _load('dg_testing_bench', f'{_DG}/bench.py').bench_kineto
count_bytes = _load('dg_testing_numeric', f'{_DG}/numeric.py').count_bytes
