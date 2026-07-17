"""Path-load DeepGEMM's pure-python testing helpers (bench_kineto, count_bytes)
without importing the deep_gemm package: its compiled _C.so was built against a
different torch ABI than this image's torch 2.13 and fails to import, but the
testing helpers themselves are torch-only python."""
import importlib.util

_DG = '/mnt/3fs/dots-pretrain/daijiangkun/DeepGEMM/deep_gemm/testing'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench_kineto = _load('dg_testing_bench', f'{_DG}/bench.py').bench_kineto
count_bytes = _load('dg_testing_numeric', f'{_DG}/numeric.py').count_bytes
