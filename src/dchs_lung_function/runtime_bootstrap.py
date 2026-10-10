"""Establish numerical thread controls before importing any scientific library."""
from __future__ import annotations
import os
import sys

THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
_STARTED = False


def bootstrap():
    """Apply the requested single-thread policy once, never retroactively."""
    global _STARTED
    if not _STARTED:
        if any(name in sys.modules for name in ('numpy', 'scipy', 'pytensor', 'pymc')):
            raise RuntimeError('Start through run.py, or call runtime_bootstrap.bootstrap() before importing NumPy/SciPy/PyTensor/PyMC. Thread controls cannot be applied retroactively.')
        for name in THREAD_ENV:
            os.environ[name] = '1'
        _STARTED = True
    if any(os.environ.get(name) != '1' for name in THREAD_ENV):
        raise RuntimeError('Numerical thread environment changed after the early runtime bootstrap.')
    return {name: os.environ[name] for name in THREAD_ENV}
