"""Recorded runtime setup for the pinned PyMC preparation and fit environment."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import sys

_RUNTIME_SETUP = None


def configure_runtime(cache_directory):
    """Use a private compiler cache and the reviewed macOS linker workaround.

    PyTensor 2.31.7 incorrectly treats macOS 15+ as requiring the linker
    library argument ``-ld64``. The narrow adapter removes that argument;
    it does not modify any model expression, prior, data or sampler setting.
    """
    global _RUNTIME_SETUP
    if _RUNTIME_SETUP is not None:
        return dict(_RUNTIME_SETUP)
    cache = Path(cache_directory).resolve()
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.setdefault('MPLCONFIGDIR', str(cache/'matplotlib'))
    desired = str(cache/'pytensor')
    if 'pytensor' not in sys.modules:
        current = os.environ.get('PYTENSOR_FLAGS','')
        os.environ['PYTENSOR_FLAGS'] = (current+',' if current else '')+'compiledir='+desired
    import pytensor
    if pytensor.config.floatX != 'float64':
        raise ValueError('Primary sampling requires float64 computations.')
    if str(pytensor.config.compiledir)!=desired:
        raise ValueError('Configure the private model runtime before importing PyMC or PyTensor.')
    adapter = False
    if platform.system() == 'Darwin' and pytensor.__version__ == '2.31.7':
        from pytensor.link.c.cmodule import GCC_compiler
        if not getattr(GCC_compiler, '_dchs_linker_adapter', False):
            original = GCC_compiler.compile_args
            def compile_args(march_flags=True):
                return [flag for flag in original(march_flags= march_flags) if flag != '-ld64']
            GCC_compiler.compile_args = staticmethod(compile_args)
            GCC_compiler._dchs_linker_adapter = True
        adapter = True
    if not pytensor.config.cxx:
        raise ValueError('Primary sampling requires an available C++ compiler.')
    _RUNTIME_SETUP = {'floatX':pytensor.config.floatX, 'compiler':pytensor.config.cxx,
                     'macos_ld64_adapter':adapter, 'blas_threads_per_chain':1}
    return dict(_RUNTIME_SETUP)


def model_runtime_identity():
    packages = ('numpy','pandas','scipy','patsy','statsmodels','pymc','pytensor',
                'arviz','xarray','threadpoolctl')
    return {'python':platform.python_version(), 'system':platform.system(),
            'machine':platform.machine(),
            'packages':{name:importlib.metadata.version(name) for name in packages}}
