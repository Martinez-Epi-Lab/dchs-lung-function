"""Recorded runtime setup for the pinned PyMC preparation and fit environment."""
from __future__ import annotations

import importlib.metadata
import functools
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
import platform
import sys

from .runtime_bootstrap import bootstrap

_RUNTIME_SETUP = None


def _compiler_settings():
    # Probe configuration in an isolated process: importing PyTensor here would
    # initialize its compiler cache before the caller selects a private cache.
    bootstrap()
    fields = ("cxx", "floatX", "blas__ldflags", "gcc__cxxflags", "openmp", "linker", "mode", "optimizer", "device")
    if "pytensor" in sys.modules:
        from pytensor.link.c.cmodule import GCC_compiler
        return {**{k:getattr(sys.modules["pytensor"].config,k) for k in fields},
                "resolved_compiler_arguments":GCC_compiler.compile_args()}
    flags = ",".join(x for x in os.environ.get("PYTENSOR_FLAGS", "").split(",")
                     if x.split("=", 1)[0].strip() not in {"compiledir", "base_compiledir"})
    rc = os.environ.get("PYTENSORRC", str(Path.home()/".pytensorrc"))
    rc_identity = tuple((name, hashlib.sha256(Path(name).read_bytes()).hexdigest()
                         if Path(name).is_file() else None) for name in rc.split(os.pathsep))
    return dict(_probe_settings(flags, rc_identity))


@functools.lru_cache(maxsize=8)
def _probe_settings(flags, rc_identity):
    with tempfile.TemporaryDirectory(prefix="dchs-runtime-probe-") as cache:
        env = dict(os.environ)
        env["PYTENSOR_FLAGS"] = (flags+"," if flags else "")+"compiledir="+cache
        code = """import json,pytensor,sys
if sys.platform=='darwin' and pytensor.__version__=='2.31.7':
 from pytensor.link.c.cmodule import GCC_compiler
 original=GCC_compiler.compile_args
 GCC_compiler.compile_args=staticmethod(lambda march_flags=True:[x for x in original(march_flags=march_flags) if x!='-ld64'])
c=pytensor.config
from pytensor.link.c.cmodule import GCC_compiler
print(json.dumps({**{k:getattr(c,k) for k in ('cxx','floatX','blas__ldflags','gcc__cxxflags','openmp','linker','mode','optimizer','device')},'resolved_compiler_arguments':GCC_compiler.compile_args()}))
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], env=env, text=True,
                                capture_output=True, check=True, timeout=60)
        return json.loads(result.stdout)


def _compiler_identity(settings):
    argv = shlex.split(settings["cxx"])
    if not argv:
        raise ValueError("Model runtime requires an available C++ compiler.")
    executable = shutil.which(argv[0])
    if not executable:
        raise ValueError("The configured C++ compiler cannot be resolved.")
    version = subprocess.run([executable, *argv[1:], "--version"], capture_output=True,
                             text=True, check=True, timeout=15)
    return {"command":argv, "resolved_executable":str(Path(executable).resolve()),
            "executable_sha256":hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
            "version":version.stdout.strip() or version.stderr.strip()}


def thread_observations():
    """Report capabilities honestly; an empty controller list is not one thread."""
    requested = bootstrap()
    import numpy as np
    import scipy.linalg  # ensure both numerical-library backends are loaded
    from threadpoolctl import threadpool_info
    build = np.show_config(mode="dicts")
    pools = threadpool_info()
    return {"requested_threads_per_chain":1, "requested_environment":requested,
            "observed_pools":pools, "numpy_blas":build.get("Build Dependencies", {}).get("blas", {}),
            "accelerate_thread_count_observed":False,
            "observation_scope":"threadpoolctl-supported loaded libraries only; Accelerate thread count is not observed"}



def configure_runtime(cache_directory):
    """Use a private compiler cache and the reviewed macOS linker workaround.

    PyTensor 2.31.7 incorrectly treats macOS 15+ as requiring the linker
    library argument ``-ld64``. The narrow adapter removes that argument;
    it does not modify any model expression, prior, data or sampler setting.
    """
    global _RUNTIME_SETUP
    bootstrap()
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
                     'macos_ld64_adapter':adapter, 'thread_controls':thread_observations(),
                     'compiler_identity':_compiler_identity({'cxx':pytensor.config.cxx})}
    return dict(_RUNTIME_SETUP)


def model_runtime_identity():
    controls = bootstrap()
    packages = ('numpy','pandas','scipy','patsy','statsmodels','pymc','pytensor',
                'arviz','xarray','threadpoolctl')
    settings = _compiler_settings()
    import numpy as np
    build = np.show_config(mode='dicts')
    import scipy.linalg
    from threadpoolctl import threadpool_info
    blas_libraries = [{k:v for k,v in pool.items() if k!='num_threads'}
                      for pool in threadpool_info() if pool.get('user_api')=='blas']
    blas_libraries.sort(key=lambda item:item.get('filepath',''))
    # Cache directories and transient pool loads are excluded; toolchain/backend
    # identity and requested controls are invariant across private run locations.
    return {'python':platform.python_version(), 'system':platform.system(),
            'os_release':platform.release(), 'os_version':platform.version(),
            'macos_version':platform.mac_ver()[0], 'machine':platform.machine(),
            'packages':{name:importlib.metadata.version(name) for name in packages},
            'compiler':_compiler_identity(settings), 'pytensor_configuration':settings,
            'macos_ld64_adapter':platform.system()=='Darwin' and importlib.metadata.version('pytensor')=='2.31.7',
            'numpy_blas':build.get('Build Dependencies', {}).get('blas', {}),
            'numpy_lapack':build.get('Build Dependencies', {}).get('lapack', {}),
            'loaded_blas_libraries':blas_libraries,
            'requested_thread_environment':controls,
            'pytensor_environment_flags':','.join(x for x in os.environ.get('PYTENSOR_FLAGS','').split(',')
                if x.split('=',1)[0].strip() not in {'compiledir','base_compiledir'}),
            'compile_environment':{k:os.environ.get(k) for k in
                ('CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','SDKROOT','MACOSX_DEPLOYMENT_TARGET')},
            'accelerate_thread_observation':'unsupported by threadpoolctl; no observed one-thread assertion'}
