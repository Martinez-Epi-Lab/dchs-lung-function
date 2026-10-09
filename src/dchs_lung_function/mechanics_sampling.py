"""Four-chain mechanics sampling with complete private posterior retention."""
from __future__ import annotations

import time
import numpy as np
from .sampling import validate_sampling
from .models.mechanics import build_model, fit_spec


def resolved_sampling(target, settings):
    settings = validate_sampling(settings)
    selected = fit_spec(target)['sampling']
    controls = {k:v for k,v in settings.items() if k not in {'schema_version', 'seed_offset'}}
    controls.update(random_seed=selected['random_seed']+settings['seed_offset'],
                    init=selected['init'], initvals=selected.get('initvals'),
                    nuts_sampler='pymc', blas_cores=settings['cores'], progressbar=False,
                    discard_tuned_samples=True)
    return controls


def sample_contract(contract, target, settings, cache_directory):
    from .model_runtime import configure_runtime, model_runtime_identity
    runtime = configure_runtime(cache_directory)
    import pymc as pm
    from threadpoolctl import threadpool_limits
    controls = resolved_sampling(target, settings)
    started = time.monotonic()
    with threadpool_limits(limits=1), build_model(contract, target):
        trace = pm.sample(draws=controls['draws'], tune=controls['tune'],
            chains=controls['chains'], cores=controls['cores'], random_seed=controls['random_seed'],
            target_accept=controls['target_accept'], init=controls['init'], initvals=controls['initvals'],
            nuts={'max_treedepth':controls['max_treedepth']}, nuts_sampler='pymc',
            blas_cores=controls['blas_cores'], progressbar=False, discard_tuned_samples=True,
            return_inferencedata=True, compute_convergence_checks=True,
            idata_kwargs={'log_likelihood':False})
    posterior = {n:np.asarray(v) for n,v in trace.posterior.data_vars.items()}
    stats = {n:np.asarray(v) for n,v in trace.sample_stats.data_vars.items() if not np.asarray(v).dtype.hasobject}
    shape = (controls['chains'], controls['draws'])
    if any(v.shape[:2] != shape for v in posterior.values()):
        raise ValueError('Mechanics posterior dimensions differ from requested sampling.')
    return posterior, stats, {'target':target, 'controls':controls, 'runtime':model_runtime_identity(),
        'runtime_setup':runtime, 'elapsed_seconds':time.monotonic()-started,
        'realized_chains':trace.posterior.sizes['chain'],
        'realized_draws_per_chain':trace.posterior.sizes['draw'],
        'realized_tuning_steps':trace.sample_stats.attrs.get('tuning_steps'),
        'posterior_variables':list(posterior), 'sample_stat_variables':list(stats)}
