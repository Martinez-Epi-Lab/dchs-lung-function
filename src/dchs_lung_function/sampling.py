"""Explicit four-chain primary sampling, preserving complete private draws."""
from __future__ import annotations

import time
import numpy as np
from .models.spirometry import build_model, fit_spec

FIELDS = {'schema_version','draws','tune','chains','cores','target_accept','max_treedepth','seed_offset'}


def validate_sampling(value):
    if not isinstance(value,dict) or set(value)!=FIELDS:
        raise ValueError('Primary sampling configuration has unexpected or missing fields.')
    for name in ('schema_version','draws','tune','chains','cores','max_treedepth','seed_offset'):
        if type(value[name]) is not int:
            raise ValueError('Primary sampling integer controls must be integers.')
    if value['schema_version']!=1 or value['chains']!=4:
        raise ValueError('Primary sampling requires schema version 1 and four chains.')
    if value['draws']<1200 or value['tune']<1600 or not 1<=value['cores']<=4:
        raise ValueError('Primary sampling requires at least 1200 draws and 1600 tuning iterations.')
    accept = value['target_accept']
    if isinstance(accept,bool) or not isinstance(accept,(float,int)) or not .95<=accept<1:
        raise ValueError('Target acceptance must be at least 0.95 and below 1.')
    if not 10<=value['max_treedepth']<=15 or not 0<=value['seed_offset']<2**30:
        raise ValueError('Unsupported tree depth or seed offset.')
    return dict(value)


def resolved_sampling(target, settings):
    settings = validate_sampling(settings)
    selected = fit_spec(target)['sampling']
    result = {k:v for k,v in settings.items() if k not in {'schema_version','seed_offset'}}
    result.update(random_seed=selected['random_seed']+settings['seed_offset'],
                  init=selected['init'], initvals=selected['initvals'],
                  nuts_sampler='pymc', blas_cores=settings['cores'],
                  progressbar=False, discard_tuned_samples=True)
    return result


def sample_contract(contract, target, settings, cache_directory):
    """Fit once; return all posterior arrays and sampler statistics privately."""
    from .model_runtime import configure_runtime,model_runtime_identity
    runtime = configure_runtime(cache_directory)
    import pymc as pm
    from threadpoolctl import threadpool_limits
    controls = resolved_sampling(target,settings)
    started = time.monotonic()
    with threadpool_limits(limits=1), build_model(contract,target):
        trace = pm.sample(draws=controls['draws'], tune=controls['tune'],
            chains=controls['chains'], cores=controls['cores'],
            random_seed=controls['random_seed'], target_accept=controls['target_accept'],
            init=controls['init'], initvals=controls['initvals'],
            nuts={'max_treedepth':controls['max_treedepth']},
            nuts_sampler='pymc', blas_cores=controls['blas_cores'],
            progressbar=False, discard_tuned_samples=True, return_inferencedata=True,
            compute_convergence_checks=True, idata_kwargs={'log_likelihood':False})
    posterior = {name:np.asarray(value) for name,value in trace.posterior.data_vars.items()}
    stats = {name:np.asarray(value) for name,value in trace.sample_stats.data_vars.items()
             if not np.asarray(value).dtype.hasobject}
    if trace.posterior.sizes['chain']!=controls['chains'] or trace.posterior.sizes['draw']!=controls['draws']:
        raise ValueError('Realized posterior dimensions differ from requested sampling.')
    if any(value.shape[:2]!=(controls['chains'],controls['draws']) for value in posterior.values()):
        raise ValueError('Posterior variables have inconsistent chain or draw dimensions.')
    return posterior,stats,{'target':target,'controls':controls,
        'runtime':model_runtime_identity(), 'runtime_setup':runtime,
        'elapsed_seconds':time.monotonic()-started,
        'realized_chains':trace.posterior.sizes['chain'],
        'realized_draws_per_chain':trace.posterior.sizes['draw'],
        'realized_tuning_steps':trace.sample_stats.attrs.get('tuning_steps'),
        'posterior_variables':list(posterior), 'sample_stat_variables':list(stats)}
