"""Recorded four-chain sampling for the ten designated sensitivity models."""
from __future__ import annotations
import time
import numpy as np
from .models.sensitivity import (TARGETS, DEFAULT_TARGETS, fit_spec, contract_key,
                                 validate_contract, build_model)
from .sampling import validate_sampling as _validate_primary_sampling


def validate_sampling(settings):
    if not isinstance(settings, dict):
        raise ValueError('Sensitivity sampling configuration must be an object.')
    overrides = settings.get('draws_by_target', {})
    base = {k:v for k,v in settings.items() if k != 'draws_by_target'}
    settings = _validate_primary_sampling(base)
    if settings['draws'] < 3200 or settings['tune'] < 2200:
        raise ValueError('Sensitivity fits require at least 3200 draws and 2200 tuning iterations.')
    if (not isinstance(overrides, dict) or not set(overrides).issubset(DEFAULT_TARGETS)
            or any(type(v) is not int or v < settings['draws'] for v in overrides.values())):
        raise ValueError('Sensitivity draw overrides require selected targets and at least the base draws.')
    if overrides:
        settings['draws_by_target'] = dict(overrides)
    return settings


def resolved_sampling(target, settings):
    settings = validate_sampling(settings)
    selected = fit_spec(target)['sampling']
    controls = {k:v for k,v in settings.items() if k not in {'schema_version','seed_offset','draws_by_target'}}
    controls['draws'] = settings.get('draws_by_target', {}).get(target, settings['draws'])
    controls.update(random_seed=selected['random_seed']+settings['seed_offset'],
        init='adapt_diag', initvals=[{'loading_ratio_mechanics':value,
            'loading_mechanics':[value]*4, 'sigma_ratio':.9} for value in (.25,.35,.45,.55)],
        nuts_sampler='pymc', blas_cores=settings['cores'], progressbar=False,
        discard_tuned_samples=True)
    return controls


def expected_posterior_shapes(contract, target, chains, draws):
    if type(chains) is not int or type(draws) is not int or min(chains,draws) < 1:
        raise ValueError('Sensitivity posterior dimensions must be positive integers.')
    model = build_model(contract,target)
    shapes = {v.name:(chains,draws,*[len(model.coords[d]) for d in model.named_vars_to_dims.get(v.name,())])
              for v in [*model.free_RVs,*model.deterministics]}
    if set(shapes) != set(fit_spec(target)['posterior_variables']):
        raise ValueError('Sensitivity posterior variables differ from the declared model inventory.')
    return shapes


def sample_contract(contract, target, settings, cache_directory):
    """Fit once and retain every posterior variable, including all child effects."""
    from .model_runtime import configure_runtime, model_runtime_identity
    runtime = configure_runtime(cache_directory)
    import pymc as pm
    from threadpoolctl import threadpool_limits
    controls = resolved_sampling(target,settings)
    expected = expected_posterior_shapes(contract,target,controls['chains'],controls['draws'])
    started = time.monotonic()
    with threadpool_limits(limits=1), build_model(contract,target):
        trace = pm.sample(draws=controls['draws'],tune=controls['tune'],chains=controls['chains'],
            cores=controls['cores'],random_seed=controls['random_seed'],target_accept=controls['target_accept'],
            init=controls['init'],initvals=controls['initvals'],nuts={'max_treedepth':controls['max_treedepth']},
            nuts_sampler='pymc',blas_cores=controls['blas_cores'],progressbar=False,
            discard_tuned_samples=True,return_inferencedata=True,compute_convergence_checks=True,
            idata_kwargs={'log_likelihood':False})
    posterior = {name:np.asarray(v) for name,v in trace.posterior.data_vars.items()}
    stats = {name:np.asarray(v) for name,v in trace.sample_stats.data_vars.items() if not np.asarray(v).dtype.hasobject}
    if set(posterior) != set(expected) or any(posterior[n].shape != shape for n,shape in expected.items()):
        raise ValueError('Sensitivity posterior inventory or dimensions differ from the model specification.')
    return posterior,stats,{'target':target,'controls':controls,'runtime':model_runtime_identity(),
        'runtime_setup':runtime,'elapsed_seconds':time.monotonic()-started,
        'realized_chains':trace.posterior.sizes['chain'],'realized_draws_per_chain':trace.posterior.sizes['draw'],
        'realized_tuning_steps':trace.sample_stats.attrs.get('tuning_steps'),
        'posterior_variables':list(posterior),'sample_stat_variables':list(stats)}
