"""Recorded sampling for the two selected wheeze joint scenarios."""
from __future__ import annotations
import time
import numpy as np
from .models.wheeze import (TARGETS, DEFAULT_TARGETS, build_model, contract_key,
                            fit_spec, validate_contract)
from .sampling import validate_sampling as _validate_primary_sampling

DEFAULT_SAMPLING = {'schema_version': 1, 'draws': 3200, 'tune': 2200, 'chains': 4,
                    'cores': 2, 'target_accept': .95, 'max_treedepth': 10, 'seed_offset': 104729}


def validate_sampling(settings):
    settings = _validate_primary_sampling(settings)
    if settings['draws'] < 1600 or settings['tune'] < 2200:
        raise ValueError('Fresh wheeze fits require at least 1600 draws and 2200 tuning iterations.')
    return settings


def resolved_sampling(target, settings):
    settings = validate_sampling(settings)
    source = fit_spec(target)['sampling']
    controls = {k: v for k, v in settings.items() if k not in {'schema_version', 'seed_offset'}}
    controls.update(random_seed=source['random_seed']+settings['seed_offset'],
                    init=source['init'], initvals=source['initvals'], nuts_sampler='pymc',
                    blas_cores=settings['cores'], progressbar=False, discard_tuned_samples=True)
    return controls


def expected_posterior_shapes(contract, target, chains, draws):
    if type(chains) is not int or type(draws) is not int or min(chains, draws) < 1:
        raise ValueError('Posterior dimensions must be positive integers.')
    info = validate_contract(contract, target)
    shapes = {name: (chains, draws) for name in fit_spec(target)['posterior_variables']}
    shapes['latent_propensity'] += (info['joint_children'],)
    shapes['scheduled_age'] += (5,)
    for name in ('scheduled_covariate', 'episode_covariate', 'tb_covariate'):
        shapes[name] += (len(info['covariates']),)
    for name in ('episode_log_baseline_rate', 'tb_log_baseline_hazard'):
        shapes[name] += (8,)
    for name in ('scheduled_infancy_post_log_or', 'scheduled_other_post_log_or',
                 'episode_infancy_post_log_rr', 'episode_other_post_log_rr',
                 'scheduled_infancy_post_or', 'scheduled_other_post_or',
                 'episode_infancy_post_rr', 'episode_other_post_rr'):
        shapes[name] += (4,)
    return shapes


def sample_contract(contract, target, settings, cache_directory):
    """Sample once and retain full private posterior and numerical sampler stats."""
    from .model_runtime import configure_runtime, model_runtime_identity
    runtime = configure_runtime(cache_directory)
    import pymc as pm
    from threadpoolctl import threadpool_limits
    controls = resolved_sampling(target, settings)
    expected = expected_posterior_shapes(contract, target, controls['chains'], controls['draws'])
    started = time.monotonic()
    with threadpool_limits(limits=1), build_model(contract, target):
        trace = pm.sample(draws=controls['draws'], tune=controls['tune'], chains=controls['chains'],
            cores=controls['cores'], random_seed=controls['random_seed'], target_accept=controls['target_accept'],
            init=controls['init'], initvals=controls['initvals'], nuts={'max_treedepth': controls['max_treedepth']},
            nuts_sampler='pymc', blas_cores=controls['blas_cores'], progressbar=False,
            discard_tuned_samples=True, return_inferencedata=True, compute_convergence_checks=True,
            idata_kwargs={'log_likelihood': False})
    posterior = {name: np.asarray(v) for name, v in trace.posterior.data_vars.items()}
    stats = {name: np.asarray(v) for name, v in trace.sample_stats.data_vars.items()
             if not np.asarray(v).dtype.hasobject}
    if set(posterior) != set(expected) or any(posterior[name].shape != shape for name, shape in expected.items()):
        raise ValueError('Wheeze posterior inventory or dimensions differ from the model specification.')
    return posterior, stats, {'target': target, 'controls': controls, 'runtime': model_runtime_identity(),
        'runtime_setup': runtime, 'elapsed_seconds': time.monotonic()-started,
        'realized_chains': trace.posterior.sizes['chain'], 'realized_draws_per_chain': trace.posterior.sizes['draw'],
        'realized_tuning_steps': trace.sample_stats.attrs.get('tuning_steps'),
        'posterior_variables': list(posterior), 'sample_stat_variables': list(stats)}
