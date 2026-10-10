"""Explicit sampling and routing for seven supporting MMEF/X6 joint models."""
from __future__ import annotations
import time
import numpy as np
from .models import mmef, x6
from .sampling import validate_sampling as _validate_primary_sampling

TARGETS = {**mmef.TARGETS, **x6.TARGETS}
DEFAULT_TARGETS = tuple(TARGETS)


def _module(target):
    if target not in TARGETS:
        raise ValueError('Unsupported supporting joint-model target.')
    return mmef if target in mmef.TARGETS else x6


def fit_spec(target):
    return _module(target).fit_spec(target)


def contract_key(target):
    return fit_spec(target)['contract_key']


def build_model(contract,target):
    return _module(target).build_model(contract,target)


def validate_contract(contract,target):
    return _module(target).validate_contract(contract,target)


def expected_posterior_shapes(contract,target,chains,draws):
    if type(chains) is not int or type(draws) is not int or min(chains,draws)<1:
        raise ValueError('Posterior dimensions must be positive integers.')
    model=build_model(contract,target)
    shapes={v.name:(chains,draws,*[len(model.coords[d]) for d in model.named_vars_to_dims.get(v.name,())])
            for v in [*model.free_RVs,*model.deterministics]}
    if set(shapes)!=set(fit_spec(target)['posterior_variables']):
        raise ValueError('Supporting model variable inventory differs from its declared specification.')
    return shapes


def validate_sampling(settings):
    """Enforce the shared minimum sampling requirements for supporting models."""
    settings=_validate_primary_sampling(settings)
    if settings['draws']<1600 or settings['tune']<2200:
        raise ValueError('Supporting fresh fits require at least 1600 draws and 2200 tuning iterations.')
    return settings


def resolved_sampling(target,settings):
    settings=validate_sampling(settings); selected=fit_spec(target)['sampling']
    controls={k:v for k,v in settings.items() if k not in {'schema_version','seed_offset'}}
    controls.update(random_seed=selected['random_seed']+settings['seed_offset'],
                    init=selected['init'],initvals=selected['initvals'],nuts_sampler='pymc',
                    blas_cores=settings['cores'],progressbar=False,discard_tuned_samples=True)
    if fit_spec(target)['family']=='mmef':
        # Use the dispersed factor starts already validated for M04. This avoids
        # jitter starting in the collapsed near-zero indicator-loading region;
        # the target loading remains an unconstrained normal model parameter.
        controls.update(init='adapt_diag', initvals=[
            {'loading_ratio_mechanics':value, 'loading_mechanics':[value]*4, 'sigma_ratio':.9}
            for value in (.25,.35,.45,.55)])
    return controls


def sample_contract(contract,target,settings,cache_directory):
    """Fit once and retain every posterior variable, including child-level arrays."""
    from .model_runtime import configure_runtime,model_runtime_identity
    runtime=configure_runtime(cache_directory)
    import pymc as pm
    from threadpoolctl import threadpool_limits
    controls=resolved_sampling(target,settings)
    expected=expected_posterior_shapes(contract,target,controls['chains'],controls['draws'])
    started=time.monotonic()
    with threadpool_limits(limits=1),build_model(contract,target):
        trace=pm.sample(draws=controls['draws'],tune=controls['tune'],chains=controls['chains'],
            cores=controls['cores'],random_seed=controls['random_seed'],target_accept=controls['target_accept'],
            init=controls['init'],initvals=controls['initvals'],nuts={'max_treedepth':controls['max_treedepth']},
            nuts_sampler='pymc',blas_cores=controls['blas_cores'],progressbar=False,
            discard_tuned_samples=True,return_inferencedata=True,compute_convergence_checks=True,
            idata_kwargs={'log_likelihood':False})
    posterior={name:np.asarray(v) for name,v in trace.posterior.data_vars.items()}
    stats={name:np.asarray(v) for name,v in trace.sample_stats.data_vars.items() if not np.asarray(v).dtype.hasobject}
    if set(posterior)!=set(expected) or any(posterior[n].shape!=shape for n,shape in expected.items()):
        raise ValueError('Supporting posterior inventory or dimensions differ from the model specification.')
    return posterior,stats,{'target':target,'controls':controls,'runtime':model_runtime_identity(),
        'runtime_setup':runtime,'elapsed_seconds':time.monotonic()-started,
        'realized_chains':trace.posterior.sizes['chain'],'realized_draws_per_chain':trace.posterior.sizes['draw'],
        'realized_tuning_steps':trace.sample_stats.attrs.get('tuning_steps'),
        'posterior_variables':list(posterior),'sample_stat_variables':list(stats)}
