"""Selected X6 shared-mechanics/first-TB joint graphs, specification 1.0.0."""
from __future__ import annotations
from typing import Mapping
import numpy as np
from .mmef import MECHANICS, SPEC_VERSION, _validate_shared_contract

TARGETS = {
    f'{outcome}_before{threshold}': {'column':column, 'contract_key':f'x6_joint/{column}/{threshold}',
        'source_target':outcome,'threshold':threshold,'outcome':label,'unit':unit}
    for outcome,column,label,unit in [('x6_z','x6_sa_z','X6 South African reference z-score','reference SD'),('x6_native','reactance_6hz','X6','hPa*s/L')]
    for threshold in (1,5)
}
DEFAULT_TARGETS = tuple(TARGETS)
CONTRASTS = {
    'Total post-TB difference':('total_target','total'),
    'Direct post-TB component':('direct_target','component'),
    'Shared-mechanics component':('mediated_target','component'),
}
POSTERIOR_VARIABLES = ('mechanics','loading_mechanics','sigma_mechanics','shift_mechanics','loading_target_mechanics','direct_target','sigma_target','log_baseline_hazard','assoc_mechanics','hazard_male','mediated_target','total_target')


def fit_spec(target):
    if target not in TARGETS:
        raise ValueError('Unsupported X6 joint target.')
    chosen=TARGETS[target];threshold=chosen['threshold']
    breaks=[v for v in [0.,1.,3.,5.] if v<=threshold]
    labels=[f'{a:g}-<{b:g}' for a,b in zip(breaks[:-1],breaks[1:])]
    return {**chosen,'target':target,'family':'x6','model_id':'X6-JOINT','specification_version':SPEC_VERSION,
        'orientation':1.,'result_rows':3,'hazard_rows':0,'target_age_bounds':[3.,17.],
        'mechanics_indicators':list(MECHANICS),'mechanics_orientation':dict(MECHANICS),
        'nuisance_formula':'1+C(visit_label)+height_z+male',
        'nuisance_training':'pre/no-TB observations, separately for each process',
        'scale_method':'sqrt(training residual SSE / (training observations minus nuisance design column count))',
        'hazard_breaks':breaks,'hazard_labels':labels,
        'log_hazard_prior_mean':np.log(np.repeat(.03,len(labels))).tolist(),'log_hazard_prior_sd':1.25,
        'priors':{'unit_normal_child':['mechanics'],'half_normal_sd1':['loading_mechanics','sigma_mechanics','sigma_target'],'normal_sd1':['loading_target_mechanics'],'normal_sd0_75':['shift_mechanics','direct_target','assoc_mechanics','hazard_male']},
        'target_mean':'loading_target_mechanics*(mechanics+shift_mechanics*post_target_tb)+direct_target*post_target_tb',
        'hazard_log_mean':'log_baseline_hazard+assoc_mechanics*mechanics+hazard_male*male',
        'contrasts':{k:list(v) for k,v in CONTRASTS.items()},
        'posterior_variables':list(POSTERIOR_VARIABLES),'participant_posterior_variables':['mechanics'],
        'sampling':{'draws':1200,'tune':1800,'chains':4,'cores':2,'target_accept':.93,'random_seed':20260815+threshold,'init':'jitter+adapt_diag','initvals':None},
        'sampling_provenance':'Source wrapper defaults; selected saved arrays confirm four chains and 1200 retained draws. Historical seed, tune and target acceptance require separately retained execution evidence.',
        'fresh_sampling':'Configured separately with full-posterior diagnostic gates and a recorded seed offset relative to the wrapper default.'}


def validate_contract(contract,target):
    return _validate_shared_contract(contract,fit_spec(target))

def build_model(prepared: Mapping, target: str):
    """Build the explicit threshold-specific X6 graph without fitting."""
    validate_contract(prepared,target)
    labels=fit_spec(target)['hazard_labels']
    import pymc as pm
    import pytensor.tensor as pt
    frames = prepared["frames"]
    children = prepared["children"]
    event = prepared["event"]
    n_children = len(children)
    mechanics_keys = list(MECHANICS)
    coords = {
        "child": np.arange(n_children),
        "mechanics_indicator": mechanics_keys,
        "hazard_interval": labels,
    }
    with pm.Model(coords=coords) as model:
        mechanics = pm.Normal("mechanics", 0.0, 1.0, dims="child")
        loading_mechanics = pm.HalfNormal(
            "loading_mechanics", 1.0, dims="mechanics_indicator"
        )
        sigma_mechanics = pm.HalfNormal(
            "sigma_mechanics", 1.0, dims="mechanics_indicator"
        )
        shift_mechanics = pm.Normal("shift_mechanics", 0.0, 0.75)

        for j, key in enumerate(mechanics_keys):
            frame = frames[key]
            idx = frame["child_index"].to_numpy(dtype=int)
            post = frame["post_target_tb"].to_numpy(dtype=float)
            current = mechanics[idx] + shift_mechanics * post
            pm.Normal(
                f"obs_{key}",
                mu=loading_mechanics[j] * current,
                sigma=sigma_mechanics[j],
                observed=frame["y_std"].to_numpy(dtype=float),
            )

        target = frames["target"]
        target_idx = target["child_index"].to_numpy(dtype=int)
        target_post = target["post_target_tb"].to_numpy(dtype=float)
        loading_target_mechanics = pm.Normal("loading_target_mechanics", 0.0, 1.0)
        direct_target = pm.Normal("direct_target", 0.0, 0.75)
        sigma_target = pm.HalfNormal("sigma_target", 1.0)
        current_target_mechanics = mechanics[target_idx] + shift_mechanics * target_post
        target_mu = (
            loading_target_mechanics * current_target_mechanics
            + direct_target * target_post
        )
        pm.Normal(
            "obs_target",
            mu=target_mu,
            sigma=sigma_target,
            observed=target["y_std"].to_numpy(dtype=float),
        )

        log_baseline_hazard = pm.Normal(
            "log_baseline_hazard",
            mu=np.log(np.repeat(0.03, len(labels))),
            sigma=1.25,
            dims="hazard_interval",
        )
        assoc_mechanics = pm.Normal("assoc_mechanics", 0.0, 0.75)
        hazard_male = pm.Normal("hazard_male", 0.0, 0.75)
        eta = assoc_mechanics * mechanics + hazard_male * children["male"].to_numpy(dtype=float)
        exposure = pt.as_tensor_variable(event["exposure"])
        event_interval = pt.as_tensor_variable(event["event_interval"])
        log_hazard = log_baseline_hazard[None, :] + eta[:, None]
        pm.Potential(
            "first_tb_loglik",
            pt.sum(event_interval * log_hazard - exposure * pt.exp(log_hazard)),
        )

        mediated_target = pm.Deterministic(
            "mediated_target", loading_target_mechanics * shift_mechanics
        )
        pm.Deterministic("total_target", direct_target + mediated_target)
    return model
