"""Draw-wise selected MMEF/X6 effects, with model-specific native scales."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .primary_results import _scalar_draws,_sample_counts,_summarize,_summarize_ratio
from ..supporting_sampling import fit_spec,validate_contract

HAZARDS={'Higher (better) jointly estimated mechanics factor':'assoc_mechanics','Male sex':'hazard_male'}
HAZARD_COLUMNS=['target','outcome','estimand','estimand_type','unit','mean','median','lower','upper',
                'probability_below_one','probability_above_one','probability_equal_one','mcse_mean']


@dataclass
class SupportingResults:
    estimates: pd.DataFrame
    hazard: pd.DataFrame
    sample_counts: pd.DataFrame
    metadata: dict


def _posterior(value):
    return value.posterior if hasattr(value,'posterior') else value


def _derived(posterior,contract,target):
    posterior=_posterior(posterior);spec=fit_spec(target);validated=validate_contract(contract,target)
    if spec['family']=='mmef':
        names=['direct_ratio','direct_ratio_infancy','loading_ratio_mechanics','shift_mechanics','shift_mechanics_infancy']
    else:
        names=['direct_target','loading_target_mechanics','shift_mechanics']
    d={name:_scalar_draws(posterior,name) for name in names}
    if len({a.shape for a in d.values()})!=1:
        raise ValueError('Supporting primitives have inconsistent chain/draw dimensions.')
    if spec['family']=='mmef':
        middle=d['direct_ratio']+d['loading_ratio_mechanics']*d['shift_mechanics']
        infant=d['direct_ratio']+d['direct_ratio_infancy']+d['loading_ratio_mechanics']*(d['shift_mechanics']+d['shift_mechanics_infancy'])
        weights=validated['weights'];w=weights[0]/sum(weights[:2])
        values=[w*infant+(1-w)*middle,infant,middle,infant-middle,d['direct_ratio'],d['direct_ratio_infancy']]
    else:
        mediated=d['loading_target_mechanics']*d['shift_mechanics']
        values=[d['direct_target']+mediated,d['direct_target'],mediated]
    contrasts=dict(zip(spec['contrasts'],values));checked=[]
    for label,value in contrasts.items():
        if not np.isfinite(value*validated['target_scale']).all():
            raise ValueError('Supporting contrast is nonfinite after rescaling.')
        variable=spec['contrasts'][label][0]
        if variable in posterior:
            stored=_scalar_draws(posterior,variable)
            if stored.shape!=value.shape or not np.allclose(stored,value,rtol=1e-11,atol=1e-12):
                raise ValueError('A stored supporting deterministic disagrees with its primitive transformation.')
            checked.append(variable)
    return contrasts,spec,validated,checked


def derive_supporting_contrast_draws(posterior,contract,target):
    """Selected source effect labels mapped to native-scale chain/draw arrays."""
    contrasts,_,validated,_=_derived(posterior,contract,target)
    return {label:value*validated['target_scale'] for label,value in contrasts.items()}


def derive_hazard_draws(posterior,target):
    """MMEF source hazard rows; X6 source exported no hazard estimands."""
    if fit_spec(target)['family']=='x6':
        return {}
    posterior=_posterior(posterior);result={}
    for label,variable in HAZARDS.items():
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            value=np.exp(_scalar_draws(posterior,variable))
        if not np.isfinite(value).all() or (value<=0).any():
            raise ValueError('Supporting hazard ratios must be finite and positive.')
        result[label]=value
    if len({v.shape for v in result.values()})!=1:
        raise ValueError('Supporting hazard draws have inconsistent chain/draw dimensions.')
    return result


def summarize_supporting(posterior,contract,target,*,compute_mcse=True):
    contrasts,spec,validated,checked=_derived(posterior,contract,target)
    scale=validated['target_scale'];rows=[]
    for label,value in contrasts.items():
        row={'target':target,'outcome':spec['source_target'],'outcome_label':spec['outcome'],
            'estimand':label,'contrast':label,'estimand_type':spec['contrasts'][label][1],
            'tb_age_cutoff_years':spec['threshold'],'unit':spec['unit'],'orientation':1.,'residual_scale':scale,
            'probability_worse':float(np.mean(value<0)),**_summarize(value*scale,compute_mcse)}
        row.update({'standardized_'+k:v for k,v in _summarize(value,compute_mcse).items()})
        rows.append(row)
    hazards=[]
    for label,value in derive_hazard_draws(posterior,target).items():
        if value.shape!=next(iter(contrasts.values())).shape:
            raise ValueError('Hazard and effect draws have inconsistent chain/draw dimensions.')
        hazards.append({'target':target,'outcome':'First TB before age five','estimand':label,
                        'estimand_type':'hazard association','unit':'hazard ratio',**_summarize_ratio(value,compute_mcse)})
    counts=_sample_counts(contract);counts.insert(0,'target',target)
    return SupportingResults(pd.DataFrame(rows),pd.DataFrame(hazards,columns=HAZARD_COLUMNS),counts,
        {**validated,'outcome':spec['source_target'],'outcome_label':spec['outcome'],'unit':spec['unit'],
         'contrasts':list(contrasts),'hazards':list(derive_hazard_draws(posterior,target)),
         'deterministic_variables_checked':checked,'interval':'equal-tailed 95% posterior',
         'quantile_method':'linear','mcse_computed':bool(compute_mcse),
         'hazard_transform':'exp(association), unit-variance mechanics factor' if hazards else None,
         'reference_group':'pre/no-TB observations after excluding post-TB measurements following first TB at or after cutoff',
         'acceptance':'full posterior diagnostics and independent scientific verification are separate'})
