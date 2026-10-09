"""Draw-wise mechanics contrasts and aggregate support on explicit scales."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .primary_results import (_scalar_draws, _prepared_summary, _sample_counts, _summarize)
from ..models.mechanics import CONTRASTS, fit_spec, validate_contract


@dataclass
class MechanicsResults:
    estimates: pd.DataFrame
    hazard: pd.DataFrame
    sample_counts: pd.DataFrame
    metadata: dict


def _derived(posterior, contract, target):
    spec = fit_spec(target)
    validate_contract(contract, target)
    posterior = posterior.posterior if hasattr(posterior, 'posterior') else posterior
    scale, weights, frame = _prepared_summary(contract)
    middle, extra, later = [_scalar_draws(posterior, n) for n in
                            ('shift_1to5', 'shift_infancy_additional', 'shift_late')]
    if len({d.shape for d in (middle, extra, later)}) != 1:
        raise ValueError('Mechanics shifts have inconsistent chain/draw dimensions.')
    infant = middle+extra
    values = {'TB <1':infant, 'TB 1-<5':middle, 'TB >=5':later,
              'TB <5 weighted':(weights[0]*infant+weights[1]*middle)/weights[:2].sum(),
              'Any TB weighted':(weights[0]*infant+weights[1]*middle+weights[2]*later)/weights.sum(),
              'Infancy minus 1-<5':extra, 'Later minus infancy':later-infant,
              'Later minus 1-<5':later-middle}
    if list(values) != list(CONTRASTS):
        raise ValueError('Mechanics contrast inventory differs from its declared specification.')
    checked = []
    for label, variable in CONTRASTS.items():
        if not np.isfinite(values[label]*scale/spec['orientation']).all():
            raise ValueError('Mechanics contrasts became nonfinite after rescaling.')
        if variable in posterior:
            stored = _scalar_draws(posterior, variable)
            if stored.shape != values[label].shape or not np.allclose(stored, values[label], atol=1e-12, rtol=1e-11):
                raise ValueError('Saved mechanics deterministic contrasts disagree with their primitives.')
            checked.append(variable)
    return posterior, spec, values, scale, weights, frame, checked


def derive_mechanics_contrast_draws(posterior, contract, target):
    """Native-scale arrays preserve chain/draw axes and reverse adverse signs."""
    _, spec, values, scale, _, _, _ = _derived(posterior, contract, target)
    return {label:d*scale/spec['orientation'] for label,d in values.items()}


def derive_hazard_draws(posterior):
    posterior = posterior.posterior if hasattr(posterior, 'posterior') else posterior
    association = _scalar_draws(posterior, 'assoc_outcome')
    tau = _scalar_draws(posterior, 'tau_child')
    if association.shape != tau.shape or (tau <= 0).any():
        raise ValueError('Mechanics hazard association requires matching positive latent scales.')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = np.exp(association*tau)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError('Mechanics hazard ratios must be finite and positive.')
    variable = 'hazard_hr_per_between_child_sd'
    if variable in posterior:
        stored = _scalar_draws(posterior, variable)
        if stored.shape != result.shape or not np.allclose(stored, result, rtol=1e-11, atol=1e-12):
            raise ValueError('Saved mechanics hazard ratios disagree with their declared transformation.')
    return result


def summarize_mechanics(posterior, contract, target, *, compute_mcse=True):
    posterior, spec, values, scale, weights, frame, checked = _derived(posterior, contract, target)
    rows = []
    for label, standardized in values.items():
        native = standardized*scale/spec['orientation']
        reported = native*spec['reporting_multiplier']
        summaries = [_summarize(d, compute_mcse) for d in (standardized, native, reported)]
        row = {'target':target, 'outcome':spec['outcome'], 'contrast':label,
               'unit':spec['unit'], 'reported_unit':spec['reporting_unit'],
               'orientation':spec['orientation'], 'residual_scale':scale,
               'reporting_multiplier':spec['reporting_multiplier'],
               'probability_worse':float(np.mean(standardized > 0))}
        for prefix, summary in zip(('', 'native_', 'reported_'), summaries):
            row.update({prefix+k:v for k,v in summary.items()})
        rows.append(row)
    hr = derive_hazard_draws(posterior)
    if hr.shape != next(iter(values.values())).shape:
        raise ValueError('Mechanics hazard and outcome draws have inconsistent dimensions.')
    hazard = {'target':target, 'outcome':spec['outcome'],
              'contrast':'First-TB hazard per between-child SD poorer mechanics',
              'unit':'hazard ratio', **_summarize(hr, compute_mcse)}
    counts = _sample_counts(contract)
    counts.insert(0, 'target', target)
    return MechanicsResults(pd.DataFrame(rows), pd.DataFrame([hazard]), counts,
        {'target':target, 'outcome':spec['outcome'], 'native_unit':spec['unit'],
         'reported_unit':spec['reporting_unit'], 'orientation':spec['orientation'],
         'residual_scale':scale, 'reporting_multiplier':spec['reporting_multiplier'],
         'contrast_weights':weights.tolist(), 'n_observations':len(frame),
         'n_children':int(frame.pid_child.nunique()), 'contrasts':list(CONTRASTS),
         'deterministic_variables_checked':checked, 'interval':'equal-tailed 95% posterior',
         'quantile_method':'linear', 'mcse_computed':bool(compute_mcse),
         'hazard_transform':'exp(assoc_outcome*tau_child)',
         'acceptance':'diagnostics and independent scientific verification are separate'})
