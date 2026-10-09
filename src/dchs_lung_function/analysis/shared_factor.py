"""Selected shared-mechanics associations derived from primary posterior draws.

These six results reuse the primary spirometry models. One unit of the primary
mechanics factor is one model-defined between-child SD. Negating its target
loading and multiplying once by the target residual scale gives the association
with one SD poorer mechanics in GLI z-score units. No sampling or file I/O occurs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TARGETS = (
    'ratio_pre_z', 'ratio_post_z', 'fev1_pre_z', 'fev1_post_z', 'fvc_pre_z', 'fvc_post_z',
)
_TARGETS = {
    f'{name}_{stage}_z': {
        'column': column + ('_post' if stage == 'post' else '') + '_gli_z',
        'outcome': outcome, 'stage': 'Pre-bronchodilator' if stage == 'pre' else 'Post-bronchodilator',
    }
    for name, column, outcome in [('ratio', 'fev1_fvc', 'FEV1/FVC'), ('fev1', 'fev1', 'FEV1'), ('fvc', 'fvc', 'FVC')]
    for stage in ('pre', 'post')
}


class SharedFactorError(ValueError):
    """A shared-factor summary lacks a valid primary target or global posterior."""


@dataclass(frozen=True)
class SharedFactorResult:
    estimates: pd.DataFrame
    metadata: dict


def _contract_details(contract, target):
    if target not in DEFAULT_TARGETS:
        raise SharedFactorError('Shared-factor reporting is restricted to the six selected GLI z-score targets.')
    try:
        audit = contract['audit']
        target_audit = audit['outcomes']['target']
        frame = contract['frames']['target']
        scale_value = target_audit['scale']
        if isinstance(scale_value, (bool, complex, np.bool_, np.complexfloating)):
            raise SharedFactorError('The target residual scale must be a real positive number.')
        scale = float(scale_value)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, SharedFactorError):
            raise
        raise SharedFactorError('A verified primary target frame and residual-scale audit are required.') from None
    if audit.get('family') != 'primary' or audit.get('target') != _TARGETS[target]['column']:
        raise SharedFactorError('The selected shared-factor target does not match its primary M04 contract.')
    if not np.isfinite(scale) or scale <= 0:
        raise SharedFactorError('The target residual scale must be finite and positive.')
    if not isinstance(frame, pd.DataFrame) or not {'pid_child', 'outcome_value'}.issubset(frame) or frame.empty:
        raise SharedFactorError('The primary target must contain nonempty child and outcome fields.')
    if frame.pid_child.isna().any() or frame.pid_child.astype('string').str.strip().eq('').any():
        raise SharedFactorError('The primary target contains a missing child linkage key.')
    values = frame.outcome_value.to_numpy()
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise SharedFactorError('The primary target outcome must contain finite real numeric values.')
    counts = {'n_observations': len(frame), 'n_children': int(frame.pid_child.nunique())}
    for name, observed in counts.items():
        declared = target_audit.get(name)
        if isinstance(declared, (bool, np.bool_)) or not isinstance(declared, (int, np.integer)) or declared != observed:
            raise SharedFactorError('The primary target sample counts disagree with its audit.')
    return scale, counts


def _loading_draws(posterior):
    source = posterior.posterior if hasattr(posterior, 'posterior') else posterior
    try:
        value = source['loading_ratio_mechanics']
    except (KeyError, TypeError):
        raise SharedFactorError('The primary posterior lacks loading_ratio_mechanics.') from None
    if hasattr(value, 'dims'):
        if set(value.dims) != {'chain', 'draw'}:
            raise SharedFactorError('The shared target loading must contain only chain and draw dimensions.')
        value = value.transpose('chain', 'draw')
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise SharedFactorError('The shared target loading requires finite real chain-by-draw values.')
    return array.astype(float, copy=False)


def derive_shared_factor_draws(posterior, contract, target: str) -> np.ndarray:
    """Return reported-scale global draws, retaining their chain-by-draw axes."""
    scale, _ = _contract_details(contract, target)
    loading = _loading_draws(posterior)
    with np.errstate(over='ignore', invalid='ignore'):
        draws = -loading * scale
    if not np.isfinite(draws).all():
        raise SharedFactorError('The transformed shared-factor draws are nonfinite.')
    return draws


def summarize_shared_factor(posterior, contract, target: str, *, compute_mcse: bool = True) -> SharedFactorResult:
    """Return a full-precision selected estimate and an explicit derivation audit.

    ``compute_mcse=False`` is for algebra tests and leaves Monte Carlo errors
    missing. The calling runner verifies source-fit diagnostics and provenance;
    this deterministic summarizer does not certify a posterior as converged.
    """
    scale, counts = _contract_details(contract, target)
    draws = derive_shared_factor_draws(posterior, contract, target)
    flattened = draws.reshape(-1)
    lower, median, upper = np.quantile(flattened, [.025, .5, .975], method='linear')
    mcse = {'mcse_mean': np.nan, 'mcse_lower': np.nan, 'mcse_upper': np.nan}
    if compute_mcse:
        try:
            import arviz as az
        except ImportError:
            raise SharedFactorError('Shared-factor Monte Carlo errors require the configured ArviZ dependency.') from None
        mcse = {
            'mcse_mean': float(az.mcse(draws, method='mean')),
            'mcse_lower': float(az.mcse(draws, method='quantile', prob=.025)),
            'mcse_upper': float(az.mcse(draws, method='quantile', prob=.975)),
        }
        if not np.isfinite(list(mcse.values())).all() or any(v < 0 for v in mcse.values()):
            raise SharedFactorError('Shared-factor Monte Carlo errors are not finite and nonnegative.')
    info = _TARGETS[target]
    row = {
        'target': target, 'outcome': info['outcome'], 'stage': info['stage'], 'unit': 'GLI z-score',
        'contrast': 'One model-defined SD poorer combined respiratory mechanics',
        'mean': float(flattened.mean()), 'median': float(median), 'lower': float(lower), 'upper': float(upper),
        'probability_below_zero': float(np.mean(flattened < 0)),
        'probability_above_zero': float(np.mean(flattened > 0)),
        'probability_equal_zero': float(np.mean(flattened == 0)), **mcse,
        'residual_scale': scale, 'n_outcome_children': counts['n_children'],
        'n_outcome_measurements': counts['n_observations'], 'figure_s5': True,
    }
    metadata = {
        'target': target, 'target_column': info['column'], 'source_model': 'SPIRO-JOINT-PRIMARY',
        'posterior_variable': 'loading_ratio_mechanics',
        'draw_transform': 'negative loading_ratio_mechanics times target residual scale',
        'factor_sd': 1., 'factor_orientation': 'one model-defined SD poorer respiratory mechanics',
        'unit': 'GLI z-score', 'residual_scale': scale, **counts,
        'chains': draws.shape[0], 'draws_per_chain': draws.shape[1], 'posterior_draws': int(draws.size),
        'interval': 'equal-tailed 95% credible interval', 'quantile_method': 'linear',
        'point_estimator': 'posterior mean', 'mcse_computed': bool(compute_mcse),
        'diagnostic_acceptance': 'not evaluated by derived-result summarizer',
        'new_fit_required': False, 'participant_records_returned': False,
    }
    return SharedFactorResult(pd.DataFrame([row]), metadata)
