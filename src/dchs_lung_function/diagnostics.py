"""Strict posterior diagnostics and Monte Carlo-aware contrast comparisons.

Only aggregate diagnostics are returned. Every posterior scalar, including all
child effects, contributes to the acceptance gates. The caller supplies the
three required sampler statistics; optional uncomputed statistics should not
be presented as diagnostic inputs. No fits or file operations occur here.
"""
from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np

DEFAULT_CRITERIA = {
    'min_chains': 4,
    'min_draws': 1200,
    'max_rhat': 1.01,
    'min_bulk_ess': 400.,
    'min_tail_ess': 400.,
    'min_bfmi': .3,
    'max_divergences': 0,
    'max_treedepth_hits': 0,
}
REQUIRED_STATS = ('diverging', 'energy', 'reached_max_treedepth')


def _criteria(criteria):
    if criteria is None:
        return dict(DEFAULT_CRITERIA)
    if not isinstance(criteria, Mapping):
        raise ValueError('Diagnostic criteria must be a mapping.')
    supplied = dict(criteria)
    supplied.pop('scope', None)  # The implementation always uses every scalar.
    if 'divergences' in supplied:
        if 'max_divergences' in supplied and supplied['max_divergences'] != supplied['divergences']:
            raise ValueError('Conflicting divergence criteria.')
        supplied['max_divergences'] = supplied.pop('divergences')
    if set(supplied)-set(DEFAULT_CRITERIA):
        raise ValueError('Unrecognized diagnostic criterion.')
    result = {**DEFAULT_CRITERIA, **supplied}
    for key, value in result.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError('Diagnostic criteria must contain finite numeric values.')
        result[key] = float(value)
    for key in ['min_chains', 'min_draws', 'max_divergences', 'max_treedepth_hits']:
        if result[key] != int(result[key]):
            raise ValueError('Count-based diagnostic criteria must be integers.')
        result[key] = int(result[key])
    if result['min_chains'] < 4 or result['min_draws'] < 4:
        raise ValueError('Diagnostics require at least four chains and four retained draws.')
    if not 1. <= result['max_rhat'] <= 1.01:
        raise ValueError('The maximum R-hat threshold must be between 1 and 1.01.')
    if result['min_bulk_ess'] < 400 or result['min_tail_ess'] < 400 or result['min_bfmi'] < .3:
        raise ValueError('ESS and BFMI thresholds cannot relax the selected acceptance gates.')
    if result['max_divergences'] != 0 or result['max_treedepth_hits'] != 0:
        raise ValueError('Accepted fits require zero divergences and zero tree-depth hits.')
    return result


def _numeric_array(value, label):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in 'biuf':
            raise ValueError(f'{label} must contain real numeric values.')
        array = array.astype(float, copy=False)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f'{label} cannot be converted to a real numeric array.') from exc
    if not np.isfinite(array).all():
        raise ValueError(f'{label} contains missing or nonfinite values.')
    return array


def _posterior_arrays(posterior):
    if not isinstance(posterior, Mapping) or not posterior:
        raise ValueError('A nonempty posterior variable mapping is required.')
    converted, shape = {}, None
    for name, value in posterior.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Posterior variable names must be nonempty strings.')
        array = _numeric_array(value, 'Posterior variable '+name)
        if array.ndim < 2 or any(size == 0 for size in array.shape):
            raise ValueError('Each posterior variable must have nonempty chain and draw axes.')
        if shape is None:
            shape = array.shape[:2]
        elif array.shape[:2] != shape:
            raise ValueError('Posterior variables have inconsistent chain/draw shapes.')
        converted[name] = array
    return converted, shape


def _stats_arrays(sample_stats, shape):
    if not isinstance(sample_stats, Mapping) or any(name not in sample_stats for name in REQUIRED_STATS):
        raise ValueError('Sampler statistics must include diverging, energy, and reached_max_treedepth.')
    converted = {}
    for name, value in sample_stats.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Sampler statistic names must be nonempty strings.')
        array = _numeric_array(value, 'Sampler statistic '+name)
        if array.shape != shape:
            raise ValueError('Each sampler statistic must match the posterior chain/draw shape.')
        if name in ['diverging', 'reached_max_treedepth'] and not np.isin(array, [0., 1.]).all():
            raise ValueError('Divergence and tree-depth indicators must be binary.')
        converted[name] = array
    return converted


def evaluate_diagnostics(posterior: Mapping, sample_stats: Mapping, criteria: Mapping | None = None) -> dict:
    """Return a JSON-safe pass/fail audit of all retained posterior scalars.

    Arrays use ``(chain, draw, ...)`` order; warm-up must already be excluded.
    The accepted-fit default requires 1,200 draws per chain. ``min_draws`` may be
    explicitly lowered for synthetic checks; the fitting runner independently
    enforces the registered sampling protocol. Input errors produce a failed
    audit, rather than an incomplete successful diagnostic summary.
    """
    result = {'status': 'failed', 'passed': False, 'criteria': None, 'metrics': {}, 'reasons': []}
    try:
        thresholds = _criteria(criteria)
        result['criteria'] = thresholds
        arrays, shape = _posterior_arrays(posterior)
        chains, draws = shape
        result['metrics'] = {
            'chains': int(chains), 'draws_per_chain': int(draws),
            'posterior_variables': len(arrays),
            'posterior_scalars': int(sum(np.prod(a.shape[2:], dtype=int) for a in arrays.values())),
            'scope': 'all posterior scalars including child effects',
        }
        stats = _stats_arrays(sample_stats, shape)
        if chains < thresholds['min_chains']:
            result['reasons'].append('Too few retained chains.')
        if draws < thresholds['min_draws']:
            result['reasons'].append('Too few retained draws per chain.')
        if chains < 4 or draws < 4:
            return result
        import arviz as az

        # User variable names stay untouched, and all trailing axes are included.
        dataset = az.from_dict(posterior=arrays)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rhats = az.rhat(dataset, method='rank')
            bulk = az.ess(dataset, method='bulk')
            tail = az.ess(dataset, method='tail')
            bfmi = np.asarray(az.bfmi(stats['energy']), dtype=float)
        by_variable = {}
        for name in arrays:
            values = [np.asarray(diagnostic[name], dtype=float) for diagnostic in [rhats, bulk, tail]]
            expected_size = int(np.prod(arrays[name].shape[2:], dtype=int))
            if any(v.size != expected_size or not np.isfinite(v).all() for v in values):
                raise ValueError('A posterior variable has missing or nonfinite R-hat/ESS diagnostics.')
            by_variable[name] = {'scalars': expected_size, 'max_rhat': float(values[0].max()),
                                 'min_bulk_ess': float(values[1].min()), 'min_tail_ess': float(values[2].min())}
        if bfmi.shape != (chains,) or not np.isfinite(bfmi).all():
            raise ValueError('One or more chains have missing or nonfinite BFMI.')
        metrics = result['metrics']
        metrics.update({
            'max_rhat': max(v['max_rhat'] for v in by_variable.values()),
            'min_bulk_ess': min(v['min_bulk_ess'] for v in by_variable.values()),
            'min_tail_ess': min(v['min_tail_ess'] for v in by_variable.values()),
            'min_bfmi': float(bfmi.min()), 'bfmi_by_chain': bfmi.tolist(),
            'divergences': int(stats['diverging'].sum()),
            'treedepth_hits': int(stats['reached_max_treedepth'].sum()),
            'by_variable': by_variable,
        })
        checks = [
            (metrics['max_rhat'] > thresholds['max_rhat'], 'Rank-normalized R-hat exceeds its maximum.'),
            (metrics['min_bulk_ess'] < thresholds['min_bulk_ess'], 'Bulk ESS is below its minimum.'),
            (metrics['min_tail_ess'] < thresholds['min_tail_ess'], 'Tail ESS is below its minimum.'),
            (metrics['min_bfmi'] < thresholds['min_bfmi'], 'BFMI is below its minimum.'),
            (metrics['divergences'] > thresholds['max_divergences'], 'Divergent transitions occurred.'),
            (metrics['treedepth_hits'] > thresholds['max_treedepth_hits'], 'Maximum tree depth was reached.'),
        ]
        result['reasons'].extend(reason for failed, reason in checks if failed)
        result['passed'] = not result['reasons']
        result['status'] = 'passed' if result['passed'] else 'failed'
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, FloatingPointError) as exc:
        result['reasons'].append(str(exc))
    return result


def _contrast_arrays(contrasts):
    arrays, shape = _posterior_arrays(contrasts)
    if shape[0] < 2 or shape[1] < 4 or any(value.ndim != 2 for value in arrays.values()):
        raise ValueError('Each contrast must be a two-dimensional array with at least two chains and four draws.')
    return arrays, shape


def compare_contrasts(current: Mapping, reference: Mapping, multiplier: float = 4):
    """Compare means and interval limits using their own Monte Carlo errors.

    Return one DataFrame row per contrast and statistic. The threshold is
    ``multiplier*sqrt(mcse_current**2+mcse_reference**2)``. Each quantile uses
    ArviZ's quantile MCSE at the same probability. This independent-run rule
    measures Monte Carlo compatibility and does not replace convergence checks.
    The current and reference runs may have different chain/draw counts.
    """
    if isinstance(multiplier, (bool, np.bool_)) or not isinstance(multiplier, (int, float, np.integer, np.floating)) or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('The MCSE comparison multiplier must be finite and positive.')
    current_arrays, current_shape = _contrast_arrays(current)
    reference_arrays, reference_shape = _contrast_arrays(reference)
    if set(current_arrays) != set(reference_arrays):
        raise ValueError('Current and reference contrast labels must match exactly.')
    import arviz as az
    import pandas as pd

    rows = []
    for label in current_arrays:
        a, b = current_arrays[label], reference_arrays[label]
        for statistic, probability in [('mean', None), ('q025', .025), ('q975', .975)]:
            if probability is None:
                estimate_a, estimate_b = float(a.mean()), float(b.mean())
                mcse_a, mcse_b = (float(az.mcse(value, method='mean')) for value in (a, b))
            else:
                estimate_a, estimate_b = (float(np.quantile(value, probability)) for value in (a, b))
                mcse_a, mcse_b = (float(az.mcse(value, method='quantile', prob=probability)) for value in (a, b))
            if not np.isfinite([estimate_a, estimate_b, mcse_a, mcse_b]).all() or mcse_a < 0 or mcse_b < 0:
                raise ValueError('Contrast estimates or Monte Carlo errors are missing, nonfinite, or invalid.')
            combined = float(np.hypot(mcse_a, mcse_b))
            threshold = float(multiplier*combined)
            difference = estimate_a-estimate_b
            rows.append({'contrast': label, 'statistic': statistic,
                         'current': estimate_a, 'reference': estimate_b,
                         'difference': difference, 'absolute_difference': abs(difference),
                         'mcse_current': mcse_a, 'mcse_reference': mcse_b,
                         'mcse_combined': combined, 'multiplier': float(multiplier),
                         'threshold': threshold, 'passed': bool(abs(difference) <= threshold),
                         'current_chains': current_shape[0], 'current_draws': current_shape[1],
                         'reference_chains': reference_shape[0], 'reference_draws': reference_shape[1]})
    return pd.DataFrame(rows)
