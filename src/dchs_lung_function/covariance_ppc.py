"""Prespecified saved-draw screen of repeated-target covariance adequacy.

Conditional Gaussian replicas use the observed pre/no-TB child/measurement
layout and posterior realized child effects. This is an adequacy screen, not a
calibrated multiple-testing procedure or a convergence diagnostic. No fitting,
participant-level outputs, or modification of posterior draws occurs here.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import numpy as np

PROTOCOL_VERSION = 'target-covariance-ppc-1.0.0'
DEFAULT_CRITERIA = {
    'replicates': 2000, 'lower_quantile': .005, 'upper_quantile': .995,
    'minimum_repeat_children': 20, 'minimum_within_df': 30,
}
SEED_NAMESPACE = 'dchs-corrective-target-covariance-ppc-v1'
METRICS = ('icc', 'within_ms')


def protocol():
    """Return a detached JSON-safe declaration of the fixed adequacy screen."""
    return {'version': PROTOCOL_VERSION, 'criteria': deepcopy(DEFAULT_CRITERIA),
        'sample': 'target rows with post_any == 0; every modeled post-TB term equals zero',
        'replication': 'one Gaussian replica per deterministically selected saved posterior draw, conditional on realized child effects',
        'selection': '2000 evenly spaced chain-major flattened draw positions, including endpoints, without replacement',
        'seed_namespace': SEED_NAMESPACE,
        'seed_rule': 'first 8 SHA256 bytes of namespace|canonical_family|target, unsigned big endian',
        'icc': 'untruncated unequal-size one-way ANOVA (MSB-MSW)/(MSB+(n0-1)*MSW)',
        'n0': '(N-sum(n_i**2)/N)/(G-1)',
        'within_ms': 'sum within-child squared deviations/(N-G)',
        'interval': 'equal-tailed 99% conditional posterior predictive interval; linear sample quantiles',
        'tails': 'separate unadjusted proportions replica<=observed and replica>=observed',
        'acceptance': 'both observed metrics inside inclusive predictive bounds',
        'not_evaluable': 'blocks completeness unless an explicit documented exception is approved',
        'interpretation': 'prespecified adequacy screen, not calibrated multiple testing'}


def _route(family, target):
    from .models import spirometry, mmef, x6, sensitivity, mechanics
    if family == 'primary':
        spec = spirometry.fit_spec(target); canonical = 'primary'
    elif family in ('supporting', 'mmef', 'x6'):
        module = mmef if target in mmef.TARGETS else x6
        spec = module.fit_spec(target); canonical = spec['family']
        if family != 'supporting' and family != canonical:
            raise ValueError('PPC family does not match the selected target.')
    elif family in ('sensitivity', 'early', 'window'):
        spec = sensitivity.fit_spec(target); canonical = spec['family']
        if family != 'sensitivity' and family != canonical:
            raise ValueError('PPC family does not match the selected target.')
    elif family == 'mechanics':
        spec = mechanics.fit_spec(target); canonical = 'mechanics'
    else:
        raise ValueError('Unknown target covariance PPC family.')
    return spec, canonical


def covariance_statistics(values, child_index):
    """Unequal-repeat one-way ANOVA moments, without truncating negative ICC.

    Both arrays contain one value per observation. Child labels must already be
    integer coordinates, and are never returned. At least two observed children
    and a positive within-child degree of freedom are required.
    """
    y = np.asarray(values, float); idx = np.asarray(child_index)
    if y.ndim != 1 or idx.shape != y.shape or not np.isfinite(y).all():
        raise ValueError('Covariance inputs must be finite vectors of equal length.')
    if not np.issubdtype(idx.dtype, np.integer) or (idx < 0).any():
        raise ValueError('Covariance child coordinates must be nonnegative integers.')
    _, group = np.unique(idx, return_inverse=True)
    counts = np.bincount(group).astype(float); n = len(y); g = len(counts)
    if g < 2 or n <= g:
        raise ValueError('Covariance statistics require repeated measurements and multiple children.')
    # Center first to protect the variance calculation against arbitrary shifts.
    centered = y - y.mean()
    group_means = np.bincount(group, weights=centered) / counts
    within = float(np.sum((centered - group_means[group])**2) / (n - g))
    between = float(np.dot(counts, group_means**2) / (g - 1))
    n0 = float((n - np.dot(counts, counts) / n) / (g - 1))
    denominator = between + (n0 - 1.) * within
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Target has no finite positive total covariance variation.')
    return {'icc': float((between - within) / denominator), 'within_ms': within,
        'between_ms': between, 'effective_repeat_count': n0,
        'total_variance': float(np.dot(centered, centered) / (n - 1))}


def evaluate_target_covariance(posterior, contract, *, family, target, criteria=None):
    """Return only aggregate adequacy evidence, with a deterministic fixed seed.

    The fitting workflow must validate the complete model contract and posterior
    inventory first. This function additionally validates every field it uses,
    the zero-exposure selection, and the saved latent transform. Passing criteria
    is permitted only to affirm the frozen constants, never to relax them.
    """
    if criteria is not None and criteria != DEFAULT_CRITERIA:
        raise ValueError('The prespecified covariance PPC criteria may not be changed.')
    spec, canonical = _route(family, target)
    frame = contract['frames']['target']; children = contract['children']
    if contract['audit']['target'] != spec['column']:
        raise ValueError('PPC outcome differs from the selected contract.')
    n_children = len(children)
    coordinates = np.asarray(children['child_index'])
    if not np.array_equal(coordinates, np.arange(n_children)):
        raise ValueError('PPC child coordinates are not in their validated order.')
    idx = np.asarray(frame['child_index'])
    y = np.asarray(frame['y_std'], float); post = np.asarray(frame['post_any'], float)
    if (idx.shape != y.shape or y.ndim != 1 or not np.issubdtype(idx.dtype, np.integer)
            or (idx < 0).any() or (idx >= n_children).any() or not np.isfinite(y).all()
            or post.shape != y.shape or not np.isin(post, [0., 1.]).all()):
        raise ValueError('Invalid target covariance sample or child coordinates.')
    selected = post == 0
    terms = contract['post_diagnosis_terms']
    expected_terms = (['post_any'] if canonical == 'x6' else
        ['post_before5', 'post_infancy'] if canonical in ('early', 'mmef') else
        ['post_before5', 'post_infancy', 'post_at_after5'])
    if list(terms) != expected_terms:
        raise ValueError('PPC post-TB term inventory differs from the selected graph.')
    for name in terms:
        indicator = np.asarray(frame[name], float)
        if indicator.shape != y.shape or not np.isin(indicator, [0., 1.]).all() or np.any(indicator[selected] != 0):
            raise ValueError('A selected pre/no-TB row has a nonzero modeled exposure.')
    y = y[selected]; idx = idx[selected]
    observed_children, repeats = np.unique(idx, return_counts=True)
    n, g = len(y), len(observed_children)
    seed = int.from_bytes(sha256(f'{SEED_NAMESPACE}|{canonical}|{target}'.encode()).digest()[:8], 'big')
    result = {'protocol': protocol(), 'family': canonical, 'target': target,
        'model_specification_version': spec['specification_version'], 'seed': seed,
        'status': 'not_evaluable', 'accepted': False, 'joint_pass': False, 'reasons': [],
        'counts': {'joint_children': n_children, 'target_observations': len(frame),
            'selected_observations': n, 'selected_children': g,
            'repeat_children': int(np.sum(repeats >= 2)), 'within_df': n - g},
        'draw_selection': {}, 'observed': {}, 'predictive': {}}
    if result['counts']['repeat_children'] < DEFAULT_CRITERIA['minimum_repeat_children']:
        result['reasons'].append('insufficient_repeat_children')
    if n - g < DEFAULT_CRITERIA['minimum_within_df']:
        result['reasons'].append('insufficient_within_df')
    if result['reasons']:
        return result
    try:
        result['observed'] = covariance_statistics(y, idx)
    except ValueError:
        result['reasons'].append('undefined_observed_covariance'); return result
    sigma_name = 'sigma_outcome' if canonical == 'mechanics' else ('sigma_target' if canonical == 'x6' else 'sigma_ratio')
    sigma = np.asarray(posterior[sigma_name], float)
    if sigma.ndim != 2 or min(sigma.shape) < 1 or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('PPC observation scale must have positive finite chain/draw values.')
    shape = sigma.shape; total = sigma.size; rep = DEFAULT_CRITERIA['replicates']
    result['draw_selection'] = {'chains': shape[0], 'draws_per_chain': shape[1],
        'available_draws': total, 'requested_replicates': rep}
    if total < rep:
        result['reasons'].append('insufficient_saved_draws'); return result
    positions = np.linspace(0, total - 1, rep, dtype=np.int64)
    result['draw_selection'].update(replicates=rep,
        positions_sha256=sha256(positions.astype('<i8').tobytes()).hexdigest())

    def scalar(name):
        value = np.asarray(posterior[name], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('PPC scalar posterior dimensions or values are invalid.')
        return value.reshape(-1)[positions]

    def child(name):
        value = np.asarray(posterior[name], float)
        if value.shape != (*shape, n_children) or not np.isfinite(value).all():
            raise ValueError('PPC child posterior dimensions or values are invalid.')
        return value.reshape(total, n_children)[positions]

    if canonical == 'mechanics':
        mean_child = child('shared_outcome')
        tau = scalar('tau_child')
        if (tau <= 0).any() or not np.allclose(mean_child, tau[:, None] * child('child_z'), rtol=1e-11, atol=1e-12):
            raise ValueError('PPC shared child effect does not reconcile with its saved transform.')
        child_variance = tau**2
    else:
        loading = scalar('loading_target_mechanics' if canonical == 'x6' else 'loading_ratio_mechanics')
        tau = scalar('tau_target'); effect = child('target_process')
        if (tau <= 0).any() or not np.allclose(effect, tau[:, None] * child('target_raw'), rtol=1e-11, atol=1e-12):
            raise ValueError('PPC target effect does not reconcile with its saved transform.')
        mean_child = loading[:, None] * child('mechanics') + effect
        child_variance = loading**2 + tau**2
    sd = scalar(sigma_name)
    population_icc = child_variance / (child_variance + sd**2)
    result['population_formula_icc'] = {'mean': float(population_icc.mean()),
        'q025': float(np.quantile(population_icc, .025)), 'q975': float(np.quantile(population_icc, .975)),
        'used_for_acceptance': False}
    rng = np.random.default_rng(seed)
    simulated = {name: np.empty(rep) for name in (*METRICS, 'between_ms', 'total_variance')}
    for i in range(rep):
        stats = covariance_statistics(mean_child[i, idx] + sd[i] * rng.standard_normal(n), idx)
        for name in simulated:
            simulated[name][i] = stats[name]
    for name, values in simulated.items():
        observed = result['observed'][name]
        low, high = np.quantile(values, [DEFAULT_CRITERIA['lower_quantile'], DEFAULT_CRITERIA['upper_quantile']])
        passed = bool(low <= observed <= high)
        result['predictive'][name] = {'mean': float(values.mean()), 'median': float(np.median(values)),
            'lower': float(low), 'upper': float(high),
            'tail_at_or_below_observed': float(np.mean(values <= observed)),
            'tail_at_or_above_observed': float(np.mean(values >= observed)),
            'used_for_acceptance': name in METRICS, 'passed': passed if name in METRICS else None}
        if name in METRICS and not passed:
            result['reasons'].append(name + '_outside_predictive_bounds')
    result['joint_pass'] = not result['reasons']; result['accepted'] = result['joint_pass']
    result['status'] = 'passed' if result['accepted'] else 'failed'
    return result
