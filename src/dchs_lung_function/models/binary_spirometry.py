"""Modified-Poisson GEE models for binary spirometry, version 1.0.0.

M03 supplies the observation sample and covariates. This module validates the
binary definitions and design, fits independence-working-correlation GEE with
child-cluster robust covariance, and returns aggregate outputs only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd

from ..preparation import repeated_design

MODEL_ID = 'SPIRO-BINARY-GEE'
SPEC_VERSION = '1.0.0'
LLN_Z = -1.645
CORE_OUTCOMES = ('ratio_LLN', 'FEV1_LLN', 'FVC_LLN', 'obstructive_pattern')
SUPPLEMENT_OUTCOMES = ('any_abnormal', 'possible_restriction')
OUTCOMES = (*CORE_OUTCOMES, *SUPPLEMENT_OUTCOMES)
EXPOSURE_TERMS = {
    'before5': ('post_before5', 'post_at_after5', 'future_tb'),
    'age_groups': ('post_infancy', 'post_age1_before5', 'post_at_after5', 'future_tb'),
    'any_tb': ('post_any', 'future_tb'),
}
DEFAULT_FIT_SELECTION = tuple(
    (stage, outcome, design)
    for stage in ('pre', 'post') for outcome in CORE_OUTCOMES for design in EXPOSURE_TERMS
) + tuple(('pre', outcome, design) for outcome in SUPPLEMENT_OUTCOMES for design in ('before5', 'age_groups'))
_BINARY_FIELDS = ('post_before5', 'post_infancy', 'post_age1_before5', 'post_at_after5', 'post_any', 'future_tb')
_PATTERNS = ('Normal', 'Obstructive', 'Possible restriction', 'Mixed')
_STATES = ('no_recorded_tb', 'future_tb', 'post_infancy', 'post_age1_before5', 'post_at_after5')


@dataclass(frozen=True)
class BinaryFit:
    """Aggregate fit outputs; coefficients/covariance include all model terms."""
    estimates: pd.DataFrame
    coefficients: pd.Series
    covariance: pd.DataFrame
    metadata: dict[str, Any]


def fit_spec(stage: str, outcome: str, design_key: str) -> dict[str, Any]:
    if stage not in ('pre', 'post') or outcome not in OUTCOMES or design_key not in EXPOSURE_TERMS:
        raise ValueError('Unsupported spirometry stage, binary outcome, or exposure design.')
    reported = {'before5': ['post_before5'],
                'age_groups': ['post_infancy', 'post_age1_before5', 'post_at_after5'],
                'any_tb': ['post_any']}[design_key]
    if outcome in SUPPLEMENT_OUTCOMES and design_key == 'age_groups':
        reported = reported[:2]
    return {
        'model_id': MODEL_ID, 'specification_version': SPEC_VERSION,
        'stage': stage, 'outcome': outcome, 'design_key': design_key,
        'family': 'Poisson', 'link': 'log', 'working_correlation': 'Independence',
        'covariance': 'robust child-cluster sandwich', 'cluster_field': 'pid_child',
        'exposure_terms': list(EXPOSURE_TERMS[design_key]), 'reported_terms': reported,
        'covariates': ['height_z', 'male', 'planned_visit'],
        'height_standardization': 'within outcome sample, sample SD (ddof=1)',
        'formula': 'outcome_value ~ 1 + '+' + '.join(EXPOSURE_TERMS[design_key])+' + height_z + male + C(planned_visit)',
        'lln_threshold': LLN_Z, 'lln_operator': '<',
        'outcome_definition': {
            'ratio_LLN': 'FEV1/FVC z < LLN', 'FEV1_LLN': 'FEV1 z < LLN', 'FVC_LLN': 'FVC z < LLN',
            'obstructive_pattern': 'FEV1/FVC z < LLN and FVC z >= LLN',
            'possible_restriction': 'FVC z < LLN and FEV1/FVC z >= LLN',
            'any_abnormal': 'FEV1/FVC z < LLN or FVC z < LLN',
        }[outcome],
        'interval': 'exp(beta +/- 1.96*robust_SE)', 'interval_multiplier': 1.96,
        'reference': 'measurements with no recorded TB; future-case measurements have a separate indicator',
        'fit_options': {'maxiter': 250, 'ctol': 1e-6, 'cov_type': 'robust'},
        'weights': None, 'offset': None, 'runtime': {'statsmodels': '0.14.5'},
        'selected_by_default': (stage, outcome, design_key) in DEFAULT_FIT_SELECTION,
    }


def _z_columns(stage, outcome):
    suffix = '' if stage == 'pre' else '_post'
    if outcome in ('ratio_LLN', 'FEV1_LLN', 'FVC_LLN'):
        stem = {'ratio_LLN': 'fev1_fvc', 'FEV1_LLN': 'fev1', 'FVC_LLN': 'fvc'}[outcome]
        return [stem+suffix+'_gli_z']
    return ['fev1_fvc'+suffix+'_gli_z', 'fvc'+suffix+'_gli_z']


def classify_outcome(frame: pd.DataFrame, stage: str, outcome: str) -> pd.Series:
    """Classify an outcome; missing required z-scores remain missing, not normal."""
    fit_spec(stage, outcome, 'before5')
    columns = _z_columns(stage, outcome)
    if not set(columns).issubset(frame):
        raise ValueError('Binary classification is missing required reference-score columns.')
    values = [pd.to_numeric(frame[column], errors='coerce').to_numpy(float) for column in columns]
    valid = np.logical_and.reduce([np.isfinite(value) for value in values])
    low = [value < LLN_Z for value in values]
    if len(low) == 1:
        event = low[0]
    elif outcome == 'obstructive_pattern':
        event = low[0] & ~low[1]
    elif outcome == 'possible_restriction':
        event = low[1] & ~low[0]
    else:
        event = low[0] | low[1]
    return pd.Series(np.where(valid, event.astype(float), np.nan), index=frame.index, name='outcome_value')


def _finite(values, shape, label, *, binary=False):
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label} must be numeric.') from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label} has an invalid shape or nonfinite values.')
    if binary and not np.isin(array, [0., 1.]).all():
        raise ValueError(f'{label} must be binary.')
    return array


def _validated(contract, design_key, stage=None, outcome=None):
    try:
        audit = contract['audit']
        chosen_stage, chosen_outcome = audit['stage'], audit['outcome']
        spec = fit_spec(chosen_stage, chosen_outcome, design_key)
        if audit['family'] != 'binary' or (stage is not None and stage != chosen_stage) or (outcome is not None and outcome != chosen_outcome):
            raise ValueError('Requested binary analysis does not match the M03 contract.')
        frame = contract['frame']
        needed = {'pid_child', 'outcome_value', 'height', 'male', 'visit_label', *_BINARY_FIELDS}
        if not isinstance(frame, pd.DataFrame) or not needed.issubset(frame) or len(frame) == 0:
            raise ValueError('Binary analysis requires a nonempty prepared frame with its covariates.')
        n = len(frame)
        groups = frame['pid_child'].to_numpy()
        if frame['pid_child'].isna().any() or frame['pid_child'].astype('string').str.strip().eq('').any():
            raise ValueError('Binary analysis contains a missing child linkage key.')
        y = _finite(frame['outcome_value'], (n,), 'Binary outcome', binary=True)
        expected_y = classify_outcome(frame, chosen_stage, chosen_outcome).to_numpy(float)
        if not np.array_equal(y, expected_y):
            raise ValueError('Prepared binary outcomes differ from the selected LLN or pattern definition.')
        _finite(frame['height'], (n,), 'Height')
        _finite(frame['male'], (n,), 'Sex', binary=True)
        if frame['visit_label'].isna().any() or frame['visit_label'].astype(str).str.strip().eq('').any():
            raise ValueError('Scheduled visit labels must be present.')
        z = {name: _finite(frame[name], (n,), 'TB exposure '+name, binary=True) for name in _BINARY_FIELDS}
        if not np.array_equal(z['post_before5'], z['post_infancy']+z['post_age1_before5']):
            raise ValueError('Before-five TB must equal the disjoint infancy and age-one-to-five groups.')
        if not np.array_equal(z['post_any'], z['post_before5']+z['post_at_after5']) or (z['post_any']+z['future_tb'] > 1).any():
            raise ValueError('Post-TB and future-TB exposure states are inconsistent.')
        x, terms, dropped = repeated_design(frame, spec['exposure_terms'])
        _finite(x, (n, len(terms)), 'Model design')
        if design_key in ('before5', 'age_groups'):
            stored = contract['designs'][design_key]
            saved_x = _finite(stored['matrix'], x.shape, 'Prepared model design')
            if list(stored['terms']) != terms or list(stored['dropped_terms']) != dropped or not np.allclose(saved_x, x, atol=1e-12, rtol=1e-12):
                raise ValueError('Prepared design differs from the selected exposure and covariate specification.')
            audit_design = audit['designs'][design_key]
            if list(audit_design['terms']) != terms or list(audit_design['dropped_terms']) != dropped:
                raise ValueError('Prepared design audit differs from its matrix.')
        counts = {'n_observations': n, 'n_children': int(frame['pid_child'].nunique()), 'events': int(y.sum())}
        if any(audit[name] != value for name, value in counts.items()):
            raise ValueError('Binary sample counts differ from their M03 audit.')
        return frame, y, x, groups, terms, dropped, spec, counts
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('The binary analysis contract is missing required M03 fields.') from exc


def validate_contract(contract: Mapping, design_key: str, *, stage=None, outcome=None) -> dict[str, Any]:
    """Validate membership-derived fields and return aggregate design metadata."""
    _, _, x, _, terms, dropped, spec, counts = _validated(contract, design_key, stage, outcome)
    return {**spec, **counts, 'terms': terms, 'dropped_terms': dropped,
            'design_rank': int(np.linalg.matrix_rank(x, tol=1e-10)), 'design_columns': x.shape[1]}


def _support(frame, term):
    exposed = frame.loc[frame[term].eq(1)]
    return {'exposed_observations': len(exposed),
            'exposed_children': int(exposed['pid_child'].nunique()),
            'exposed_events': int(exposed['outcome_value'].sum()),
            'exposed_children_ever_positive': int(exposed.groupby('pid_child', observed=True)['outcome_value'].max().sum())}


def _boundary_terms(x, terms, y):
    result = []
    for j, term in enumerate(terms):
        column = x[:, j]
        if term == 'Intercept' or not np.isin(column, [0., 1.]).all():
            continue
        selected = column == 1
        n, events = int(selected.sum()), int(y[selected].sum())
        if n and events in (0, n):
            result.append({'term': term, 'observations': n, 'events': events,
                           'status': 'zero_events' if events == 0 else 'full_events'})
    return result


def _stable_poisson_covariance(y, x, groups, beta):
    """Compute the same child sandwich through equilibrated influence solves.

    For independence Poisson GEE, A=X' diag(mu) X and each child's score is
    U_i=sum_j X_ij*(y_ij-mu_ij). Solving A*I_i=U_i then summing I_i*I_i' avoids
    multiplying two separately evaluated inverses. Diagonal equilibration is
    an algebraic change of coordinates; neither the estimator nor its scale
    changes. It is particularly useful for zero-event nuisance categories.
    """
    mu = np.exp(x@beta)
    if not np.isfinite(mu).all() or (mu <= 0).any():
        raise ValueError('The fitted Poisson means are not finite and positive.')
    weighted = x*np.sqrt(mu)[:, None]
    scale = np.sqrt(np.sum(weighted**2, axis=0))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Poisson information has a nonpositive diagonal.')
    normalized = weighted/scale
    information = normalized.T@normalized
    _, group_index = np.unique(groups, return_inverse=True)
    scores = np.zeros((int(group_index.max())+1, x.shape[1]))
    np.add.at(scores, group_index, x*(y-mu)[:, None])
    influences = np.linalg.solve(information, scores.T/scale[:, None])/scale[:, None]
    covariance = influences@influences.T
    return covariance, {'method': 'equilibrated child-score influence sandwich',
                        'normalized_information_condition': float(np.linalg.cond(information)),
                        'raw_information_condition': float(np.linalg.cond(weighted.T@weighted)),
                        'scale_factor': 1.0, 'finite_sample_correction': False}


def fit_binary(contract: Mapping, design_key: str, *, stage=None, outcome=None) -> BinaryFit:
    """Fit one selected design and distinguish boundary outcomes from failures.

    Coefficients and the complete robust covariance are retained even when a
    reported contrast is non-estimable. Non-estimable result cells are null;
    all exact support counts remain. A numerical fitting failure always sets
    ``metadata['accepted']`` false, including when another contrast has a known
    boundary condition. Neither outcome nor exposure prevalence is suppressed.
    """
    frame, y, x, groups, terms, dropped, spec, counts = _validated(contract, design_key, stage, outcome)
    p = len(terms)
    coefficients = np.full(p, np.nan)
    covariance = np.full((p, p), np.nan)
    boundary = _boundary_terms(x, terms, y)
    metadata = {**spec, **counts, 'terms': terms, 'dropped_terms': dropped,
                'design_rank': int(np.linalg.matrix_rank(x, tol=1e-10)),
                'boundary_terms': boundary, 'converged': False, 'iterations': 0,
                'fit_status': 'not_fitted', 'warning_categories': {},
                'fit_failure_reasons': [], 'expected_nonestimable_reason': None}
    expected = None
    if np.unique(y).size == 1:
        expected = 'constant_outcome'
    elif counts['n_children'] < 2:
        expected = 'insufficient_independent_clusters'
    elif not any(term in terms for term in spec['reported_terms']):
        expected = 'reported_exposures_absent'
    metadata['expected_nonestimable_reason'] = expected
    if expected is None:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Poisson
        from statsmodels.genmod.families.links import Log
        from statsmodels.genmod.cov_struct import Independence

        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                model = GEE(y, x, groups=groups, family=Poisson(link=Log()), cov_struct=Independence())
                result = model.fit(**spec['fit_options'])
            metadata['warning_categories'] = dict(Counter(w.category.__name__ for w in recorded))
            coefficients = np.asarray(result.params, dtype=float)
            covariance = np.asarray(result.cov_params(), dtype=float)
            metadata['converged'] = bool(result.converged)
            metadata['iterations'] = len(result.fit_history.get('params', []))
            metadata['fit_status'] = 'converged' if metadata['converged'] else 'not_converged'
            if not metadata['converged']:
                metadata['fit_failure_reasons'].append('GEE did not converge.')
            if coefficients.shape != (p,) or covariance.shape != (p, p):
                metadata['fit_failure_reasons'].append('GEE returned coefficient or covariance dimensions inconsistent with the design.')
                coefficients = np.full(p, np.nan); covariance = np.full((p, p), np.nan)
            elif not np.isfinite(coefficients).all() or not np.isfinite(covariance).all():
                metadata['fit_failure_reasons'].append('GEE returned nonfinite coefficients or robust covariance.')
            else:
                raw_covariance = covariance.copy()
                covariance, covariance_audit = _stable_poisson_covariance(y, x, groups, coefficients)
                covariance_audit.update({
                    'raw_statsmodels_covariance': raw_covariance.tolist(),
                    'raw_symmetry_max_abs_difference': float(np.max(np.abs(raw_covariance-raw_covariance.T))),
                    'raw_stable_max_abs_difference': float(np.max(np.abs(raw_covariance-covariance))),
                    'raw_stable_diagonal_max_abs_difference': float(np.max(np.abs(np.diag(raw_covariance)-np.diag(covariance)))),
                })
                metadata['covariance_evaluation'] = covariance_audit
                norm = max(1., float(np.max(np.abs(covariance))))
                if not np.isfinite(covariance).all():
                    metadata['fit_failure_reasons'].append('Child-score influence covariance is nonfinite.')
                if not np.allclose(covariance, covariance.T, atol=1e-10*norm, rtol=1e-8):
                    metadata['fit_failure_reasons'].append('Robust covariance is not symmetric.')
                if np.min(np.diag(covariance)) < -1e-10*norm or np.linalg.eigvalsh((covariance+covariance.T)/2).min() < -1e-8*norm:
                    metadata['fit_failure_reasons'].append('Robust covariance has invalid negative variance.')
                scores = result.fit_history.get('score', [])
                if scores:
                    score_norm = float(np.linalg.norm(scores[-1]))
                    metadata['score_norm'] = score_norm if np.isfinite(score_norm) else None
        except Exception as exc:
            # Exception text can include data-dependent details; preserve only its type.
            metadata['fit_status'] = 'exception'
            metadata['fit_failure_reasons'].append('GEE fitting raised '+type(exc).__name__+'.')
    failure = bool(metadata['fit_failure_reasons'])
    rows = []
    reference = frame.loc[frame[list(spec['exposure_terms'])].sum(axis=1).eq(0)]
    for term in spec['reported_terms']:
        support = _support(frame, term)
        reason = None
        if support['exposed_observations'] == 0:
            reason = 'no_exposed_observations'
        elif support['exposed_events'] == 0:
            reason = 'zero_exposed_events'
        elif support['exposed_events'] == support['exposed_observations']:
            reason = 'full_exposed_events'
        elif term not in terms:
            reason = 'exposure_not_identifiable'
        elif expected is not None:
            reason = expected
        elif failure:
            reason = 'numerical_fit_failure'
        values = {'beta': None, 'se': None, 'rr': None, 'lower': None, 'upper': None, 'p_value': None}
        if reason is None:
            from scipy.stats import norm
            j = terms.index(term)
            beta, variance = float(coefficients[j]), float(covariance[j, j])
            if not np.isfinite(beta+variance) or variance <= 0:
                reason = 'invalid_standard_error'
                failure = True
                metadata['fit_failure_reasons'].append('A reported contrast has an invalid robust standard error.')
            else:
                se = float(np.sqrt(variance))
                with np.errstate(over='ignore', under='ignore'):
                    rr, low, high = np.exp([beta, beta-1.96*se, beta+1.96*se])
                if not np.isfinite([rr, low, high]).all() or min(rr, low, high) <= 0:
                    reason = 'nonfinite_risk_ratio'
                    failure = True
                    metadata['fit_failure_reasons'].append('A reported contrast has an invalid exponentiated estimate or interval.')
                else:
                    values = {'beta': beta, 'se': se, 'rr': float(rr), 'lower': float(low), 'upper': float(high),
                              'p_value': float(2*norm.sf(abs(beta/se)))}
        rows.append({'stage': spec['stage'], 'outcome': spec['outcome'], 'design_key': design_key,
                     'term': term, 'estimable': reason is None,
                     'selected_for_figure2': spec['outcome'] in ('ratio_LLN', 'obstructive_pattern') and term in ('post_before5', 'post_infancy'),
                     'status': 'estimable' if reason is None else 'non_estimable',
                     'reason_code': reason, 'reason': '' if reason is None else reason.replace('_', ' '),
                     **values, **counts, **support,
                     'reference_observations': len(reference),
                     'reference_children': int(reference['pid_child'].nunique()),
                     'reference_events': int(reference['outcome_value'].sum())})
    estimable = sum(row['estimable'] for row in rows)
    metadata['accepted'] = not failure
    metadata['status'] = ('fit_failed' if failure else 'success' if estimable == len(rows)
                          else 'non_estimable' if estimable == 0 else 'partial')
    metadata['estimable_contrasts'] = estimable
    metadata['non_estimable_contrasts'] = len(rows)-estimable
    metadata['coefficients_finite'] = bool(np.isfinite(coefficients).all())
    metadata['covariance_finite'] = bool(np.isfinite(covariance).all())
    return BinaryFit(pd.DataFrame(rows), pd.Series(coefficients, index=terms, name='coefficient'),
                     pd.DataFrame(covariance, index=terms, columns=terms), metadata)


def pattern_counts(contract: Mapping) -> pd.DataFrame:
    """Exact mutually exclusive pattern counts in every TB state, including zeros."""
    frame, _, _, _, _, _, spec, _ = _validated(contract, 'before5', outcome='obstructive_pattern')
    columns = _z_columns(spec['stage'], 'obstructive_pattern')
    ratio = frame[columns[0]].to_numpy(float) < LLN_Z
    fvc = frame[columns[1]].to_numpy(float) < LLN_Z
    pattern = np.select([ratio & fvc, ratio & ~fvc, fvc & ~ratio],
                        ['Mixed', 'Obstructive', 'Possible restriction'], default='Normal')
    state = np.select([frame['future_tb'].eq(1), frame['post_infancy'].eq(1),
                       frame['post_age1_before5'].eq(1), frame['post_at_after5'].eq(1)],
                      list(_STATES[1:]), default=_STATES[0])
    rows = []
    for exposure_state in _STATES:
        group = state == exposure_state
        for category in _PATTERNS:
            event = group & (pattern == category)
            n = int(group.sum())
            rows.append({'stage': spec['stage'], 'exposure_state': exposure_state, 'pattern': category,
                         'observations': n, 'children': int(frame.loc[group, 'pid_child'].nunique()),
                         'event_observations': int(event.sum()),
                         'event_children': int(frame.loc[event, 'pid_child'].nunique()),
                         'prevalence': float(event.sum()/n) if n else None})
    return pd.DataFrame(rows)
