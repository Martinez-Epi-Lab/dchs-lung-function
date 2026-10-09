"""Selected pre-TB Cox analyses and the source child-cluster sandwich.

The partial likelihood uses Efron ties and closed entry risk sets.  Its
uncertainty deliberately preserves PHReg 0.14.5 score_residuals, including
its reverse-time set bookkeeping, rather than substituting Efron scores or
a newly defined martingale estimator.  This distinction is audited below.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels
from statsmodels.base.model import LikelihoodModel
from statsmodels.duration.hazard_regression import PHReg

TERMS = ('lf_adverse_sd', 'male', 'mbekweni')
_TARGET_ROWS = (
    ('resistance', 'R', 'Model-derived resistance', 1, False),
    ('compliance', 'C', 'Model-derived compliance', -1, False),
    ('ree', 'ReE', 'End-expiratory resistance', 1, False),
    ('rei', 'ReI', 'End-inspiratory resistance', 1, False),
    ('xee', 'XeE', 'End-expiratory reactance', -1, False),
    ('xei', 'XeI', 'End-inspiratory reactance', -1, False),
    ('dr_mean', 'delta R', 'Within-breath resistance difference', 1, False),
    ('dx_mean', 'delta X', 'Within-breath reactance difference', 1, False),
    ('r6_sa_z', 'R6', 'Resistance at 6 Hz', 1, True),
    ('r8_sa_z', 'R8', 'Resistance at 8 Hz', 1, True),
    ('r10_sa_z', 'R10', 'Resistance at 10 Hz', 1, True),
    ('x6_sa_z', 'X6', 'Reactance at 6 Hz', -1, True),
    ('x8_sa_z', 'X8', 'Reactance at 8 Hz', -1, True),
    ('x10_sa_z', 'X10', 'Reactance at 10 Hz', -1, True),
    ('fres_sa_z', 'Fres', 'Resonant frequency', 1, True),
    ('r6_r20_sa_z', 'R6-R20', 'Frequency dependence of resistance', 1, True),
    ('ax_sa_z', 'AX', 'Area of reactance', 1, True),
)
TARGETS = {column: {'measure': measure, 'measure_label': label,
                   'source_column': column, 'orientation': orientation,
                   'reference_z': reference,
                   'scale': 'reference_z' if reference else 'residual_sd'}
           for column, measure, label, orientation, reference in _TARGET_ROWS}
DEFAULT_TARGETS = tuple(TARGETS)
ESTIMATE_COLUMNS = (
    'target', 'measure', 'measure_label', 'source_column', 'scale',
    'carry_forward_window_years', 'n_children', 'n_intervals',
    'n_subsequent_tb', 'person_years', 'status', 'accepted', 'reason',
    'coefficient', 'robust_se', 'hr_per_adverse_sd', 'ci_low', 'ci_high',
    'p_value', 'fdr_q_value', 'warning')
COEFFICIENT_COLUMNS = (
    'target', 'term', 'coefficient', 'robust_se', 'model_se', 'hazard_ratio',
    'ci_low', 'ci_high', 'p_value', 'status', 'accepted', 'reason')
COVARIANCE_COLUMNS = (
    'target', 'covariance_type', 'row_term', 'column_term', 'value', 'status', 'accepted')


@dataclass
class CoxFit:
    estimates: pd.DataFrame
    coefficients: pd.Series
    covariance: pd.DataFrame
    model_covariance: pd.DataFrame
    metadata: dict


def contract_key(target):
    if target not in TARGETS:
        raise ValueError('Unsupported pre-TB Cox target.')
    return 'pretb_cox/' + target


def fit_spec(target):
    contract_key(target)
    return {'target': target, **TARGETS[target], 'terms': list(TERMS),
            'carry_forward_years': 2.0, 'time_scale': 'attained_age_years',
            'ties': 'efron', 'risk_membership': 'entry <= event_age <= stop',
            'likelihood': 'Cox partial likelihood', 'optimizer': 'newton',
            'maxiter': 100, 'tol': 1e-8, 'ridge_factor': 1e-10,
            'score_residuals': 'statsmodels.PHReg.score_residuals',
            'score_residual_version': '0.14.5',
            'covariance': 'child-cluster source score-residual sandwich',
            'covariance_method': 'Efron information bread; legacy PHReg 0.14.5 Breslow-style scores with reverse-time residual membership',
            'covariance_correction': 'G/(G-1)*(N-1)/(N-3)',
            'confidence_multiplier': 1.96, 'minimum_source_events': 3,
            'multiplicity': 'Benjamini-Hochberg over the 17 selected finite LF p-values'}


def _require(condition, message):
    if not condition:
        raise ValueError('Pre-TB Cox contract ' + message)


def validate_contract(contract, target):
    """Validate the frozen interval contract without changing its sample."""
    contract_key(target)
    _require(isinstance(contract, dict), 'must be a mapping.')
    frame, audit = contract.get('frame'), contract.get('audit')
    _require(isinstance(frame, pd.DataFrame) and isinstance(audit, dict), 'is incomplete.')
    columns = ('pid_child', 'start', 'stop', 'event', *TERMS)
    _require(set(columns).issubset(frame), 'is missing required interval fields.')
    _require(tuple(contract.get('terms', ())) == TERMS, 'terms differ from the selected design.')
    _require(audit.get('family') == 'pretb_cox' and audit.get('column') == target,
             'target does not match its preparation audit.')
    _require(audit.get('carry_forward_years') == 2.0, 'carry-forward window differs.')
    _require(not frame.pid_child.isna().any(), 'contains missing cluster labels.')
    numeric = frame.loc[:, ['start', 'stop', 'event', *TERMS]].to_numpy(dtype=float)
    _require(np.isfinite(numeric).all(), 'contains nonfinite values.')
    start, stop, event = numeric[:, 0], numeric[:, 1], numeric[:, 2]
    _require(np.all(start >= 0) and np.all(stop > start), 'has invalid attained-age intervals.')
    _require(np.all(stop-start <= 2.0 + 1e-12), 'exceeds the selected carry-forward window.')
    _require(np.isin(event, [0., 1.]).all(), 'event indicators are not binary.')
    _require(np.isin(numeric[:, 4:], [0., 1.]).all(), 'sex/site indicators are not binary.')
    design = np.asarray(contract.get('design'), dtype=float)
    _require(design.shape == (len(frame), 3) and np.array_equal(design, numeric[:, 3:]),
             'matrix differs from the named interval covariates.')
    _require(not frame.duplicated(['pid_child', 'start']).any(), 'has duplicate child/start intervals.')
    for _, part in frame.groupby('pid_child', sort=False):
        starts, stops = part.start.to_numpy(float), part.stop.to_numpy(float)
        _require(np.all(np.diff(starts) > 0), 'within-child intervals are not ordered.')
        _require(np.all(stops[:-1] <= starts[1:]), 'within-child intervals overlap in their interiors.')
        _require(int(part.event.sum()) <= 1, 'contains more than one first event per child.')
        if part.event.any():
            _require(bool(part.event.iloc[-1]), 'contains follow-up after its first event.')
    counts = {'n_children': int(frame.pid_child.nunique()), 'n_intervals': len(frame),
              'n_subsequent_tb': int(event.sum()), 'person_years': float(np.sum(stop-start))}
    _require(audit.get('n_children') == counts['n_children']
             and audit.get('n_intervals') == counts['n_intervals']
             and audit.get('events') == counts['n_subsequent_tb']
             and np.isclose(audit.get('person_years', np.nan), counts['person_years'], rtol=1e-12, atol=1e-12),
             'counts disagree with the preparation audit.')
    return {'target': target, 'contract_key': contract_key(target), 'terms': list(TERMS),
            'counts': counts, 'specification': fit_spec(target)}


def _bit_hash(matrix):
    return hashlib.sha256(np.packbits(matrix, axis=None, bitorder='little').tobytes()).hexdigest()


def risk_set_audit(frame, model=None):
    """Return aggregate mathematical risk sets and actual source score support."""
    start, stop = frame.start.to_numpy(float), frame.stop.to_numpy(float)
    event = frame.event.to_numpy(bool)
    times = np.unique(stop[event])
    risk = (start[None, :] <= times[:, None]) & (stop[None, :] >= times[:, None])
    failures = event[None, :] & (stop[None, :] == times[:, None])
    codes = pd.Categorical(frame.pid_child.to_numpy()).codes
    at_entry = start[None, :] == times[:, None]
    rows = []
    for i, age in enumerate(times):
        nrows = int(risk[i].sum())
        nclusters = len(np.unique(codes[risk[i]]))
        entry = risk[i] & at_entry[i]
        rows.append({'event_age_years': float(age), 'events': int(failures[i].sum()),
                     'risk_rows': nrows, 'risk_children': nclusters,
                     'entry_at_event_rows': int(entry.sum()),
                     'entry_at_event_children': len(np.unique(codes[entry])),
                     'additional_same_child_risk_rows': nrows-nclusters})
    result = {'membership_shape': list(risk.shape), 'membership_sha256': _bit_hash(risk),
              'event_membership_sha256': _bit_hash(failures), 'event_times': rows,
              'intervals_outside_event_risk_sets': int((~risk.any(axis=0)).sum())}
    if model is not None:
        # Replay only the source membership bookkeeping, not its floating-point
        # residual calculations. Rows with no intervening failure can leave the
        # reverse-time set before they enter it; later membership is preserved.
        support = np.zeros_like(risk)
        surv = model.surv
        for stx in range(surv.nstrat):
            active = set()
            for i in range(len(surv.ufailt[stx]))[::-1]:
                active.update(surv.risk_enter[stx][i])
                global_i = int(np.searchsorted(times, surv.ufailt[stx][i]))
                support[global_i, surv.stratum_rows[stx][list(active)]] = True
                active.difference_update(surv.risk_exit[stx][i])
        result['likelihood_membership_contributions'] = int(risk.sum())
        result['source_score_membership_contributions'] = int(support.sum())
        result['source_score_membership_sha256'] = _bit_hash(support)
        result['source_score_membership_difference_cells'] = int(np.count_nonzero(support != risk))
        result['source_score_support_intervals'] = int(support.any(axis=0).sum())
        result['source_undefined_score_intervals'] = int((~support.any(axis=0)).sum())
        return result, support.any(axis=0)
    return result


def _clean_source_scores(scores, supported):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape != (len(supported), 3):
        raise ArithmeticError('source_score_shape')
    undefined = np.isnan(scores).all(axis=1)
    if not np.array_equal(undefined, ~supported):
        raise ArithmeticError('source_score_undefined_support')
    if not np.isfinite(scores[supported]).all():
        raise ArithmeticError('nonfinite_source_scores')
    clean = scores.copy()
    clean[undefined] = 0.0
    return clean


def _matrix_valid(matrix, *, positive_definite=False):
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        return False
    eig = np.linalg.eigvalsh(matrix)
    if positive_definite:
        return bool(eig.min() > 0)
    return bool(eig.min() >= -1e-10 * max(1., np.linalg.norm(matrix, ord=2))
                and np.all(np.diag(matrix) > 0))


def _result(metadata, beta=None, robust=None, model_cov=None):
    beta = np.full(3, np.nan) if beta is None else np.asarray(beta)
    robust = np.full((3, 3), np.nan) if robust is None else robust
    model_cov = np.full((3, 3), np.nan) if model_cov is None else model_cov
    estimated = metadata['status'] == 'estimated'
    rows = []
    for i, term in enumerate(TERMS):
        se = float(np.sqrt(robust[i, i])) if estimated else np.nan
        m_se = float(np.sqrt(model_cov[i, i])) if estimated else np.nan
        b = float(beta[i]) if estimated else np.nan
        rows.append({'target': metadata['target'], 'term': term, 'coefficient': b,
                     'robust_se': se, 'model_se': m_se,
                     'hazard_ratio': float(np.exp(b)), 'ci_low': float(np.exp(b-1.96*se)),
                     'ci_high': float(np.exp(b+1.96*se)),
                     'p_value': float(2*norm.sf(abs(b/se))) if estimated else np.nan,
                     'status': metadata['status'], 'accepted': metadata['accepted'],
                     'reason': metadata['reason']})
    # Failed fits expose no inferential estimates, even if an optimizer returned
    # a finite candidate. Diagnostic candidate metrics stay in metadata.
    if not estimated:
        beta, robust, model_cov = np.full(3, np.nan), np.full((3,3), np.nan), np.full((3,3), np.nan)
    return CoxFit(pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS),
                  pd.Series(beta, index=TERMS, name='coefficient'),
                  pd.DataFrame(robust, index=TERMS, columns=TERMS),
                  pd.DataFrame(model_cov, index=TERMS, columns=TERMS), metadata)


def fit_cox(contract, target):
    """Fit the selected source model with explicit acceptance diagnostics."""
    metadata = validate_contract(contract, target)
    metadata.update({'accepted': False, 'status': 'fit_failed', 'reason': '',
                     'warnings': [], 'optimizer': {}, 'diagnostics': {},
                     'covariance_correction': None,
                     'covariance_method': fit_spec(target)['covariance_method']})
    frame = contract['frame']
    metadata['risk_sets'] = risk_set_audit(frame)
    n, g, events = len(frame), metadata['counts']['n_children'], metadata['counts']['n_subsequent_tb']
    if events < 3:
        metadata.update(accepted=True, status='non_estimable', reason='source_minimum_event_rule')
        metadata['warnings'] = ['Fewer than three subsequent TB events']
        return _result(metadata)
    if statsmodels.__version__ != '0.14.5':
        raise RuntimeError('Selected Cox score-residual behavior requires statsmodels 0.14.5.')
    if n <= 3 or g <= 1 or np.linalg.matrix_rank(contract['design']) != 3:
        metadata['reason'] = 'singular_design_or_insufficient_support'
        return _result(metadata)
    model = PHReg(frame.stop.to_numpy(float), np.asarray(contract['design'], float),
                  status=frame.event.to_numpy(int), entry=frame.start.to_numpy(float), ties='efron')
    metadata['risk_sets'], supported = risk_set_audit(frame, model)
    correction = g/(g-1)*(n-1)/(n-3)
    metadata['covariance_correction'] = float(correction)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            # PHReg.fit delegates to this identical base optimizer, then drops
            # mle_retvals. Calling the base retains convergence evidence.
            fitted = LikelihoodModel.fit(model, start_params=None, method='newton',
                       maxiter=100, full_output=True, disp=False, tol=1e-8, ridge_factor=1e-10)
        metadata['warnings'] = [type(w.message).__name__ + ': ' + str(w.message) for w in caught]
        ret = fitted.mle_retvals
        metadata['optimizer'] = {'method': 'newton', 'iterations': int(ret['iterations']),
            'converged': bool(ret['converged']), 'warnflag': int(ret['warnflag']),
            'maxiter': 100, 'tol': 1e-8, 'ridge_factor': 1e-10}
        beta = np.asarray(fitted.params, float)
        if not np.isfinite(beta).all():
            raise ArithmeticError('nonfinite_coefficients')
        score = model.score(beta)
        info = -model.hessian(beta)
        loglike = float(model.loglike(beta))
        metadata['diagnostics'] = {'log_partial_likelihood': loglike if np.isfinite(loglike) else None,
              'score': [float(x) if np.isfinite(x) else None for x in score],
              'max_absolute_score': float(np.max(abs(score))) if np.isfinite(score).all() else None,
              'information_condition_number': float(np.linalg.cond(info)) if np.isfinite(info).all() else None}
        if not ret['converged'] or ret['warnflag'] != 0:
            raise ArithmeticError('optimizer_did_not_converge')
        if not np.isfinite(loglike) or not np.isfinite(score).all() or np.max(abs(score)) > 1e-5:
            raise ArithmeticError('likelihood_or_score_failed')
        if not _matrix_valid(info, positive_definite=True):
            raise ArithmeticError('information_not_positive_definite')
        bread = np.asarray(fitted.cov_params(), float)
        scores = _clean_source_scores(model.score_residuals(beta), supported)
        codes = pd.Categorical(frame.pid_child.to_numpy()).codes
        grouped = np.zeros((g, 3), float)
        np.add.at(grouped, codes, scores)
        robust = bread @ (grouped.T @ grouped) @ bread * correction
        if not _matrix_valid(bread, positive_definite=True) or not _matrix_valid(robust):
            raise ArithmeticError('invalid_covariance')
        metadata['diagnostics'].update({'model_covariance_min_eigenvalue': float(np.linalg.eigvalsh(bread).min()),
              'robust_covariance_min_eigenvalue': float(np.linalg.eigvalsh(robust).min()),
              'robust_covariance_max_asymmetry': float(np.max(abs(robust-robust.T))),
              'score_residual_group_sum': grouped.sum(axis=0).tolist(),
              'undefined_score_rows_zeroed': int((~supported).sum())})
        # Exponentiation must produce finite complete inference for every term.
        se = np.sqrt(np.diag(robust))
        if not np.isfinite(np.exp(np.r_[beta, beta-1.96*se, beta+1.96*se])).all():
            raise ArithmeticError('nonfinite_hazard_ratio_inference')
        metadata.update(accepted=True, status='estimated', reason='')
        return _result(metadata, beta, robust, bread)
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
        # Do not disclose library exception messages that might contain data.
        metadata['reason'] = str(exc) if isinstance(exc, ArithmeticError) and not isinstance(exc, np.linalg.LinAlgError) else 'numerical_fit_failure'
        return _result(metadata)


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        if np.any((p[valid] < 0) | (p[valid] > 1)):
            raise ValueError('P-values must lie between zero and one.')
        order = valid[np.argsort(p[valid])]
        adjusted = p[order]*len(order)/np.arange(1, len(order)+1)
        q[order] = np.minimum(1., np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def fit_selected(contracts):
    """Fit exactly the selected 17 outcomes, retaining every output row."""
    expected = {contract_key(t) for t in DEFAULT_TARGETS}
    if not isinstance(contracts, dict) or set(contracts) != expected:
        raise ValueError('Pre-TB Cox selection requires exactly the 17 selected contracts.')
    fits = [fit_cox(contracts[contract_key(t)], t) for t in DEFAULT_TARGETS]
    estimates, covariance = [], []
    for target, fit in zip(DEFAULT_TARGETS, fits):
        spec, meta, row = TARGETS[target], fit.metadata, fit.estimates.iloc[0]
        estimates.append({'target': target, 'measure': spec['measure'], 'measure_label': spec['measure_label'],
            'source_column': target, 'scale': spec['scale'], 'carry_forward_window_years': 2.,
            **meta['counts'], 'status': meta['status'], 'accepted': meta['accepted'], 'reason': meta['reason'],
            'coefficient': row.coefficient, 'robust_se': row.robust_se,
            'hr_per_adverse_sd': row.hazard_ratio, 'ci_low': row.ci_low, 'ci_high': row.ci_high,
            'p_value': row.p_value, 'fdr_q_value': np.nan, 'warning': '; '.join(meta['warnings'])})
        for kind, matrix in (('robust', fit.covariance), ('model', fit.model_covariance)):
            for term1 in TERMS:
                for term2 in TERMS:
                    covariance.append({'target': target, 'covariance_type': kind, 'row_term': term1,
                        'column_term': term2, 'value': matrix.loc[term1, term2],
                        'status': meta['status'], 'accepted': meta['accepted']})
    estimates = pd.DataFrame(estimates, columns=ESTIMATE_COLUMNS)
    estimates['fdr_q_value'] = benjamini_hochberg(estimates.p_value)
    accepted = all(f.metadata['accepted'] for f in fits)
    return {'tables': {'estimates': estimates,
                      'coefficients': pd.concat([f.estimates for f in fits], ignore_index=True),
                      'covariance': pd.DataFrame(covariance, columns=COVARIANCE_COLUMNS)},
            'metadata': {'schema_version': 1, 'accepted': accepted,
                         'status': 'accepted' if accepted else 'fit_failed',
                         'selection': list(DEFAULT_TARGETS), 'models': [f.metadata for f in fits],
                         'multiplicity': {'method': 'Benjamini-Hochberg', 'family': 'selected_17',
                                          'finite_tests': int(estimates.p_value.notna().sum())}}}
