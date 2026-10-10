"""Correct pre-TB Cox score membership without fitting or replacing source results.

The accepted Efron coefficients and inverse information stay fixed. Child
scores use the existing Breslow martingale convention, evaluated only when an
interval belongs to the actual inclusive event-time risk set. This correction
does not replace the selected tie or interval-boundary conventions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..execution import canonical_bytes, file_hash, load_artifacts
from ..models import pretb_cox as source_model
from ..pretb_execution import check_result, contract_descriptor

ESTIMATE_COLUMNS = source_model.ESTIMATE_COLUMNS
COEFFICIENT_COLUMNS = source_model.COEFFICIENT_COLUMNS
COVARIANCE_COLUMNS = source_model.COVARIANCE_COLUMNS
DEFAULT_TARGETS = source_model.DEFAULT_TARGETS
TERMS = source_model.TERMS
SPECIFICATION_VERSION = '1.0.0'
METHOD_ID = 'actual_risk_child_sandwich'
COMPARISON_RTOL = 1e-7
COMPARISON_ATOL = 1e-8
RESULT_FILES = ('estimates.csv', 'coefficients.csv', 'covariance.csv',
                'diagnostics.json', 'summary.json')


@dataclass
class CorrectedPreTBCovariance:
    estimates: pd.DataFrame
    coefficients: pd.DataFrame
    covariance: pd.DataFrame
    metadata: dict


def specification():
    return {'specification_version': SPECIFICATION_VERSION, 'method_id': METHOD_ID,
            'likelihood': 'accepted Efron Cox partial likelihood; no refitting',
            'risk_membership': 'start <= event_age <= stop',
            'score_method': 'Breslow weighted covariate means and d/S0 martingale increments',
            'bread': 'unchanged accepted inverse Efron observed information',
            'cluster': 'child', 'finite_correction': 'G/(G-1)*(N-1)/(N-3)',
            'confidence_multiplier': 1.96,
            'multiplicity': 'Benjamini-Hochberg over the selected 17 finite target p-values',
            'source_minimum_events': 3,
            'correction_scope': 'Only score-residual risk-set membership changes.',
            'remaining_conventions': ['Efron-fit/Breslow-score combination retained',
                                      'Inclusive entry/stop event-risk boundaries retained']}


def _require(condition, reason):
    if not condition:
        raise ValueError('Pre-TB uncertainty correction: '+reason)


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise ValueError('Pre-TB uncertainty correction: unreadable source metadata.') from None


def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=COMPARISON_RTOL, atol=COMPARISON_ATOL))


def _matrix(matrix, *, positive_definite=False):
    a = np.asarray(matrix, float)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        return False
    if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
        return False
    eigenvalues = np.linalg.eigvalsh(a)
    return bool(eigenvalues.min() > 0 if positive_definite else
                eigenvalues.min() >= -1e-12*max(1., np.linalg.norm(a, 2))
                and np.all(np.diag(a) > 0))


def _bit_hash(matrix):
    return hashlib.sha256(np.packbits(matrix, axis=None, bitorder='little').tobytes()).hexdigest()


def correct_covariance(contract, target, coefficients, model_covariance):
    """Return a corrected 3-by-3 covariance and aggregate validation evidence.

    Uses the validated intervals, fitted coefficients, and inverse information
    without refitting. Invalid inputs or numerical diagnostics raise ValueError
    and block publication.
    """
    validation = source_model.validate_contract(contract, target)
    frame = contract['frame']
    identifiers = frame.pid_child.astype('string').str.strip()
    _require(not identifiers.isna().any() and not identifiers.eq('').any(),
             'cluster labels must be nonblank.')
    counts = validation['counts']
    n, g = counts['n_intervals'], counts['n_children']
    _require(counts['n_subsequent_tb'] >= 3 and n > 3 and g > 1,
             'numerical correction requires an estimable accepted source fit.')
    x = np.asarray(contract['design'], float)
    beta, bread = np.asarray(coefficients, float), np.asarray(model_covariance, float)
    _require(beta.shape == (3,) and np.isfinite(beta).all(), 'coefficient vector is invalid.')
    _require(_matrix(bread, positive_definite=True), 'source model covariance is invalid.')
    start, stop = frame.start.to_numpy(float), frame.stop.to_numpy(float)
    event = frame.event.to_numpy(bool)
    times = np.unique(stop[event])
    risk = (start[None, :] <= times[:, None]) & (times[:, None] <= stop[None, :])
    failures = event[None, :] & (stop[None, :] == times[:, None])
    _require(np.all(risk[failures]), 'events fall outside their risk intervals.')
    supported = risk.any(axis=0)
    _require(np.linalg.matrix_rank(x[supported]) == 3, 'risk-supported design is rank deficient.')
    active_eta = x[supported] @ beta
    _require(np.isfinite(active_eta).all(), 'risk-supported linear predictors are nonfinite.')
    # A common shift cancels from weighted means, likelihood and d/S0 weights.
    # Irrelevant intervals cannot set that shift or cause risk-weight underflow.
    eta, weights = np.zeros(n), np.zeros(n)
    eta[supported] = active_eta-active_eta.max()
    weights[supported] = np.exp(eta[supported])
    scores, gradient = np.zeros_like(x), np.zeros(3)
    information = np.zeros((3, 3))
    loglike = 0.
    for members, dead in zip(risk, failures):
        xr, wr = x[members], weights[members]
        xd, wd = x[dead], weights[dead]
        d = int(dead.sum())
        s0, s1 = float(wr.sum()), wr @ xr
        s2 = xr.T @ (wr[:, None]*xr)
        e0, e1 = float(wd.sum()), wd @ xd
        e2 = xd.T @ (wd[:, None]*xd)
        _require(d > 0 and np.isfinite(s0) and s0 > 0, 'event risk weights are invalid.')
        mean = s1/s0
        # No score increment exists outside actual event-time risk membership.
        scores[members] += (xr-mean)*(dead[members].astype(float)-wr*d/s0)[:, None]
        loglike += float(eta[dead].sum())
        gradient += xd.sum(axis=0)
        for fraction in np.arange(d, dtype=float)/d:
            denominator = s0-fraction*e0
            _require(np.isfinite(denominator) and denominator > 0,
                     'Efron denominator is invalid.')
            average = (s1-fraction*e1)/denominator
            loglike -= float(np.log(denominator))
            gradient -= average
            information += (s2-fraction*e2)/denominator-np.outer(average, average)
    _require(np.isfinite(loglike) and np.isfinite(gradient).all()
             and np.max(np.abs(gradient)) <= 1e-5, 'accepted coefficient score check failed.')
    _require(_matrix(information, positive_definite=True), 'Efron information is invalid.')
    inverse_information = np.linalg.solve(information, np.eye(3))
    _require(_close(bread, inverse_information), 'accepted bread differs from Efron information.')
    codes = pd.Categorical(frame.pid_child).codes
    _require(np.all(codes >= 0) and len(np.unique(codes)) == g, 'cluster coding differs from support.')
    grouped = np.zeros((g, 3))
    np.add.at(grouped, codes, scores)
    correction = g/(g-1)*(n-1)/(n-3)
    # Multiplication by the unchanged accepted bread preserves its source
    # provenance. Influence crossproducts avoid asymmetric inverse products.
    influence = grouped @ bread.T
    covariance = influence.T @ influence * correction
    _require(_matrix(covariance), 'corrected robust covariance is invalid.')
    se = np.sqrt(np.diag(covariance))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        ratios = np.exp(np.r_[beta, beta-1.96*se, beta+1.96*se])
    _require(np.isfinite(ratios).all() and np.all(ratios > 0), 'corrected HR inference is invalid.')
    empty = ~risk.any(axis=0)
    _require(np.all(scores[empty] == 0), 'event-free intervals acquired score increments.')
    retained = (start <= times[-1]) & (stop >= times[0])
    legacy_extra = (retained & empty)[None, :] & (times[:, None] <= stop[None, :])
    evidence = {'target': target, 'contract_key': source_model.contract_key(target),
                'status': 'estimated', 'accepted': True, 'counts': counts,
                'finite_sample_correction': float(correction),
                'log_partial_likelihood': float(loglike),
                'max_absolute_efron_score': float(np.max(np.abs(gradient))),
                'information_condition_number': float(np.linalg.cond(information)),
                'inverse_information_max_difference': float(np.max(np.abs(bread-inverse_information))),
                'robust_covariance_min_eigenvalue': float(np.linalg.eigvalsh(covariance).min()),
                'robust_covariance_max_asymmetry': float(np.max(np.abs(covariance-covariance.T))),
                'risk_membership_shape': list(risk.shape),
                'risk_membership_sha256': _bit_hash(risk),
                'event_membership_sha256': _bit_hash(failures),
                'event_time_count': len(times), 'risk_memberships': int(risk.sum()),
                'entry_at_event_memberships': int((risk & (start[None, :] == times[:, None])).sum()),
                'intervals_outside_all_event_risk_sets': int(empty.sum()),
                'outside_risk_scores_exactly_zero': True,
                'removed_legacy_score_memberships': int(legacy_extra.sum()),
                'removed_legacy_scored_event_free_intervals': int((retained & empty).sum()),
                'score_cluster_sum': grouped.sum(axis=0).tolist()}
    return covariance, evidence


def _read_tables(directory):
    schemas = {'estimates': ESTIMATE_COLUMNS, 'coefficients': COEFFICIENT_COLUMNS,
               'covariance': COVARIANCE_COLUMNS}
    text = {'target', 'measure', 'measure_label', 'source_column', 'scale', 'status',
            'reason', 'warning', 'term', 'covariance_type', 'row_term', 'column_term', 'accepted'}
    result = {}
    for name, columns in schemas.items():
        frame = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip',
                            keep_default_na=False, na_values={k:[''] for k in columns if k not in text})
        _require(tuple(frame.columns) == columns, 'source table schema differs.')
        result[name] = frame
    return result


def calculate(prep, source_directory):
    """Recalculate complete uncertainty from a verified accepted M09 publication.

    Output schemas and ordering match the source. Explicit source-policy
    non-estimable rows remain null with their support intact. Malformed,
    unaccepted or numerically invalid source results raise ValueError.
    """
    prep, source_directory = Path(prep), Path(source_directory)
    _require(source_directory.is_dir() and not source_directory.is_symlink(), 'source directory is invalid.')
    receipt_path = source_directory/'receipt.json'
    _require(receipt_path.is_file() and not receipt_path.is_symlink(), 'source receipt is invalid.')
    receipt = _read(receipt_path)
    _require(isinstance(receipt, dict) and receipt.get('schema_version') == 1
             and receipt.get('fingerprint') == source_directory.name
             and _digest(receipt.get('identity')) == receipt.get('fingerprint'), 'source identity differs.')
    _require(set(receipt.get('files', {})) == set(RESULT_FILES)
             and {p.name for p in source_directory.iterdir()} == set(RESULT_FILES) | {'receipt.json'},
             'source publication inventory differs.')
    source_hashes = {'receipt.json': file_hash(receipt_path)}
    for name, expected in receipt['files'].items():
        path = source_directory/name
        _require(path.is_file() and not path.is_symlink() and file_hash(path) == expected,
                 'source artifact hash differs.')
        source_hashes[name] = expected
    descriptor = contract_descriptor(prep)
    _require(receipt['identity'].get('selection') == list(DEFAULT_TARGETS)
             and receipt['identity'].get('contracts') == descriptor['contracts'],
             'source sample contracts differ from current preparation.')
    summary = _read(source_directory/'summary.json')
    _require(summary == receipt.get('summary') and summary.get('accepted') is True
             and summary.get('status') == 'accepted', 'source model publication is not accepted.')
    tables = _read_tables(source_directory)
    contracts = {source_model.contract_key(t): load_artifacts(prep, ('M03', 'pft', 'samples', source_model.contract_key(t)))
                 for t in DEFAULT_TARGETS}
    source_metadata = _read(source_directory/'diagnostics.json')
    check_result({'tables': tables, 'metadata': source_metadata}, contracts)
    estimates, coefficients, covariance = [tables[n].copy(deep=True) for n in ('estimates', 'coefficients', 'covariance')]
    models = []
    for target in DEFAULT_TARGETS:
        contract = contracts[source_model.contract_key(target)]
        validation = source_model.validate_contract(contract, target)
        selected = estimates.target.eq(target)
        row = estimates.loc[selected].iloc[0]
        if row.status == 'non_estimable':
            _require(row.reason == 'source_minimum_event_rule' and validation['counts']['n_subsequent_tb'] < 3,
                     'non-estimable status is inconsistent with the source guard.')
            models.append({**validation, 'status': 'non_estimable', 'accepted': True,
                           'reason': 'source_minimum_event_rule', 'correction_performed': False})
            continue
        _require(row.status == 'estimated', 'source contains an unaccepted fit.')
        coefficient_mask = coefficients.target.eq(target)
        source_coefficients = coefficients.loc[coefficient_mask].set_index('term').loc[list(TERMS)]
        beta = source_coefficients.coefficient.to_numpy(float)
        source_model_cov = covariance.loc[covariance.target.eq(target) & covariance.covariance_type.eq('model')]
        bread = source_model_cov.pivot(index='row_term', columns='column_term', values='value').loc[list(TERMS), list(TERMS)].to_numpy(float)
        corrected, evidence = correct_covariance(contract, target, beta, bread)
        se = np.sqrt(np.diag(corrected))
        low, high = np.exp(beta-1.96*se), np.exp(beta+1.96*se)
        p_values = 2*norm.sf(np.abs(beta/se))
        for index, term in enumerate(TERMS):
            mask = coefficient_mask & coefficients.term.eq(term)
            coefficients.loc[mask, ['robust_se', 'ci_low', 'ci_high', 'p_value']] = [se[index], low[index], high[index], p_values[index]]
            for j, other in enumerate(TERMS):
                mask = covariance.target.eq(target) & covariance.covariance_type.eq('robust') & covariance.row_term.eq(term) & covariance.column_term.eq(other)
                covariance.loc[mask, 'value'] = corrected[index, j]
        estimates.loc[selected, ['robust_se', 'ci_low', 'ci_high', 'p_value']] = [se[0], low[0], high[0], p_values[0]]
        evidence.update({'correction_performed': True, 'reason': '',
                         'source_target_robust_se': float(row.robust_se),
                         'corrected_target_robust_se': float(se[0]),
                         'target_relative_se_change': float(se[0]/row.robust_se-1),
                         'source_target_ci': [float(row.ci_low), float(row.ci_high)],
                         'corrected_target_ci': [float(low[0]), float(high[0])],
                         'source_target_p': float(row.p_value), 'corrected_target_p': float(p_values[0]),
                         'accepted_coefficients_and_hazard_ratios_preserved': True})
        models.append(evidence)
    estimates['fdr_q_value'] = source_model.benjamini_hochberg(estimates.p_value)
    unchanged = {
        'estimates': set(ESTIMATE_COLUMNS)-{'robust_se', 'ci_low', 'ci_high', 'p_value', 'fdr_q_value'},
        'coefficients': set(COEFFICIENT_COLUMNS)-{'robust_se', 'ci_low', 'ci_high', 'p_value'},
    }
    for name, frame in [('estimates', estimates), ('coefficients', coefficients)]:
        columns = sorted(unchanged[name])
        _require(frame[columns].equals(tables[name][columns]), 'an accepted point estimate or support value changed.')
    mask = covariance.covariance_type.eq('model')
    _require(covariance.loc[mask].equals(tables['covariance'].loc[mask]), 'accepted model covariance changed.')
    _require(contract_descriptor(prep) == descriptor
             and all(file_hash(source_directory/n) == h for n, h in source_hashes.items()),
             'source or preparation changed during correction.')
    metadata = {'schema_version': 1, 'status': 'accepted', 'accepted': True,
                'specification': specification(), 'selection': list(DEFAULT_TARGETS),
                'rows': {'estimates': len(estimates), 'coefficients': len(coefficients), 'covariance': len(covariance)},
                'source_fingerprint': receipt['fingerprint'], 'source_files': source_hashes,
                'preparation_fingerprint': descriptor['preparation_fingerprint'],
                'contracts': descriptor['contracts'], 'models': models,
                'point_estimates_preserved_exactly': True, 'model_covariance_preserved_exactly': True,
                'source_outputs_modified': False, 'model_fits_performed': 0,
                'multiplicity': {'method': 'Benjamini-Hochberg', 'family': 'selected_17',
                                 'finite_tests': int(estimates.p_value.notna().sum())},
                'inference_provenance': 'Accepted source coefficients and Efron bread; corrected actual-risk child-score covariance, SE, normal Wald CI/p and family BH q; original source artifacts remain separate.',
                'remaining_limitations': specification()['remaining_conventions']}
    # Metadata must not carry participant labels or nonfinite diagnostic values.
    json.dumps(metadata, allow_nan=False)
    return CorrectedPreTBCovariance(estimates, coefficients, covariance, metadata)
