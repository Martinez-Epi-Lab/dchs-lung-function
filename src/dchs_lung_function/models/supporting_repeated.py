"""Selected paired-MMEF GEE and repeated/trajectory X6 models.

M03 supplies verified samples and designs. This module neither reads files nor
returns participant records. The paired change is calculated from float64
operands; all source-scale transformations and resampling controls are explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2, norm
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from ..preparation import repeated_design
from .spirometry_trajectories import (
    FULL_TERMS, REDUCED_TERMS, random_intercept_fit, height_standardization,
    expected_height_z, prediction_vector, estimate_prediction,
)

SPEC_VERSION = '1.0.0'
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260903
X6_SPLINE_FORMULA = 'cr(age, df=3, constraints="center", lower_bound=3, upper_bound=10) - 1'
CONTRACT_KEYS = ('mmef/change',
                 'x6_repeated/x6_sa_z/1', 'x6_repeated/x6_sa_z/5',
                 'x6_repeated/reactance_6hz/1', 'x6_repeated/reactance_6hz/5',
                 'trajectory/x6/1', 'trajectory/x6/5')
TABLE_NAMES = ('mmef_change', 'mmef_descriptive', 'x6_repeated', 'x6_global_tests',
               'x6_predictions', 'x6_age_contrasts', 'coefficients', 'covariance')
MMEF_SCOPES = (
    ('before5', 'TB before age 5', ('post_before5', 'post_at_after5', 'future_tb'), 'post_before5', 5),
    ('infancy', 'TB during infancy', ('post_infancy', 'post_age1_before5', 'post_at_after5', 'future_tb'), 'post_infancy', 1),
)
MMEF_GROUPS = ('No recorded TB', 'Before future TB', 'TB before age 5',
               'TB during infancy', 'TB at ages 1 to <5', 'TB at age >=5')
COEFFICIENT_COLUMNS = ('model_id', 'family', 'target', 'threshold', 'fit', 'term',
                       'coefficient', 'robust_se', 'model_se')
COVARIANCE_COLUMNS = ('model_id', 'family', 'target', 'threshold', 'fit', 'covariance_type',
                      'row_term', 'column_term', 'covariance')


class NumericalFailure(ValueError):
    """A valid sample could not yield accepted numerical inference."""


@dataclass
class GaussianFit:
    coefficients: pd.Series
    covariance: pd.DataFrame
    model_covariance: pd.DataFrame
    metadata: dict


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _same(actual, expected, label, *, exact=False):
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape or (not np.array_equal(a, e) if exact else
                              not np.allclose(a, e, rtol=1e-12, atol=1e-12)):
        raise ValueError(label+' does not match the selected prepared specification.')


def _frame(contract, family, column, date_column, bounds=None):
    try:
        frame, audit = contract['frame'], contract['audit']
        needed = {'pid_child', 'test_date', date_column, 'child_dob', 'tb_first_dx',
                  'tb_age_first_years', 'height', 'male', 'visit_label', 'age_years',
                  'post_any', 'future_tb', 'post_before5', 'post_infancy',
                  'post_age1_before5', 'post_at_after5', 'outcome_value', column}
        if not isinstance(frame, pd.DataFrame) or frame.empty or not needed.issubset(frame) or audit['family'] != family:
            raise ValueError('A complete nonempty selected M03 contract is required.')
        if frame.pid_child.isna().any() or frame.pid_child.astype('string').str.strip().eq('').any():
            raise ValueError('The supporting sample contains a missing child linkage key.')
        for name in ['height', 'male', 'age_years', 'outcome_value', column]:
            if not np.isfinite(frame[name].to_numpy(float)).all():
                raise ValueError('Supporting outcomes and covariates must be finite and complete.')
        if not frame.male.isin([0, 1]).all() or frame.visit_label.isna().any():
            raise ValueError('Supporting sex or visit covariates are invalid.')
        dates, modality_dates, dob, dx = [pd.to_datetime(frame[name]) for name in
                                         ['test_date', date_column, 'child_dob', 'tb_first_dx']]
        if dates.isna().any() or dob.isna().any() or not dates.equals(modality_dates):
            raise ValueError('The supporting measurement date differs from its selected modality.')
        age = (dates-dob).dt.days/365.25
        _same(frame.age_years, age, 'Exact measurement age', exact=True)
        if bounds is not None and not age.between(*bounds, inclusive='both').all():
            raise ValueError('The supporting sample is outside its selected exact-age window.')
        post, future = dx.notna() & dates.ge(dx), dx.notna() & dates.lt(dx)
        tb_age = pd.to_numeric(frame.tb_age_first_years, errors='raise')
        if (dx.notna() & (~np.isfinite(tb_age) | tb_age.lt(0))).any():
            raise ValueError('Recorded first TB requires a finite nonnegative diagnosis age.')
        if frame.groupby('pid_child', observed=True).tb_age_first_years.nunique(dropna=False).gt(1).any():
            raise ValueError('A supporting child has inconsistent first-diagnosis ages.')
        truth = {'post_any': post, 'future_tb': future, 'post_before5': post & tb_age.lt(5),
                 'post_infancy': post & tb_age.lt(1),
                 'post_age1_before5': post & tb_age.ge(1) & tb_age.lt(5),
                 'post_at_after5': post & tb_age.ge(5)}
        for name, expected in truth.items():
            _same(frame[name], expected.astype(float), name, exact=True)
        counts = {'n_observations': len(frame), 'n_children': int(frame.pid_child.nunique())}
        if any(audit[k] != v for k, v in counts.items()):
            raise ValueError('Supporting sample counts differ from their M03 audit.')
        return frame, counts
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('The supporting contract lacks required M03 fields.') from exc


def _design(frame, terms, given, names, dropped=None):
    expected, expected_names, expected_dropped = repeated_design(frame, list(terms))
    _same(given, expected, 'Repeated-measures design')
    if list(names) != expected_names or (dropped is not None and list(dropped) != expected_dropped):
        raise ValueError('Supporting design terms or dropped-term evidence have changed.')
    if any(t not in names for t in terms):
        raise ValueError('A required supporting exposure term is absent from the selected design.')
    if not np.isfinite(expected).all() or np.linalg.matrix_rank(expected, tol=1e-10) != expected.shape[1]:
        raise ValueError('The supporting design is nonfinite or rank deficient.')
    return np.asarray(given, dtype=float), list(names), expected_dropped


def validate_mmef(contract):
    frame, counts = _frame(contract, 'mmef_change', 'mmef', 'date_of_test')
    if 'mmef_post' not in frame or not np.isfinite(frame.mmef_post.to_numpy(float)).all():
        raise ValueError('Paired MMEF requires finite pre- and post-bronchodilator observations.')
    if frame.duplicated(['pid_child', 'date_of_test']).any():
        raise ValueError('Paired MMEF contains duplicate child/test-date observations.')
    # M03 can store float32 subtraction. Validate that original contract first,
    # then perform the selected source calculation with float64 operands.
    prepared_delta = frame.mmef_post-frame.mmef
    _same(frame.outcome_value, prepared_delta, 'Prepared paired change', exact=True)
    y = frame.mmef_post.to_numpy(float)-frame.mmef.to_numpy(float)
    designs = {}
    if set(contract['designs']) != {'before5', 'infancy'}:
        raise ValueError('Paired MMEF requires exactly the two selected exposure designs.')
    for key, _, terms, _, _ in MMEF_SCOPES:
        saved = contract['designs'][key]
        designs[key] = _design(frame, terms, saved['matrix'], saved['terms'], saved['dropped_terms'])
    difference = y-frame.outcome_value.to_numpy(float)
    precision = {'prepared_delta_dtype': str(frame.outcome_value.dtype), 'model_delta_dtype': 'float64',
                 'formula': 'float64(mmef_post) - float64(mmef)',
                 'changed_numeric_values': int(np.count_nonzero(difference)),
                 'maximum_absolute_rounding_difference': float(np.max(np.abs(difference))),
                 'sample_and_designs_unchanged': True}
    return frame, y, designs, counts, precision


def validate_repeated(contract, outcome, threshold):
    if outcome not in ('x6_sa_z', 'reactance_6hz') or threshold not in (1, 5):
        raise ValueError('Unsupported repeated X6 selection.')
    frame, counts = _frame(contract, 'x6_repeated', outcome, 'date_of_test_ilf2', (3., 17.))
    _same(frame.outcome_value, frame[outcome], 'Native X6 outcome', exact=True)
    target = frame.post_any.eq(1) & frame.tb_age_first_years.lt(threshold)
    later = frame.post_any.eq(1) & frame.tb_age_first_years.ge(threshold)
    _same(frame.target, target.astype(float), 'Target X6 state', exact=True)
    _same(frame.later, later.astype(float), 'Later X6 state', exact=True)
    if list(contract['audit']['terms']) != list(contract['terms']):
        raise ValueError('Repeated X6 audit terms differ from the design.')
    design = _design(frame, ('target', 'later', 'future_tb'), contract['design'],
                     contract['terms'], contract['audit']['dropped_terms'])
    return frame, design, counts


def x6_spline_basis(ages):
    age = np.asarray(ages, dtype=float)
    if age.ndim != 1 or not np.isfinite(age).all() or not ((age >= 3) & (age <= 10)).all():
        raise ValueError('X6 trajectory ages must be within three through ten years.')
    matrix = patsy.dmatrix(X6_SPLINE_FORMULA, {'age': age}, return_type='dataframe', NA_action='raise')
    return matrix.to_numpy(float), matrix.design_info


def x6_basis_at(info, ages):
    age = np.asarray(ages, dtype=float)
    if age.ndim != 1 or not np.isfinite(age).all() or not ((age >= 3) & (age <= 10)).all():
        raise ValueError('X6 prediction ages must be within three through ten years.')
    return np.asarray(patsy.build_design_matrices([info], {'age': age})[0], dtype=float)


def validate_trajectory(contract, threshold):
    if threshold not in (1, 5):
        raise ValueError('Unsupported X6 trajectory cutoff.')
    frame, counts = _frame(contract, 'trajectory', 'x6_sa_z', 'date_of_test_ilf2', (3., 10.))
    audit = contract['audit']
    if audit['outcome'] != 'x6_sa_z' or audit['threshold'] != threshold:
        raise ValueError('The X6 trajectory selector differs from its preparation audit.')
    _same(frame.outcome_value, frame.x6_sa_z, 'X6 trajectory outcome', exact=True)
    basis, info = x6_spline_basis(frame.age_years)
    target = (frame.post_any.eq(1) & frame.tb_age_first_years.lt(threshold)).to_numpy(float)
    later = (frame.post_any.eq(1) & frame.tb_age_first_years.ge(threshold)).to_numpy(float)
    height = frame.height.to_numpy(float)
    sd = height.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('X6 trajectory height has no positive finite scale.')
    expected = np.column_stack([np.ones(len(frame)), target, later, frame.future_tb,
                                (height-height.mean())/sd, frame.male, basis, target[:, None]*basis])
    for name, values in [('basis', basis), ('design', expected), ('reduced_design', expected[:, :9])]:
        _same(contract[name], values, 'X6 trajectory '+name)
    if (list(contract['terms']) != list(FULL_TERMS) or list(audit['terms']) != list(FULL_TERMS)
            or list(contract['reduced_terms']) != list(REDUCED_TERMS)
            or contract['spline_formula'] != X6_SPLINE_FORMULA or audit['spline_formula'] != X6_SPLINE_FORMULA):
        raise ValueError('The X6 trajectory terms or spline formula differ from the selected specification.')
    if np.linalg.matrix_rank(expected, tol=1e-10) != 12:
        raise ValueError('The X6 trajectory design is rank deficient.')
    return frame, counts, info


def _failed_gaussian(terms, counts, reason):
    p = len(terms)
    return GaussianFit(pd.Series(np.full(p, np.nan), index=terms),
                       pd.DataFrame(np.full((p, p), np.nan), index=terms, columns=terms),
                       pd.DataFrame(np.full((p, p), np.nan), index=terms, columns=terms),
                       {**counts, 'terms': list(terms), 'n_parameters': p, 'accepted': False,
                        'status': 'fit_failed', 'failure_reasons': [reason], 'converged': False,
                        'finite_sample_correction': 1., 'covariance': 'child HC0 sandwich'})


def fit_gaussian_gee(y, x, groups, terms):
    """Gaussian independence GEE, with the uncorrected child HC0 covariance."""
    y, x, groups = np.asarray(y, float), np.asarray(x, float), np.asarray(groups)
    counts = {'n_observations': len(y), 'n_children': len(np.unique(groups))}
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape or x.shape[1] != len(terms):
        raise ValueError('Gaussian GEE arrays or terms have incompatible dimensions.')
    if not np.isfinite(y).all() or not np.isfinite(x).all() or pd.isna(groups).any():
        raise ValueError('Gaussian GEE arrays must be finite with complete child linkage.')
    if len(y) <= x.shape[1] or counts['n_children'] <= 2 or np.linalg.matrix_rank(x, tol=1e-10) != x.shape[1]:
        return _failed_gaussian(terms, counts, 'Insufficient Gaussian GEE sample or rank.')
    try:
        with warnings.catch_warnings(record=True) as caught:
            fitted = GEE(y, x, groups=groups, family=Gaussian(), cov_struct=Independence()).fit(maxiter=250, cov_type='robust')
        beta, covariance, model_cov = np.asarray(fitted.params), np.asarray(fitted.cov_params()), np.asarray(fitted.cov_naive)
        ols = np.linalg.lstsq(x, y, rcond=None)[0]
        if not fitted.converged or not np.allclose(beta, ols, atol=1e-7, rtol=1e-7):
            raise NumericalFailure('Gaussian independence GEE did not converge to its OLS estimator.')
        for matrix in [covariance, model_cov]:
            scale = max(1., float(np.max(np.abs(matrix))))
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10*scale, rtol=1e-8):
                raise NumericalFailure('Gaussian GEE covariance is nonfinite or asymmetric.')
            if np.linalg.eigvalsh((matrix+matrix.T)/2).min() < -1e-8*scale:
                raise NumericalFailure('Gaussian GEE covariance is not positive semidefinite.')
        if not np.isfinite(beta).all() or not np.isfinite(fitted.scale) or fitted.scale <= 0:
            raise NumericalFailure('Gaussian GEE coefficient or residual scale is invalid.')
        metadata = {**counts, 'terms': list(terms), 'n_parameters': len(terms), 'accepted': True,
                    'status': 'accepted', 'failure_reasons': [], 'converged': True,
                    'covariance': 'child HC0 sandwich', 'finite_sample_correction': 1.,
                    'condition_number': float(np.linalg.cond(x.T@x)), 'residual_variance': float(fitted.scale),
                    'maximum_OLS_coefficient_difference': float(np.max(np.abs(beta-ols))),
                    'working_correlation': 'Independence', 'family': 'Gaussian',
                    'warning_categories': sorted({type(w.message).__name__ for w in caught})}
        return GaussianFit(pd.Series(beta, index=terms), pd.DataFrame(covariance, index=terms, columns=terms),
                           pd.DataFrame(model_cov, index=terms, columns=terms), metadata)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        reason = str(error) if isinstance(error, NumericalFailure) else 'Gaussian GEE numerical fitting failure.'
        return _failed_gaussian(terms, counts, reason)


def cluster_resampling(frame, x, y, term_index, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Source stratified whole-child bootstrap and exact leave-one-child range."""
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates <= 0:
        raise ValueError('Bootstrap repetitions must be a positive integer.')
    group, children = pd.factorize(frame.pid_child, sort=True)
    p = x.shape[1]
    xtx, xty = np.zeros((len(children), p, p)), np.zeros((len(children), p))
    np.add.at(xtx, group, x[:, :, None]*x[:, None, :])
    np.add.at(xty, group, x*y[:, None])
    age = pd.to_numeric(frame.groupby('pid_child', observed=True).tb_age_first_years.first().reindex(children), errors='coerce').to_numpy()
    strata = np.select([age < 1, (age >= 1) & (age < 5), age >= 5], [1, 2, 3], default=0)
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(replicates):
        weights = np.zeros(len(children))
        for level in np.unique(strata):
            index = np.flatnonzero(strata == level)
            weights[index] = rng.multinomial(len(index), np.full(len(index), 1./len(index)))
        information, score = np.einsum('g,gij->ij', weights, xtx), weights@xty
        if np.linalg.matrix_rank(information) == p:
            estimates.append(np.linalg.solve(information, score)[term_index])
    full_a, full_b = x.T@x, x.T@y
    jack = [np.linalg.solve(full_a-a, full_b-b)[term_index] for a, b in zip(xtx, xty)
            if np.linalg.matrix_rank(full_a-a) == p]
    if not estimates or not jack or not np.isfinite(estimates).all() or not np.isfinite(jack).all():
        raise NumericalFailure('Paired-change resampling has no finite estimable distribution.')
    row = {'bootstrap_replicates': len(estimates), 'bootstrap_ci_low': float(np.quantile(estimates, .025)),
           'bootstrap_ci_high': float(np.quantile(estimates, .975)),
           'leave_one_child_out_min': float(np.min(jack)), 'leave_one_child_out_max': float(np.max(jack))}
    audit = {'seed': seed, 'requested_replicates': replicates, 'accepted_replicates': len(estimates),
             'rank_deficient_replicates': replicates-len(estimates), 'resampling_unit': 'whole child',
             'stratum_child_counts': {str(level): int((strata == level).sum()) for level in np.unique(strata)},
             'leave_one_child_out_estimable': len(jack), 'leave_one_child_out_rank_deficient': len(children)-len(jack),
             'table_interval': 'Gaussian GEE robust normal interval; bootstrap is supporting evidence'}
    return row, audit


def _collect(fit, model_id, family, target, threshold, variant, coefficients, covariances, models):
    identity = {'model_id': model_id, 'family': family, 'target': target, 'threshold': threshold, 'fit': variant}
    metadata = {**fit.metadata, **identity}
    models.append(_safe(metadata))
    for term in fit.coefficients.index:
        se = math.sqrt(max(float(fit.covariance.loc[term, term]), 0.))
        model_se = math.sqrt(max(float(fit.model_covariance.loc[term, term]), 0.))
        coefficients.append({**identity, 'term': term, 'coefficient': float(fit.coefficients[term]),
                             'robust_se': se, 'model_se': model_se})
    for kind, covariance in [('robust', fit.covariance), ('model', fit.model_covariance)]:
        for a in covariance.index:
            for b in covariance.columns:
                covariances.append({**identity, 'covariance_type': kind, 'row_term': a, 'column_term': b,
                                    'covariance': float(covariance.loc[a, b])})


def _failed_inference(fit, reason):
    """Retain the expected coefficient inventory while nulling failed inference."""
    fit.coefficients[:] = np.nan
    fit.covariance.iloc[:, :] = np.nan
    fit.model_covariance.iloc[:, :] = np.nan
    fit.metadata.update(accepted=False, status='fit_failed', failure_reasons=[reason])


def _normal_effect(fit, term):
    estimate = float(fit.coefficients[term])
    variance = float(fit.covariance.loc[term, term])
    if not np.isfinite(estimate+variance) or variance <= 0:
        raise NumericalFailure('A supporting exposure contrast has no finite positive robust variance.')
    se = math.sqrt(variance)
    return estimate, se, estimate-1.96*se, estimate+1.96*se, float(2*norm.sf(abs(estimate/se)))


def _mmef(checked, tables, models, coefficients, covariance):
    frame, y, designs, counts, precision = checked
    descriptions = []
    masks = [frame.tb_first_dx.isna(), frame.future_tb.eq(1), frame.post_before5.eq(1),
             frame.post_infancy.eq(1), frame.post_age1_before5.eq(1), frame.post_at_after5.eq(1)]
    pre, post = frame.mmef.astype(float), frame.mmef_post.astype(float)
    delta = pd.Series(y, index=frame.index)
    for label, mask in zip(MMEF_GROUPS, masks):
        values = delta.loc[mask]
        descriptions.append({'target': 'mmef', 'outcome': 'MMEF (FEF25–75)', 'unit': 'L/s', 'group': label,
                             'n_pairs': int(mask.sum()), 'n_children': int(frame.loc[mask, 'pid_child'].nunique()),
                             'mean_pre': float(pre.loc[mask].mean()), 'mean_post': float(post.loc[mask].mean()),
                             'mean_change': float(values.mean()), 'sd_change': float(values.std(ddof=1)),
                             'median_change': float(values.median())})
    resampling = {}
    inferential = ('estimate', 'se', 'ci_low', 'ci_high', 'p_value', 'standardized_reference_change',
                   'standardized_reference_ci_low', 'standardized_reference_ci_high', 'standardized_exposed_change',
                   'standardized_exposed_ci_low', 'standardized_exposed_ci_high', 'bootstrap_replicates',
                   'bootstrap_ci_low', 'bootstrap_ci_high', 'leave_one_child_out_min', 'leave_one_child_out_max')
    for key, exposure, terms, term, threshold in MMEF_SCOPES:
        x, names, dropped = designs[key]
        model_id = 'mmef/change/'+key
        fit = fit_gaussian_gee(y, x, frame.pid_child.to_numpy(), names)
        exposed = frame[term].eq(1)
        row = {'model_id': model_id, 'fit': 'full', 'design': key, 'target': 'mmef',
               'outcome': 'MMEF (FEF25–75)', 'unit': 'L/s', 'exposure': exposure,
               'sensitivity': 'Primary paired post-minus-pre response',
               **dict.fromkeys(inferential, np.nan), 'n_pairs': len(frame), 'n_children': counts['n_children'],
               'n_exposed_pairs': int(exposed.sum()), 'n_exposed_children': int(frame.loc[exposed, 'pid_child'].nunique()),
               'converged': False, 'dropped_terms': '; '.join(dropped),
               'adjustment': 'visit category, height, sex, future TB and later TB states'}
        if fit.metadata['accepted']:
            try:
                beta, se, low, high, p = _normal_effect(fit, term)
                ref_x = x.copy()
                for t in terms:
                    ref_x[:, names.index(t)] = 0.
                ref = np.mean(ref_x, axis=0); exp = ref.copy(); exp[names.index(term)] = 1.
                rmean, _, rlow, rhigh = estimate_prediction(ref, fit)
                emean, _, elow, ehigh = estimate_prediction(exp, fit)
                bootstrap, audit = cluster_resampling(frame, x, y, names.index(term))
                resampling[key] = audit
                row.update(estimate=beta, se=se, ci_low=low, ci_high=high, p_value=p,
                           standardized_reference_change=rmean, standardized_reference_ci_low=rlow,
                           standardized_reference_ci_high=rhigh, standardized_exposed_change=emean,
                           standardized_exposed_ci_low=elow, standardized_exposed_ci_high=ehigh,
                           converged=True, **bootstrap)
            except (NumericalFailure, ValueError, np.linalg.LinAlgError) as error:
                _failed_inference(fit, str(error) if isinstance(error, NumericalFailure) else 'Paired-change inference failed.')
        tables['mmef_change'].append(row)
        _collect(fit, model_id, 'MMEF-CHANGE-GEE', 'mmef', threshold, 'full', coefficients, covariance, models)
    tables['mmef_descriptive'].extend(descriptions)
    return precision, resampling


def _x6_counts(frame, threshold):
    target = frame.post_any.eq(1) & frame.tb_age_first_years.lt(threshold)
    reference = frame.tb_first_dx.isna()
    counts = {'n_observations': len(frame), 'n_children': int(frame.pid_child.nunique())}
    for label, mask in [('exposed', target), ('reference', reference)]:
        counts['n_'+label+'_observations'] = int(mask.sum())
        counts['n_'+label+'_children'] = int(frame.loc[mask, 'pid_child'].nunique())
    return counts, reference


def _repeated(checked, outcome, threshold, tables, models, coefficients, covariance):
    frame, (x, names, dropped), _ = checked
    model_id = f'x6_repeated/{outcome}/{threshold}'
    fit = random_intercept_fit(frame.outcome_value.to_numpy(float), x, frame.pid_child.to_numpy(), names)
    counts, reference = _x6_counts(frame, threshold)
    z = outcome == 'x6_sa_z'
    row = {'model_id': model_id, 'fit': 'full', 'outcome': 'x6_z' if z else 'x6_native',
           'outcome_label': 'X6 South African reference z-score' if z else 'X6',
           'unit': 'reference SD' if z else 'hPa*s/L', 'tb_age_cutoff_years': threshold,
           'contrast': f'Post-TB after first TB <{threshold} vs no recorded TB',
           **dict.fromkeys(['estimate', 'robust_se', 'ci_low', 'ci_high', 'p_value', 'icc', 'condition_number'], np.nan),
           **counts, 'reference_measurement_mean': float(frame.loc[reference, 'outcome_value'].mean()),
           'reference_measurement_sd': float(frame.loc[reference, 'outcome_value'].std(ddof=1)),
           'interpretation': 'Negative estimate means more negative/worse X6'}
    if fit.metadata['accepted']:
        try:
            beta, se, low, high, p = _normal_effect(fit, 'target')
            row.update(estimate=beta, robust_se=se, ci_low=low, ci_high=high, p_value=p,
                       icc=fit.metadata['icc'], condition_number=fit.metadata['condition_number'])
        except NumericalFailure as error:
            _failed_inference(fit, str(error))
    fit.metadata['dropped_design_terms'] = dropped
    tables['x6_repeated'].append(row)
    _collect(fit, model_id, 'X6-REPEATED', outcome, threshold, 'full', coefficients, covariance, models)


def _wald(fit, indexes):
    beta = fit.coefficients.to_numpy()[indexes]
    covariance = fit.covariance.to_numpy()[np.ix_(indexes, indexes)]
    rank = int(np.linalg.matrix_rank(covariance))
    statistic = float(beta@np.linalg.pinv(covariance)@beta)
    if rank != len(indexes) or not np.isfinite(statistic) or statistic < 0:
        raise NumericalFailure('The X6 trajectory Wald covariance is rank deficient or invalid.')
    return statistic, rank, float(chi2.sf(statistic, rank))


def _trajectory(contract, checked, threshold, tables, models, coefficients, covariance):
    frame, _, info = checked
    model_id = f'trajectory/x6/{threshold}'
    full = random_intercept_fit(frame.outcome_value.to_numpy(float), contract['design'], frame.pid_child.to_numpy(), FULL_TERMS)
    reduced = random_intercept_fit(frame.outcome_value.to_numpy(float), contract['reduced_design'], frame.pid_child.to_numpy(), REDUCED_TERMS)
    counts, _ = _x6_counts(frame, threshold)
    base = {'model_id': model_id, 'fit': 'full', 'outcome': 'x6_sa_z',
            'outcome_label': 'X6 South African-reference z-score', 'unit': 'reference SD', 'tb_age_cutoff_years': threshold}
    fields = ('overall_trajectory_wald_chi2', 'overall_trajectory_df', 'overall_trajectory_p',
              'interaction_wald_chi2', 'interaction_df', 'interaction_p', 'parallel_trajectory_difference',
              'parallel_trajectory_robust_se', 'parallel_trajectory_ci_low', 'parallel_trajectory_ci_high',
              'parallel_trajectory_p', 'icc', 'condition_number', 'random_intercept_variance', 'residual_variance')
    row = {**base, **counts, **dict.fromkeys(fields, np.nan)}
    height = height_standardization(frame)
    transform = next(iter(next(iter(info.factor_infos.values())).state['transforms'].values()))
    standardization = {'height': height, 'spline': {'formula': X6_SPLINE_FORMULA,
                       'knots': np.asarray(transform._all_knots).tolist(),
                       'centering_constraints': np.asarray(transform._constraints).tolist()},
                       'reference': 'no recorded TB', 'future_and_later_indicators': 0.}
    predictions, contrasts = [], []
    groups = [('No recorded TB', False), (f'First TB <{threshold} years', True)]
    dense, ages = np.linspace(3., 10., 101), np.arange(3., 11.)
    if full.metadata['accepted'] and reduced.metadata['accepted']:
        try:
            interaction, overall = _wald(full, [9, 10, 11]), _wald(full, [1, 9, 10, 11])
            parallel, pse, plow, phigh, pp = _normal_effect(reduced, 'target')
            row.update(overall_trajectory_wald_chi2=overall[0], overall_trajectory_df=overall[1], overall_trajectory_p=overall[2],
                       interaction_wald_chi2=interaction[0], interaction_df=interaction[1], interaction_p=interaction[2],
                       parallel_trajectory_difference=parallel, parallel_trajectory_robust_se=pse,
                       parallel_trajectory_ci_low=plow, parallel_trajectory_ci_high=phigh, parallel_trajectory_p=pp,
                       icc=full.metadata['icc'], condition_number=full.metadata['condition_number'],
                       random_intercept_variance=full.metadata['tau2'], residual_variance=full.metadata['sigma2'])
            for i, (basis, height_z) in enumerate(zip(x6_basis_at(info, dense), expected_height_z(dense, height))):
                for label, exposed in groups:
                    vector = prediction_vector(FULL_TERMS, basis, height_z, height['male_mean'], exposed)
                    estimate, se, low, high = estimate_prediction(vector, full)
                    predictions.append({**base, 'age_years': float(dense[i]), 'group': label,
                                        'predicted_mean': estimate, 'robust_se': se, 'ci_low': low, 'ci_high': high})
            for age, basis, height_z in zip(ages, x6_basis_at(info, ages), expected_height_z(ages, height)):
                exposed = prediction_vector(FULL_TERMS, basis, height_z, height['male_mean'], True)
                reference = prediction_vector(FULL_TERMS, basis, height_z, height['male_mean'], False)
                estimate, se, low, high = estimate_prediction(exposed-reference, full)
                contrasts.append({**base, 'age_years': float(age), 'difference_tb_minus_reference': estimate,
                                  'robust_se': se, 'ci_low': low, 'ci_high': high})
        except (NumericalFailure, ValueError, np.linalg.LinAlgError):
            _failed_inference(full, 'X6 trajectory inference failed.')
            _failed_inference(reduced, 'X6 trajectory inference failed.')
    if not full.metadata['accepted'] or not reduced.metadata['accepted']:
        row.update(dict.fromkeys(fields, np.nan))
        predictions = [{**base, 'age_years': float(age), 'group': label,
                        **dict.fromkeys(['predicted_mean', 'robust_se', 'ci_low', 'ci_high'], np.nan)}
                       for age in dense for label, _ in groups]
        contrasts = [{**base, 'age_years': float(age),
                      **dict.fromkeys(['difference_tb_minus_reference', 'robust_se', 'ci_low', 'ci_high'], np.nan)} for age in ages]
    tables['x6_global_tests'].append(row)
    tables['x6_predictions'].extend(predictions)
    tables['x6_age_contrasts'].extend(contrasts)
    for label, fit in [('full', full), ('reduced', reduced)]:
        _collect(fit, model_id, 'X6-TRAJECTORY', 'x6_sa_z', threshold, label, coefficients, covariance, models)
    return standardization


def fit_selected(contracts: Mapping) -> dict:
    """Fit the seven selected M03 contracts, returning ten aggregate fits."""
    if not isinstance(contracts, Mapping) or set(contracts) != set(CONTRACT_KEYS):
        raise ValueError('Supporting repeated models require exactly the seven selected M03 contracts.')
    # Validate every sample before starting any model calculation.
    checked = {'mmef/change': validate_mmef(contracts['mmef/change'])}
    for key in CONTRACT_KEYS[1:5]:
        _, outcome, threshold = key.split('/')
        checked[key] = validate_repeated(contracts[key], outcome, int(threshold))
    for key in CONTRACT_KEYS[5:]:
        checked[key] = validate_trajectory(contracts[key], int(key.rsplit('/', 1)[1]))
    tables = {name: [] for name in TABLE_NAMES}
    models, coefficients, covariance = [], tables['coefficients'], tables['covariance']
    precision, resampling = _mmef(checked['mmef/change'], tables, models, coefficients, covariance)
    for key in CONTRACT_KEYS[1:5]:
        _, outcome, threshold = key.split('/')
        _repeated(checked[key], outcome, int(threshold), tables, models, coefficients, covariance)
    standardization = {}
    for key in CONTRACT_KEYS[5:]:
        threshold = int(key.rsplit('/', 1)[1])
        standardization[str(threshold)] = _trajectory(contracts[key], checked[key], threshold, tables, models, coefficients, covariance)
    result_tables = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    result_tables['coefficients'] = result_tables['coefficients'].reindex(columns=COEFFICIENT_COLUMNS)
    result_tables['covariance'] = result_tables['covariance'].reindex(columns=COVARIANCE_COLUMNS)
    failures = [m['model_id']+'/'+m['fit']+': '+reason for m in models for reason in m['failure_reasons']]
    metadata = {'specification_version': SPEC_VERSION, 'accepted': not failures,
                'status': 'fit_failed' if failures else 'accepted', 'failure_reasons': failures,
                'fitted_models': 10, 'analyses': 8, 'contract_keys': list(CONTRACT_KEYS), 'models': models,
                'mmef_delta_precision': precision, 'mmef_resampling': resampling,
                'x6_prediction_standardization': standardization,
                'row_counts': {name: len(frame) for name, frame in result_tables.items()},
                'p_value_adjustment': None, 'interval_multiplier': 1.96,
                'MMEF_table_interval': 'uncorrected robust Gaussian GEE; bootstrap not substituted',
                'X6_covariance': 'child sandwich with G/(G-1)*(N-1)/(N-P)',
                'participant_records_returned': False}
    return {'tables': result_tables, 'metadata': _safe(metadata)}
