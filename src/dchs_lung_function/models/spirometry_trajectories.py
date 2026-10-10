"""Gaussian random-intercept spirometry trajectories, version 1.0.0.

REML variance profiling, the child-cluster sandwich, and standardized
predictions are explicit. The four selected families retain raw Wald p values;
no multiple-testing family is created from this subset. Outputs are aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2

MODEL_ID = 'SPIRO-TRAJECTORY'
SPEC_VERSION = '1.0.0'
AGE_MIN, AGE_MAX = 5., 10.
SPLINE_FORMULA = 'cr(age, df=3, constraints="center", lower_bound=5, upper_bound=10) - 1'
FULL_TERMS = ('Intercept', 'target', 'later', 'future_tb', 'height_z', 'sex:Male',
              'age_spline_1', 'age_spline_2', 'age_spline_3',
              'target:age_spline_1', 'target:age_spline_2', 'target:age_spline_3')
REDUCED_TERMS = FULL_TERMS[:9]
DEFAULT_FIT_SELECTION = tuple((outcome, threshold) for outcome in ('fev1_fvc', 'fev1_fvc_gli_z') for threshold in (1, 5))


@dataclass(frozen=True)
class RandomInterceptFit:
    coefficients: pd.Series
    covariance: pd.DataFrame
    model_covariance: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryFit:
    full: RandomInterceptFit
    reduced: RandomInterceptFit
    global_tests: pd.DataFrame
    predictions: pd.DataFrame
    age_contrasts: pd.DataFrame
    metadata: dict[str, Any]


class _NumericalFailure(ValueError):
    pass


def fit_spec(outcome: str, threshold: int) -> dict[str, Any]:
    if isinstance(threshold, (bool, np.bool_)) or (outcome, threshold) not in DEFAULT_FIT_SELECTION:
        raise ValueError('Only the selected ratio or GLI-ratio trajectories with cutoffs one and five are supported.')
    native = outcome == 'fev1_fvc'
    return {'model_id': MODEL_ID, 'specification_version': SPEC_VERSION,
            'target': outcome, 'threshold': int(threshold),
            'outcome': 'fev1_fvc_percent' if native else outcome,
            'outcome_label': 'FEV1/FVC' if native else 'FEV1/FVC GLI-Global z-score',
            'unit': 'percentage points' if native else 'reference SD',
            'measurement_multiplier': 100. if native else 1.,
            'family': 'Gaussian random-intercept model', 'estimation': 'profile REML',
            'age_range': [AGE_MIN, AGE_MAX], 'spline_formula': SPLINE_FORMULA,
            'full_terms': list(FULL_TERMS), 'reduced_terms': list(REDUCED_TERMS),
            'covariance': 'child-cluster robust sandwich',
            'finite_sample_correction_formula': 'g/(g-1)*(n-1)/(n-p)',
            'optimizer': {'positive_grid_points': 90, 'log_grid_min': -10., 'log_grid_max': 12.,
                          'include_zero': True, 'golden_steps': 60},
            'prediction_grid': {'minimum': AGE_MIN, 'maximum': AGE_MAX, 'points': 101},
            'contrast_ages': [5., 6., 7., 8., 9., 10.],
            'interval_multiplier': 1.96, 'p_value_adjustment': None,
            'prediction_standardization': 'measurement-weighted sex mean and cubic expected height; future/later TB set to zero'}


def spline_basis(ages):
    """Fit the selected spline transform to the full analysis-age sample once."""
    age = np.asarray(ages, dtype=float)
    if age.ndim != 1 or not np.isfinite(age).all() or not ((age >= AGE_MIN) & (age <= AGE_MAX)).all():
        raise ValueError('Spline training ages must be finite and within five through ten years.')
    basis = patsy.dmatrix(SPLINE_FORMULA, {'age': age}, return_type='dataframe', NA_action='raise')
    return basis.to_numpy(dtype=float), basis.design_info


def basis_at(design_info, ages):
    """Evaluate an existing training transform; never refit knots or centering."""
    age = np.asarray(ages, dtype=float)
    if age.ndim != 1 or not np.isfinite(age).all() or not ((age >= AGE_MIN) & (age <= AGE_MAX)).all():
        raise ValueError('Prediction ages must be finite and within the selected age range.')
    return np.asarray(patsy.build_design_matrices([design_info], {'age': age})[0], dtype=float)


def height_standardization(frame):
    """Cubic expected height and measurement-weighted standardization constants."""
    age = frame['age_years'].to_numpy(dtype=float)-7.5
    male = frame['male'].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(frame)), age, age**2, age**3, male])
    beta = np.linalg.lstsq(x, frame['height'].to_numpy(dtype=float), rcond=None)[0]
    # Preserve the prepared height dtype for the source's mean/SD calculation.
    state = {'coefficients': beta.tolist(), 'mean': float(frame['height'].mean()),
             'sd': float(frame['height'].std(ddof=1)), 'male_mean': float(male.mean()), 'age_center': 7.5}
    if not np.isfinite([*state['coefficients'], state['mean'], state['sd'], state['male_mean']]).all() or state['sd'] <= 0:
        raise ValueError('Prediction height standardization has no finite positive scale.')
    return state


def expected_height_z(ages, state):
    age = np.asarray(ages, dtype=float)-state['age_center']
    x = np.column_stack([np.ones(len(age)), age, age**2, age**3, np.repeat(state['male_mean'], len(age))])
    return (x@np.asarray(state['coefficients'])-state['mean'])/state['sd']


def prediction_vector(terms, age_basis, height_z, male_mean, exposed):
    basis = np.asarray(age_basis, dtype=float)
    if basis.shape != (3,) or not np.isfinite([*basis, height_z, male_mean]).all() or not set(terms).issubset(FULL_TERMS):
        raise ValueError('Prediction vector requires the fitted three-column basis and valid model terms.')
    values = {'Intercept': 1., 'target': float(exposed), 'later': 0., 'future_tb': 0.,
              'height_z': float(height_z), 'sex:Male': float(male_mean)}
    for i in range(3):
        values[f'age_spline_{i+1}'] = basis[i]
        values[f'target:age_spline_{i+1}'] = float(exposed)*basis[i]
    return np.asarray([values[name] for name in terms], dtype=float)


def estimate_prediction(vector, fit: RandomInterceptFit):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != fit.coefficients.shape or not np.isfinite(vector).all() or not fit.metadata['accepted']:
        raise ValueError('Predictions require a valid contrast vector and accepted fit.')
    estimate = float(vector@fit.coefficients.to_numpy())
    variance = float(vector@fit.covariance.to_numpy()@vector)
    tolerance = 1e-10*max(1., float(np.max(np.abs(fit.covariance.to_numpy())))*float(vector@vector))
    if not np.isfinite(estimate+variance) or variance < -tolerance:
        raise _NumericalFailure('Prediction variance is invalid.')
    se = math.sqrt(max(variance, 0.))
    return estimate, se, estimate-1.96*se, estimate+1.96*se


def _failed_fit(terms, n, g, reason):
    p = len(terms)
    metadata = {'accepted': False, 'status': 'fit_failed', 'failure_reasons': [reason],
                'terms': list(terms), 'n_observations': n, 'n_children': g, 'n_parameters': p,
                'lambda': None, 'sigma2': None, 'tau2': None, 'icc': None, 'objective': None,
                'condition_number': None, 'finite_sample_correction': None, 'optimizer': None}
    return RandomInterceptFit(pd.Series(np.full(p, np.nan), index=terms, name='coefficient'),
                              pd.DataFrame(np.full((p, p), np.nan), index=terms, columns=terms),
                              pd.DataFrame(np.full((p, p), np.nan), index=terms, columns=terms), metadata)


def random_intercept_fit(y, x, groups, terms=None) -> RandomInterceptFit:
    """Selected 91-candidate REML profile and 60-step log-ratio refinement."""
    y, x, groups = np.asarray(y, dtype=float), np.asarray(x, dtype=float), np.asarray(groups)
    if x.ndim != 2 or y.shape != (x.shape[0],) or groups.shape != y.shape:
        raise ValueError('Gaussian model arrays have incompatible dimensions.')
    n, p = x.shape
    names = list(terms) if terms is not None else [f'x{i}' for i in range(p)]
    if len(names) != p or len(set(names)) != p or not np.isfinite(x).all() or not np.isfinite(y).all() or pd.isna(groups).any():
        raise ValueError('Gaussian model arrays or term names are invalid.')
    order = np.argsort(groups, kind='stable')
    y, x, groups = y[order], x[order], groups[order]
    unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    g = len(unique)
    if n <= p or g <= 2:
        return _failed_fit(names, n, g, 'Insufficient observations or child clusters.')
    if np.linalg.matrix_rank(x, tol=1e-10) != p:
        return _failed_fit(names, n, g, 'The Gaussian fixed-effects design is singular.')
    sx, sy = np.add.reduceat(x, starts, axis=0), np.add.reduceat(y, starts)
    xtx, xty, yty = x.T@x, x.T@y, float(y@y)

    def components(lam):
        a = lam/(1.+counts*lam)
        xt_v_x = xtx-sx.T@(a[:, None]*sx)
        xt_v_y = xty-sx.T@(a*sy)
        y_v_y = yty-float(np.sum(a*sy*sy))
        try:
            beta = np.linalg.solve(xt_v_x, xt_v_y)
        except np.linalg.LinAlgError:
            return math.inf, None
        sse = y_v_y-float(beta@xt_v_y)
        logdet_v = float(np.sum(np.log1p(counts*lam)))
        sign, logdet_x = np.linalg.slogdet(xt_v_x)
        if sign <= 0 or sse <= 0 or not np.isfinite([sse, logdet_v, logdet_x, *beta]).all():
            return math.inf, None
        objective = (n-p)*math.log(sse/(n-p))+logdet_v+logdet_x
        return objective, (beta, sse, xt_v_x, a)

    try:
        lambdas = np.concatenate(([0.], np.exp(np.linspace(-10., 12., 90))))
        objectives = np.asarray([components(float(lam))[0] for lam in lambdas])
        if not np.isfinite(objectives).any():
            raise _NumericalFailure('All REML profile candidates are invalid.')
        best = int(np.nanargmin(objectives))
        if best == 0:
            lam = 0.
        else:
            lo, hi = math.log(lambdas[max(1, best-1)]), math.log(lambdas[min(len(lambdas)-1, best+1)])
            phi = (1.+math.sqrt(5.))/2.
            c, d = hi-(hi-lo)/phi, lo+(hi-lo)/phi
            fc, fd = components(math.exp(c))[0], components(math.exp(d))[0]
            for _ in range(60):
                if fc < fd:
                    hi, d, fd = d, c, fc
                    c = hi-(hi-lo)/phi; fc = components(math.exp(c))[0]
                else:
                    lo, c, fc = c, d, fd
                    d = lo+(hi-lo)/phi; fd = components(math.exp(d))[0]
            lam = math.exp((lo+hi)/2.)
        objective, fitted = components(lam)
        if fitted is None:
            raise _NumericalFailure('Random-intercept REML optimization failed.')
        beta, sse, information, a = fitted
        sigma2, tau2 = sse/(n-p), lam*sse/(n-p)
        bread = np.linalg.inv(information)
        model_cov = sigma2*bread
        residual = y-x@beta
        meat = np.zeros((p, p))
        for start, count, ai in zip(starts, counts, a):
            xi, ri = x[start:start+count], residual[start:start+count]
            score = xi.T@(ri-ai*ri.sum())
            meat += np.outer(score, score)
        correction = (g/(g-1))*((n-1)/(n-p))
        robust_cov = correction*bread@meat@bread
        if not np.isfinite([sigma2, tau2, objective, *beta]).all() or sigma2 <= 0 or tau2 < 0:
            raise _NumericalFailure('Gaussian coefficient or variance estimates are invalid.')
        for covariance in (model_cov, robust_cov):
            scale = max(1., float(np.max(np.abs(covariance))))
            if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-10*scale, rtol=1e-8):
                raise _NumericalFailure('Gaussian covariance is nonfinite or asymmetric.')
            if np.linalg.eigvalsh((covariance+covariance.T)/2).min() < -1e-8*scale:
                raise _NumericalFailure('Gaussian covariance is not positive semidefinite.')
        condition = float(np.linalg.cond(information))
        if not np.isfinite(condition):
            raise _NumericalFailure('Gaussian information matrix has an invalid condition number.')
        metadata = {'accepted': True, 'status': 'accepted', 'failure_reasons': [],
                    'terms': names, 'n_observations': n, 'n_children': g, 'n_parameters': p,
                    'lambda': float(lam), 'sigma2': float(sigma2), 'tau2': float(tau2),
                    'icc': float(tau2/(tau2+sigma2)), 'objective': float(objective),
                    'condition_number': condition, 'finite_sample_correction': float(correction),
                    'optimizer': {'grid_candidates': 91, 'positive_grid': 90,
                                  'golden_steps': 0 if best == 0 else 60, 'grid_winner_index': best,
                                  'boundary': 'zero' if best == 0 else 'upper_grid' if best == 90 else 'interior',
                                  'finite_grid_candidates': int(np.isfinite(objectives).sum())}}
        return RandomInterceptFit(pd.Series(beta, index=names, name='coefficient'),
                                  pd.DataFrame(robust_cov, index=names, columns=names),
                                  pd.DataFrame(model_cov, index=names, columns=names), metadata)
    except (_NumericalFailure, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        return _failed_fit(names, n, g, str(exc))


def _validated(contract):
    try:
        audit = contract['audit']
        spec = fit_spec(audit['outcome'], audit['threshold'])
        frame = contract['frame']
        needed = {'pid_child', 'age_years', 'height', 'male', 'outcome_value', 'test_date',
                  'child_dob', 'tb_first_dx', 'tb_age_first_years', 'future_tb', 'post_any',
                  'fev1_fvc', 'fev1_fvc_gli_z'}
        if audit['family'] != 'trajectory' or not isinstance(frame, pd.DataFrame) or not needed.issubset(frame) or frame.empty:
            raise ValueError('A nonempty M03 spirometry trajectory contract is required.')
        if frame.pid_child.isna().any() or frame.pid_child.astype('string').str.strip().eq('').any():
            raise ValueError('The trajectory frame contains missing child linkage keys.')
        for name in ['age_years', 'height', 'male', 'outcome_value', 'fev1_fvc', 'fev1_fvc_gli_z']:
            if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
                raise ValueError('Trajectory measurements and covariates must be finite and complete.')
        if not frame.age_years.between(AGE_MIN, AGE_MAX).all() or not frame.male.isin([0, 1]).all():
            raise ValueError('Trajectory age or sex is outside the selected specification.')
        dates, dob, dx = (pd.to_datetime(frame[name]) for name in ['test_date', 'child_dob', 'tb_first_dx'])
        if dates.isna().any() or dob.isna().any() or not np.array_equal(((dates-dob).dt.days/365.25).to_numpy(), frame.age_years.to_numpy()):
            raise ValueError('Trajectory measurement age does not match its dates.')
        post = dx.notna() & dates.ge(dx)
        future = dx.notna() & dates.lt(dx)
        if not np.array_equal(post.astype(float), frame.post_any) or not np.array_equal(future.astype(float), frame.future_tb):
            raise ValueError('Trajectory TB timing states do not match diagnosis and measurement dates.')
        if (post & frame.tb_age_first_years.isna()).any():
            raise ValueError('Post-TB measurements require a first-diagnosis age.')
        target = (post & frame.tb_age_first_years.lt(spec['threshold'])).to_numpy(float)
        later = (post & frame.tb_age_first_years.ge(spec['threshold'])).to_numpy(float)
        expected_y = (pd.to_numeric(frame[spec['target']])*spec['measurement_multiplier']).to_numpy(float)
        if not np.array_equal(frame.outcome_value.to_numpy(float), expected_y):
            raise ValueError('Trajectory outcome values do not match the requested outcome and scale.')
        basis, design_info = spline_basis(frame.age_years.to_numpy(float))
        height = frame.height.to_numpy(float)
        sd = height.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError('Trajectory height has no positive finite scale.')
        expected_x = np.column_stack([np.ones(len(frame)), target, later, future.to_numpy(float),
                                      (height-height.mean())/sd, frame.male.to_numpy(float), basis, target[:, None]*basis])
        for name, expected in [('basis', basis), ('design', expected_x), ('reduced_design', expected_x[:, :9])]:
            actual = np.asarray(contract[name], dtype=float)
            if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError('The prepared trajectory basis or design differs from the selected specification.')
        if list(contract['terms']) != list(FULL_TERMS) or list(contract['reduced_terms']) != list(REDUCED_TERMS) or list(audit['terms']) != list(FULL_TERMS):
            raise ValueError('Trajectory design terms differ from the fixed specification.')
        if contract['spline_formula'] != SPLINE_FORMULA or audit['spline_formula'] != SPLINE_FORMULA:
            raise ValueError('Trajectory spline specification has changed.')
        if np.linalg.matrix_rank(expected_x, tol=1e-10) != len(FULL_TERMS):
            raise ValueError('The prepared trajectory design is singular.')
        counts = {'n_observations': len(frame), 'n_children': int(frame.pid_child.nunique())}
        if any(audit[key] != value for key, value in counts.items()):
            raise ValueError('Trajectory sample counts differ from their M03 audit.')
        for label, selected in [('exposed', target == 1), ('reference', dx.isna().to_numpy()),
                                ('future', future.to_numpy()), ('later', later == 1)]:
            counts[f'n_{label}_observations'] = int(selected.sum())
            counts[f'n_{label}_children'] = int(frame.loc[selected, 'pid_child'].nunique())
        return frame, spec, counts, basis, design_info
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('The trajectory contract is missing required M03 fields.') from exc


def validate_contract(contract: Mapping):
    _, spec, counts, _, _ = _validated(contract)
    return {**spec, 'counts': counts}


def _wald(fit, indices):
    beta = fit.coefficients.to_numpy()[indices]
    covariance = fit.covariance.to_numpy()[np.ix_(indices, indices)]
    rank = int(np.linalg.matrix_rank(covariance))
    statistic = float(beta@np.linalg.pinv(covariance)@beta)
    if rank != len(indices) or not np.isfinite(statistic) or statistic < 0:
        raise _NumericalFailure('The trajectory Wald contrast has invalid covariance rank or statistic.')
    return statistic, rank, float(chi2.sf(statistic, rank))


def _spline_state(design_info):
    # Patsy's fitted CR state is retained as aggregate constants for audit;
    # prediction evaluation itself uses build_design_matrices/DesignInfo.
    transform = next(iter(next(iter(design_info.factor_infos.values())).state['transforms'].values()))
    return {'formula': SPLINE_FORMULA, 'knots': np.asarray(transform._all_knots).tolist(),
            'centering_constraints': np.asarray(transform._constraints).tolist()}


def fit_trajectory(contract: Mapping) -> TrajectoryFit:
    frame, spec, counts, _, design_info = _validated(contract)
    y, groups = frame.outcome_value.to_numpy(float), frame.pid_child.to_numpy()
    full = random_intercept_fit(y, contract['design'], groups, FULL_TERMS)
    reduced = random_intercept_fit(y, contract['reduced_design'], groups, REDUCED_TERMS)
    reasons = [f'{label}: {reason}' for label, fitted in [('full', full), ('reduced', reduced)] for reason in fitted.metadata['failure_reasons']]
    height_state = height_standardization(frame)
    base = {'outcome': spec['outcome'], 'outcome_label': spec['outcome_label'], 'unit': spec['unit'],
            'tb_age_cutoff_years': spec['threshold']}
    test_row = {**base, **counts,
                **{key: None for key in ['overall_trajectory_wald_chi2', 'overall_trajectory_df', 'overall_trajectory_p',
                    'interaction_wald_chi2', 'interaction_df', 'interaction_p', 'parallel_trajectory_difference',
                    'parallel_trajectory_robust_se', 'parallel_trajectory_ci_low', 'parallel_trajectory_ci_high',
                    'parallel_trajectory_p', 'icc', 'condition_number', 'random_intercept_variance', 'residual_variance']}}
    prediction_rows, contrast_rows = [], []
    try:
        if reasons:
            raise _NumericalFailure('A component trajectory fit was not accepted.')
        interaction, overall = _wald(full, [9, 10, 11]), _wald(full, [1, 9, 10, 11])
        parallel = float(reduced.coefficients['target'])
        parallel_variance = float(reduced.covariance.loc['target', 'target'])
        if parallel_variance <= 0 or not np.isfinite(parallel_variance):
            raise _NumericalFailure('The reduced trajectory contrast has invalid robust variance.')
        parallel_se = math.sqrt(parallel_variance)
        test_row.update(overall_trajectory_wald_chi2=overall[0], overall_trajectory_df=overall[1], overall_trajectory_p=overall[2],
                        interaction_wald_chi2=interaction[0], interaction_df=interaction[1], interaction_p=interaction[2],
                        parallel_trajectory_difference=parallel, parallel_trajectory_robust_se=parallel_se,
                        parallel_trajectory_ci_low=parallel-1.96*parallel_se,
                        parallel_trajectory_ci_high=parallel+1.96*parallel_se,
                        parallel_trajectory_p=math.erfc(abs(parallel/parallel_se)/math.sqrt(2.)),
                        icc=full.metadata['icc'], condition_number=full.metadata['condition_number'],
                        random_intercept_variance=full.metadata['tau2'], residual_variance=full.metadata['sigma2'])
        dense = np.linspace(AGE_MIN, AGE_MAX, 101)
        dense_basis, dense_height = basis_at(design_info, dense), expected_height_z(dense, height_state)
        for i, age in enumerate(dense):
            for group, exposed in [('No recorded TB', False), (f"First TB <{spec['threshold']} years", True)]:
                vector = prediction_vector(FULL_TERMS, dense_basis[i], dense_height[i], height_state['male_mean'], exposed)
                estimate, se, low, high = estimate_prediction(vector, full)
                prediction_rows.append({**base, 'age_years': float(age), 'group': group,
                                        'predicted_mean': estimate, 'robust_se': se, 'ci_low': low, 'ci_high': high})
        ages = np.asarray(spec['contrast_ages'])
        age_basis, height_z = basis_at(design_info, ages), expected_height_z(ages, height_state)
        for i, age in enumerate(ages):
            exposed = prediction_vector(FULL_TERMS, age_basis[i], height_z[i], height_state['male_mean'], True)
            reference = prediction_vector(FULL_TERMS, age_basis[i], height_z[i], height_state['male_mean'], False)
            estimate, se, low, high = estimate_prediction(exposed-reference, full)
            contrast_rows.append({**base, 'age_years': float(age), 'difference_tb_minus_reference': estimate,
                                  'robust_se': se, 'ci_low': low, 'ci_high': high})
    except _NumericalFailure as exc:
        if not reasons:
            reasons.append(str(exc))
    if reasons:
        # Preserve the complete expected row inventory; failed inference is null.
        for key in set(test_row)-set(base)-set(counts):
            test_row[key] = None
        prediction_rows = [{**base, 'age_years': float(age), 'group': group,
                            'predicted_mean': None, 'robust_se': None, 'ci_low': None, 'ci_high': None}
                           for age in np.linspace(AGE_MIN, AGE_MAX, 101)
                           for group in ['No recorded TB', f"First TB <{spec['threshold']} years"]]
        contrast_rows = [{**base, 'age_years': age, 'difference_tb_minus_reference': None,
                          'robust_se': None, 'ci_low': None, 'ci_high': None} for age in spec['contrast_ages']]
    metadata = {**spec, 'accepted': not reasons, 'status': 'fit_failed' if reasons else 'accepted',
                'failure_reasons': reasons, 'counts': counts,
                'standardization': {'height': height_state, 'spline': _spline_state(design_info)},
                'full_terms': list(FULL_TERMS), 'reduced_terms': list(REDUCED_TERMS)}
    return TrajectoryFit(full, reduced, pd.DataFrame([test_row]), pd.DataFrame(prediction_rows),
                         pd.DataFrame(contrast_rows), metadata)
