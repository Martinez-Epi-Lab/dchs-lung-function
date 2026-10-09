"""Numerical trajectory tests on invented longitudinal Gaussian observations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.preparation import trajectory_inputs
from dchs_lung_function.models.spirometry_trajectories import (
    DEFAULT_FIT_SELECTION, FULL_TERMS, REDUCED_TERMS, basis_at, fit_spec,
    fit_trajectory, random_intercept_fit, spline_basis, validate_contract,
)


def fabricated_contract(outcome='fev1_fvc', threshold=1):
    rng = np.random.default_rng(6749)
    child = np.repeat(np.arange(100), 4)
    n = len(child)
    days = np.tile([1860, 2350, 2850, 3470], 100)+rng.integers(-20, 21, n)
    dob = pd.Timestamp('2020-01-01')
    dates = dob+pd.to_timedelta(days, unit='D')
    group = child % 5
    diagnosis_days = np.select([group == 1, group == 2, group == 3, group == 4],
                               [4200., 183., 730., 2192.], default=np.nan)
    diagnosis = dob+pd.to_timedelta(diagnosis_days, unit='D')
    age = days/365.25
    male = np.repeat(rng.integers(0, 2, 100), 4).astype(float)
    height = 115.+5*(age-5)+2*male+rng.normal(0, 3, n)
    post = np.isfinite(diagnosis_days) & (days >= diagnosis_days)
    target = post & (diagnosis_days/365.25 < threshold)
    intercept = np.repeat(rng.normal(0, 1.3, 100), 4)
    y = 89.-.5*(age-7.5)+.09*(age-7.5)**2-.15*male-.04*(height-height.mean())
    y += target*(-2.+.45*(age-7.5))+intercept+rng.normal(0, 1.1, n)
    frame = pd.DataFrame({'pid_child': ['fabricated_'+str(i) for i in child],
                          'date_of_test': dates, 'child_dob': dob,
                          'tb_first_dx': diagnosis, 'tb_age_first_years': diagnosis_days/365.25,
                          'male': male, 'height': height, 'visit_label': 'spirometry',
                          'fev1_fvc': y/100., 'fev1_fvc_gli_z': (y-90)/8.})
    return trajectory_inputs(frame, outcome=outcome, threshold=threshold)


class TrajectoryTests(unittest.TestCase):
    def test_four_selected_families_and_full_output_inventory(self):
        self.assertEqual(len(DEFAULT_FIT_SELECTION), 4)
        self.assertEqual(len(FULL_TERMS), 12); self.assertEqual(len(REDUCED_TERMS), 9)
        with self.assertRaises(ValueError): fit_spec('fev1_fvc', 3)
        with self.assertRaises(ValueError): fit_spec('x6_sa_z', 1)
        result = fit_trajectory(fabricated_contract())
        self.assertTrue(result.metadata['accepted'], result.metadata['failure_reasons'])
        self.assertEqual(len(result.global_tests), 1)
        self.assertEqual(len(result.predictions), 202)
        self.assertEqual(len(result.age_contrasts), 6)
        self.assertEqual(set(result.predictions.age_years.agg(['min', 'max'])), {5., 10.})
        self.assertEqual(result.age_contrasts.age_years.tolist(), [5., 6., 7., 8., 9., 10.])
        self.assertEqual(result.global_tests.interaction_df.iloc[0], 3)
        self.assertEqual(result.global_tests.overall_trajectory_df.iloc[0], 4)
        self.assertFalse(any('bh' in column.lower() for column in result.global_tests))
        json.dumps(result.metadata, allow_nan=False)
        json.dumps(result.full.metadata, allow_nan=False)

    def test_gls_objective_model_and_robust_covariance_match_dense_independent_matrices(self):
        rng = np.random.default_rng(554)
        groups = np.repeat(np.arange(30), 3)
        x = np.column_stack([np.ones(len(groups)), rng.normal(size=len(groups))])
        y = 1.+.5*x[:, 1]+np.repeat(rng.normal(0, 1.7, 30), 3)+rng.normal(0, .8, len(groups))
        fit = random_intercept_fit(y, x, groups, ['Intercept', 'x'])
        self.assertTrue(fit.metadata['accepted'])
        lam = fit.metadata['lambda']
        v = np.eye(len(y))+lam*(groups[:, None] == groups[None, :])
        inverse = np.linalg.inv(v); information = x.T@inverse@x
        beta = np.linalg.solve(information, x.T@inverse@y)
        residual = y-x@beta
        sigma2 = float(residual@inverse@residual)/(len(y)-2)
        logdet_v = 2*np.log(np.diag(np.linalg.cholesky(v))).sum()
        logdet_information = 2*np.log(np.diag(np.linalg.cholesky(information))).sum()
        objective = (len(y)-2)*np.log(sigma2)+logdet_v+logdet_information
        meat = np.zeros((2, 2))
        for group in np.unique(groups):
            take = groups == group
            score = x[take].T@np.linalg.solve(v[np.ix_(take, take)], residual[take])
            meat += np.outer(score, score)
        bread = np.linalg.inv(information)
        correction = 30/29*(len(y)-1)/(len(y)-2)
        np.testing.assert_allclose(fit.coefficients, beta, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.model_covariance, sigma2*bread, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit.covariance, correction*bread@meat@bread, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(fit.metadata['objective'], objective, places=10)
        self.assertAlmostEqual(fit.metadata['finite_sample_correction'], correction, places=14)

    def test_zero_random_intercept_boundary_is_retained_exactly(self):
        groups = np.repeat(np.arange(12), 2)
        x = np.ones((24, 1))
        y = np.tile([-1., 1.], 12)
        result = random_intercept_fit(y, x, groups, ['Intercept'])
        self.assertTrue(result.metadata['accepted'])
        self.assertEqual(result.metadata['lambda'], 0.)
        self.assertEqual(result.metadata['tau2'], 0.)
        self.assertEqual(result.metadata['optimizer']['boundary'], 'zero')
        self.assertEqual(result.metadata['optimizer']['golden_steps'], 0)

    def test_singular_and_no_residual_variance_models_fail_explicitly(self):
        groups = np.repeat(np.arange(10), 3)
        y = np.arange(30, dtype=float)
        result = random_intercept_fit(y, np.ones((30, 2)), groups, ['a', 'b'])
        self.assertFalse(result.metadata['accepted']); self.assertEqual(result.metadata['status'], 'fit_failed')
        self.assertTrue(result.coefficients.isna().all())
        result = random_intercept_fit(np.zeros(30), np.ones((30, 1)), groups, ['Intercept'])
        self.assertFalse(result.metadata['accepted'])
        self.assertIn('profile', result.metadata['failure_reasons'][0])

    def test_training_spline_transform_is_fixed_at_grid_endpoints_and_subsets(self):
        training = np.r_[np.linspace(5.1, 6.2, 60), np.linspace(7.2, 9.9, 20)]
        basis, info = spline_basis(training)
        np.testing.assert_allclose(basis_at(info, training), basis, atol=0., rtol=0.)
        grid = np.linspace(5., 10., 101)
        whole = basis_at(info, grid)
        np.testing.assert_array_equal(basis_at(info, np.array([5., 10.])), whole[[0, 100]])
        refitted, _ = spline_basis(grid)
        self.assertGreater(np.max(np.abs(whole-refitted)), .05)
        with self.assertRaises(ValueError): basis_at(info, [4.99])

    def test_contract_identity_and_tampered_dates_basis_or_design_fail(self):
        for change in ['cutoff', 'basis', 'design', 'tb_age', 'measurement_age', 'outcome', 'count']:
            contract = fabricated_contract()
            if change == 'cutoff': contract['audit']['threshold'] = 3
            elif change == 'basis':
                contract['basis'] = contract['basis'].copy()
                contract['basis'][0, 0] += .1
            elif change == 'design': contract['design'][0, 1] = 1-contract['design'][0, 1]
            elif change == 'tb_age':
                row = contract['frame'].post_any.eq(1).idxmax()
                contract['frame'].loc[row, 'tb_age_first_years'] = np.nan
            elif change == 'measurement_age': contract['frame'].loc[0, 'age_years'] += .01
            elif change == 'outcome': contract['frame'].loc[0, 'outcome_value'] += 1.
            else: contract['audit']['n_children'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError): validate_contract(contract)

    def test_numerical_failure_keeps_null_output_rows_and_exact_counts(self):
        contract = fabricated_contract(outcome='fev1_fvc_gli_z')
        contract['frame'].loc[:, ['fev1_fvc_gli_z', 'outcome_value']] = 0.
        result = fit_trajectory(contract)
        self.assertFalse(result.metadata['accepted'])
        self.assertEqual(result.metadata['counts']['n_observations'], 400)
        self.assertEqual(len(result.predictions), 202); self.assertEqual(len(result.age_contrasts), 6)
        self.assertTrue(result.predictions.predicted_mean.isna().all())
        self.assertTrue(result.age_contrasts.difference_tb_minus_reference.isna().all())
        self.assertTrue(result.global_tests.interaction_p.isna().all())
        json.dumps(result.metadata, allow_nan=False)

    def test_fitting_preserves_frames_designs_and_common_sample(self):
        contract = fabricated_contract(); before = deepcopy(contract)
        result = fit_trajectory(contract)
        pd.testing.assert_frame_equal(contract['frame'], before['frame'])
        for key in ['design', 'reduced_design', 'basis']:
            np.testing.assert_array_equal(contract[key], before[key])
        self.assertEqual(result.full.metadata['n_observations'], result.reduced.metadata['n_observations'])
        self.assertEqual(result.full.metadata['n_children'], result.reduced.metadata['n_children'])


if __name__ == '__main__':
    unittest.main()
