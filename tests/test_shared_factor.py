"""Selected shared-factor orientation, scope, validation and MCSE checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.analysis.shared_factor import (
    DEFAULT_TARGETS, SharedFactorError, derive_shared_factor_draws, summarize_shared_factor,
)


def fixture(target='ratio_pre_z', scale=2.):
    name, stage, _ = target.split('_')
    column = {'ratio': 'fev1_fvc', 'fev1': 'fev1', 'fvc': 'fvc'}[name]
    column += ('_post' if stage == 'post' else '') + '_gli_z'
    frame = pd.DataFrame({'pid_child': ['a', 'a', 'b'], 'outcome_value': [.1, .2, -.1]})
    contract = {'frames': {'target': frame}, 'audit': {'family': 'primary', 'target': column,
                'outcomes': {'target': {'scale': scale, 'n_observations': 3, 'n_children': 2}}}}
    posterior = {'loading_ratio_mechanics': np.array([[-2., -1., 0., 0.], [0., 1., 3., 4.]])}
    return posterior, contract


class SharedFactorTests(unittest.TestCase):
    def test_draw_sign_and_scale_applied_once(self):
        posterior, contract = fixture(scale=3.)
        draws = derive_shared_factor_draws(posterior, contract, 'ratio_pre_z')
        np.testing.assert_array_equal(draws, np.array([[6., 3., 0., 0.], [0., -3., -9., -12.]]))
        np.testing.assert_array_equal(posterior['loading_ratio_mechanics'][0], [-2., -1., 0., 0.])

    def test_quantiles_reordered_after_negative_transform(self):
        posterior, contract = fixture()
        result = summarize_shared_factor(posterior, contract, 'ratio_pre_z', compute_mcse=False)
        row = result.estimates.iloc[0]
        self.assertAlmostEqual(row.lower, -7.65)
        self.assertAlmostEqual(row.upper, 3.65)
        self.assertEqual(row['median'], 0.)
        self.assertEqual(row.probability_below_zero, 3 / 8)
        self.assertEqual(row.probability_above_zero, 2 / 8)
        self.assertEqual(row.probability_equal_zero, 3 / 8)

    def test_exact_small_counts_and_no_child_fields_returned(self):
        posterior, contract = fixture()
        result = summarize_shared_factor(posterior, contract, 'ratio_pre_z', compute_mcse=False)
        row = result.estimates.iloc[0]
        self.assertEqual(row.n_outcome_children, 2)
        self.assertEqual(row.n_outcome_measurements, 3)
        self.assertNotIn('pid_child', result.estimates.columns)
        self.assertFalse(result.metadata['new_fit_required'])
        self.assertEqual(result.metadata['diagnostic_acceptance'], 'not evaluated by derived-result summarizer')

    def test_exact_six_target_selection_and_units(self):
        self.assertEqual(DEFAULT_TARGETS, ('ratio_pre_z', 'ratio_post_z', 'fev1_pre_z', 'fev1_post_z', 'fvc_pre_z', 'fvc_post_z'))
        for target in DEFAULT_TARGETS:
            posterior, contract = fixture(target)
            row = summarize_shared_factor(posterior, contract, target, compute_mcse=False).estimates.iloc[0]
            self.assertEqual(row.unit, 'GLI z-score')
            self.assertTrue(row.figure_s5)
        posterior, contract = fixture()
        with self.assertRaises(SharedFactorError):
            derive_shared_factor_draws(posterior, contract, 'ratio_pre')

    def test_wrong_contract_family_or_target_is_rejected(self):
        for field, value in [('family', 'mechanics'), ('target', 'fvc_gli_z')]:
            posterior, contract = fixture()
            contract['audit'][field] = value
            with self.subTest(field=field), self.assertRaises(SharedFactorError):
                derive_shared_factor_draws(posterior, contract, 'ratio_pre_z')

    def test_invalid_scales_rejected(self):
        for scale in [0., -1., np.nan, np.inf, True, 1 + 2j]:
            posterior, contract = fixture(scale=scale)
            with self.subTest(scale=scale), self.assertRaises(SharedFactorError):
                derive_shared_factor_draws(posterior, contract, 'ratio_pre_z')

    def test_non_global_or_nonfinite_loadings_rejected(self):
        for value in [np.ones(8), np.ones((2, 4, 1)), np.empty((0, 4)), np.full((2, 4), np.nan),
                      np.full((2, 4), np.inf), np.ones((2, 4), dtype=complex), np.ones((2, 4), dtype=bool)]:
            posterior, contract = fixture()
            posterior['loading_ratio_mechanics'] = value
            with self.subTest(shape=value.shape), self.assertRaises(SharedFactorError):
                derive_shared_factor_draws(posterior, contract, 'ratio_pre_z')
        _, contract = fixture()
        with self.assertRaises(SharedFactorError):
            derive_shared_factor_draws({}, contract, 'ratio_pre_z')

    def test_bad_sample_counts_and_missing_data_rejected(self):
        for mutation in ['count', 'child', 'outcome', 'empty']:
            posterior, contract = fixture()
            if mutation == 'count':
                contract['audit']['outcomes']['target']['n_children'] = 3
            elif mutation == 'child':
                contract['frames']['target'].loc[0, 'pid_child'] = ' '
            elif mutation == 'outcome':
                contract['frames']['target'].loc[0, 'outcome_value'] = np.nan
            else:
                contract['frames']['target'] = contract['frames']['target'].iloc[:0]
            with self.subTest(mutation=mutation), self.assertRaises(SharedFactorError):
                summarize_shared_factor(posterior, contract, 'ratio_pre_z', compute_mcse=False)

    def test_input_frames_unchanged_and_skipped_mcse_explicit(self):
        posterior, contract = fixture()
        before = deepcopy(contract)
        result = summarize_shared_factor(posterior, contract, 'ratio_pre_z', compute_mcse=False)
        pd.testing.assert_frame_equal(before['frames']['target'], contract['frames']['target'])
        self.assertFalse(result.metadata['mcse_computed'])
        self.assertTrue(result.estimates[['mcse_mean', 'mcse_lower', 'mcse_upper']].isna().all().all())

    def test_xarray_axis_order_preserved(self):
        import xarray as xr
        posterior, contract = fixture()
        values = posterior['loading_ratio_mechanics']
        source = xr.Dataset({'loading_ratio_mechanics': (('draw', 'chain'), values.T)})
        np.testing.assert_array_equal(derive_shared_factor_draws(source, contract, 'ratio_pre_z'), -2 * values)

    def test_mean_mcse_matches_sample_sd_over_mean_ess(self):
        import arviz as az
        posterior, contract = fixture()
        posterior['loading_ratio_mechanics'] = np.random.default_rng(293).normal(size=(4, 400))
        draws = -2 * posterior['loading_ratio_mechanics']
        row = summarize_shared_factor(posterior, contract, 'ratio_pre_z').estimates.iloc[0]
        expected = np.std(draws, ddof=1) / np.sqrt(float(az.ess(draws, method='mean')))
        self.assertAlmostEqual(row.mcse_mean, expected, places=12)
        self.assertGreater(row.mcse_lower, 0)
        self.assertGreater(row.mcse_upper, 0)


if __name__ == '__main__':
    unittest.main()
