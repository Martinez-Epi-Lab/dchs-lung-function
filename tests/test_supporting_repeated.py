"""Fabricated-data checks for paired MMEF and selected repeated X6 models."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.preparation import exposure_frame, repeated_design, trajectory_inputs
from dchs_lung_function.models import spirometry_trajectories as primary_trajectories
from dchs_lung_function.models.supporting_repeated import (
    CONTRACT_KEYS, MMEF_SCOPES, MMEF_GROUPS, TABLE_NAMES, fit_selected,
    validate_mmef, validate_repeated, validate_trajectory, fit_gaussian_gee,
    cluster_resampling, x6_spline_basis, x6_basis_at,
)


def fabricated_frame():
    rng = np.random.default_rng(708)
    birth = pd.Timestamp('2010-01-01')
    days = [1169, 1534, 1899, 2264, 2995, 3579, 3726, 6136]
    diagnoses = [None, 180, 700, 1461, 2000, 2557, 6575]
    records = []
    for state, diagnosis in enumerate(diagnoses):
        for child in range(8):
            male = float(child % 2)
            latent = rng.normal(scale=.3)
            for visit, day in enumerate(days):
                # Unequal numbers of visits distinguish child resampling and
                # measurement-weighted standardization from simpler shortcuts.
                if state == 0 and child == 0 and visit > 4:
                    continue
                age = day/365.25
                post = diagnosis is not None and day >= diagnosis
                early = post and diagnosis/365.25 < 5
                infant = post and diagnosis/365.25 < 1
                pre = 1.2 + .12*age + .08*male + latent + rng.normal(scale=.1)
                change = .16 + .01*age + .08*early + .04*infant + rng.normal(scale=.13)
                x6 = -.8 + .09*age + latent + .12*early + rng.normal(scale=.25)
                records.append({'pid_child': f'fabricated_support_{state}_{child}',
                                'child_dob': birth, 'tb_first_dx': birth+pd.Timedelta(days=diagnosis) if diagnosis else pd.NaT,
                                'tb_age_first_years': diagnosis/365.25 if diagnosis else np.nan,
                                'date_of_test': birth+pd.Timedelta(days=day),
                                'date_of_test_ilf2': birth+pd.Timedelta(days=day),
                                'visit_label': f'fabricated_visit_{visit}', 'male': male,
                                'height': 75+6*age+3*male+rng.normal(scale=3.),
                                'mmef': pre, 'mmef_post': pre+change,
                                'x6_sa_z': x6, 'reactance_6hz': 2*x6-1.5, 'exit_date': pd.NaT})
    frame = pd.DataFrame(records)
    frame['mmef'] = frame.mmef.astype('float32')
    frame['mmef_post'] = frame.mmef_post.astype('float32')
    frame.loc[0, ['mmef', 'mmef_post']] = [np.float32(.1), np.float32(1.)]
    return frame


def fabricated_contracts(source=None):
    """Reusable selected M03 contracts, made entirely from invented records."""
    source = fabricated_frame() if source is None else source.copy(deep=True)
    frame = exposure_frame(source, 'mmef')
    frame = frame.loc[frame.mmef_post.notna()].copy()
    frame['outcome_value'] = frame.mmef_post-frame.mmef
    designs = {}
    for key, _, terms, _, _ in MMEF_SCOPES:
        matrix, names, dropped = repeated_design(frame, terms)
        designs[key] = {'matrix': matrix, 'terms': names, 'dropped_terms': dropped}
    result = {'mmef/change': {'frame': frame, 'designs': designs,
                              'audit': {'family': 'mmef_change', 'n_observations': len(frame),
                                        'n_children': int(frame.pid_child.nunique())}}}
    for key in CONTRACT_KEYS[1:5]:
        _, outcome, cutoff = key.split('/'); cutoff = int(cutoff)
        frame = exposure_frame(source, outcome, 'date_of_test_ilf2', age_range=(3, 17))
        frame['target'] = (frame.post_any.eq(1) & frame.tb_age_first_years.lt(cutoff)).astype(float)
        frame['later'] = (frame.post_any.eq(1) & frame.tb_age_first_years.ge(cutoff)).astype(float)
        matrix, names, dropped = repeated_design(frame, ['target', 'later', 'future_tb'])
        result[key] = {'frame': frame, 'design': matrix, 'terms': names,
                       'audit': {'family': 'x6_repeated', 'n_observations': len(frame),
                                 'n_children': int(frame.pid_child.nunique()), 'terms': names, 'dropped_terms': dropped}}
    for cutoff in (1, 5):
        result[f'trajectory/x6/{cutoff}'] = trajectory_inputs(source, outcome='x6_sa_z', threshold=cutoff, x6=True)
    return result


def independent_ols_hc0(x, y, groups):
    q, r = np.linalg.qr(x, mode='reduced')
    beta = np.linalg.solve(r, q.T@y)
    rinv = np.linalg.inv(r)
    bread = rinv@rinv.T
    residual = y-x@beta
    covariance = np.zeros((x.shape[1], x.shape[1]))
    for child in np.unique(groups):
        selected = groups == child
        influence = bread@(x[selected].T@residual[selected])
        covariance += np.outer(influence, influence)
    return beta, covariance


class SupportingRepeatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = fabricated_contracts()
        cls.result = fit_selected(cls.contracts)

    def test_selected_fit_and_table_inventories(self):
        result = self.result
        self.assertEqual(set(result), {'tables', 'metadata'})
        self.assertEqual(set(result['tables']), set(TABLE_NAMES))
        self.assertTrue(result['metadata']['accepted'])
        self.assertEqual(result['metadata']['fitted_models'], 10)
        self.assertEqual(len(result['metadata']['models']), 10)
        expected = {'mmef_change': 2, 'mmef_descriptive': 6, 'x6_repeated': 4,
                    'x6_global_tests': 2, 'x6_predictions': 404, 'x6_age_contrasts': 16}
        for name, count in expected.items():
            self.assertEqual(len(result['tables'][name]), count)
        terms = [len(m['terms']) for m in result['metadata']['models']]
        self.assertEqual(len(result['tables']['coefficients']), sum(terms))
        self.assertEqual(len(result['tables']['covariance']), 2*sum(p*p for p in terms))
        self.assertIsNone(result['metadata']['p_value_adjustment'])
        json.dumps(result['metadata'], allow_nan=False)

    def test_mmef_float64_operands_before_subtraction_preserve_prepared_input(self):
        contract = deepcopy(self.contracts['mmef/change'])
        original = contract['frame'].copy(deep=True)
        frame, y, _, _, audit = validate_mmef(contract)
        wanted = original.mmef_post.astype(float)-original.mmef.astype(float)
        np.testing.assert_array_equal(y, wanted)
        self.assertGreater(audit['changed_numeric_values'], 0)
        self.assertGreater(audit['maximum_absolute_rounding_difference'], 0.)
        self.assertEqual(audit['prepared_delta_dtype'], 'float32')
        pd.testing.assert_frame_equal(frame, original)

    def test_gaussian_gee_coefficients_and_full_hc0_covariance_match_independent_qr(self):
        frame, y, designs, _, _ = validate_mmef(self.contracts['mmef/change'])
        for key in ['before5', 'infancy']:
            x, names, _ = designs[key]
            beta, covariance = independent_ols_hc0(x, y, frame.pid_child.to_numpy())
            fit = fit_gaussian_gee(y, x, frame.pid_child.to_numpy(), names)
            self.assertTrue(fit.metadata['accepted'])
            np.testing.assert_allclose(fit.coefficients, beta, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(fit.covariance, covariance, atol=1e-12, rtol=1e-12)
            correction = frame.pid_child.nunique()/(frame.pid_child.nunique()-1)*(len(frame)-1)/(len(frame)-len(names))
            self.assertFalse(np.allclose(fit.covariance, correction*covariance, rtol=1e-3, atol=1e-12))
            self.assertEqual(fit.metadata['finite_sample_correction'], 1.)

    def test_standardized_change_levels_use_all_sample_covariates_and_full_covariance(self):
        frame, y, designs, _, _ = validate_mmef(self.contracts['mmef/change'])
        for key, _, terms, term, _ in MMEF_SCOPES:
            x, names, _ = designs[key]
            beta, covariance = independent_ols_hc0(x, y, frame.pid_child.to_numpy())
            reference = x.mean(axis=0)
            for name in terms:
                reference[names.index(name)] = 0.
            exposed = reference.copy(); exposed[names.index(term)] = 1.
            row = self.result['tables']['mmef_change'].set_index('design').loc[key]
            for prefix, vector in [('reference', reference), ('exposed', exposed)]:
                estimate, se = vector@beta, np.sqrt(vector@covariance@vector)
                np.testing.assert_allclose([row[f'standardized_{prefix}_change'], row[f'standardized_{prefix}_ci_low'],
                                            row[f'standardized_{prefix}_ci_high']],
                                           [estimate, estimate-1.96*se, estimate+1.96*se], atol=1e-12)
            self.assertAlmostEqual(row.standardized_exposed_change-row.standardized_reference_change, row.estimate)

    def test_bootstrap_whole_children_matches_independent_resampled_rows(self):
        frame, y, designs, _, _ = validate_mmef(self.contracts['mmef/change'])
        x, names, _ = designs['before5']; j = names.index('post_before5')
        result, audit = cluster_resampling(frame, x, y, j, replicates=40)
        child_ids = np.unique(frame.pid_child.to_numpy())
        ages = frame.groupby('pid_child', observed=True).tb_age_first_years.first().reindex(child_ids).to_numpy()
        strata = np.select([ages < 1, (ages >= 1) & (ages < 5), ages >= 5], [1, 2, 3], default=0)
        rng = np.random.default_rng(20260903); estimates = []
        for _ in range(40):
            multiplicity = np.zeros(len(child_ids), dtype=int)
            for level in np.unique(strata):
                members = np.flatnonzero(strata == level)
                multiplicity[members] = rng.multinomial(len(members), np.full(len(members), 1/len(members)))
            observations = np.concatenate([np.tile(np.flatnonzero(frame.pid_child.eq(child).to_numpy()), count)
                                            for child, count in zip(child_ids, multiplicity) if count])
            if np.linalg.matrix_rank(x[observations]) == x.shape[1]:
                estimates.append(np.linalg.lstsq(x[observations], y[observations], rcond=None)[0][j])
        self.assertEqual(audit['accepted_replicates'], len(estimates))
        np.testing.assert_allclose([result['bootstrap_ci_low'], result['bootstrap_ci_high']],
                                   np.quantile(estimates, [.025, .975]), atol=1e-11)
        self.assertEqual(self.result['metadata']['mmef_resampling']['before5']['requested_replicates'], 2000)

    def test_leave_one_child_range_matches_direct_deleted_sample_fits(self):
        frame, y, designs, _, _ = validate_mmef(self.contracts['mmef/change'])
        x, names, _ = designs['infancy']; j = names.index('post_infancy')
        values = []
        for child in frame.pid_child.unique():
            keep = frame.pid_child.ne(child).to_numpy()
            if np.linalg.matrix_rank(x[keep]) == x.shape[1]:
                values.append(np.linalg.lstsq(x[keep], y[keep], rcond=None)[0][j])
        row = self.result['tables']['mmef_change'].set_index('design').loc['infancy']
        np.testing.assert_allclose([row.leave_one_child_out_min, row.leave_one_child_out_max], [min(values), max(values)], atol=1e-11)

    def test_robust_table_interval_is_not_replaced_by_bootstrap_interval(self):
        for row in self.result['tables']['mmef_change'].itertuples():
            self.assertAlmostEqual(row.ci_low, row.estimate-1.96*row.se)
            self.assertAlmostEqual(row.ci_high, row.estimate+1.96*row.se)
            self.assertNotAlmostEqual(row.ci_low, row.bootstrap_ci_low, places=6)
            self.assertNotAlmostEqual(row.ci_high, row.bootstrap_ci_high, places=6)

    def test_mmef_preserves_nominal_sample_without_adding_exact_age_limits(self):
        frame, _, _, counts, _ = validate_mmef(self.contracts['mmef/change'])
        self.assertLess(frame.age_years.min(), 5.)
        self.assertGreater(frame.age_years.max(), 10.)
        self.assertEqual(counts['n_observations'], len(frame))
        descriptions = self.result['tables']['mmef_descriptive']
        self.assertEqual(descriptions.group.tolist(), list(MMEF_GROUPS))

    def test_x6_repeated_and_trajectory_age_windows_are_distinct(self):
        repeated = self.contracts['x6_repeated/x6_sa_z/1']['frame']
        trajectory = self.contracts['trajectory/x6/1']['frame']
        self.assertGreater(repeated.age_years.max(), 10.)
        self.assertLessEqual(repeated.age_years.max(), 17.)
        self.assertTrue(trajectory.age_years.between(3., 10.).all())
        self.assertLess(len(trajectory), len(repeated))
        bad = deepcopy(self.contracts['x6_repeated/x6_sa_z/1'])
        bad['frame'].loc[0, 'date_of_test_ilf2'] = bad['frame'].loc[0, 'child_dob']+pd.Timedelta(days=1095)
        bad['frame'].loc[0, 'test_date'] = bad['frame'].loc[0, 'date_of_test_ilf2']
        bad['frame'].loc[0, 'age_years'] = 1095/365.25
        with self.assertRaisesRegex(ValueError, 'window'):
            validate_repeated(bad, 'x6_sa_z', 1)

    def test_future_later_and_modality_dates_are_validated(self):
        for change in ['future', 'modality', 'later']:
            contract = deepcopy(self.contracts['x6_repeated/x6_sa_z/1'])
            frame = contract['frame']
            if change == 'future':
                i = frame.index[frame.future_tb.eq(1)][0]; frame.loc[i, 'future_tb'] = 0.
            elif change == 'modality':
                frame.loc[0, 'date_of_test_ilf2'] += pd.Timedelta(days=1)
            else:
                i = frame.index[frame.later.eq(1)][0]; frame.loc[i, 'later'] = 0.
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_repeated(contract, 'x6_sa_z', 1)

    def test_x6_retains_outcome_specific_missingness(self):
        source = fabricated_frame()
        source.loc[0, 'reactance_6hz'] = np.nan
        contracts = fabricated_contracts(source)
        z, _, zcounts = validate_repeated(contracts['x6_repeated/x6_sa_z/1'], 'x6_sa_z', 1)
        native, _, ncounts = validate_repeated(contracts['x6_repeated/reactance_6hz/1'], 'reactance_6hz', 1)
        self.assertEqual(zcounts['n_observations']-ncounts['n_observations'], 1)
        self.assertEqual(len(z)-len(native), 1)

    def test_x6_training_basis_reuse_and_primary_constants_are_unchanged(self):
        frame, _, info = validate_trajectory(self.contracts['trajectory/x6/1'], 1)
        _, fitted_info = x6_spline_basis(frame.age_years)
        np.testing.assert_array_equal(x6_basis_at(info, [3., 6.5, 10.]), x6_basis_at(fitted_info, [3., 6.5, 10.]))
        self.assertEqual((primary_trajectories.AGE_MIN, primary_trajectories.AGE_MAX), (5., 10.))
        self.assertIn('lower_bound=5', primary_trajectories.SPLINE_FORMULA)
        with self.assertRaises(ValueError):
            x6_basis_at(info, [2.99])

    def test_x6_predictions_raw_wald_tests_and_parallel_outputs(self):
        tables = self.result['tables']
        self.assertEqual(set(tables['x6_predictions'].age_years), set(np.linspace(3., 10., 101)))
        self.assertEqual(set(tables['x6_age_contrasts'].age_years), set(np.arange(3., 11.)))
        self.assertTrue(tables['x6_global_tests'].interaction_df.eq(3).all())
        self.assertTrue(tables['x6_global_tests'].overall_trajectory_df.eq(4).all())
        self.assertFalse(any('bh' in name.lower() for name in tables['x6_global_tests'].columns))
        for model_id, group in tables['x6_age_contrasts'].groupby('model_id'):
            parallel = tables['x6_global_tests'].set_index('model_id').loc[model_id, 'parallel_trajectory_difference']
            self.assertGreater(np.ptp(group.difference_tb_minus_reference), 0.)
            self.assertFalse(np.allclose(group.difference_tb_minus_reference, parallel))

    def test_mmef_and_x6_finite_sample_covariance_rules_are_separate(self):
        for model in self.result['metadata']['models']:
            if model['model_id'].startswith('mmef'):
                self.assertEqual(model['finite_sample_correction'], 1.)
            else:
                n, g, p = model['n_observations'], model['n_children'], model['n_parameters']
                self.assertAlmostEqual(model['finite_sample_correction'], g/(g-1)*(n-1)/(n-p))

    def test_malformed_selections_designs_precision_and_duplicate_pairs_rejected(self):
        missing = dict(self.contracts); missing.pop('trajectory/x6/1')
        with self.assertRaises(ValueError): fit_selected(missing)
        for change in ['delta', 'terms', 'duplicate']:
            contract = deepcopy(self.contracts['mmef/change'])
            if change == 'delta':
                contract['frame'].loc[0, 'outcome_value'] += 1.
            elif change == 'terms':
                contract['designs']['before5']['terms'][0] = 'wrong'
            else:
                contract['frame'].loc[1, 'date_of_test'] = contract['frame'].loc[0, 'date_of_test']
                contract['frame'].loc[1, 'test_date'] = contract['frame'].loc[0, 'test_date']
                contract['frame'].loc[1, 'age_years'] = contract['frame'].loc[0, 'age_years']
            with self.subTest(change=change), self.assertRaises(ValueError): validate_mmef(contract)

    def test_zero_residual_failure_preserves_complete_result_inventory(self):
        source = fabricated_frame()
        source['mmef_post'] = source.mmef.copy()
        source['x6_sa_z'] = 0.; source['reactance_6hz'] = 0.
        result = fit_selected(fabricated_contracts(source))
        self.assertFalse(result['metadata']['accepted'])
        self.assertEqual(result['metadata']['status'], 'fit_failed')
        self.assertEqual(result['metadata']['fitted_models'], 10)
        self.assertTrue(result['metadata']['failure_reasons'])
        for name in TABLE_NAMES:
            self.assertEqual(result['tables'][name].shape, self.result['tables'][name].shape)
        self.assertTrue(result['tables']['x6_predictions'].predicted_mean.isna().all())
        self.assertTrue(result['tables']['mmef_change'].estimate.isna().all())
        json.dumps(result['metadata'], allow_nan=False)

    def test_fit_does_not_mutate_m03_frames_matrices_or_basis(self):
        contracts = fabricated_contracts(); before = deepcopy(contracts)
        fit_selected(contracts)
        for key in CONTRACT_KEYS:
            pd.testing.assert_frame_equal(contracts[key]['frame'], before[key]['frame'])
            for name in ['design', 'reduced_design', 'basis']:
                if name in contracts[key]:
                    np.testing.assert_array_equal(contracts[key][name], before[key][name])


if __name__ == '__main__':
    unittest.main()
