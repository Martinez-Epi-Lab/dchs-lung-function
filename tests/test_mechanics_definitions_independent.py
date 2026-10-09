"""Independent manuscript-scale mechanics tests using fabricated observations."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.analysis.mechanics_results import (
    derive_mechanics_contrast_draws, derive_hazard_draws, summarize_mechanics,
)
from dchs_lung_function.analysis.shared_factor import (
    derive_shared_factor_draws, summarize_shared_factor, SharedFactorError,
)


MEASURES = {
    'R': ('resistance', 1., 2., 10.),
    'C': ('compliance', -1., .001, .03),
    'ReE': ('ree', 1., 2., 10.),
    'ReI': ('rei', 1., 2., 10.),
    'XeE': ('xee', -1., .2, -2.),
    'XeI': ('xei', -1., .2, -2.),
}


def fabricated_mechanics(target='R'):
    column, orientation, scale, baseline = MEASURES[target]
    children = pd.DataFrame({'pid_child': [f'fabricated_mechanics_{i}' for i in range(6)],
                             'child_index': np.arange(6), 'male': [0., 1., 0., 1., 0., 1.]})
    indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5])
    states = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0],
                       [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1],
                       [0, 0, 0], [0, 0, 0]], dtype=float)
    frame = pd.DataFrame(states, columns=['post_before5', 'post_infancy', 'post_at_after5'])
    frame['child_index'] = indices
    frame['pid_child'] = children.pid_child.to_numpy()[indices]
    frame['post_any'] = [0., 1., 0., 1., 0., 1., 0., 1., 0., 0.]
    frame['tb_age_first_years'] = np.array([.5, 2., 3., 6., np.nan, 8.])[indices]
    pre_values = np.array([-5., -3., -1., 1., 3., 5.])
    pre_values /= pre_values.std(ddof=1)
    frame['y_std'] = 0.
    frame.loc[frame.post_any.eq(0), 'y_std'] = pre_values
    frame.loc[frame.post_any.eq(1), 'y_std'] = [.4, -.2, .7, -.1]
    intercept = baseline * orientation
    frame['outcome_value'] = intercept + frame.y_std * scale
    frame[column] = frame.outcome_value / orientation
    stops = np.array([.5, 2., 3., 6., 9., 8.])
    events = np.array([1., 1., 1., 1., 0., 1.])
    breaks = np.array([0., 1., 3., 5., 8., 10., 20.])
    exposure = np.maximum(0., np.minimum(stops[:, None], breaks[1:]) - breaks[:-1])
    event_interval = events[:, None] * ((stops[:, None] >= breaks[:-1]) & (stops[:, None] < breaks[1:]))
    return {
        'children': children, 'frames': {'target': frame},
        'designs': {'target': pd.DataFrame({'Intercept': np.ones(10)})},
        'post_diagnosis_terms': ['post_before5', 'post_infancy', 'post_at_after5'],
        'post_diagnosis_designs': {'target': states}, 'hazard_terms': ['male'],
        'hazard_design': children[['male']].to_numpy(), 'weights': np.array([1, 2, 1]),
        'event': {'breaks': breaks, 'stop': stops, 'event': events,
                  'exposure': exposure, 'event_interval': event_interval},
        'audit': {'family': 'mechanics', 'target': column, 'covariates': [],
                  'joint_children': 6, 'tb_events': 5, 'weights': [1, 2, 1],
                  'outcomes': {'target': {'scale': scale, 'nuisance_terms': ['Intercept'],
                                         'nuisance_coefficients': [intercept],
                                         'n_training': 6, 'n_observations': 10, 'n_children': 6}}},
    }


def fabricated_draws():
    return {'shift_1to5': np.full((2, 4), 2.),
            'shift_infancy_additional': np.full((2, 4), 3.),
            'shift_late': np.full((2, 4), 8.),
            'assoc_outcome': np.full((2, 4), np.log(2.)),
            'tau_child': np.tile([1., 2., 3., 4.], (2, 1))}


def set_frame(contract, frame, weights):
    """Update fabricated contract counts after a deliberate sample change."""
    frame = frame.reset_index(drop=True)
    contract['frames']['target'] = frame
    contract['designs']['target'] = pd.DataFrame({'Intercept': np.ones(len(frame))})
    contract['post_diagnosis_designs']['target'] = frame[['post_before5', 'post_infancy', 'post_at_after5']].to_numpy()
    contract['weights'] = np.asarray(weights)
    contract['audit']['weights'] = list(weights)
    contract['audit']['outcomes']['target'].update(
        n_observations=len(frame), n_children=int(frame.pid_child.nunique()),
        n_training=int(frame.post_any.eq(0).sum()))


def fabricated_primary(target, *, scale=.5, n=3):
    columns = {'ratio_pre_z': 'fev1_fvc_gli_z', 'ratio_post_z': 'fev1_fvc_post_gli_z',
               'fev1_pre_z': 'fev1_gli_z', 'fev1_post_z': 'fev1_post_gli_z',
               'fvc_pre_z': 'fvc_gli_z', 'fvc_post_z': 'fvc_post_gli_z'}
    frame = pd.DataFrame({'pid_child': [f'fabricated_primary_{i}' for i in range(n)],
                          'outcome_value': np.linspace(-1., 1., n)})
    return {'frames': {'target': frame},
            'audit': {'family': 'primary', 'target': columns[target],
                      'outcomes': {'target': {'scale': scale, 'n_observations': n, 'n_children': n}}}}


class MechanicsDefinitionsIndependentTests(unittest.TestCase):
    def test_all_native_orientations_and_credible_interval_endpoint_reversal(self):
        draws = fabricated_draws()
        draws['shift_late'] = np.tile([-4., -1., 2., 7.], (2, 1))
        for target, (_, orientation, scale, _) in MEASURES.items():
            with self.subTest(target=target):
                result = summarize_mechanics(draws, fabricated_mechanics(target), target, compute_mcse=False)
                row = result.estimates.set_index('contrast').loc['TB >=5']
                actual = derive_mechanics_contrast_draws(draws, fabricated_mechanics(target), target)['TB >=5']
                wanted = scale * draws['shift_late'] if orientation == 1 else -scale * draws['shift_late']
                np.testing.assert_array_equal(actual, wanted)
                self.assertAlmostEqual(row.native_mean, wanted.mean())
                np.testing.assert_allclose([row.native_lower, row.native_upper], np.quantile(wanted, [.025, .975]))
                if orientation == -1:
                    self.assertAlmostEqual(row.native_lower, -scale * row.upper)
                    self.assertAlmostEqual(row.native_upper, -scale * row.lower)
                self.assertEqual(row.probability_worse, .5)

    def test_compliance_millilitres_conversion_is_applied_once_after_sign_reversal(self):
        posterior = fabricated_draws()
        result = summarize_mechanics(posterior, fabricated_mechanics('C'), 'C', compute_mcse=False)
        row = result.estimates.set_index('contrast').loc['TB <1']
        self.assertEqual(row['mean'], 5.)
        self.assertAlmostEqual(row.native_mean, -.005)
        self.assertAlmostEqual(row.reported_mean, -5.)
        self.assertEqual(row.unit, 'L/cmH2O')
        self.assertEqual(row.reported_unit, 'mL/cmH2O')
        self.assertEqual(row.probability_worse, 1.)
        self.assertEqual(row.native_probability_below_zero, 1.)
        for target in ['R', 'ReE', 'ReI', 'XeE', 'XeI']:
            other = summarize_mechanics(posterior, fabricated_mechanics(target), target, compute_mcse=False)
            np.testing.assert_array_equal(other.estimates.native_mean, other.estimates.reported_mean)

    def test_pooled_effects_use_outcome_specific_children_not_measurement_frequency(self):
        original = fabricated_mechanics()
        repeated = deepcopy(original)
        f = original['frames']['target']
        set_frame(repeated, pd.concat([f, *[f.iloc[[1]]] * 19]), [1, 2, 1])
        baseline = derive_mechanics_contrast_draws(fabricated_draws(), original, 'R')
        duplicated = derive_mechanics_contrast_draws(fabricated_draws(), repeated, 'R')
        for name in baseline:
            np.testing.assert_array_equal(baseline[name], duplicated[name])
        np.testing.assert_array_equal(baseline['TB <5 weighted'], np.full((2, 4), 6.))
        np.testing.assert_array_equal(baseline['Any TB weighted'], np.full((2, 4), 8.5))
        fewer_middle_children = deepcopy(original)
        set_frame(fewer_middle_children, f.loc[f.child_index.ne(2)], [1, 1, 1])
        changed = derive_mechanics_contrast_draws(fabricated_draws(), fewer_middle_children, 'R')
        np.testing.assert_array_equal(changed['TB <5 weighted'], np.full((2, 4), 7.))
        self.assertEqual(len(fewer_middle_children['children']), 6)

    def test_pooled_intervals_preserve_posterior_dependence_between_age_groups(self):
        draws = fabricated_draws()
        draws['shift_1to5'] = np.tile([-1., 1., -1., 1.], (2, 1))
        draws['shift_infancy_additional'] = -3 * draws['shift_1to5']
        result = summarize_mechanics(draws, fabricated_mechanics(), 'R', compute_mcse=False).estimates.set_index('contrast')
        # Infant=-2*middle, with one infant and two middle-age children. The
        # weighted posterior cancels exactly, while weighted marginal bounds do not.
        pooled = result.loc['TB <5 weighted']
        np.testing.assert_array_equal([pooled.native_mean, pooled.native_lower, pooled.native_upper], [0., 0., 0.])
        weighted_lower = (result.loc['TB <1', 'native_lower'] + 2 * result.loc['TB 1-<5', 'native_lower']) / 3
        self.assertLess(weighted_lower, 0.)
        self.assertEqual(pooled.native_probability_equal_zero, 1.)

    def test_hazard_ratio_transforms_each_joint_draw_and_retains_between_child_scale(self):
        posterior = fabricated_draws()
        actual = derive_hazard_draws(posterior)
        np.testing.assert_allclose(actual, np.tile([2., 4., 8., 16.], (2, 1)), atol=1e-14)
        result = summarize_mechanics(posterior, fabricated_mechanics('C'), 'C', compute_mcse=False)
        self.assertAlmostEqual(result.hazard['mean'].item(), 7.5)
        self.assertAlmostEqual(result.hazard['median'].item(), 6.)
        self.assertNotAlmostEqual(result.hazard['mean'].item(), np.exp(np.mean(posterior['assoc_outcome'] * posterior['tau_child'])))
        self.assertNotAlmostEqual(result.hazard['mean'].item(), 2.)
        result_r = summarize_mechanics(posterior, fabricated_mechanics('R'), 'R', compute_mcse=False)
        self.assertEqual(result.hazard['mean'].item(), result_r.hazard['mean'].item())

    def test_eight_contrasts_and_small_support_survive_while_corrupt_deterministics_fail(self):
        posterior, contract = fabricated_draws(), fabricated_mechanics()
        result = summarize_mechanics(posterior, contract, 'R', compute_mcse=False)
        wanted = ['TB <1', 'TB 1-<5', 'TB >=5', 'TB <5 weighted', 'Any TB weighted',
                  'Infancy minus 1-<5', 'Later minus infancy', 'Later minus 1-<5']
        self.assertEqual(result.estimates.contrast.tolist(), wanted)
        np.testing.assert_array_equal(result.estimates.native_mean, [10., 4., 16., 6., 8.5, 6., 6., 12.])
        support = result.sample_counts.set_index('group').loc['post_tb_age_five_or_later']
        self.assertEqual((support.measurements, support.children), (1, 1))
        bad = deepcopy(posterior); bad['shift_infancy'] = np.full((2, 4), 3.)
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            derive_mechanics_contrast_draws(bad, contract, 'R')

    def test_shared_factor_uses_only_loading_and_one_residual_scale(self):
        posterior = {'loading_ratio_mechanics': np.array([[-2., -1., 0., 1., 2.]])}
        contract = fabricated_primary('ratio_pre_z', scale=.5)
        expected = np.array([[1., .5, 0., -.5, -1.]])
        np.testing.assert_array_equal(derive_shared_factor_draws(posterior, contract, 'ratio_pre_z'), expected)
        posterior.update({'direct_ratio': np.full((1, 5), 1000.),
                          'shift_mechanics': np.full((1, 5), 1000.),
                          'tau_child': np.full((1, 5), 1000.)})
        np.testing.assert_array_equal(derive_shared_factor_draws(posterior, contract, 'ratio_pre_z'), expected)
        row = summarize_shared_factor(posterior, contract, 'ratio_pre_z', compute_mcse=False).estimates.iloc[0]
        self.assertEqual(row.unit, 'GLI z-score')
        self.assertEqual(row.probability_below_zero, .4)
        self.assertEqual(row.probability_above_zero, .4)
        self.assertEqual(row.probability_equal_zero, .2)

    def test_shared_factor_retains_six_separate_available_case_zscore_samples(self):
        targets = ['ratio_pre_z', 'ratio_post_z', 'fev1_pre_z', 'fev1_post_z', 'fvc_pre_z', 'fvc_post_z']
        posterior = {'loading_ratio_mechanics': np.ones((2, 4))}
        rows = []
        for n, target in enumerate(targets, 1):
            result = summarize_shared_factor(posterior, fabricated_primary(target, scale=.1*n, n=n), target, compute_mcse=False)
            rows.append(result.estimates.iloc[0])
            self.assertEqual(result.estimates.n_outcome_children.item(), n)
            self.assertEqual(result.estimates.n_outcome_measurements.item(), n)
            self.assertAlmostEqual(result.estimates['mean'].item(), -.1*n)
        self.assertEqual([row.target for row in rows], targets)
        self.assertEqual([row.stage for row in rows], ['Pre-bronchodilator', 'Post-bronchodilator'] * 3)
        with self.assertRaises(SharedFactorError):
            summarize_shared_factor(posterior, fabricated_primary('ratio_pre_z'), 'ratio_pre', compute_mcse=False)


if __name__ == '__main__':
    unittest.main()
