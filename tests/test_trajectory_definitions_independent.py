"""Independent trajectory-definition checks; all observations are fabricated."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.preparation import trajectory_inputs


def fabricated_trajectory_frame():
    """Unequal timing states and nonlinear responses, with no study data."""
    birth = pd.Timestamp('2010-01-01')
    states = [('never', None, np.nan), ('infancy', 183, .5),
              ('just_under_one', 365, np.nextafter(1., 0.)),
              ('exactly_one', 366, 1.), ('just_under_five', 1826, np.nextafter(5., 0.)),
              ('exactly_five', 1827, 5.), ('later', 2557, 7.), ('future', 4383, 12.)]
    records = []
    for state, diagnosis_day, diagnosis_age in states:
        for male in (0., 1.):
            for visit, day in enumerate((1827, 2192, 2557, 2922, 3287, 3652)):
                age = day / 365.25
                post = diagnosis_day is not None and day >= diagnosis_day
                early = post and diagnosis_age < 1.
                index = len(records)
                value = (.88 - .025 * early + .003 * (age - 7.5)
                         + .007 * early * (age - 7.5) + .002 * np.sin(index * .73))
                records.append({
                    'pid_child': f'fabricated_{state}_{int(male)}',
                    'scenario': state, 'visit_number': visit,
                    'child_dob': birth, 'date_of_test': birth + pd.Timedelta(days=day),
                    'tb_first_dx': birth + pd.Timedelta(days=diagnosis_day) if diagnosis_day else pd.NaT,
                    'tb_age_first_years': diagnosis_age,
                    'male': male, 'height': 110. + 3.2 * age + 3 * male + .8 * np.sin(index),
                    'fev1_fvc': value, 'fev1_fvc_gli_z': 4 * (value - .9),
                    'visit_label': 'fabricated unscheduled visit', 'exit_date': pd.NaT,
                })
    return pd.DataFrame(records)


class TrajectoryDefinitionsIndependentTests(unittest.TestCase):
    def test_date_resolution_age_boundaries_and_no_nominal_visit_filter(self):
        source = fabricated_trajectory_frame()
        boundary = source.iloc[[0, 0, 0, 0]].copy().reset_index(drop=True)
        boundary['scenario'] = ['below_five', 'above_five', 'below_ten', 'above_ten']
        boundary['pid_child'] = ['fabricated_boundary_' + s for s in boundary.scenario]
        boundary['date_of_test'] = boundary.child_dob + pd.to_timedelta([1826, 1827, 3652, 3653], unit='D')
        # M06 uses exact ages, not M05's nominal school-visit labels or exit rule.
        boundary['visit_label'] = '60 months'
        boundary['exit_date'] = boundary.date_of_test - pd.Timedelta(days=1)
        actual = trajectory_inputs(pd.concat([source, boundary], ignore_index=True))['frame']
        present = set(actual.loc[actual.scenario.str.contains('five|ten'), 'scenario'])
        self.assertIn('above_five', present)
        self.assertIn('below_ten', present)
        self.assertNotIn('below_five', present)
        self.assertNotIn('above_ten', present)
        self.assertTrue(actual.age_years.between(5., 10., inclusive='both').all())
        # Whole-day dates straddle 5 and 10 because 365.25*5/10 is fractional.
        self.assertAlmostEqual(actual.loc[actual.scenario.eq('above_five'), 'age_years'].item(), 1827 / 365.25)

    def test_shared_complete_cases_and_native_percentage_points(self):
        source = fabricated_trajectory_frame()
        source.loc[0, 'fev1_fvc'] = np.nan
        source.loc[1, 'fev1_fvc_gli_z'] = np.nan
        source.loc[2, 'height'] = np.nan
        source.loc[3, 'male'] = np.nan
        raw = trajectory_inputs(source, outcome='fev1_fvc')
        standardized = trajectory_inputs(source, outcome='fev1_fvc_gli_z')
        self.assertEqual(len(raw['frame']), len(source) - 4)
        pd.testing.assert_frame_equal(raw['frame'][['pid_child', 'test_date']],
                                      standardized['frame'][['pid_child', 'test_date']])
        np.testing.assert_array_equal(raw['design'], standardized['design'])
        np.testing.assert_allclose(raw['frame'].outcome_value, 100 * raw['frame'].fev1_fvc, atol=0, rtol=0)
        np.testing.assert_allclose(standardized['frame'].outcome_value,
                                   standardized['frame'].fev1_fvc_gli_z, atol=0, rtol=0)

    def test_same_day_later_diagnosis_future_and_never_states_remain_distinct(self):
        prepared = trajectory_inputs(fabricated_trajectory_frame(), threshold=1)
        f = prepared['frame']
        x = prepared['design'][:, [prepared['terms'].index(c) for c in ('target', 'later', 'future_tb')]]
        truth = {'never': [0, 0, 0], 'future': [0, 0, 1], 'infancy': [1, 0, 0]}
        for label, expected in truth.items():
            np.testing.assert_array_equal(x[f.scenario.eq(label)], np.tile(expected, (12, 1)))
        later = f.scenario.eq('later')
        same_day = later & f.test_date.eq(f.tb_first_dx)
        before = later & f.test_date.lt(f.tb_first_dx)
        self.assertEqual(int(same_day.sum()), 2)
        np.testing.assert_array_equal(x[same_day], [[0, 1, 0], [0, 1, 0]])
        np.testing.assert_array_equal(x[before], np.tile([0, 0, 1], (4, 1)))

    def test_first_diagnosis_cutoffs_are_strict_and_all_interactions_switch(self):
        for cutoff, below, equality in ((1, 'just_under_one', 'exactly_one'),
                                       (5, 'just_under_five', 'exactly_five')):
            prepared = trajectory_inputs(fabricated_trajectory_frame(), threshold=cutoff)
            frame, design = prepared['frame'], prepared['design']
            target = prepared['terms'].index('target')
            later = prepared['terms'].index('later')
            interaction = [prepared['terms'].index(f'target:age_spline_{j}') for j in (1, 2, 3)]
            is_below = frame.scenario.eq(below)
            is_equal = frame.scenario.eq(equality)
            np.testing.assert_array_equal(design[is_below, target], np.ones(12))
            np.testing.assert_array_equal(design[is_equal, target], np.zeros(12))
            np.testing.assert_array_equal(design[is_equal, later], np.ones(12))
            np.testing.assert_array_equal(design[is_below][:, interaction], prepared['basis'][is_below])
            np.testing.assert_array_equal(design[is_equal][:, interaction], np.zeros((12, 3)))
            self.assertEqual(design.shape[1], 12)
            self.assertEqual(prepared['reduced_design'].shape[1], 9)

    def test_cubic_height_and_measurement_weighted_sex_standardization(self):
        from dchs_lung_function.models.spirometry_trajectories import height_standardization, expected_height_z
        # One female has six visits; two males have three each. Measurement sex
        # mean is 1/2, but the mean across children is 2/3.
        ages = np.array([5, 6, 7, 8, 9, 10, 5, 7, 9, 6, 8, 10], dtype=float)
        male = np.array([0.] * 6 + [1.] * 6)
        centered = ages - 7.5
        height = 120 + 5 * centered + .5 * centered**2 + .125 * centered**3 + 4 * male
        frame = pd.DataFrame({'age_years': ages, 'male': male, 'height': height,
                              'pid_child': ['fabricated_female'] * 6 + ['fabricated_male_a'] * 3 + ['fabricated_male_b'] * 3})
        state = height_standardization(frame)
        np.testing.assert_allclose(state['coefficients'], [120, 5, .5, .125, 4], atol=1e-11)
        self.assertEqual(state['male_mean'], .5)
        self.assertNotEqual(state['male_mean'], frame.drop_duplicates('pid_child').male.mean())
        wanted_ages = np.array([5., 7.5, 10.])
        t = wanted_ages - 7.5
        wanted_height = 122 + 5 * t + .5 * t**2 + .125 * t**3
        expected = (wanted_height - height.mean()) / height.std(ddof=1)
        np.testing.assert_allclose(expected_height_z(wanted_ages, state), expected, atol=1e-12)

    def test_common_standardization_changes_levels_but_not_group_differences(self):
        from dchs_lung_function.models.spirometry_trajectories import FULL_TERMS, prediction_vector
        terms = list(FULL_TERMS)
        beta = np.zeros(len(terms))
        values = {'Intercept': 85., 'target': -2., 'later': 50., 'future_tb': -50.,
                  'height_z': 4., 'sex:Male': 1., 'target:age_spline_1': .4,
                  'target:age_spline_2': -.8, 'target:age_spline_3': .3}
        for term, value in values.items():
            beta[terms.index(term)] = value
        differences, references = [], []
        for height, male in ((-1., .1), (2., .9)):
            reference = prediction_vector(terms, np.array([-1., .5, 2.]), height, male, False)
            exposed = prediction_vector(terms, np.array([-1., .5, 2.]), height, male, True)
            self.assertEqual(reference[terms.index('later')], 0.)
            self.assertEqual(exposed[terms.index('future_tb')], 0.)
            differences.append((exposed - reference) @ beta)
            references.append(reference @ beta)
        np.testing.assert_allclose(differences, [-2.2, -2.2], atol=1e-13)
        self.assertAlmostEqual(references[1] - references[0], 12.8)

    def test_difference_interval_preserves_between_curve_covariance(self):
        from dchs_lung_function.models.spirometry_trajectories import FULL_TERMS, prediction_vector, estimate_prediction
        terms = list(FULL_TERMS)
        intercept, target = terms.index('Intercept'), terms.index('target')
        beta = np.zeros(len(terms)); beta[intercept] = 85.; beta[target] = -2.
        covariance = np.eye(len(terms))
        covariance[intercept, intercept] = 9.; covariance[target, target] = 4.
        covariance[intercept, target] = covariance[target, intercept] = -3.
        fit = SimpleNamespace(coefficients=pd.Series(beta, index=terms),
                              covariance=pd.DataFrame(covariance, index=terms, columns=terms),
                              metadata={'accepted': True})
        reference = prediction_vector(terms, np.zeros(3), 0., 0., False)
        exposed = prediction_vector(terms, np.zeros(3), 0., 0., True)
        actual = estimate_prediction(exposed - reference, fit)
        # Curve variances are 9 and 7, but their shared covariance is 6.
        # Therefore difference variance is 9+7-2*6=4, not 9+7=16.
        np.testing.assert_allclose(actual, [-2., 2., -5.92, 1.92], atol=1e-13)
        separate_curve_variances = (estimate_prediction(exposed, fit)[1]**2
                                    + estimate_prediction(reference, fit)[1]**2)
        self.assertAlmostEqual(separate_curve_variances, 16.)
        self.assertNotEqual(actual[1]**2, separate_curve_variances)

    def test_full_age_contrasts_use_interactions_parallel_contrasts_do_not(self):
        from dchs_lung_function.models.spirometry_trajectories import (
            FULL_TERMS, REDUCED_TERMS, spline_basis, basis_at, prediction_vector,
        )
        ages = np.linspace(5., 10., 31)
        basis, info = spline_basis(ages)
        # A straight line lies in this spline space. Construct a known full
        # difference -2 + .6*(age-7.5), distinct from a reduced shift of -1.25.
        interaction = np.linalg.lstsq(basis, .6 * (ages - 7.5), rcond=None)[0]
        full_beta = np.zeros(len(FULL_TERMS)); full_beta[list(FULL_TERMS).index('target')] = -2.
        for i in range(3):
            full_beta[list(FULL_TERMS).index(f'target:age_spline_{i+1}')] = interaction[i]
        reduced_beta = np.zeros(len(REDUCED_TERMS)); reduced_beta[list(REDUCED_TERMS).index('target')] = -1.25
        points = np.array([5., 7.5, 10.])
        full, parallel = [], []
        for row in basis_at(info, points):
            full.append((prediction_vector(FULL_TERMS, row, 0., .5, True)
                         - prediction_vector(FULL_TERMS, row, 0., .5, False)) @ full_beta)
            parallel.append((prediction_vector(REDUCED_TERMS, row, 0., .5, True)
                             - prediction_vector(REDUCED_TERMS, row, 0., .5, False)) @ reduced_beta)
        np.testing.assert_allclose(full, [-3.5, -2., -.5], atol=1e-12)
        np.testing.assert_allclose(parallel, [-1.25, -1.25, -1.25], atol=0)


if __name__ == '__main__':
    unittest.main()
