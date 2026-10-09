"""Independent manuscript-definition checks using entirely fabricated records."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.preparation import binary_inputs
from dchs_lung_function.models.binary_spirometry import classify_outcome, fit_binary, pattern_counts


def fabricated_definition_frame():
    """A hand-specified nine-cell diagnostic table crossed with five TB states."""
    cases = [
        ('mixed', -2., -2.),
        ('obstruction_equal_fvc', -2., -1.645),
        ('obstruction_high_fvc', -2., 0.),
        ('restriction_equal_ratio', -1.645, -2.),
        ('both_equal', -1.645, -1.645),
        ('equal_ratio_high_fvc', -1.645, 0.),
        ('restriction_high_ratio', 0., -2.),
        ('high_ratio_equal_fvc', 0., -1.645),
        ('both_high', 0., 0.),
    ]
    states = [('never', None, np.nan), ('infancy', '2014-07-01', .5),
              ('age_one', '2015-01-01', 1.), ('age_five', '2019-01-01', 5.),
              ('future', '2021-01-01', 7.)]
    records = []
    for state, diagnosis, age in states:
        for case, ratio, fvc in cases:
            index = len(records)
            records.append({
                'pid_child': f'fabricated_{index:03d}', 'case': case, 'test_state': state,
                'visit_label': '60 months', 'male': float(index % 2),
                'height': 101. + index * .23 + (index % 3) * .11,
                'child_dob': pd.Timestamp('2014-01-01'),
                'date_of_test': pd.Timestamp('2020-01-01'),
                'tb_first_dx': pd.Timestamp(diagnosis) if diagnosis else pd.NaT,
                'tb_age_first_years': age, 'exit_date': pd.NaT,
                'fev1_fvc_gli_z': ratio, 'fvc_gli_z': fvc,
                'fev1_gli_z': -2. if case == 'both_high' else 0.,
                'fev1_fvc_post_gli_z': 0., 'fvc_post_gli_z': 0., 'fev1_post_gli_z': 0.,
            })
    return pd.DataFrame(records)


class BinaryDefinitionsIndependentTests(unittest.TestCase):
    def test_hand_classified_boundary_and_mixed_patterns(self):
        # These sets are a diagnostic truth table, independent of implementation
        # expressions. Equality at the lower limit is never a low index.
        positive_cases = {
            'ratio_LLN': {'mixed', 'obstruction_equal_fvc', 'obstruction_high_fvc'},
            'FVC_LLN': {'mixed', 'restriction_equal_ratio', 'restriction_high_ratio'},
            'FEV1_LLN': {'both_high'},
            'obstructive_pattern': {'obstruction_equal_fvc', 'obstruction_high_fvc'},
            'possible_restriction': {'restriction_equal_ratio', 'restriction_high_ratio'},
            'any_abnormal': {'mixed', 'obstruction_equal_fvc', 'obstruction_high_fvc',
                             'restriction_equal_ratio', 'restriction_high_ratio'},
        }
        source = fabricated_definition_frame()
        for outcome, expected in positive_cases.items():
            prepared = binary_inputs(source, outcome=outcome)
            frame = prepared['frame']
            with self.subTest(outcome=outcome):
                self.assertEqual(set(frame.loc[frame.outcome_value.eq(1), 'case']), expected)
                self.assertEqual(prepared['audit']['events'], 5 * len(expected))
                self.assertEqual(len(frame), 45)
                classified = classify_outcome(source, 'pre', outcome)
                self.assertEqual(set(source.loc[classified.eq(1), 'case']), expected)

    def test_missing_index_does_not_change_other_index_or_cross_stage(self):
        source = fabricated_definition_frame()
        source.loc[0, 'fvc_gli_z'] = np.nan
        source.loc[1, 'fev1_fvc_gli_z'] = np.nan
        source.loc[2, 'fev1_fvc_post_gli_z'] = np.nan
        source.loc[3, 'fvc_post_gli_z'] = np.nan
        expected_membership = {
            ('pre', 'ratio_LLN'): {1}, ('pre', 'FVC_LLN'): {0},
            ('pre', 'FEV1_LLN'): set(), ('pre', 'obstructive_pattern'): {0, 1},
            ('pre', 'possible_restriction'): {0, 1}, ('pre', 'any_abnormal'): {0, 1},
            ('post', 'ratio_LLN'): {2}, ('post', 'FVC_LLN'): {3},
            ('post', 'FEV1_LLN'): set(), ('post', 'obstructive_pattern'): {2, 3},
            ('post', 'possible_restriction'): {2, 3}, ('post', 'any_abnormal'): {2, 3},
        }
        for (stage, outcome), excluded in expected_membership.items():
            result = binary_inputs(source, stage=stage, outcome=outcome)['frame']
            with self.subTest(stage=stage, outcome=outcome):
                self.assertEqual(set(result.index), set(source.index) - excluded)
                if stage == 'post':
                    self.assertEqual(int(result.outcome_value.sum()), 0)

    def test_never_tb_and_future_tb_have_distinct_design_states(self):
        prepared = binary_inputs(fabricated_definition_frame())
        frame = prepared['frame']
        expected = {
            'never': [0, 0, 0, 0, 0], 'infancy': [1, 1, 0, 0, 0],
            'age_one': [1, 0, 1, 0, 0], 'age_five': [0, 0, 0, 1, 0],
            'future': [0, 0, 0, 0, 1],
        }
        terms = ['post_before5', 'post_infancy', 'post_age1_before5',
                 'post_at_after5', 'future_tb']
        for state, row in expected.items():
            values = frame.loc[frame.test_state.eq(state), terms].to_numpy()
            np.testing.assert_array_equal(values, np.tile(row, (9, 1)))
        for design in prepared['designs'].values():
            future = design['terms'].index('future_tb')
            np.testing.assert_array_equal(design['matrix'][:, future], frame.test_state.eq('future'))
        # Before-five is a separate pooled regression exposure. In the age-group
        # design, infancy and age one to under five are disjoint, not nested.
        self.assertNotIn('post_before5', prepared['designs']['age_groups']['terms'])
        self.assertNotIn('post_infancy', prepared['designs']['before5']['terms'])

    def test_same_day_diagnosis_and_exit_dates_are_inclusive(self):
        source = fabricated_definition_frame()
        source.loc[36, 'date_of_test'] = source.loc[36, 'tb_first_dx']
        source.loc[0, 'exit_date'] = source.loc[0, 'date_of_test']
        source.loc[1, 'exit_date'] = source.loc[1, 'date_of_test'] - pd.Timedelta(days=1)
        frame = binary_inputs(source)['frame']
        self.assertIn(0, frame.index)
        self.assertNotIn(1, frame.index)
        self.assertEqual(frame.loc[36, 'future_tb'], 0)
        self.assertEqual(frame.loc[36, 'post_at_after5'], 1)
        self.assertEqual(frame.loc[36, 'post_any'], 1)

    def test_nominal_school_visits_do_not_add_an_exact_age_window(self):
        source = fabricated_definition_frame()
        source.loc[0, 'child_dob'] = pd.Timestamp('2015-06-01')
        source.loc[1, 'child_dob'] = pd.Timestamp('2009-06-01')
        prepared = binary_inputs(source)
        self.assertLess(prepared['frame'].loc[0, 'age_years'], 5.)
        self.assertGreater(prepared['frame'].loc[1, 'age_years'], 10.)
        self.assertEqual(prepared['audit']['n_observations'], 45)

    def test_nonfinite_required_scores_remain_unclassified(self):
        source = fabricated_definition_frame()
        source.loc[[0, 1, 2], 'fvc_gli_z'] = [np.nan, np.inf, -np.inf]
        for outcome in ['FVC_LLN', 'obstructive_pattern', 'possible_restriction', 'any_abnormal']:
            with self.subTest(outcome=outcome):
                self.assertTrue(classify_outcome(source, 'pre', outcome).loc[[0, 1, 2]].isna().all())
        self.assertTrue(classify_outcome(source, 'pre', 'ratio_LLN').loc[[0, 1, 2]].notna().all())

    def test_pattern_ledger_retains_mixed_small_and_zero_cells(self):
        source = fabricated_definition_frame()
        counts = pattern_counts(binary_inputs(source, outcome='obstructive_pattern'))
        expected = {'Normal': 4, 'Obstructive': 2, 'Possible restriction': 2, 'Mixed': 1}
        self.assertEqual(len(counts), 20)
        for _, group in counts.groupby('exposure_state'):
            self.assertEqual(dict(zip(group.pattern, group.event_observations)), expected)
            self.assertEqual(int(group.event_observations.sum()), 9)
        post = pattern_counts(binary_inputs(source, stage='post', outcome='obstructive_pattern'))
        self.assertEqual(len(post), 20)
        self.assertEqual(int(post.loc[post.pattern.ne('Normal'), 'event_observations'].sum()), 0)

    def test_known_risk_ratio_and_cluster_variance_keep_future_cases_separate(self):
        # Ten invented children per state, with two identical records per child:
        # risks are 2/10 never-TB, 4/10 infancy-TB and 9/10 future-TB.
        # The adjusted RR is exactly 2, whereas the odds ratio would be 8/3.
        records = []
        for state, positive_children, diagnosis, age in [
            ('never', 2, None, np.nan), ('infancy', 4, '2014-07-01', .5),
            ('future', 9, '2021-01-01', 7.),
        ]:
            for child in range(10):
                for _ in range(2):
                    records.append({
                        'pid_child': f'fabricated_{state}_{child}', 'visit_label': '60 months',
                        'male': 0., 'height': 120., 'child_dob': pd.Timestamp('2014-01-01'),
                        'date_of_test': pd.Timestamp('2020-01-01'), 'exit_date': pd.NaT,
                        'tb_first_dx': pd.Timestamp(diagnosis) if diagnosis else pd.NaT,
                        'tb_age_first_years': age, 'fev1_fvc_gli_z': -2. if child < positive_children else 0.,
                    })
        result = fit_binary(binary_inputs(pd.DataFrame(records)), 'before5')
        row = result.estimates.iloc[0]
        self.assertTrue(result.metadata['accepted'])
        self.assertTrue(row.estimable)
        self.assertAlmostEqual(row.rr, 2., delta=1e-10)
        self.assertNotAlmostEqual(row.rr, 8 / 3)
        # Exact child sandwich variance for two independent binomial proportions;
        # duplicated within-child records do not double the independent sample.
        variance = (1 - .4) / (10 * .4) + (1 - .2) / (10 * .2)
        self.assertAlmostEqual(row.se ** 2, variance, delta=1e-10)
        self.assertAlmostEqual(row.lower, np.exp(np.log(2.) - 1.96 * np.sqrt(variance)), delta=1e-10)
        self.assertAlmostEqual(row.upper, np.exp(np.log(2.) + 1.96 * np.sqrt(variance)), delta=1e-10)
        self.assertEqual(row.reference_observations, 20)
        self.assertEqual(row.reference_children, 10)
        self.assertEqual(row.reference_events, 4)
        self.assertEqual(row.exposed_events, 8)
        self.assertEqual(row.exposed_children_ever_positive, 4)
        self.assertNotIn('pid_child', result.estimates.columns)


if __name__ == '__main__':
    unittest.main()
