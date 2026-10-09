"""Fabricated binary spirometry tests, including clustered covariance boundaries."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.preparation import repeated_design
from dchs_lung_function.models.binary_spirometry import (
    DEFAULT_FIT_SELECTION, EXPOSURE_TERMS, classify_outcome, fit_binary,
    fit_spec, pattern_counts, validate_contract,
)


def make_contract(stage='pre', outcome='ratio_LLN', *, boundary=None, zero_visit=False):
    rng = np.random.default_rng(5516)
    children = np.repeat(np.arange(180), 3)
    group = children % 5
    n = len(children)
    height = 120.+rng.normal(0, 10, n)
    male = np.repeat(rng.integers(0, 2, 180), 3)
    frame = pd.DataFrame({'pid_child': ['fabricated_'+str(i) for i in children],
                          'height': height, 'male': male,
                          'visit_label': np.tile(['60 months', '72 months', '84 months'], 180)})
    frame['future_tb'] = (group == 1).astype(float)
    frame['post_infancy'] = (group == 2).astype(float)
    frame['post_age1_before5'] = (group == 3).astype(float)
    frame['post_at_after5'] = (group == 4).astype(float)
    frame['post_before5'] = frame.post_infancy+frame.post_age1_before5
    frame['post_any'] = frame.post_before5+frame.post_at_after5
    cluster_effect = np.repeat(rng.normal(0, .25, 180), 3)
    probability = .22*np.exp(.12*(height-120)/10+.15*male+.3*frame.post_before5+cluster_effect)
    event = (rng.uniform(size=n) < probability).astype(int)
    if boundary == 'zero': event[group == 2] = 0
    elif boundary == 'full': event[group == 2] = 1
    elif boundary == 'all_zero': event[:] = 0
    elif boundary == 'absent':
        frame.loc[:, ['post_infancy', 'post_age1_before5', 'post_at_after5', 'post_before5', 'post_any']] = 0.
    for suffix in ['', '_post']:
        frame['fev1_fvc'+suffix+'_gli_z'] = np.where(event, -2., 0.)
        frame['fev1'+suffix+'_gli_z'] = np.where(event, -2., 0.)
        frame['fvc'+suffix+'_gli_z'] = 0.
    if zero_visit:
        extra = frame.iloc[:10].copy(); extra['visit_label'] = 'FeNO'
        extra['fev1_fvc_gli_z'] = 0.; extra['fev1_fvc_post_gli_z'] = 0.
        frame = pd.concat([frame, extra], ignore_index=True)
    frame['outcome_value'] = classify_outcome(frame, stage, outcome).astype(int)
    designs = {}
    for key in ['before5', 'age_groups']:
        x, names, dropped = repeated_design(frame, EXPOSURE_TERMS[key])
        designs[key] = {'matrix': x, 'terms': names, 'dropped_terms': dropped}
    audit = {'family': 'binary', 'stage': stage, 'outcome': outcome,
             'n_observations': len(frame), 'n_children': frame.pid_child.nunique(),
             'events': int(frame.outcome_value.sum()),
             'designs': {key: {name: value[name] for name in ['terms', 'dropped_terms']} for key, value in designs.items()}}
    return {'frame': frame, 'designs': designs, 'audit': audit}


class BinarySpirometryTests(unittest.TestCase):
    def test_selected_specifications_and_complete_parameter_metadata(self):
        self.assertEqual(len(DEFAULT_FIT_SELECTION), 28)
        self.assertTrue(all(stage == 'pre' for stage, outcome, _ in DEFAULT_FIT_SELECTION if outcome in ('any_abnormal', 'possible_restriction')))
        self.assertEqual(sum(len(fit_spec(*selection)['reported_terms']) for selection in DEFAULT_FIT_SELECTION), 46)
        result = fit_binary(make_contract(), 'before5', stage='pre', outcome='ratio_LLN')
        self.assertTrue(result.metadata['accepted'])
        self.assertEqual(result.metadata['status'], 'success')
        self.assertEqual(result.coefficients.index.tolist(), result.covariance.index.tolist())
        self.assertEqual(result.coefficients.index.tolist(), result.covariance.columns.tolist())
        self.assertTrue(bool(result.estimates.selected_for_figure2.iloc[0]))
        json.dumps(result.metadata, allow_nan=False)

    def test_robust_covariance_matches_independent_cluster_score_sandwich(self):
        contract = make_contract(); result = fit_binary(contract, 'age_groups')
        frame = contract['frame']; x = contract['designs']['age_groups']['matrix']
        y = frame.outcome_value.to_numpy(); beta = result.coefficients.to_numpy()
        mu = np.exp(x@beta); bread = x.T@(mu[:, None]*x)
        scores = []
        for _, group in frame.groupby('pid_child', sort=True):
            index = group.index.to_numpy(); scores.append(x[index].T@(y[index]-mu[index]))
        influences = np.linalg.solve(bread, np.asarray(scores).T)
        expected = influences@influences.T
        np.testing.assert_allclose(result.covariance, expected, atol=1e-11, rtol=1e-10)
        self.assertLess(np.linalg.norm(x.T@(y-mu)), 1e-6)
        self.assertFalse(np.allclose(expected, np.linalg.inv(bread)))

    def test_zero_event_nuisance_has_stable_symmetric_covariance_without_changing_fit(self):
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Poisson
        from statsmodels.genmod.cov_struct import Independence
        contract = make_contract(zero_visit=True)
        result = fit_binary(contract, 'before5')
        f = contract['frame']; x = contract['designs']['before5']['matrix']
        original = GEE(f.outcome_value.to_numpy(), x, groups=f.pid_child.to_numpy(), family=Poisson(), cov_struct=Independence()).fit(maxiter=250, cov_type='robust')
        np.testing.assert_array_equal(result.coefficients, original.params)
        np.testing.assert_allclose(np.diag(result.covariance), np.diag(original.cov_params()), atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(result.covariance, result.covariance.T)
        evaluation = result.metadata['covariance_evaluation']
        self.assertGreater(evaluation['raw_information_condition'], evaluation['normalized_information_condition']*1e4)
        self.assertIn('visit:FeNO', [row['term'] for row in result.metadata['boundary_terms']])
        self.assertTrue(result.metadata['accepted'])

    def test_boundary_and_absent_exposures_are_explicit_and_keep_counts(self):
        for boundary, reason in [('zero', 'zero_exposed_events'), ('full', 'full_exposed_events'), ('absent', 'no_exposed_observations')]:
            result = fit_binary(make_contract(boundary=boundary), 'age_groups')
            row = result.estimates.loc[result.estimates.term.eq('post_infancy')].iloc[0]
            self.assertFalse(bool(row.estimable)); self.assertEqual(row.reason_code, reason)
            self.assertTrue(pd.isna(row.rr)); self.assertTrue(pd.isna(row.p_value))
            self.assertEqual(row.exposed_observations, 0 if boundary == 'absent' else 108)
            self.assertTrue(result.metadata['accepted'], result.metadata)
        result = fit_binary(make_contract(boundary='all_zero'), 'before5')
        self.assertEqual(result.metadata['status'], 'non_estimable')
        self.assertEqual(result.metadata['fit_status'], 'not_fitted')
        self.assertTrue(result.metadata['accepted'])

    def test_nonconvergence_and_numerical_exceptions_cannot_be_accepted(self):
        contract = make_contract(); p = len(contract['designs']['before5']['terms'])
        bad = SimpleNamespace(params=np.zeros(p), cov_params=lambda: np.eye(p), converged=False, fit_history={'params': []})
        with patch('statsmodels.genmod.generalized_estimating_equations.GEE.fit', return_value=bad):
            result = fit_binary(contract, 'before5')
        self.assertFalse(result.metadata['accepted']); self.assertEqual(result.metadata['status'], 'fit_failed')
        self.assertFalse(result.estimates.estimable.any())
        with patch('statsmodels.genmod.generalized_estimating_equations.GEE.fit', side_effect=np.linalg.LinAlgError('fabricated failure')):
            result = fit_binary(contract, 'before5')
        self.assertFalse(result.metadata['accepted']); self.assertEqual(result.metadata['fit_status'], 'exception')

    def test_mutated_classification_design_or_requested_identity_is_rejected(self):
        contract = make_contract()
        with self.assertRaises(ValueError): fit_binary(contract, 'before5', stage='post')
        with self.assertRaises(ValueError): fit_binary(contract, 'before5', outcome='FEV1_LLN')
        for change in ['outcome', 'design', 'count', 'exposure']:
            p = deepcopy(contract)
            if change == 'outcome': p['frame'].loc[0, 'outcome_value'] = 1-p['frame'].loc[0, 'outcome_value']
            elif change == 'design': p['designs']['before5']['matrix'][0, 0] = 2.
            elif change == 'count': p['audit']['events'] += 1
            else: p['frame'].loc[0, 'post_any'] = 1.
            with self.subTest(change=change), self.assertRaises(ValueError): validate_contract(p, 'before5')

    def test_pattern_counts_reconcile_all_states_and_retain_zeros(self):
        contract = make_contract(outcome='obstructive_pattern')
        counts = pattern_counts(contract)
        self.assertEqual(len(counts), 20)
        self.assertEqual(int(counts.event_observations.sum()), len(contract['frame']))
        for _, group in counts.groupby('exposure_state'):
            self.assertEqual(int(group.event_observations.sum()), group.observations.iloc[0])
        self.assertTrue((counts.loc[counts.pattern.eq('Mixed'), 'event_observations'] == 0).all())
        self.assertEqual(counts.loc[counts.exposure_state.eq('future_tb'), 'children'].iloc[0], 36)

    def test_any_tb_is_derived_from_the_same_sample_and_inputs_unchanged(self):
        contract = make_contract(); before = deepcopy(contract)
        result = fit_binary(contract, 'any_tb')
        self.assertTrue(result.metadata['accepted'])
        self.assertEqual(result.metadata['n_observations'], len(contract['frame']))
        self.assertEqual(result.metadata['exposure_terms'], ['post_any', 'future_tb'])
        self.assertEqual(result.estimates.term.tolist(), ['post_any'])
        pd.testing.assert_frame_equal(contract['frame'], before['frame'])
        for name in ['before5', 'age_groups']:
            np.testing.assert_array_equal(contract['designs'][name]['matrix'], before['designs'][name]['matrix'])


if __name__ == '__main__':
    unittest.main()
