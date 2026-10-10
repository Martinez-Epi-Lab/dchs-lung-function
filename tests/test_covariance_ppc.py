"""Independent synthetic checks of the frozen repeated-target adequacy screen."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.covariance_ppc import (
    DEFAULT_CRITERIA, covariance_statistics, evaluate_target_covariance, protocol,
)


def fabricated_case(*, omitted=False, children=80, repeats=4, family='primary'):
    """Invented data with persistent target variability independent of mechanics."""
    rng = np.random.default_rng(567841)
    counts = np.resize(np.array([repeats, repeats+1, repeats+2]), children)
    idx = np.repeat(np.arange(children), counts)
    raw = rng.normal(size=children); factor = rng.normal(size=children)
    mean = .25 * factor + .9 * raw
    y = mean[idx] + .35 * rng.normal(size=len(idx))
    frame = pd.DataFrame({'child_index': idx, 'y_std': y, 'post_any': 0.,
        'post_before5': 0., 'post_infancy': 0., 'post_at_after5': 0.})
    contract = {'children': pd.DataFrame({'child_index': np.arange(children)}),
        'frames': {'target': frame}, 'audit': {'target': 'fev1_fvc'},
        'post_diagnosis_terms': ['post_before5', 'post_infancy', 'post_at_after5']}
    shape = (4, 600)
    posterior = {'mechanics': np.broadcast_to(factor, (*shape, children)).copy(),
        'target_raw': np.broadcast_to(raw, (*shape, children)).copy(),
        'tau_target': np.full(shape, .0001 if omitted else .9),
        'loading_ratio_mechanics': np.full(shape, .25),
        'sigma_ratio': np.full(shape, np.sqrt(.9**2+.35**2) if omitted else .35)}
    posterior['target_process'] = posterior['tau_target'][..., None] * posterior['target_raw']
    if family == 'mechanics':
        contract['audit']['target'] = 'resistance'
        posterior = {'shared_outcome': np.broadcast_to(mean, (*shape, children)).copy(),
            'tau_child': np.full(shape, .9), 'child_z': np.broadcast_to(mean/.9, (*shape, children)).copy(),
            'sigma_outcome': np.full(shape, .35)}
    return contract, posterior


class CovarianceStatisticTests(unittest.TestCase):
    def test_unequal_repeat_anova_matches_independent_group_calculation(self):
        idx = np.array([0,0,1,1,1,2,2,2,2]); y = np.array([1.,3.,2.,4.,6.,-2.,0.,2.,4.])
        groups = [y[idx == k] for k in range(3)]; n = len(y)
        msw = sum(sum((g-g.mean())**2) for g in groups)/(n-3)
        msb = sum(len(g)*(g.mean()-y.mean())**2 for g in groups)/2
        n0 = (n-sum(len(g)**2 for g in groups)/n)/2
        observed = covariance_statistics(y, idx)
        self.assertAlmostEqual(observed['within_ms'], msw, places=14)
        self.assertAlmostEqual(observed['icc'], (msb-msw)/(msb+(n0-1)*msw), places=14)
        self.assertAlmostEqual(observed['effective_repeat_count'], n0, places=14)
        shifted = covariance_statistics(17+3*y, idx)
        self.assertAlmostEqual(shifted['icc'], observed['icc'], places=14)
        self.assertAlmostEqual(shifted['within_ms'], 9*msw, places=14)

    def test_negative_icc_is_not_suppressed_and_undefined_inputs_rejected(self):
        result = covariance_statistics(np.array([-1.,1.,-1.,1.]), np.array([0,0,1,1]))
        self.assertEqual(result['icc'], -1.)
        for y, idx in [(np.ones(4),np.array([0,0,1,1])), (np.arange(4.),np.arange(4)),
                       (np.array([1.,np.nan,2.,3.]),np.array([0,0,1,1])),
                       (np.arange(4.),np.array([0.,0.,1.,1.]))]:
            with self.assertRaises(ValueError): covariance_statistics(y,idx)


class CovariancePPCTests(unittest.TestCase):
    def test_well_specified_covariance_passes_and_omitted_persistence_fails(self):
        contract, posterior = fabricated_case()
        result = evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertTrue(result['accepted']); self.assertEqual(result['status'],'passed')
        self.assertEqual(result['counts']['repeat_children'],80)
        self.assertEqual(result['draw_selection']['replicates'],2000)
        contract, wrong = fabricated_case(omitted=True)
        failed = evaluate_target_covariance(wrong,contract,family='primary',target='ratio_pre')
        self.assertFalse(failed['accepted']); self.assertEqual(failed['status'],'failed')
        self.assertEqual(set(failed['reasons']),{'icc_outside_predictive_bounds','within_ms_outside_predictive_bounds'})
        self.assertEqual(failed['predictive']['icc']['tail_at_or_above_observed'],0.)
        self.assertEqual(failed['predictive']['within_ms']['tail_at_or_below_observed'],0.)
        self.assertFalse(result['population_formula_icc']['used_for_acceptance'])
        json.dumps(result,allow_nan=False)

    def test_conditioning_uses_realized_selected_child_effects_not_prior_formula(self):
        contract, posterior = fabricated_case()
        # A nonrepresentative observed subset can have factor variance unrelated
        # to its prior unit variance; predictions must retain those actual effects.
        posterior['target_raw'] *= 2.
        posterior['target_process'] *= 2.
        frame=contract['frames']['target']; idx=frame.child_index.to_numpy()
        frame['y_std'] += .9*posterior['target_raw'][0,0,idx]/2.
        result=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertTrue(result['accepted'])
        self.assertGreater(result['predictive']['icc']['mean'],result['population_formula_icc']['mean']+.04)

    def test_seed_draw_selection_reproducibility_and_no_input_or_identifier_exports(self):
        contract, posterior=fabricated_case(); before=deepcopy(contract)
        contract['children']['pid_child']=['private_'+str(i) for i in range(80)]
        one=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        two=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertEqual(one,two)
        pd.testing.assert_frame_equal(contract['frames']['target'],before['frames']['target'])
        rendered=json.dumps(one,allow_nan=False)
        self.assertNotIn('private_',rendered); self.assertNotIn('pid_child',rendered)
        p=protocol();p['criteria']['replicates']=1
        self.assertEqual(protocol()['criteria']['replicates'],2000)
        with self.assertRaises(ValueError): evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre',criteria={**DEFAULT_CRITERIA,'replicates':10})

    def test_ineligible_samples_and_draw_counts_are_explicit_not_passed(self):
        contract,posterior=fabricated_case(children=9,repeats=2)
        result=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertEqual(result['counts']['selected_children'],9)
        self.assertEqual(result['status'],'not_evaluable');self.assertFalse(result['accepted'])
        self.assertIn('insufficient_repeat_children',result['reasons'])
        self.assertIn('insufficient_within_df',result['reasons'])
        contract,posterior=fabricated_case()
        posterior={k:v[:,:20] for k,v in posterior.items()}
        result=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertEqual(result['reasons'],['insufficient_saved_draws']);self.assertFalse(result['accepted'])

    def test_nonzero_selected_exposure_and_malformed_posterior_are_rejected(self):
        contract,posterior=fabricated_case()
        bad=deepcopy(contract);bad['frames']['target'].loc[0,'post_infancy']=1.
        with self.assertRaises(ValueError):evaluate_target_covariance(posterior,bad,family='primary',target='ratio_pre')
        for name, change in [('target_process',lambda v:v+.1),('tau_target',lambda v:-v),
                              ('mechanics',lambda v:v[:,:,:-1]),('sigma_ratio',lambda v:v*np.nan)]:
            bad=dict(posterior);bad[name]=change(posterior[name])
            with self.subTest(variable=name),self.assertRaises(ValueError):evaluate_target_covariance(bad,contract,family='primary',target='ratio_pre')
        with self.assertRaises(ValueError):evaluate_target_covariance(posterior,contract,family='unknown',target='ratio_pre')

    def test_post_tb_rows_are_excluded_without_dropping_pre_tb_rows_of_same_child(self):
        contract,posterior=fabricated_case(); original=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        post=contract['frames']['target'].iloc[:12].copy()
        post['post_any']=1.;post['post_before5']=1.;post['y_std']=1e5
        contract['frames']['target']=pd.concat([contract['frames']['target'],post],ignore_index=True)
        result=evaluate_target_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertEqual(original['observed'],result['observed']);self.assertEqual(original['predictive'],result['predictive'])
        self.assertEqual(result['counts']['target_observations'],original['counts']['target_observations']+12)

    def test_mechanics_existing_random_intercept_is_supported_without_new_effect(self):
        contract,posterior=fabricated_case(family='mechanics')
        result=evaluate_target_covariance(posterior,contract,family='mechanics',target='R')
        self.assertTrue(result['accepted']);self.assertEqual(result['family'],'mechanics')
        altered=dict(posterior);altered['shared_outcome']=altered['shared_outcome']+.1
        with self.assertRaises(ValueError):evaluate_target_covariance(altered,contract,family='mechanics',target='R')


if __name__=='__main__':unittest.main()
