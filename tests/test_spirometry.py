"""Fabricated data check the primary graph against independent probability math."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.models.spirometry import (
    BASELINE_HAZARDS, CONTRASTS, HAZARD_BREAKS, POST_TERMS, PROCESSES,
    TARGETS, build_model, fit_spec, validate_contract,
)


def fabricated_contract(target='ratio_pre'):
    """Five invented children span all exposure groups and censoring intervals."""
    children = pd.DataFrame({'child_index': np.arange(5), 'male': [0., 1., 0., 1., 1.]})
    # TB at infancy, age two, age six, or never; child four contributes early TB.
    stop = np.array([.5, 2., 6., 9., 3.])
    event = np.array([1, 1, 1, 0, 1])
    breaks = np.asarray(HAZARD_BREAKS)
    exposure = np.maximum(0., np.minimum(stop[:, None], breaks[1:])-breaks[:-1])
    intervals = event[:, None]*((stop[:, None] >= breaks[:-1]) & (stop[:, None] < breaks[1:]))
    indices = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    z = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0],
                  [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    frames, audits = {}, {}
    for i, name in enumerate(PROCESSES):
        f = pd.DataFrame(z, columns=POST_TERMS)
        f['child_index'] = indices
        f['y_std'] = np.array([-.4, .3, .2, -.1, .6, -.2, .1, -.3])+.07*i
        frames[name] = f
        audits[name] = {'scale': 1.3+.1*i, 'n_observations': len(f), 'n_children': 5}
    return {'children': children, 'frames': frames,
            'event': {'breaks': breaks, 'exposure': exposure, 'event': event,
                      'event_interval': intervals, 'stop': stop},
            'weights': np.array([1, 2, 1]), 'hazard_terms': ['male'],
            'hazard_design': children[['male']].to_numpy(copy=True),
            'post_diagnosis_terms': list(POST_TERMS),
            'post_diagnosis_designs': {k: f[list(POST_TERMS)].to_numpy(copy=True) for k, f in frames.items()},
            'audit': {'family': 'primary', 'target': TARGETS[target][0], 'covariates': [],
                      'joint_children': 5, 'tb_events': 4, 'weights': [1, 2, 1], 'outcomes': audits}}


def normal_logp(y, mean, sd):
    return -.5*np.log(2*np.pi)-np.log(sd)-.5*((np.asarray(y)-mean)/sd)**2


class ContractTests(unittest.TestCase):
    def test_fixed_twelve_targets_seeds_and_json_metadata(self):
        self.assertEqual(len(TARGETS), 12)
        for target in TARGETS:
            spec = fit_spec(target)
            json.dumps(spec, allow_nan=False)
            self.assertEqual(validate_contract(fabricated_contract(target), target)['weights'], [1, 2, 1])
        self.assertEqual(fit_spec('ratio_pre')['sampling']['random_seed'], 2026092308)
        self.assertEqual(fit_spec('fvc_post_native')['sampling']['random_seed'], 2026092407)
        spec['sampling']['initvals'][0]['loading_mechanics'][0] = 99
        self.assertEqual(fit_spec('fvc_post_native')['sampling']['initvals'][0]['loading_mechanics'][0], .25)
        with self.assertRaises(ValueError): fit_spec('unsupported')

    def test_contract_rejects_altered_samples_or_matrices(self):
        def change_frame(p, column, value):
            frame = p['frames']['target']
            frame[column] = value
        mutations = [
            lambda p: p['weights'].__setitem__(0, 2),
            lambda p: p['hazard_design'].__setitem__((0, 0), 1),
            lambda p: p['event']['exposure'].__setitem__((0, 0), .1),
            lambda p: p['event']['event_interval'].__setitem__((0, 0), 0),
            lambda p: p['event']['breaks'].__setitem__(1, 2),
            lambda p: change_frame(p, 'y_std', np.nan),
            lambda p: change_frame(p, 'child_index', 999),
            lambda p: p['post_diagnosis_designs']['target'].__setitem__((0, 0), 1),
            lambda p: p['audit']['outcomes']['target'].__setitem__('n_observations', 9),
            lambda p: p['audit'].__setitem__('family', 'mechanics'),
        ]
        for mutate in mutations:
            p = fabricated_contract(); mutate(p)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError): validate_contract(p, 'ratio_pre')

    def test_boundary_events_and_child_level_weights(self):
        p = fabricated_contract()
        p['event']['stop'][0] = 1.
        p['event']['exposure'][0] = [1., 0., 0., 0., 0., 0.]
        p['event']['event_interval'][0] = [0, 1, 0, 0, 0, 0]
        self.assertEqual(validate_contract(p, 'ratio_pre')['tb_events'], 4)
        p['frames']['target'].loc[7, 'child_index'] = 0
        p['audit']['outcomes']['target']['n_children'] = 4
        with self.assertRaisesRegex(ValueError, 'age groups'): validate_contract(p, 'ratio_pre')


class ModelGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = fabricated_contract()
        cls.model = build_model(cls.contract, 'ratio_pre')
        cls.point = cls.model.initial_point()
        values = {
            'mechanics': [-.4, .2, .5, -.1, .3], 'lci_raw': [.1, -.2, .3, .4, -.5],
            'tau_lci': .8, 'tau_target': .45, 'target_raw': [.2, -.5, .4, .1, -.3], 'loading_mechanics': [.6, .7, .8, .9],
            'sigma_mechanics': [.8, .9, 1.1, 1.2], 'sigma_lci': .9,
            'shift_mechanics': -.2, 'shift_mechanics_infancy': -.3,
            'shift_mechanics_late': -.1, 'shift_lci': .15, 'shift_lci_infancy': .25,
            'shift_lci_late': -.12, 'direct_ratio': -.18, 'direct_ratio_infancy': -.21,
            'direct_ratio_late': -.14, 'loading_ratio_mechanics': .65, 'sigma_ratio': .85,
            'log_baseline_hazard': np.log([.03, .02, .01, .008, .006, .004]),
            'assoc_mechanics': -.4, 'hazard_male': .2,
        }
        cls.values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        for name, value in cls.values.items():
            key = name if name in cls.point else name+'_log__'
            cls.point[key] = value if key == name else np.log(value)
        cls.factor_logp = cls.model.compile_logp(sum=False, jacobian=False)

    def test_all_prior_and_likelihood_factors_match_independent_math(self):
        v, p = self.values, self.contract
        rvs = self.model.free_RVs+self.model.observed_RVs+self.model.potentials
        actual = {rv.name: np.sum(value) for rv, value in zip(rvs, self.factor_logp(self.point))}
        expected = {}
        for name, (mean, sd) in fit_spec('ratio_pre')['normal_priors_mean_sd'].items():
            expected[name] = np.sum(normal_logp(v[name], mean, sd))
        for name, sd in fit_spec('ratio_pre')['half_normal_prior_sd'].items():
            expected[name] = np.sum(np.log(2.)+normal_logp(v[name], 0., sd))
        expected['log_baseline_hazard'] = np.sum(normal_logp(v['log_baseline_hazard'], np.log(BASELINE_HAZARDS), 1.25))
        for j, name in enumerate(['resistance', 'compliance', 'xee', 'ree']):
            f = p['frames'][name]; ix = f.child_index.to_numpy()
            z = f[list(POST_TERMS)].to_numpy()
            change = z@np.array([v['shift_mechanics'], v['shift_mechanics_infancy'], v['shift_mechanics_late']])
            mean = v['loading_mechanics'][j]*(v['mechanics'][ix]+change)
            expected['obs_'+name] = np.sum(normal_logp(f.y_std.to_numpy(), mean, v['sigma_mechanics'][j]))
        f = p['frames']['lci']; z = f[list(POST_TERMS)].to_numpy()
        mean = v['tau_lci']*v['lci_raw'][f.child_index]+z@np.array([v['shift_lci'], v['shift_lci_infancy'], v['shift_lci_late']])
        expected['obs_lci'] = np.sum(normal_logp(f.y_std.to_numpy(), mean, v['sigma_lci']))
        f = p['frames']['target']; z = f[list(POST_TERMS)].to_numpy()
        change = z@np.array([v['shift_mechanics'], v['shift_mechanics_infancy'], v['shift_mechanics_late']])
        direct = z@np.array([v['direct_ratio'], v['direct_ratio_infancy'], v['direct_ratio_late']])
        mean = v['loading_ratio_mechanics']*(v['mechanics'][f.child_index]+change)+v['tau_target']*v['target_raw'][f.child_index]+direct
        expected['obs_fev1_fvc'] = np.sum(normal_logp(f.y_std.to_numpy(), mean, v['sigma_ratio']))
        log_h = v['log_baseline_hazard'][None, :]+(v['assoc_mechanics']*v['mechanics']+v['hazard_male']*p['children'].male.to_numpy())[:, None]
        expected['first_tb_loglik'] = np.sum(p['event']['event_interval']*log_h-p['event']['exposure']*np.exp(log_h))
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            # PyTensor constant-folds scalar prior SD logs in float32; the
            # independent float64 density differs by at most 3e-8 here.
            places = 11 if name.startswith('obs_') or name == 'first_tb_loglik' else 7
            with self.subTest(factor=name): self.assertAlmostEqual(actual[name], expected[name], places=places)

    def test_contrasts_and_dimensions_match_hand_calculation(self):
        v = self.values
        early = float(v['direct_ratio']+v['loading_ratio_mechanics']*v['shift_mechanics'])
        infant = float(v['direct_ratio']+v['direct_ratio_infancy']+v['loading_ratio_mechanics']*(v['shift_mechanics']+v['shift_mechanics_infancy']))
        late = float(v['direct_ratio_late']+v['loading_ratio_mechanics']*v['shift_mechanics_late'])
        expected = [infant, early, late, (infant+2*early)/3, (infant+2*early+late)/4,
                    infant-early, late-infant, late-early]
        outputs = self.model.replace_rvs_by_values([self.model[name] for name in CONTRASTS.values()])
        fn = self.model.compile_fn(outputs, inputs=self.model.value_vars, on_unused_input='ignore')
        np.testing.assert_allclose(fn(self.point), expected, atol=1e-14, rtol=0.)
        self.assertEqual(len(self.model.coords['child']), 5)
        self.assertEqual(len(self.model.coords['hazard_interval']), 6)
        self.assertEqual(len(self.model.coords['mechanics_indicator']), 4)

    def test_hazard_gradient_and_lci_independence(self):
        # The analytic derivative includes the N(0,.75) prior and survival term.
        v, p = self.values, self.contract
        log_h = v['log_baseline_hazard'][None, :]+(v['assoc_mechanics']*v['mechanics']+v['hazard_male']*p['children'].male.to_numpy())[:, None]
        expected = -float(v['assoc_mechanics'])/.75**2+np.sum((p['event']['event_interval']-p['event']['exposure']*np.exp(log_h))*v['mechanics'][:, None])
        fn = self.model.compile_dlogp(vars=[self.model['assoc_mechanics']], jacobian=False)
        self.assertAlmostEqual(float(fn(self.point)[0]), expected, places=11)
        rvs = self.model.free_RVs+self.model.observed_RVs+self.model.potentials
        before = {rv.name: np.sum(value) for rv, value in zip(rvs, self.factor_logp(self.point))}
        changed = deepcopy(self.point); changed['lci_raw'] = changed['lci_raw']+.5
        after = {rv.name: np.sum(value) for rv, value in zip(rvs, self.factor_logp(changed))}
        for name in ['first_tb_loglik', 'obs_fev1_fvc', 'obs_resistance', 'obs_compliance', 'obs_xee', 'obs_ree']:
            self.assertEqual(before[name], after[name])
        self.assertNotEqual(before['obs_lci'], after['obs_lci'])

    def test_builder_preserves_input_and_keeps_identifier_free_coordinates(self):
        p = fabricated_contract(); before = deepcopy(p)
        model = build_model(p, 'ratio_pre')
        for name in PROCESSES: pd.testing.assert_frame_equal(p['frames'][name], before['frames'][name])
        pd.testing.assert_frame_equal(p['children'], before['children'])
        self.assertEqual(tuple(model.coords['child']), (0, 1, 2, 3, 4))


if __name__ == '__main__':
    unittest.main()
