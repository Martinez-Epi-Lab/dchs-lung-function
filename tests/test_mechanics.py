"""Fabricated mechanics data test the joint likelihood and adverse orientation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.models.mechanics import (
    BASELINE_HAZARDS, CONTRASTS, DEFAULT_TARGETS, HAZARD_BREAKS,
    POSTERIOR_VARIABLES, POST_TERMS, build_model, fit_spec, validate_contract,
)


def fabricated_contract(target='R'):
    spec = fit_spec(target)
    children = pd.DataFrame({'pid_child': ['fabricated_'+str(i) for i in range(5)],
                             'child_index': np.arange(5), 'male': [0., 1., 0., 1., 1.]})
    index = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    z = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0],
                  [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    frame = pd.DataFrame(z, columns=POST_TERMS)
    frame['pid_child'] = children.pid_child.to_numpy()[index]
    frame['child_index'] = index
    frame['post_any'] = z[:, 0]+z[:, 2]
    frame['y_std'] = [-.4, .3, .2, -.1, .6, -.2, .1, -.3]
    scale = .01 if target == 'C' else .7
    native_mean = .03 if target == 'C' else -2. if target.startswith('X') else 2.
    intercept = native_mean*spec['orientation']
    frame['outcome_value'] = intercept+frame.y_std*scale
    frame[spec['column']] = frame.outcome_value/spec['orientation']
    design = pd.DataFrame({'Intercept': np.ones(len(frame))})
    stop = np.array([.5, 2., 6., 9., 3.]); event = np.array([1, 1, 1, 0, 1])
    breaks = np.asarray(HAZARD_BREAKS)
    exposure = np.maximum(0., np.minimum(stop[:, None], breaks[1:])-breaks[:-1])
    intervals = event[:, None]*((stop[:, None] >= breaks[:-1]) & (stop[:, None] < breaks[1:]))
    return {'children': children, 'frames': {'target': frame}, 'designs': {'target': design},
            'event': {'breaks': breaks, 'stop': stop, 'event': event,
                      'exposure': exposure, 'event_interval': intervals},
            'weights': np.array([1, 2, 1]), 'hazard_terms': ['male'],
            'hazard_design': children[['male']].to_numpy(copy=True),
            'post_diagnosis_terms': list(POST_TERMS),
            'post_diagnosis_designs': {'target': z.copy()},
            'audit': {'family': 'mechanics', 'target': spec['column'], 'covariates': [],
                      'joint_children': 5, 'tb_events': 4, 'weights': [1, 2, 1],
                      'outcomes': {'target': {'scale': scale, 'nuisance_terms': ['Intercept'],
                                              'nuisance_coefficients': [intercept], 'n_training': 4,
                                              'n_observations': 8, 'n_children': 5}}}}


def normal_logp(value, mean, sd):
    return -.5*np.log(2*np.pi)-np.log(sd)-.5*((np.asarray(value)-mean)/sd)**2


class MechanicsContractTests(unittest.TestCase):
    def test_six_fixed_targets_seeds_and_native_reporting_orientation(self):
        self.assertEqual(DEFAULT_TARGETS, ('R', 'C', 'ReE', 'ReI', 'XeE', 'XeI'))
        for i, target in enumerate(DEFAULT_TARGETS):
            spec = fit_spec(target); validation = validate_contract(fabricated_contract(target), target)
            self.assertEqual(spec['sampling']['random_seed'], 2026092612+i)
            self.assertEqual(spec['sampling']['init'], 'jitter+adapt_diag')
            self.assertIsNone(spec['sampling']['initvals'])
            self.assertEqual(validation['weights'], [1, 2, 1])
            self.assertEqual(spec['orientation'], -1. if target in ('C', 'XeE', 'XeI') else 1.)
            self.assertEqual(spec['reporting_multiplier'], 1000. if target == 'C' else 1.)
            json.dumps(spec, allow_nan=False); json.dumps(validation, allow_nan=False)
        self.assertEqual(fit_spec('C')['unit'], 'L/cmH2O')
        self.assertEqual(fit_spec('C')['reporting_unit'], 'mL/cmH2O')
        with self.assertRaises(ValueError): fit_spec('X6')

    def test_orientation_residuals_design_and_child_linkage_are_checked(self):
        def change_frame(p, name, value):
            frame = p['frames']['target']; frame[name] = value
        mutations = [
            lambda p: change_frame(p, 'outcome_value', -p['frames']['target'].outcome_value),
            lambda p: change_frame(p, 'y_std', p['frames']['target'].y_std+.1),
            lambda p: change_frame(p, 'child_index', 99),
            lambda p: change_frame(p, 'pid_child', 'unknown'),
            lambda p: p['audit']['outcomes']['target'].__setitem__('n_training', 3),
            lambda p: p['hazard_design'].__setitem__((0, 0), 1),
            lambda p: p['post_diagnosis_designs']['target'].__setitem__((0, 0), 1),
            lambda p: p['audit'].__setitem__('family', 'primary'),
        ]
        for mutate in mutations:
            contract = fabricated_contract(); mutate(contract)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError): validate_contract(contract, 'R')
        with self.assertRaises(ValueError): validate_contract(fabricated_contract('R'), 'C')

    def test_first_event_boundary_and_exposed_child_weights(self):
        contract = fabricated_contract()
        contract['event']['stop'][0] = 1.
        contract['event']['exposure'][0] = [1., 0., 0., 0., 0., 0.]
        contract['event']['event_interval'][0] = [0, 1, 0, 0, 0, 0]
        self.assertEqual(validate_contract(contract, 'R')['tb_events'], 4)
        contract['event']['event_interval'][0] = [1, 0, 0, 0, 0, 0]
        with self.assertRaises(ValueError): validate_contract(contract, 'R')
        contract = fabricated_contract(); contract['weights'][0] = 2
        with self.assertRaises(ValueError): validate_contract(contract, 'R')
        contract = fabricated_contract(); contract['event']['exposure'][0, 0] = np.nan
        with self.assertRaises(ValueError): validate_contract(contract, 'R')


class MechanicsGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = fabricated_contract()
        cls.model = build_model(cls.contract, 'R')
        cls.values = {name: np.asarray(value, dtype=float) for name, value in {
            'child_z': [-.4, .2, .5, -.1, .3], 'tau_child': 1.4,
            'shift_1to5': .2, 'shift_infancy_additional': .3,
            'shift_late': -.1, 'sigma_outcome': .85,
            'log_baseline_hazard': np.log([.03, .02, .01, .008, .006, .004]),
            'assoc_outcome': .4, 'hazard_male': .2,
        }.items()}
        cls.point = cls.model.initial_point()
        for name, value in cls.values.items():
            key = name if name in cls.point else name+'_log__'
            cls.point[key] = value.copy() if key == name else np.log(value)
        cls.factor_logp = cls.model.compile_logp(sum=False, jacobian=False)

    def test_all_priors_and_joint_likelihood_against_independent_math(self):
        v, p = self.values, self.contract
        rvs = self.model.free_RVs+self.model.observed_RVs+self.model.potentials
        actual = {rv.name: np.sum(value) for rv, value in zip(rvs, self.factor_logp(self.point))}
        expected = {}
        for name, (mean, sd) in fit_spec('R')['normal_priors_mean_sd'].items():
            expected[name] = np.sum(normal_logp(v[name], mean, sd))
        for name, sd in fit_spec('R')['half_normal_prior_sd'].items():
            expected[name] = np.sum(np.log(2.)+normal_logp(v[name], 0., sd))
        expected['log_baseline_hazard'] = np.sum(normal_logp(v['log_baseline_hazard'], np.log(BASELINE_HAZARDS), 1.25))
        f = p['frames']['target']; latent = v['child_z']*v['tau_child']
        mean = latent[f.child_index]+f[list(POST_TERMS)].to_numpy()@np.array([v['shift_1to5'], v['shift_infancy_additional'], v['shift_late']])
        expected['obs_outcome'] = np.sum(normal_logp(f.y_std.to_numpy(), mean, v['sigma_outcome']))
        loghaz = v['log_baseline_hazard'][None, :]+(v['assoc_outcome']*latent+v['hazard_male']*p['children'].male.to_numpy())[:, None]
        expected['first_tb_loglik'] = np.sum(p['event']['event_interval']*loghaz-p['event']['exposure']*np.exp(loghaz))
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            # Scalar prior log constants can be folded in float32 by PyTensor.
            places = 11 if name in ('obs_outcome', 'first_tb_loglik') else 7
            with self.subTest(factor=name): self.assertAlmostEqual(actual[name], expected[name], places=places)

    def test_hazard_uses_scaled_shared_effect_and_gradient_matches(self):
        v, p = self.values, self.contract
        latent = v['child_z']*v['tau_child']
        loghaz = v['log_baseline_hazard'][None, :]+(v['assoc_outcome']*latent+v['hazard_male']*p['children'].male.to_numpy())[:, None]
        expected = -float(v['assoc_outcome'])/.75**2+np.sum((p['event']['event_interval']-p['event']['exposure']*np.exp(loghaz))*latent[:, None])
        fn = self.model.compile_dlogp(vars=[self.model['assoc_outcome']], jacobian=False)
        self.assertAlmostEqual(float(fn(self.point)[0]), expected, places=11)
        deterministic = self.model.replace_rvs_by_values([self.model['hazard_hr_per_between_child_sd'], self.model['shared_outcome']])
        evaluate = self.model.compile_fn(deterministic, inputs=self.model.value_vars, on_unused_input='ignore')
        hazard_hr, shared = evaluate(self.point)
        self.assertAlmostEqual(float(hazard_hr), float(np.exp(.4*1.4)), places=14)
        np.testing.assert_allclose(shared, latent, atol=0., rtol=0.)
        self.assertNotAlmostEqual(float(hazard_hr), float(np.exp(.4)), places=5)

    def test_eight_contrasts_use_child_weights_and_infancy_addition(self):
        expected = [.5, .2, -.1, (.5+2*.2)/3, (.5+2*.2-.1)/4, .3, -.6, -.3]
        outputs = self.model.replace_rvs_by_values([self.model[name] for name in CONTRASTS.values()])
        fn = self.model.compile_fn(outputs, inputs=self.model.value_vars, on_unused_input='ignore')
        np.testing.assert_allclose(fn(self.point), expected, atol=1e-14, rtol=0.)

    def test_positive_scale_transforms_have_exact_log_jacobian(self):
        with_jacobian = float(self.model.compile_logp(jacobian=True)(self.point))
        without_jacobian = float(self.model.compile_logp(jacobian=False)(self.point))
        self.assertAlmostEqual(with_jacobian-without_jacobian, float(np.log(1.4)+np.log(.85)), places=12)

    def test_all_posterior_variables_and_input_nonmutation(self):
        posterior_names = {v.name for v in self.model.free_RVs+self.model.deterministics}
        self.assertEqual(posterior_names, set(POSTERIOR_VARIABLES))
        self.assertEqual(len(posterior_names), 16)
        contract = fabricated_contract(); before = deepcopy(contract)
        model = build_model(contract, 'R')
        pd.testing.assert_frame_equal(contract['frames']['target'], before['frames']['target'])
        pd.testing.assert_frame_equal(contract['children'], before['children'])
        np.testing.assert_array_equal(contract['event']['exposure'], before['event']['exposure'])
        self.assertEqual(tuple(model.coords['child']), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(model.coords['hazard_interval']), (0, 1, 2, 3, 4, 5))


if __name__ == '__main__':
    unittest.main()
