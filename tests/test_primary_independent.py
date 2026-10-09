"""Independent density, Jacobian and gradient checks on fabricated contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.models.spirometry import build_model
from test_spirometry import fabricated_contract


def independent_log_density(point, contract):
    """Probability math in value-variable space, independent of graph metadata."""
    positive = ('tau_lci', 'loading_mechanics', 'sigma_mechanics', 'sigma_lci', 'sigma_ratio')
    values = {key: np.asarray(value, float) for key, value in point.items() if not key.endswith('_log__')}
    for key in positive:
        values[key] = np.exp(point[key + '_log__'])
    def normal(x, mu, sd):
        return np.sum(-np.log(sd)-.5*np.log(2*np.pi)-.5*((x-mu)/sd)**2)
    total = normal(values['mechanics'], 0., 1.) + normal(values['lci_raw'], 0., 1.)
    total += normal(values['loading_ratio_mechanics'], 0., 1.)
    for key in ('shift_mechanics', 'shift_mechanics_infancy', 'shift_mechanics_late',
                'shift_lci', 'shift_lci_infancy', 'shift_lci_late',
                'direct_ratio', 'direct_ratio_infancy', 'direct_ratio_late',
                'assoc_mechanics', 'hazard_male'):
        total += normal(values[key], 0., .75)
    for key in positive:
        total += normal(values[key], 0., 1.) + values[key].size*np.log(2.)
        total += np.sum(point[key + '_log__'])  # log-transform Jacobian
    total += normal(values['log_baseline_hazard'], np.log([.045, .018, .006, .006, .006, .006]), 1.25)
    m_shift = np.array([values['shift_mechanics'], values['shift_mechanics_infancy'], values['shift_mechanics_late']])
    l_shift = np.array([values['shift_lci'], values['shift_lci_infancy'], values['shift_lci_late']])
    direct = np.array([values['direct_ratio'], values['direct_ratio_infancy'], values['direct_ratio_late']])
    for j, key in enumerate(('resistance', 'compliance', 'xee', 'ree', 'lci', 'target')):
        frame = contract['frames'][key]
        child = frame.child_index.to_numpy(int)
        terms = frame[['post_before5', 'post_infancy', 'post_at_after5']].to_numpy(float)
        if j < 4:
            mu = values['loading_mechanics'][j]*(values['mechanics'][child]+terms @ m_shift)
            sd = values['sigma_mechanics'][j]
        elif key == 'lci':
            mu = values['tau_lci']*values['lci_raw'][child]+terms @ l_shift
            sd = values['sigma_lci']
        else:
            mu = values['loading_ratio_mechanics']*(values['mechanics'][child]+terms @ m_shift)+terms @ direct
            sd = values['sigma_ratio']
        total += normal(frame.y_std.to_numpy(float), mu, sd)
    log_h = (values['log_baseline_hazard'][None, :]
             + values['assoc_mechanics']*values['mechanics'][:, None]
             + values['hazard_male']*contract['children'].male.to_numpy(float)[:, None])
    total += np.sum(contract['event']['event_interval']*log_h-contract['event']['exposure']*np.exp(log_h))
    return float(total)


def perturbed_point(model, seed):
    rng = np.random.default_rng(seed)
    return {key: np.asarray(value)+rng.normal(0, .23, np.shape(value))
            for key, value in model.initial_point().items()}


class IndependentPrimaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = fabricated_contract()
        cls.model = build_model(cls.contract, 'ratio_pre')
        cls.logp = cls.model.compile_logp(jacobian=True)
        cls.gradient = cls.model.compile_dlogp(jacobian=True)

    def test_transformed_density_includes_every_positive_parameter_jacobian(self):
        for seed in (31, 59, 97):
            point = perturbed_point(self.model, seed)
            # Scalar-prior log constants may be folded at lower precision by
            # PyTensor; this tolerance is far below a missing Jacobian term.
            self.assertAlmostEqual(self.logp(point), independent_log_density(point, self.contract), delta=1e-6)
        point = perturbed_point(self.model, 31)
        no_jacobian = self.model.compile_logp(jacobian=False)(point)
        expected_jacobian = sum(np.sum(value) for key, value in point.items() if key.endswith('_log__'))
        self.assertAlmostEqual(float(self.logp(point)-no_jacobian), expected_jacobian, delta=1e-10)

    def test_every_gradient_coordinate_matches_independent_finite_differences(self):
        point = perturbed_point(self.model, 73)
        numerical = []
        step = 1e-5
        for variable in self.model.value_vars:
            key = variable.name
            shape = point[key].shape
            for flat_index in range(point[key].size):
                upper, lower = deepcopy(point), deepcopy(point)
                up = upper[key].reshape(-1); down = lower[key].reshape(-1)
                up[flat_index] += step; down[flat_index] -= step
                upper[key] = up.reshape(shape); lower[key] = down.reshape(shape)
                numerical.append((independent_log_density(upper, self.contract)-independent_log_density(lower, self.contract))/(2*step))
        np.testing.assert_allclose(self.gradient(point), numerical, rtol=1e-7, atol=1e-7)

    def test_all_lci_parameters_leave_hazard_and_target_likelihood_unchanged(self):
        point = perturbed_point(self.model, 19)
        changed = deepcopy(point)
        for key in ('lci_raw', 'tau_lci_log__', 'sigma_lci_log__', 'shift_lci', 'shift_lci_infancy', 'shift_lci_late'):
            changed[key] = changed[key]+.37
        variables = self.model.free_RVs + self.model.observed_RVs + self.model.potentials
        fn = self.model.compile_logp(sum=False, jacobian=False)
        before = {v.name: np.sum(x) for v, x in zip(variables, fn(point))}
        after = {v.name: np.sum(x) for v, x in zip(variables, fn(changed))}
        for key in ('obs_fev1_fvc', 'first_tb_loglik'):
            self.assertEqual(before[key], after[key])
        self.assertNotEqual(before['obs_lci'], after['obs_lci'])


if __name__ == '__main__':
    unittest.main()
