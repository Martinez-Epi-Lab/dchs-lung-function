"""Fabricated-data checks for the selected sensitivity graphs and contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.preparation import joint_inputs
from dchs_lung_function.models import sensitivity, spirometry
from dchs_lung_function.sensitivity_sampling import resolved_sampling, validate_sampling, expected_posterior_shapes


def fabricated_pft():
    """Invented repeated measures with independent baseline covariate patterns."""
    rng = np.random.default_rng(120812)
    rows = []
    ages = [.4, .8, 1., 2., 4., 5., 6., np.nan] + [np.nan] * 16
    for child, tb_age in enumerate(ages):
        dob = pd.Timestamp('2000-01-01') + pd.Timedelta(days=child * 2)
        dx = dob + pd.Timedelta(days=round(tb_age * 365.25)) if np.isfinite(tb_age) else pd.NaT
        for visit, age in enumerate([.15, .6, 1.5, 4., 6., 9.]):
            date = dob + pd.Timedelta(days=round(age * 365.25))
            noise = rng.normal()
            rows.append({'pid_child': f'invented_{child:02}', 'child_dob': dob,
                'male': float(child % 2), 'maternal_hiv': float((child // 2) % 2),
                'maternal_smoking': float((child // 4) % 2),
                'birthweight_z': (child % 5 - 2) / 1.4, 'ever_lrti': float((child // 3) % 2),
                'tb_first_dx': dx, 'tb_age_first_years': tb_age, 'exit_date': pd.NaT,
                'visit_label': f'visit_{visit}', 'height': 55 + age * 7 + child * .3 + rng.normal(),
                'date_of_test': date, 'date_of_test_ilf1': date, 'date_of_test_ilf2': date,
                'fev1_fvc': .88 - .002 * age + .01 * noise,
                'fev1_fvc_post': .91 - .001 * age + .008 * noise,
                'i_lci_2_5': 7. + .3 * noise, 'resistance': 9. - .3 * age + .8 * noise,
                'compliance': .01 + .001 * age + .002 * noise,
                'xee': -3. + .03 * age + .3 * noise, 'ree': 8. - .3 * age + .7 * noise})
    return pd.DataFrame(rows)


def fabricated_contract(target='early_core'):
    spec = sensitivity.fit_spec(target)
    pft = fabricated_pft()
    return joint_inputs(pft, (spec['column'], 100.),
        family='primary' if spec['family'] == 'window' else 'early',
        covariates=spec['covariates'], event_end='2007-01-01',
        observation_end='2007-01-01' if spec['observation_window'] == 'restricted' else None)


def replace_column(contract, column, value):
    frame = contract['frames']['target']
    frame[column] = value


def change_hazard(contract):
    matrix = contract['hazard_design'].copy()
    matrix[0, 1] = 1.
    contract['hazard_design'] = matrix


def assigned_point(model):
    values = {}
    for name, initial in model.initial_point().items():
        raw = name[:-6] if name.endswith('_log__') else name
        shape = np.shape(initial)
        if name.endswith('_log__'):
            values[raw] = np.full(shape, .8)
        elif name == 'log_baseline_hazard':
            values[raw] = np.log(np.linspace(.02, .04, np.size(initial))).reshape(shape)
        elif shape:
            values[raw] = np.linspace(-.3, .3, np.size(initial)).reshape(shape)
        else:
            values[raw] = np.asarray(.17)
    for name, value in {'loading_ratio_mechanics': .6, 'shift_mechanics': -.12,
                        'shift_mechanics_infancy': .09, 'direct_ratio': -.2,
                        'direct_ratio_infancy': .13, 'tau_lci': 1.2,
                        'assoc_mechanics': .35, 'hazard_male': -.15}.items():
        values[name] = np.asarray(value)
    point = {name: np.log(values[name[:-6]]) if name.endswith('_log__') else values[name]
             for name in model.initial_point()}
    return values, point


def normal_logp(value, mean, sd):
    return -.5 * np.log(2 * np.pi) - np.log(sd) - .5 * ((np.asarray(value) - mean) / sd) ** 2


def independent_factors(contract, target, values):
    """Direct NumPy density; no PyMC operations or production graph helpers."""
    spec = sensitivity.fit_spec(target)
    result = {}
    for name, (mean, sd) in spec['normal_priors_mean_sd'].items():
        result[name] = float(np.sum(normal_logp(values[name], mean, sd)))
    for name, sd in spec['half_normal_prior_sd'].items():
        result[name] = float(np.sum(normal_logp(values[name], 0., sd) + np.log(2.)))
    prior = spec['log_baseline_hazard_prior']
    result['log_baseline_hazard'] = float(np.sum(normal_logp(values['log_baseline_hazard'], prior['mean'], prior['sd'])))
    for j, key in enumerate(spec['mechanics_indicators']):
        frame = contract['frames'][key]
        current = values['mechanics'][frame.child_index] + values['shift_mechanics'] * frame.post_before5 + values['shift_mechanics_infancy'] * frame.post_infancy
        result['obs_' + key] = float(np.sum(normal_logp(frame.y_std, values['loading_mechanics'][j] * current, values['sigma_mechanics'][j])))
    frame = contract['frames']['lci']
    mu = values['tau_lci'] * values['lci_raw'][frame.child_index] + values['shift_lci'] * frame.post_before5 + values['shift_lci_infancy'] * frame.post_infancy
    result['obs_lci'] = float(np.sum(normal_logp(frame.y_std, mu, values['sigma_lci'])))
    frame = contract['frames']['target']
    current = values['mechanics'][frame.child_index] + values['shift_mechanics'] * frame.post_before5 + values['shift_mechanics_infancy'] * frame.post_infancy
    mu = values['loading_ratio_mechanics'] * current + values['tau_target'] * values['target_raw'][frame.child_index] + values['direct_ratio'] * frame.post_before5 + values['direct_ratio_infancy'] * frame.post_infancy
    result['obs_fev1_fvc'] = float(np.sum(normal_logp(frame.y_std, mu, values['sigma_ratio'])))
    eta = values['assoc_mechanics'] * values['mechanics'] + values['hazard_male'] * contract['children'].male.to_numpy()
    for variable, column in spec['hazard_variable_columns'].items():
        eta = eta + values[variable] * contract['children'][column].to_numpy()
    loghazard = values['log_baseline_hazard'][None, :] + eta[:, None]
    result['first_tb_loglik'] = float(np.sum(contract['event']['event_interval'] * loghazard - contract['event']['exposure'] * np.exp(loghazard)))
    return result


class SensitivityContractTests(unittest.TestCase):
    def test_all_selected_contracts_counts_inventory_and_metadata(self):
        self.assertEqual(len(sensitivity.DEFAULT_TARGETS), 10)
        for target in sensitivity.DEFAULT_TARGETS:
            with self.subTest(target=target):
                spec = sensitivity.fit_spec(target)
                contract = fabricated_contract(target)
                valid = sensitivity.validate_contract(contract, target)
                self.assertEqual(spec['contract_key'], sensitivity.contract_key(target))
                self.assertEqual(spec['artifact_path'], ['M03', 'pft', 'samples', spec['contract_key']])
                self.assertEqual(valid['joint_children'], 24)
                self.assertEqual(valid['tb_events'], 7 if spec['family'] == 'window' else 5)
                self.assertEqual(valid['weights'], [2, 3, 2] if spec['family'] == 'window' else [2, 3, 0])
                json.dumps(spec, allow_nan=False); json.dumps(valid, allow_nan=False)
        full = sensitivity.validate_contract(fabricated_contract(), 'early_core')
        restricted = sensitivity.validate_contract(fabricated_contract('early_core_window'), 'early_core_window')
        self.assertLess(restricted['process_counts']['target']['n_observations'], full['process_counts']['target']['n_observations'])
        with self.assertRaises(ValueError): sensitivity.fit_spec('early_post')
        spec['covariates'].append('invalid')
        self.assertEqual(sensitivity.fit_spec('window_ratio_post')['covariates'], [])

    def test_covariate_complete_case_population_and_exploratory_hazard_exclusion(self):
        frame = fabricated_pft()
        frame.loc[frame.pid_child.eq('invented_23'), 'birthweight_z'] = np.nan
        core = joint_inputs(frame, ('fev1_fvc', 100.), family='early', event_end='2007-01-01')
        adjusted = joint_inputs(frame, ('fev1_fvc', 100.), family='early', event_end='2007-01-01', covariates=('maternal_hiv', 'maternal_smoking', 'birthweight_z', 'ever_lrti'))
        self.assertEqual(sensitivity.validate_contract(core, 'early_core')['joint_children'], 24)
        self.assertEqual(sensitivity.validate_contract(adjusted, 'early_all_four')['joint_children'], 23)
        spec = sensitivity.fit_spec('early_all_four')
        self.assertIn('ever_lrti', spec['nuisance_formula'])
        self.assertNotIn('ever_lrti', spec['hazard_covariates'])
        self.assertNotIn('hazard_ever_lrti', spec['posterior_variables'])
        self.assertIn('hazard_prenatal_smoking', sensitivity.fit_spec('early_maternal_smoking')['posterior_variables'])
        self.assertIn('hazard_maternal_smoking', spec['posterior_variables'])

    def test_adversarial_linkage_hazard_covariates_and_outcomes_rejected(self):
        mutations = [
            lambda c: c.update(children=c['children'].assign(pid_child='')),
            lambda c: replace_column(c, 'child_index', 999),
            lambda c: replace_column(c, 'maternal_hiv', 0.),
            change_hazard,
            lambda c: c['hazard_terms'].append('ever_lrti'),
            lambda c: c['event']['breaks'].__setitem__(1, .9),
            lambda c: c['event']['exposure'].__setitem__((0, 0), 0.),
            lambda c: c['event']['event_interval'].__setitem__((0, 0), 0.),
            lambda c: replace_column(c, 'outcome_value', 100.),
            lambda c: replace_column(c, 'y_std', np.nan),
            lambda c: c['weights'].__setitem__(0, 99),
            lambda c: c['audit'].__setitem__('covariates', []),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                contract = fabricated_contract('early_maternal_hiv'); mutate(contract)
                with self.assertRaises(ValueError): sensitivity.validate_contract(contract, 'early_maternal_hiv')

    def test_nuisance_training_covariate_design_and_scale_are_verified(self):
        mutations = [
            lambda c: c['designs']['target'].iloc.__setitem__((0, 0), 2.),
            lambda c: c['audit']['outcomes']['target']['nuisance_coefficients'].__setitem__(0, 999.),
            lambda c: c['audit']['outcomes']['target'].__setitem__('scale', 2.),
            lambda c: replace_column(c, 'height_z', 0.),
            lambda c: c['audit']['outcomes']['target'].__setitem__('n_training', 1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                contract = fabricated_contract('early_all_baseline'); mutate(contract)
                with self.assertRaises(ValueError): sensitivity.validate_contract(contract, 'early_all_baseline')

    def test_exact_dates_nested_infancy_aliases_and_late_exclusion(self):
        for kind in ['post_date', 'infancy', 'alias', 'late', 'family']:
            with self.subTest(kind=kind):
                contract = fabricated_contract(); frame = contract['frames']['target']
                if kind == 'post_date': frame.loc[frame.post_any.eq(1), 'test_date'] = pd.Timestamp('1999-01-01')
                if kind == 'infancy': frame['post_infancy'] = 0.
                if kind == 'alias': frame['post_tb_lt5'] = 0.
                if kind == 'late': frame.loc[frame.post_any.eq(1), 'tb_age_first_years'] = 5.
                if kind == 'family': contract['audit']['family'] = 'primary'
                with self.assertRaises(ValueError): sensitivity.validate_contract(contract, 'early_core')
        early = fabricated_contract()
        for frame in early['frames'].values():
            self.assertFalse((frame.post_any.eq(1) & frame.tb_age_first_years.ge(5)).any())
        all_age = fabricated_contract('window_ratio_pre')
        self.assertGreater(all_age['frames']['target'].post_at_after5.sum(), 0)

    def test_float32_native_multiplication_is_preserved(self):
        frame = fabricated_pft()
        frame['fev1_fvc'] = frame.fev1_fvc.astype('float32')
        contract = joint_inputs(frame, ('fev1_fvc', 100.), family='early', event_end='2007-01-01')
        sensitivity.validate_contract(contract, 'early_core')
        target = contract['frames']['target']
        self.assertGreater(np.max(np.abs(target.outcome_value.to_numpy(float) - target.fev1_fvc.to_numpy(float) * 100.)), 0.)

    def test_recorded_fresh_sampler_controls_and_invalid_settings(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'config/sensitivity_sampling.json').read_text())
        self.assertEqual(validate_sampling(config), config)
        for target in sensitivity.DEFAULT_TARGETS:
            spec = sensitivity.fit_spec(target); controls = resolved_sampling(target, config)
            self.assertEqual(controls['draws'], config.get('draws_by_target', {}).get(target, config['draws']))
            self.assertEqual(controls['tune'], config['tune'])
            self.assertEqual(controls['random_seed'], spec['sampling']['random_seed'] + 104729)
            self.assertEqual(controls['init'], 'adapt_diag')
            self.assertEqual([v['loading_ratio_mechanics'] for v in controls['initvals']], [.25, .35, .45, .55])
        for key, value in [('draws', 1600), ('tune', 1800), ('target_accept', .94), ('chains', 3)]:
            with self.subTest(key=key), self.assertRaises(ValueError): validate_sampling({**config, key:value})


class SensitivityGraphTests(unittest.TestCase):
    def test_complete_posterior_shapes_and_early_hazard_inventory(self):
        for target in sensitivity.DEFAULT_TARGETS:
            contract = fabricated_contract(target); spec = sensitivity.fit_spec(target)
            shapes = expected_posterior_shapes(contract, target, 4, 7)
            self.assertEqual(set(shapes), set(spec['posterior_variables']))
            self.assertEqual(shapes['mechanics'], (4, 7, 24))
            self.assertEqual(shapes['lci_process'], (4, 7, 24))
            self.assertEqual(shapes['log_baseline_hazard'], (4, 7, 6 if spec['family'] == 'window' else 3))
            if spec['family'] == 'early': self.assertNotIn('direct_ratio_late', shapes)

    def test_all_early_factors_against_independent_density_and_covariate_gradient(self):
        for target in ['early_core', 'early_maternal_smoking', 'early_all_four']:
            with self.subTest(target=target):
                contract = fabricated_contract(target); model = sensitivity.build_model(contract, target)
                values, point = assigned_point(model)
                names = [v.name for v in [*model.free_RVs, *model.observed_RVs, *model.potentials]]
                factors = model.compile_logp(sum=False, jacobian=False)(point)
                expected = independent_factors(contract, target, values)
                for name, factor in zip(names, factors):
                    observed = float(np.sum(factor))
                    # PyMC constant-folds log(.75) as float32. Account only for
                    # that parameter-independent prior normalizer in the oracle.
                    if name in sensitivity.fit_spec(target)['normal_priors_mean_sd'] and sensitivity.fit_spec(target)['normal_priors_mean_sd'][name][1] == .75:
                        expected[name] += np.size(values[name]) * (np.log(.75) - float(np.log(np.float32(.75))))
                    if name == 'log_baseline_hazard':
                        expected[name] += np.size(values[name]) * (np.log(1.25) - float(np.log(np.float32(1.25))))
                    self.assertAlmostEqual(observed, expected[name], places=9, msg=name)
                gradient = model.compile_dlogp()(point)
                start = 0; gradient_by_name = {}
                for name, value in point.items():
                    width = np.size(value); gradient_by_name[name] = gradient[start:start + width]; start += width
                eta = values['assoc_mechanics'] * values['mechanics'] + values['hazard_male'] * contract['children'].male.to_numpy()
                for variable, column in sensitivity.fit_spec(target)['hazard_variable_columns'].items(): eta += values[variable] * contract['children'][column].to_numpy()
                residual = contract['event']['event'] - np.sum(contract['event']['exposure'] * np.exp(values['log_baseline_hazard'][None, :] + eta[:, None]), axis=1)
                for variable, column in {'hazard_male':'male', **sensitivity.fit_spec(target)['hazard_variable_columns']}.items():
                    expected_derivative = -float(values[variable]) / .75**2 + np.dot(contract['children'][column].to_numpy(), residual)
                    np.testing.assert_allclose(gradient_by_name[variable], expected_derivative, rtol=1e-11, atol=1e-11)

    def test_contrasts_use_distinct_child_weights_and_no_lci_loading(self):
        contract = fabricated_contract('early_all_baseline'); model = sensitivity.build_model(contract, 'early_all_baseline')
        values, point = assigned_point(model)
        names = [entry[0] for entry in sensitivity.EARLY_CONTRASTS.values()]
        compiled = model.compile_fn(model.replace_rvs_by_values([model[n] for n in names]), inputs=model.value_vars, on_unused_input='ignore')
        middle = -.2 + .6 * -.12
        infancy = -.2 + .13 + .6 * (-.12 + .09)
        expected = [(2 * infancy + 3 * middle) / 5, infancy, middle, infancy - middle, -.2, .13]
        np.testing.assert_allclose(compiled(point), expected, atol=1e-14, rtol=1e-14)
        changed = deepcopy(point); changed['lci_raw'] += 1.2; changed['shift_lci'] = np.asarray(4.2)
        np.testing.assert_array_equal(compiled(changed), compiled(point))
        named = [model['first_tb_loglik'], model['obs_fev1_fvc']]
        factors = model.compile_fn(model.logp(vars=named, sum=False, jacobian=False), inputs=model.value_vars, on_unused_input='ignore')
        for left, right in zip(factors(point), factors(changed)): np.testing.assert_array_equal(left, right)

    def test_window_builder_delegates_exact_primary_graph(self):
        for target in ['window_ratio_pre', 'window_ratio_post']:
            contract = fabricated_contract(target)
            new = sensitivity.build_model(contract, target)
            original = spirometry.build_model(contract, sensitivity.fit_spec(target)['source_target'])
            _, point = assigned_point(new)
            self.assertEqual([v.name for v in new.value_vars], [v.name for v in original.value_vars])
            self.assertEqual(float(new.compile_logp()(point)), float(original.compile_logp()(point)))
            np.testing.assert_array_equal(new.compile_dlogp()(point), original.compile_dlogp()(point))


if __name__ == '__main__':
    unittest.main()
