"""Fabricated-data checks for the selected three-process wheeze graph."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.respiratory_preparation import Sources, prepare_scenario
from dchs_lung_function.models import wheeze
from dchs_lung_function.wheeze_sampling import DEFAULT_SAMPLING, expected_posterior_shapes, resolved_sampling, validate_sampling


def fabricated_contract(target='primary_sex'):
    """Small full prepared contract; IDs are explicitly fabricated labels."""
    children, scheduled, contacts = [], [], []
    for i in range(12):
        dob = pd.Timestamp('2000-01-01')+pd.Timedelta(days=i*9)
        censor = dob+pd.DateOffset(years=10, months=6)
        tb_days = [120, 300, 365, 730, 1100, 2200, None, None, None, None, None, None][i]
        tb = dob+pd.Timedelta(days=tb_days) if tb_days is not None else pd.NaT
        child = {'pid_child': f'fabricated-{i:02}', 'dob': dob, 'censor_date': censor,
                 'censor_age': (censor-dob).days/365.25, 'tb_date': tb,
                 'tb_event': int(tb_days is not None), 'tb_age': tb_days/365.25 if tb_days is not None else np.nan,
                 'male': float(i%2), 'prenatal_smoking': float((i//2)%2),
                 'birthweight_z': (i-5.5)/4 if i != 11 else np.nan}
        children.append(child)
        days = [30, 180, 365, 730, 1100, 1460, 1830, 2560, 3290, 3700]
        if tb_days is not None and tb_days not in days:
            days.append(tb_days)
        for j, day in enumerate(sorted(days)):
            scheduled.append({**child, 'date': dob+pd.Timedelta(days=day), 'age': day/365.25,
                              'visit_category': 1., 'wheeze': int((i+j)%6 == 0)})
        for day in [80+i, 100+i, 120+i, 800+i, 835+i, 2000+i]:
            contacts.append({'pid_child': child['pid_child'], 'date': dob+pd.Timedelta(days=day),
                             'age': day/365.25, 'episode_source': 'dedicated wheeze visit'})
    sources = Sources(pd.DataFrame(children), pd.DataFrame(scheduled), pd.DataFrame(contacts),
                      {'administrative_end': '2012-01-01'})
    return asdict(prepare_scenario(sources, target))


def assigned_point(model):
    rng = np.random.default_rng(90211)
    point = {key: np.asarray(value).copy() + rng.normal(0., .18, np.shape(value))
             for key, value in model.initial_point().items()}
    point['sigma_scheduled_log__'] = np.asarray(np.log(.8))
    point['sigma_episode_log__'] = np.asarray(np.log(1.1))
    point['scheduled_intercept'] = np.asarray(-2.3)
    point['episode_log_baseline_rate'] = np.log(np.linspace(.03, .12, 8))
    point['tb_log_baseline_hazard'] = np.log(np.linspace(.008, .03, 8))
    values = {key[:-6] if key.endswith('_log__') else key:
              np.exp(value) if key.endswith('_log__') else value for key, value in point.items()}
    return values, point


def normal_log(value, mean, sd):
    return -.5*np.log(2*np.pi)-np.log(sd)-.5*((value-mean)/sd)**2


def independent_factors(contract, target, values):
    v = values
    terms = {}
    for name, prior in wheeze.fit_spec(target)['priors'].items():
        if prior[0] == 'Normal':
            terms[name] = float(normal_log(v[name], prior[1], prior[2]).sum())
        else:
            terms[name] = float((np.log(2)+normal_log(v[name], 0., prior[1])).sum())
    c, s, e = contract['children'], contract['scheduled'], contract['episode_intervals']
    covariates = contract['covariates']
    xc, xs = c[covariates].to_numpy(float), s[covariates].to_numpy(float)
    si, ei = s.child_index.to_numpy(int), e.child_index.to_numpy(int)
    sb, eb = s.effect_band.to_numpy(int), e.effect_band.to_numpy(int)
    tb = c.tb_age.to_numpy(float)
    sinf, einf = s.post_tb.to_numpy()*(tb[si] < 1), e.post_tb.to_numpy()*(tb[ei] < 1)
    sother, eother = s.post_tb.to_numpy()*(tb[si] >= 1), e.post_tb.to_numpy()*(tb[ei] >= 1)
    seta = (v['scheduled_intercept'] + contract['spline']@v['scheduled_age']
            + xs@v['scheduled_covariate'] + v['sigma_scheduled']*v['latent_propensity'][si]
            + v['scheduled_infancy_post_log_or'][sb]*sinf + v['scheduled_other_post_log_or'][sb]*sother)
    eeta = (v['episode_log_baseline_rate'][e.base_band.to_numpy(int)] + xc[ei]@v['episode_covariate']
            + v['sigma_episode']*v['latent_propensity'][ei]
            + v['episode_infancy_post_log_rr'][eb]*einf + v['episode_other_post_log_rr'][eb]*eother)
    mu = e.person_years.to_numpy(float)*np.exp(eeta)
    logh = v['tb_log_baseline_hazard'][None, :] + (xc@v['tb_covariate']+v['tb_latent_association']*v['latent_propensity'])[:, None]
    terms['scheduled_wheeze_observed'] = float(np.sum(s.wheeze.to_numpy()*seta-np.logaddexp(0., seta)))
    terms['episode_count_observed'] = float(np.sum(e.events.to_numpy()*np.log(mu)-mu-gammaln(e.events.to_numpy()+1)))
    terms['first_tb_loglikelihood'] = float(np.sum(contract['survival']['event_interval']*logh-contract['survival']['exposure']*np.exp(logh)))
    return terms, seta, mu, logh


class WheezeContractTests(unittest.TestCase):
    def test_both_scenarios_counts_and_complete_posterior_inventory(self):
        for target, children, episodes in [('primary_sex', 12, 48), ('secondary_sex_smoking_birthweight', 11, 44)]:
            c = fabricated_contract(target)
            info = wheeze.validate_contract(c, target)
            self.assertEqual(info['joint_children'], children)
            self.assertEqual(info['counts']['recurrent_episodes'], episodes)
            shapes = expected_posterior_shapes(c, target, 4, 3200)
            self.assertEqual(len(shapes), 22)
            self.assertEqual(shapes['latent_propensity'], (4, 3200, children))
            self.assertEqual(shapes['scheduled_infancy_post_or'], (4, 3200, 4))
            self.assertEqual(shapes['scheduled_other_post_or'], (4, 3200, 4))
            self.assertEqual(shapes['episode_other_post_rr'], (4, 3200, 4))
            self.assertEqual(shapes['tb_log_baseline_hazard'], (4, 3200, 8))

    def test_adjustment_keeps_original_birthweight_scaling(self):
        primary, adjusted = fabricated_contract(), fabricated_contract('secondary_sex_smoking_birthweight')
        original = primary['children'].set_index('pid_child').birthweight_z
        actual = adjusted['children'].set_index('pid_child').birthweight_z
        pd.testing.assert_series_equal(actual, original.loc[actual.index])
        self.assertNotAlmostEqual(actual.mean(), 0.)
        self.assertEqual(adjusted['audit']['children_excluded_for_covariate_missingness'], 1)

    def test_input_dates_exposure_covariates_and_arrays_cannot_be_tampered(self):
        examples = []
        c = fabricated_contract(); c['episodes'].loc[0, 'date'] += pd.Timedelta(days=1); examples.append(c)
        c = fabricated_contract(); c['episodes'].loc[0, 'age'] += 1/365.25; examples.append(c)
        c = fabricated_contract(); c['scheduled'].loc[0, 'male'] = 1-c['scheduled'].loc[0, 'male']; examples.append(c)
        c = fabricated_contract(); c['episode_intervals'].loc[0, 'events'] += 1; examples.append(c)
        c = fabricated_contract(); c['survival']['exposure'][0, 0] += .01; examples.append(c)
        c = fabricated_contract(); c['spline'][0, 0] += .01; examples.append(c)
        c = fabricated_contract(); c['scheduled'].loc[0, 'child_index'] = 1; examples.append(c)
        c = fabricated_contract(); c['episode_intervals'].loc[0, 'post_tb'] = 1; examples.append(c)
        for i, c in enumerate(examples):
            with self.subTest(perturbation=i), self.assertRaises(ValueError):
                wheeze.validate_contract(c, 'primary_sex')

    def test_same_day_scheduled_exposure_and_all_four_bands(self):
        c = fabricated_contract()
        s = c['scheduled']
        same_day = s.date.eq(s.tb_date)
        self.assertGreater(int(same_day.sum()), 0)
        self.assertTrue(s.loc[same_day, 'post_tb'].eq(1).all())
        self.assertEqual(set(s.effect_band), {0, 1, 2, 3})
        self.assertEqual(set(c['episode_intervals'].effect_band), {0, 1, 2, 3})
        s.loc[same_day, 'post_tb'] = 0
        with self.assertRaisesRegex(ValueError, 'same-day'):
            wheeze.validate_contract(c, 'primary_sex')

    def test_calendar_censor_person_time_preserved(self):
        c = fabricated_contract()
        censor = c['children'].censor_age.to_numpy()
        self.assertTrue((censor > 10.5).any())
        self.assertAlmostEqual(c['episode_intervals'].person_years.sum(), censor.sum(), places=12)
        self.assertAlmostEqual(c['survival']['full_exposure'].sum(), np.minimum(censor, 10.5).sum(), places=12)
        wheeze.validate_contract(c, 'primary_sex')

    def test_sampling_protocol_source_seed_offset_and_validation(self):
        for target in wheeze.DEFAULT_TARGETS:
            controls = resolved_sampling(target, DEFAULT_SAMPLING)
            self.assertEqual(controls['draws'], 3200)
            self.assertEqual(controls['tune'], 2200)
            self.assertEqual(controls['random_seed'], wheeze.TARGETS[target]['source_seed']+104729)
            self.assertEqual(controls['init'], 'jitter+adapt_diag')
            self.assertIsNone(controls['initvals'])
            self.assertEqual(controls['chains'], 4)
        for key, value in [('draws', 1599), ('tune', 2199), ('target_accept', .94), ('chains', 3), ('draws', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_sampling({**DEFAULT_SAMPLING, key: value})


class WheezeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {}
        for target in wheeze.DEFAULT_TARGETS:
            c = fabricated_contract(target)
            model = wheeze.build_model(c, target)
            values, point = assigned_point(model)
            cls.examples[target] = c, model, values, point

    def test_all_priors_and_likelihood_factors_against_numpy(self):
        for target, (c, m, v, p) in self.examples.items():
            actual = {rv.name: float(np.sum(value)) for rv, value in
                      zip(m.free_RVs+m.observed_RVs+m.potentials, m.compile_logp(sum=False, jacobian=False)(p))}
            expected, _, _, _ = independent_factors(c, target, v)
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                with self.subTest(target=target, factor=name):
                    # PyMC constant-folds scalar prior log(SD) in float32.
                    # Account explicitly for that parameter-independent
                    # normalizer; do not relax the equation tolerance.
                    prior = wheeze.fit_spec(target)['priors'].get(name)
                    correction = 0.
                    if prior is not None:
                        sd = prior[-1]
                        correction = np.size(v[name])*(np.log(sd)-float(np.log(np.float32(sd))))
                    self.assertAlmostEqual(actual[name], expected[name]+correction, places=9)

    def test_all_three_processes_enter_shared_child_gradient(self):
        for target, (c, m, v, p) in self.examples.items():
            _, seta, mu, logh = independent_factors(c, target, v)
            s, e = c['scheduled'], c['episode_intervals']
            hazard_residual = c['survival']['event_interval']-c['survival']['exposure']*np.exp(logh)
            expected = -v['latent_propensity'].copy()
            np.add.at(expected, s.child_index, v['sigma_scheduled']*(s.wheeze.to_numpy()-expit(seta)))
            np.add.at(expected, e.child_index, v['sigma_episode']*(e.events.to_numpy()-mu))
            expected += v['tb_latent_association']*hazard_residual.sum(axis=1)
            actual = m.compile_dlogp(vars=[m['latent_propensity']], jacobian=False)(p)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
            association = -v['tb_latent_association']/.75**2 + np.sum(hazard_residual*v['latent_propensity'][:, None])
            self.assertAlmostEqual(m.compile_dlogp(vars=[m['tb_latent_association']], jacobian=False)(p)[0], association, places=10)

    def test_positive_scale_jacobians_and_exponentiated_deterministics(self):
        pairs = {'scheduled_infancy_post_or': 'scheduled_infancy_post_log_or',
                 'scheduled_other_post_or': 'scheduled_other_post_log_or',
                 'episode_infancy_post_rr': 'episode_infancy_post_log_rr',
                 'episode_other_post_rr': 'episode_other_post_log_rr',
                 'tb_propensity_hr': 'tb_latent_association',
                 'scheduled_or_per_propensity_sd': 'sigma_scheduled',
                 'episode_rr_per_propensity_sd': 'sigma_episode'}
        for target, (c, m, v, p) in self.examples.items():
            diff = m.compile_logp(jacobian=True)(p)-m.compile_logp(jacobian=False)(p)
            self.assertAlmostEqual(diff, np.log(v['sigma_scheduled'])+np.log(v['sigma_episode']), places=10)
            values = m.compile_fn(m.replace_rvs_by_values(m.deterministics), inputs=m.value_vars, on_unused_input='ignore')(p)
            for rv, value in zip(m.deterministics, values):
                np.testing.assert_allclose(value, np.exp(v[pairs[rv.name]]), rtol=1e-14, atol=1e-14)

    def test_nuisance_age_bands_and_complete_inventory_are_retained(self):
        for target, (c, m, v, p) in self.examples.items():
            names = {rv.name for rv in m.free_RVs+m.deterministics}
            self.assertEqual(names, set(wheeze.POSTERIOR_VARIABLES))
            self.assertEqual(len(m.coords['effect_age_band']), 4)
            self.assertEqual(len(m.coords['baseline_age_interval']), 8)
            before = deepcopy(c)
            wheeze.process_arrays(c, target)
            for name in ('children', 'scheduled', 'episodes', 'episode_intervals'):
                pd.testing.assert_frame_equal(c[name], before[name])
            np.testing.assert_array_equal(c['spline'], before['spline'])
            # The undisplayed infancy attained-age band is a real likelihood term.
            p2 = deepcopy(p); p2['scheduled_infancy_post_log_or'][0] += .5
            factor_names = [rv.name for rv in m.free_RVs+m.observed_RVs+m.potentials]
            scheduled_factor = factor_names.index('scheduled_wheeze_observed')
            likelihood = m.compile_logp(sum=False, jacobian=False)
            self.assertGreater(abs(np.sum(likelihood(p2)[scheduled_factor])-np.sum(likelihood(p)[scheduled_factor])), .01)


if __name__ == '__main__':
    unittest.main()
