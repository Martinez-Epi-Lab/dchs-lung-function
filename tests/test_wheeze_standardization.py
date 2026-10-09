"""Estimand and numerical-integration checks using fabricated posterior draws."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from test_wheeze import fabricated_contract
from test_wheeze_results import fabricated_posterior
from dchs_lung_function.analysis import wheeze_standardization as ws


class WheezeStandardizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=fabricated_contract()
        cls.posterior=fabricated_posterior(cls.contract)

    def simple_posterior(self):
        p=deepcopy(self.posterior)
        p['scheduled_intercept'][:]=-2.
        p['sigma_scheduled'][:]=0.
        p['scheduled_covariate'][:]=1.2
        p['scheduled_age'][:]=0.
        p['scheduled_infancy_post_log_or'][:]=[0.,np.log(3),np.log(2),np.log(.7)]
        return p

    def test_zero_variance_probability_ratio_has_analytic_sex_mixture(self):
        p=self.simple_posterior()
        draws,meta=ws.standardized_draws(p,self.contract,'primary_sex')
        male=self.contract['children'].male.mean()
        baseline=(1-male)*expit(-2)+male*expit(-.8)
        for band,odds in ((1,3.),(2,2.),(3,.7)):
            exposed=(1-male)*expit(-2+np.log(odds))+male*expit(-.8+np.log(odds))
            np.testing.assert_allclose(draws[band]['p0'],baseline,rtol=1e-14)
            np.testing.assert_allclose(draws[band]['p1'],exposed,rtol=1e-14)
            np.testing.assert_allclose(draws[band]['rr'],exposed/baseline,rtol=1e-14)
            self.assertGreater(abs(exposed/baseline-odds),.05)
        self.assertEqual(meta['cohort_children'],12)
        self.assertEqual(meta['male_fraction'],float(male))
        self.assertNotEqual(float(male),float(self.contract['scheduled'].male.mean()))

    def test_latent_integration_matches_independent_adaptive_quadrature(self):
        flat={'scheduled_intercept':np.array([-1.7]),'scheduled_age':np.zeros((1,5)),
              'scheduled_covariate':np.array([[.9]]),'sigma_scheduled':np.array([1.3])}
        p0,p1=ws.integrated_probabilities(flat,np.zeros((1,5)),np.array([[.8]]),.4)
        for actual,shift in ((p0[0,0],0.),(p1[0,0],.8)):
            expected=sum(weight*quad(lambda z:expit(-1.7+sex*.9+1.3*z+shift)*norm.pdf(z),
                                     -12,12,epsabs=1e-12)[0] for sex,weight in ((0,.6),(1,.4)))
            self.assertAlmostEqual(actual,expected,places=12)
        self.assertGreater(abs(p0[0,0]-(.6*expit(-1.7)+.4*expit(-.8))),.03)

    def test_band_rr_is_ratio_after_probability_averaging(self):
        p=self.simple_posterior()
        p['scheduled_covariate'][:]=0.
        p['scheduled_age'][:]=[1.5,-.9,.7,-.3,1.1]
        result,_=ws.standardized_draws(p,self.contract,'primary_sex')
        transform,_=ws.baseline_transform(self.contract)
        x,w=leggauss(48);ages=1+(x+1)
        eta=-2+transform(ages)@p['scheduled_age'][0,0]
        no,yes=expit(eta),expit(eta+np.log(3.))
        correct=(yes@(w/2))/(no@(w/2));wrong=(yes/no)@(w/2)
        np.testing.assert_allclose(result[1]['rr'],correct,rtol=1e-14)
        self.assertGreater(abs(correct-wrong),.01)

    def test_training_spline_is_reused_and_boundaries_are_explicit(self):
        transform,error=ws.baseline_transform(self.contract)
        np.testing.assert_allclose(transform(self.contract['scheduled'].age.to_numpy(float)),self.contract['spline'],atol=1e-14)
        self.assertLess(error,1e-12)
        same=transform(np.array([1.,3.,5.]))
        expanded=transform(np.array([0.,1.,3.,5.,10.5]))
        np.testing.assert_allclose(same,expanded[1:4],atol=1e-14)
        for ages in (np.array([-1.]),np.array([10.5001]),np.array([np.nan])):
            with self.assertRaises(ValueError):transform(ages)
        corrupt=deepcopy(self.contract);corrupt['spline'][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'reconstruction'):ws.baseline_transform(corrupt)

    def test_primary_rr_summary_quantiles_and_convergence_receipt(self):
        table,meta=ws.summarize_standardized_rr(self.posterior,self.contract,'primary_sex')
        draws,_=ws.standardized_draws(self.posterior,self.contract,'primary_sex')
        self.assertEqual(list(table),ws.RR_COLUMNS)
        self.assertEqual(len(table),3)
        for row,band in zip(table.itertuples(),(1,2,3)):
            np.testing.assert_array_equal([row.rr_low,row.rr_median,row.rr_high],np.quantile(draws[band]['rr'],[.025,.5,.975]))
            np.testing.assert_allclose([row.difference_pp_low,row.difference_pp_median,row.difference_pp_high],
                np.quantile(100*(draws[band]['p1']-draws[band]['p0']),[.025,.5,.975]),rtol=0,atol=0)
        self.assertEqual((meta['latent_nodes'],meta['age_nodes'],meta['convergence_nodes']),(80,48,120))
        self.assertTrue(meta['convergence_passed'])
        self.assertLess(meta['maximum_draw_level_RR_quadrature_change_80_vs_120'],1e-4)

    def test_other_post_effects_and_fitted_child_latents_do_not_enter_standardization(self):
        p=deepcopy(self.posterior)
        original,_=ws.standardized_draws(p,self.contract,'primary_sex')
        p['latent_propensity'][:]=100.
        p['scheduled_other_post_log_or'][:]=50.
        changed,_=ws.standardized_draws(p,self.contract,'primary_sex')
        for band in original:
            for key in original[band]:np.testing.assert_array_equal(original[band][key],changed[band][key])

    def test_no_exposure_effect_yields_unit_rr_and_zero_difference(self):
        p=self.simple_posterior();p['scheduled_infancy_post_log_or'][:]=0.
        draws,_=ws.standardized_draws(p,self.contract,'primary_sex')
        for d in draws.values():
            np.testing.assert_array_equal(d['rr'],np.ones((2,12)))
            np.testing.assert_array_equal(d['p1']-d['p0'],np.zeros((2,12)))

    def test_adjusted_rr_is_explicitly_unselected(self):
        c=fabricated_contract('secondary_sex_smoking_birthweight')
        p=fabricated_posterior(c,'secondary_sex_smoking_birthweight')
        table,meta=ws.summarize_standardized_rr(p,c,'secondary_sex_smoking_birthweight')
        self.assertEqual(table.shape,(0,len(ws.RR_COLUMNS)))
        self.assertFalse(meta['applicable'])
        self.assertIn('primary',meta['reason'])

    def test_invalid_draw_scale_or_quadrature_arguments_rejected(self):
        p=self.simple_posterior();p['sigma_scheduled'][:]=-.2
        with self.assertRaisesRegex(ValueError,'scales'):ws.standardized_draws(p,self.contract,'primary_sex')
        p=self.simple_posterior();p['scheduled_covariate']=np.zeros((2,12,2))
        with self.assertRaisesRegex(ValueError,'shape'):ws.standardized_draws(p,self.contract,'primary_sex')
        flat={'scheduled_intercept':np.array([-2.]),'scheduled_age':np.zeros((1,5)),
              'scheduled_covariate':np.array([[0.]]),'sigma_scheduled':np.array([.3])}
        for kwargs in ({'nodes':1},{'nodes':True},{'nodes':2.5}):
            with self.assertRaises(ValueError):ws.integrated_probabilities(flat,np.zeros((1,5)),np.zeros((1,1)),.5,**kwargs)
        with self.assertRaises(ValueError):ws.integrated_probabilities(flat,np.zeros((1,5)),np.zeros((1,1)),1.1)

    def test_inadequate_quadrature_is_not_silently_accepted(self):
        p=self.simple_posterior();p['sigma_scheduled'][:]=12.
        with self.assertRaisesRegex(ValueError,'convergence criterion'):
            ws.summarize_standardized_rr(p,self.contract,'primary_sex')


if __name__=='__main__':unittest.main()
