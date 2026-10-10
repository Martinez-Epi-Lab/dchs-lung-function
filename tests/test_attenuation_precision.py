"""Independent delta arithmetic and prospective precision configuration."""
import math
import unittest

from dchs_lung_function.analysis.attenuation_precision import attenuation_precision
from dchs_lung_function.analysis.primary_results import _summarize, _summarize_ratio
import numpy as np
from dchs_lung_function.sensitivity_sampling import resolved_sampling, validate_sampling


class AttenuationPrecisionTests(unittest.TestCase):
    def test_ratio_probabilities_use_one_and_preserve_all_numeric_summaries(self):
        log_draws=np.array([[-2.,0.,1.],[.5,-.1,0.]])
        ratios=np.exp(log_draws)
        original=_summarize(ratios,False)
        corrected=_summarize_ratio(ratios,False)
        for name in ('mean','median','lower','upper'):
            self.assertEqual(corrected[name],original[name])
        for direction,expected in [('below',2/6),('above',2/6),('equal',2/6)]:
            self.assertEqual(corrected['probability_'+direction+'_one'],expected)
            self.assertNotIn('probability_'+direction+'_zero',corrected)
        for invalid in (np.array([[0.]]),np.array([[-1.]]),np.array([[np.nan]])):
            with self.assertRaises(ValueError):_summarize_ratio(invalid,False)

    def test_hand_calculated_gradient_and_covariance_bound(self):
        # A=-1, C=-2: gradients are50 and-25; errors .05 and.1
        # contribute2.5 each, hence an unrestricted covariance bound of5.
        r=attenuation_precision(-1.,-2.,.05,.1)
        self.assertEqual(r['attenuation_percent'],50.)
        self.assertEqual(r['attenuation_mcse_bound_pp'],5.)
        self.assertAlmostEqual(r['attenuation_mcse_independent_pp'],math.sqrt(12.5))
        self.assertFalse(r['attenuation_precision_passed'])

    def test_scale_invariance_and_sign_reversal_are_separate(self):
        a=attenuation_precision(-1.,-2.,.001,.002)
        b=attenuation_precision(-100.,-200.,.1,.2)
        self.assertAlmostEqual(a['attenuation_mcse_bound_pp'],b['attenuation_mcse_bound_pp'])
        self.assertTrue(a['attenuation_precision_passed'])
        r=attenuation_precision(1.,-2.,.001,.002)
        self.assertEqual(r['attenuation_status'],'sign_reversal')
        self.assertEqual(r['attenuation_percent'],50.)

    def test_undefined_and_nondifferentiable_means_are_explicit(self):
        self.assertEqual(attenuation_precision(-1.,0.,.1,.1)['attenuation_status'],'undefined_zero_core')
        r=attenuation_precision(0.,-1.,.1,.1)
        self.assertIsNone(r['attenuation_mcse_bound_pp'])
        self.assertFalse(r['attenuation_precision_passed'])
        self.assertEqual(r['attenuation_status'],'zero_adjusted_mean')

    def test_invalid_precision_inputs_fail(self):
        for args in [(True,-2.,.1,.1),(-1.,-2.,-.1,.1),(-1.,-2.,float('nan'),.1),
                     (-1.,1e-200,.1,.1)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                attenuation_precision(*args)

    def test_target_draw_override_keeps_other_sampling_controls(self):
        base={'schema_version':1,'chains':4,'draws':3200,'tune':2200,'cores':2,
              'target_accept':.95,'max_treedepth':10,'seed_offset':104729}
        settings={**base,'draws_by_target':{'early_core':12800}}
        core=resolved_sampling('early_core',settings)
        old=resolved_sampling('early_core',base)
        self.assertEqual(core['draws'],12800)
        self.assertEqual({k:v for k,v in core.items() if k!='draws'},
                         {k:v for k,v in old.items() if k!='draws'})
        self.assertEqual(resolved_sampling('early_ever_lrti',settings)['draws'],3200)
        for overrides in ({'unknown':12800},{'early_core':True},{'early_core':3199},[]):
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):
                validate_sampling({**base,'draws_by_target':overrides})


if __name__=='__main__':unittest.main()
