"""Independent aggregate-summary checks using entirely fabricated observations."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from test_wheeze import fabricated_contract
from dchs_lung_function.models.wheeze import validate_contract, process_arrays
from dchs_lung_function.wheeze_sampling import expected_posterior_shapes
from dchs_lung_function.analysis import wheeze_results as wr


def fabricated_posterior(contract=None,target='primary_sex',*,chains=2,draws=12):
    """Complete graph arrays from arbitrary synthetic numbers, without fitting."""
    if contract is None:contract=fabricated_contract(target)
    rng=np.random.default_rng(3917)
    result={name:rng.normal(0,.12,shape) for name,shape in
            expected_posterior_shapes(contract,target,chains,draws).items()}
    result['scheduled_intercept']-=2.
    result['episode_log_baseline_rate']-=1.
    result['tb_log_baseline_hazard']-=4.
    result['sigma_scheduled']=np.full((chains,draws),.4)
    result['sigma_episode']=np.full((chains,draws),.3)
    result['scheduled_infancy_post_log_or']+=np.array([.2,.8,.3,-.2])
    result['episode_infancy_post_log_rr']+=np.array([.1,.6,.2,-.1])
    for ratio,log in wr.TRANSFORMS.items():result[ratio]=np.exp(result[log])
    return result


class WheezeResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=fabricated_contract()
        cls.posterior=fabricated_posterior(cls.contract)
        cls.info=validate_contract(cls.contract,'primary_sex')
        cls.results=wr.summarize_wheeze(cls.posterior,cls.contract,'primary_sex',ppc_seed=431)

    def test_exact_table_inventory_and_source_order(self):
        r=self.results
        for key,count in {'effects':6,'contrasts':6,'global_tests':2,'support_counts':22,
                          'posterior_predictive_cells':22,'standardized_rr':3}.items():
            table=getattr(r,key)
            self.assertEqual(len(table),count)
            self.assertEqual(list(table),wr.TABLE_COLUMNS[key])
            self.assertFalse(any('child_index' in c or c=='pid_child' for c in table))
        self.assertEqual(r.support_counts.iloc[2].exposure_group,'post-TB after TB at age >=1')
        self.assertEqual(r.posterior_predictive_cells.iloc[2].exposure_group,'post infancy TB')
        self.assertEqual(set(r.ppc_replicates),set(wr.PPC_REPLICATE_KEYS))

    def test_effect_estimands_use_all_chains_and_nonlinear_draw_summaries(self):
        for row,name,band in ((self.results.effects.iloc[0],'scheduled_infancy_post_log_or',1),
                              (self.results.effects.iloc[5],'episode_infancy_post_log_rr',3)):
            a=np.exp(self.posterior[name][...,band]).ravel()
            self.assertEqual(row.posterior_mean,float(a.mean()))
            self.assertEqual(row.posterior_median,float(np.median(a)))
            np.testing.assert_array_equal([row.cri_2_5,row.cri_97_5],np.quantile(a,[.025,.975]))
            self.assertEqual(row.posterior_probability_above_1,float((a>1).mean()))
            self.assertNotAlmostEqual(row.posterior_mean,float(np.exp(np.log(a).mean())),places=6)

    def test_global_test_preserves_correlated_contrasts(self):
        beta=self.posterior['scheduled_infancy_post_log_or'].reshape(-1,4)[:,1:]
        c=np.array([[-1,1,0],[-1,0,1]])
        mean=c@beta.mean(axis=0)
        covariance=c@np.cov(beta,rowvar=False)@c.T
        statistic=float(mean@np.linalg.inv(covariance)@mean)
        row=self.results.global_tests.iloc[0]
        self.assertAlmostEqual(row.posterior_wald_chi_square,statistic,places=10)
        self.assertAlmostEqual(row.reference_p_value,chi2.sf(statistic,2),places=12)
        pair=np.exp(beta[:,2]-beta[:,0])
        self.assertAlmostEqual(self.results.contrasts.iloc[1].posterior_mean,pair.mean(),places=14)
        self.assertEqual(self.results.contrasts.iloc[1].posterior_probability_later_effect_smaller,
                         float((beta[:,2]<beta[:,0]).mean()))

    def test_every_stored_deterministic_is_verified(self):
        for variable in wr.TRANSFORMS:
            with self.subTest(variable=variable):
                modified={**self.posterior,variable:self.posterior[variable]*1.001}
                with self.assertRaisesRegex(ValueError,'deterministic'):
                    wr.validate_posterior(modified,self.info)

    def test_nonfinite_shapes_missing_and_scales_rejected(self):
        cases=[]
        missing=dict(self.posterior);missing.pop('tb_propensity_hr');cases.append(missing)
        cases.append({**self.posterior,'scheduled_age':self.posterior['scheduled_age'][...,:4]})
        cases.append({**self.posterior,'sigma_episode':np.full((2,12),-.1)})
        cases.append({**self.posterior,'scheduled_intercept':np.full((2,12),np.nan)})
        cases.append({**self.posterior,'tb_latent_association':np.full((2,12),1000.)})
        for case in cases:
            with self.subTest(case=len(cases)),self.assertRaises(ValueError):wr.validate_posterior(case,self.info)

    def test_support_uses_exact_small_counts_and_preserves_empty_cells(self):
        source=self.contract['scheduled']
        self.assertEqual(self.results.support_counts.iloc[:11].observations_or_intervals.sum(),len(source))
        self.assertEqual(self.results.support_counts.iloc[:11].events_or_positive.sum(),source.wheeze.sum())
        altered={**self.contract,'scheduled':source[source.post_tb==0].copy()}
        table=wr.support_counts(altered,'primary_sex')
        selected=table[(table.process==wr.PROCESSES[0])&(table.exposure_group!='pre/no-TB')]
        self.assertEqual(len(table),22)
        self.assertEqual(int(selected.observations_or_intervals.sum()),0)
        self.assertTrue((selected.children==0).all())
        self.assertTrue(table.iloc[:11].person_years.isna().all())

    def test_ppc_matches_independent_source_order_rng_and_counts(self):
        p=fabricated_posterior(self.contract,chains=2,draws=110)
        table,reps,meta=wr.posterior_predictive_cells(p,self.contract,'primary_sex',ppc_seed=727)
        keep=np.linspace(0,219,200,dtype=int)
        np.testing.assert_array_equal(reps['draw_indices'],keep)
        arrays=process_arrays(self.contract,'primary_sex')
        flat={key:value.reshape((-1,)+value.shape[2:]) for key,value in p.items()}
        rng=np.random.default_rng(727)
        expected=np.zeros((200,22),int)
        for k,scheduled in enumerate((True,False)):
            f=self.contract['scheduled' if scheduled else 'episode_intervals']
            child=f.child_index.to_numpy(int);band=f.effect_band.to_numpy(int)
            prefix='sched' if scheduled else 'epi';inf=arrays[prefix+'_inf'];other=arrays[prefix+'_other']
            cats=np.where(inf==1,'post infancy TB',np.where(other==1,'post other TB','pre/no TB'))
            for j,d in enumerate(keep):
                if scheduled:
                    eta=flat['scheduled_intercept'][d]+self.contract['spline']@flat['scheduled_age'][d]
                    eta+=arrays['X_sched']@flat['scheduled_covariate'][d]
                    eta+=flat['sigma_scheduled'][d]*flat['latent_propensity'][d,child]
                    eta+=flat['scheduled_infancy_post_log_or'][d,band]*inf+flat['scheduled_other_post_log_or'][d,band]*other
                    sim=rng.binomial(1,expit(eta))
                else:
                    eta=flat['episode_log_baseline_rate'][d,f.base_band.to_numpy(int)]+arrays['X_epi']@flat['episode_covariate'][d]
                    eta+=flat['sigma_episode'][d]*flat['latent_propensity'][d,child]
                    eta+=flat['episode_infancy_post_log_rr'][d,band]*inf+flat['episode_other_post_log_rr'][d,band]*other
                    sim=rng.poisson(f.person_years.to_numpy(float)*np.exp(eta))
                for col,(b,g) in enumerate(wr.PPC_CELL_KEYS):
                    expected[j,k*11+col]=sim[(band==b)&(cats==g)].sum()
        np.testing.assert_array_equal(reps['counts'],expected)
        for i,row in table.iterrows():
            scale=(1 if i<11 else 1000)/row.observed_denominator
            np.testing.assert_array_equal(reps['statistics'][:,i],expected[:,i]*scale)
            self.assertEqual(row.replicated_mean,float(reps['statistics'][:,i].mean()))
            np.testing.assert_array_equal([row.replicated_2_5,row.replicated_97_5],np.quantile(reps['statistics'][:,i],[.025,.975]))
        self.assertEqual(meta['replicates'],200)

    def test_ppc_analytic_conditional_variance_and_seed(self):
        p={**self.posterior,'scheduled_age':np.zeros((2,12,5)),
           'scheduled_covariate':np.zeros((2,12,1)),'sigma_scheduled':np.zeros((2,12)),
           'scheduled_infancy_post_log_or':np.zeros((2,12,4)),
           'scheduled_other_post_log_or':np.zeros((2,12,4))}
        t,a,_=wr.posterior_predictive_cells(p,self.contract,'primary_sex',ppc_seed=23)
        _,b,_=wr.posterior_predictive_cells(p,self.contract,'primary_sex',ppc_seed=24)
        self.assertFalse(np.array_equal(a['counts'],b['counts']))
        for i in range(11):
            prob=expit(p['scheduled_intercept']).ravel()
            np.testing.assert_allclose(a['conditional_means'][:,i],prob,rtol=1e-15)
            np.testing.assert_allclose(a['conditional_variances'][:,i],prob*(1-prob)/t.iloc[i].observed_denominator,rtol=1e-15)
        np.testing.assert_array_equal(a['conditional_means'],b['conditional_means'])

    def test_empty_predictive_support_has_zero_counts_and_nan_normalization(self):
        f=self.contract['scheduled'];mask=f.post_tb==0
        c={**self.contract,'scheduled':f[mask].copy(),'spline':self.contract['spline'][mask]}
        t,rep,meta=wr.posterior_predictive_cells(self.posterior,c,'primary_sex',ppc_seed=12)
        empty=t.observed_denominator.eq(0).to_numpy()
        self.assertTrue(empty.any())
        self.assertTrue((rep['counts'][:,empty]==0).all())
        self.assertTrue(np.isnan(rep['statistics'][:,empty]).all())
        self.assertTrue(t.loc[empty,'observed_statistic'].isna().all())
        self.assertTrue(all(row['simulation_mean_error'] is None for row in meta['conditional_mean_checks'] if row['empty']))

    def test_adjusted_result_has_empty_rr_with_schema_and_no_mutation(self):
        c=fabricated_contract('secondary_sex_smoking_birthweight');before=deepcopy(c)
        p=fabricated_posterior(c,'secondary_sex_smoking_birthweight');p_before=deepcopy(p)
        result=wr.summarize_wheeze(p,c,'secondary_sex_smoking_birthweight',ppc_seed=96)
        self.assertEqual(result.standardized_rr.shape,(0,len(wr.RR_COLUMNS)))
        self.assertFalse(result.metadata['standardization']['applicable'])
        for key in ('children','scheduled','episode_intervals','episodes'):pd.testing.assert_frame_equal(c[key],before[key])
        for key in p:np.testing.assert_array_equal(p[key],p_before[key])

    def test_ppc_overflow_and_invalid_seed_fail_safely(self):
        p={**self.posterior,'episode_log_baseline_rate':np.full((2,12,8),1000.)}
        with self.assertRaisesRegex(ValueError,'intensity'):
            wr.posterior_predictive_cells(p,self.contract,'primary_sex',ppc_seed=1)
        for seed in (-1,True,1.5):
            with self.assertRaisesRegex(ValueError,'seed'):
                wr.posterior_predictive_cells(self.posterior,self.contract,'primary_sex',ppc_seed=seed)


if __name__=='__main__':unittest.main()
