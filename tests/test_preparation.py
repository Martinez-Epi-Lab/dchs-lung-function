"""Synthetic repeated records exercise dates, eligibility, and matrix contracts."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.preparation import (BASE_TB,link_longitudinal,enrich_covariates,
    normalize_pft,exposure_frame,event_arrays,drop_design_columns,repeated_design,binary_inputs,pretb_cox_inputs,joint_inputs)


def linkage_data():
    dates=pd.to_datetime(['2021-01-01','2021-01-02','2022-01-01',None])
    p=pd.DataFrame({'pid_child':['A','A','A','B'],'visit':[2,2,3,1],
                    'date_of_test':dates,'date_of_test_ilf1':dates,'date_of_test_ilf2':dates})
    b=pd.DataFrame({'pid_child':['A','B'],**{key:[np.nan,np.nan] for key in BASE_TB}})
    e=pd.DataFrame({'pid_child':['A','A'],'date_tb_diagnosed':pd.to_datetime(['2021-01-02','2021-06-01']),
                    'date_tb_investigated':pd.to_datetime(['2021-01-01','2021-05-30']),
                    'childdob':pd.to_datetime(['2020-01-01']*2),
                    'tbcat':['Unconfirmed tb','Confirmed tb'],'tbtrt':['yes','yes']})
    return p,b,e


def model_frame():
    n=12
    f=pd.DataFrame({'pid_child':['A','A','B','B','C','C','D','D','E','E','F','F'],
                    'visit':[6,9]*6,'visit_label':['60 months','72 months']*6,
                    'sex':[0,0,1,1]*3,'male':[1.,1.,0.,0.]*3,
                    'height':np.arange(n)+100.,'site':[1]*n,
                    'child_dob':pd.to_datetime(['2015-01-01']*n),
                    'date_of_test':pd.to_datetime(['2020-01-01','2021-01-01']*6),
                    'tb_first_dx':pd.to_datetime(['2015-07-01']*2+['2020-07-01']*2+['2021-07-01']*2+[None]*6),
                    'tb_age_first_years':[.5,.5,5.5,5.5,6.5,6.5]+[np.nan]*6,
                    'exit_date':pd.to_datetime([None]*n),
                    'fev1_fvc_gli_z':[-1.645,-1.646]+[0.]*10,'fvc_gli_z':[0.]*n})
    f['date_of_test_ilf2']=f.date_of_test;f['date_of_test_ilf1']=f.date_of_test
    f['base_survivaltb_final']=8.;f['x6_sa_z']=np.linspace(-1,1,n)
    return f


class PreparationTests(unittest.TestCase):
    def test_exact_diagnosis_date_and_recurrent_history(self):
        p,b,e=linkage_data();before=p.copy(deep=True)
        actual=link_longitudinal(p,b,e,ascertainment_end='2021-12-31')
        self.assertEqual(actual.tb_state_visit.iloc[:3].tolist(),[1,2,2])
        self.assertEqual(actual.tb_prior_n_visit.iloc[:3].tolist(),[0,1,2])
        self.assertEqual(actual.tb_prior_conf_n_visit.iloc[:3].tolist(),[0,0,1])
        self.assertEqual(actual.tb_state_inv_visit.iloc[0],2)
        self.assertTrue(pd.isna(actual.tb_state_visit.iloc[3]))
        self.assertEqual(actual.visit_label.iloc[0],'12 months')
        pd.testing.assert_frame_equal(p,before)

    def test_ambiguous_child_linkage_and_undated_episode_fail(self):
        p,b,e=linkage_data()
        with self.assertRaisesRegex(ValueError,'duplicate'):link_longitudinal(p,pd.concat([b,b]),e,ascertainment_end='2022-01-01')
        e.loc[0,'date_tb_diagnosed']=pd.NaT
        with self.assertRaisesRegex(ValueError,'undated'):link_longitudinal(p,b,e,ascertainment_end='2022-01-01')

    def test_reference_covariate_scale_uses_entire_baseline(self):
        b=pd.DataFrame({'pid_child':['A','B','C'],'maternal_hiv':[0,1,0],'Prenatal_smoking_composite':[0,1,0],'lrti':[1,0,0],'birth_weight':[2000.,3000.,4000.]})
        p=pd.DataFrame({'pid_child':['A','A']})
        result=enrich_covariates(p,b)
        self.assertEqual(result.birthweight_z.tolist(),[-1.,-1.])
        self.assertEqual(result.prenatal_smoking.tolist(),[0,0])

    def test_later_tb_rules_and_exit_boundary_are_model_specific(self):
        f=model_frame();f.loc[0,'exit_date']=pd.Timestamp('2020-01-01')
        full=exposure_frame(f,'fev1_fvc_gli_z',respect_exit=True)
        early=exposure_frame(f,'fev1_fvc_gli_z',threshold=5)
        self.assertEqual(len(full),12);self.assertEqual(len(early),11)
        self.assertEqual(full.post_at_after5.sum(),1)
        self.assertEqual(early.future_tb.sum(),3)

    def test_lln_is_strict_and_no_small_count_suppression(self):
        actual=binary_inputs(model_frame(),outcome='ratio_LLN')
        self.assertEqual(actual['frame'].outcome_value.iloc[:2].tolist(),[0,1])
        self.assertEqual(actual['audit']['events'],1)
        self.assertEqual(actual['audit']['n_children'],6)

    def test_survival_interval_boundary_and_censoring(self):
        f=model_frame().drop_duplicates('pid_child').copy()
        a=event_arrays(f,threshold=5,event_end='2022-01-01')
        self.assertEqual(int(a['event'].sum()),1)
        np.testing.assert_allclose(a['exposure'].sum(axis=1),a['stop'])
        f.loc[f.index[0],'tb_first_dx']=pd.Timestamp('2030-01-01')
        with self.assertRaisesRegex(ValueError,'outside'):event_arrays(f,event_end='2022-01-01')

    def test_required_nonconstant_collinear_exposure_rejected(self):
        x=np.column_stack([np.ones(4),[0.,1.,0.,1.],[0.,1.,0.,1.]])
        with self.assertRaisesRegex(ValueError,'collinear'):drop_design_columns(x,['Intercept','a','b'],{'b'})

    def test_required_constant_terms_fail_but_zero_support_is_explicit(self):
        for value in (0.,1.):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'constant'):
                drop_design_columns(np.column_stack([np.ones(4),np.repeat(value,4)]),['Intercept','target'],{'target'})
        f=model_frame();f['zero_exposure']=0.
        matrix,terms,dropped=repeated_design(f,['zero_exposure'])
        self.assertNotIn('zero_exposure',terms)
        self.assertIn('zero_exposure (constant)',dropped)
        self.assertEqual(matrix.shape[0],len(f))
        f['zero_exposure']=1.
        with self.assertRaisesRegex(ValueError,'constant'):repeated_design(f,['zero_exposure'])

    def test_threshold_followup_requires_endpoint_and_positive_time(self):
        f=model_frame().drop_duplicates('pid_child').copy()
        with self.assertRaisesRegex(ValueError,'explicit endpoint'):event_arrays(f,threshold=5)
        for endpoint in ['2015-01-01','2014-12-31',pd.NaT]:
            with self.subTest(endpoint=endpoint),self.assertRaises(ValueError):
                event_arrays(f,threshold=5,event_end=endpoint)
        f.loc[f.index[-1],'exit_date']=f.child_dob.iloc[-1]
        with self.assertRaisesRegex(ValueError,'follow-up duration'):event_arrays(f,threshold=5,event_end='2022-01-01')

    def test_threshold_event_and_censor_boundaries(self):
        # Ages are supplied explicitly because the source TB-age role differs
        # from the measurement-DOB role. Dates here use a common birth date.
        birth=pd.Timestamp('2000-01-01');end=pd.Timestamp('2000-10-01')
        diagnosis=pd.to_datetime([None,None,'2000-07-01','2000-10-01','2000-07-02','2000-10-02','2000-06-01'])
        f=pd.DataFrame({'child_dob':[birth]*7,'exit_date':pd.to_datetime(['2000-07-01',None,'2000-07-01',None,'2000-07-01',None,None]),
                        'tb_first_dx':diagnosis,'tb_age_first_years':(diagnosis-birth).days.to_numpy()/365.25})
        result=event_arrays(f,threshold=1,event_end=end)
        np.testing.assert_array_equal(result['event'],[0,0,1,1,0,0,1])
        # Independent hand-specified day durations: both endpoint events count;
        # the known next-day diagnoses are censored and add no event.
        np.testing.assert_allclose(result['stop'],np.array([182,274,182,274,182,274,152])/365.25)
        np.testing.assert_allclose(result['exposure'][:,0],result['stop'])
        self.assertEqual(result['ascertainment_end'],'2000-10-01')
        self.assertEqual(int(result['event_interval'].sum()),3)

    def test_exact_threshold_is_not_an_event(self):
        for threshold in (1,5):
            f=pd.DataFrame({'child_dob':pd.to_datetime(['2000-01-01']*2),'exit_date':pd.to_datetime([None,None]),
                'tb_first_dx':pd.to_datetime(['2000-12-31','2004-12-31']),
                'tb_age_first_years':[float(threshold),float(threshold)-.01]})
            a=event_arrays(f,threshold=threshold,event_end='2010-01-01')
            np.testing.assert_array_equal(a['event'],[0,1])
            np.testing.assert_allclose(a['stop'],[threshold,threshold-.01])

    def test_threshold_malformed_diagnosis_fails(self):
        f=model_frame().drop_duplicates('pid_child').copy()
        for date,age in [(pd.Timestamp('2014-01-01'),.5),(pd.Timestamp('2016-01-01'),np.nan),(pd.NaT,.5)]:
            altered=f.copy();altered.loc[altered.index[0],['tb_first_dx','tb_age_first_years']]=[date,age]
            with self.subTest(date=date,age=age),self.assertRaisesRegex(ValueError,'chronology'):
                event_arrays(altered,threshold=5,event_end='2022-01-01')

    def test_joint_process_exit_filter_preserves_admin_independent_observations(self):
        f=model_frame().copy()
        for i,col in enumerate(['target','i_lci_2_5','resistance','compliance','xee','ree']):
            f[col]=np.sin(np.arange(len(f))*.7+i)+np.arange(len(f))*.03
        # No TB so every process has enough pre/no-TB residual observations.
        f['tb_first_dx']=pd.NaT;f['tb_age_first_years']=np.nan
        f.loc[f.pid_child.eq('F'),'exit_date']=pd.Timestamp('2020-06-01')
        # This row's LCI measurement occurs on exit, while target/mechanics
        # occur later and must be removed only from their own processes.
        f.loc[11,'date_of_test_ilf1']=pd.Timestamp('2020-06-01')
        result=joint_inputs(f,('target',1.),family='early',event_end='2019-01-01')
        self.assertEqual(len(result['frames']['target']),11)
        self.assertEqual(len(result['frames']['lci']),12)
        self.assertEqual(len(result['frames']['resistance']),11)
        self.assertTrue(result['frames']['lci'].test_date.eq(pd.Timestamp('2020-06-01')).any())
        self.assertTrue(result['frames']['target'].test_date.gt(pd.Timestamp('2019-01-01')).all())
        # Event-risk time is nevertheless capped at the earlier admin end.
        np.testing.assert_allclose(result['event']['stop'],1461/365.25)

    def test_cox_two_year_carry_forward_and_event_time(self):
        f=model_frame();r=pretb_cox_inputs(f,'x6_sa_z',-1,True,observation_end='2022-01-01')
        interval=r['frame']
        self.assertTrue(((interval.stop-interval.start)<=2).all())
        self.assertTrue((interval.stop>interval.start).all())
        self.assertEqual(int(interval.event.sum()),2)


if __name__=='__main__':unittest.main()
