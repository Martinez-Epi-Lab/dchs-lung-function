"""Independent equations and source-rule tests using fabricated FeNO data."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import sys,unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.models import feno


def fabricated_contract(target='main_12months', *, repeat=True, no_exposure=False):
    """Source-shaped main/supplement contract with genuine repeated clusters."""
    spec=feno.fit_spec(target);rng=np.random.default_rng(60911);rows=[]
    nominal=spec['nominal_age_years']
    for child in range(32):
        dob=pd.Timestamp('2000-01-01')+pd.Timedelta(days=child*3)
        censor=dob+pd.DateOffset(years=10,months=6)
        group=child%4
        tbday=None if no_exposure or group==3 else 120 if group==0 else max(380,int(nominal*365.25)-50) if group==1 else int(nominal*365.25)+700
        tb=dob+pd.Timedelta(days=tbday) if tbday is not None else pd.NaT
        tb_age=tbday/365.25 if tbday is not None and tbday/365.25<10.5 and tb<censor else np.nan
        child_error=rng.normal(0,.23)
        for occasion in range(2 if repeat else 1):
            day=max(7,int(nominal*365.25)+int(rng.integers(-15,16))+occasion*31)
            date=dob+pd.Timedelta(days=day);age=day/365.25;height=50+nominal*6+rng.normal(0,3)
            infancy=int(tb_age<1 and date>=tb);later=int(tb_age>=1 and date>=tb)
            before=int(tbday is not None and date>=tb);after=int(tbday is not None and date<tb)
            log_value=2+.18*infancy+.12*before+.08*(child%2)+.02*(height-50-nominal*6)+child_error+rng.normal(0,.12)
            value=np.float32(np.exp(log_value))
            rows.append({'pid_child':f'fabricated-{child:02}','visit_label':spec['visit'],'date':date,
                'child_dob':dob,'dob':dob,'censor_date':censor,'tb_date':tb,'tb_first_dx':tb,'tb_age':tb_age,
                'age':age,'age_years':age,'height':height,'male':float(child%2),'feno':value,
                'infancy_post':infancy,'later_post':later,'tb_before_measurement':float(before),
                'tb_after_measurement':float(after)})
    frame=pd.DataFrame(rows);frame['feno']=frame.feno.astype(np.float32);frame['log_feno']=np.log(frame.feno)
    if spec['family']=='main':
        if spec['visit']!='12 months':frame=frame.loc[frame.age.ge(1)&frame.age.lt(10.5)&frame.date.lt(frame.censor_date)].copy()
        else:frame=frame.loc[frame.date.lt(frame.censor_date)&(~frame.tb_age.lt(1)|frame.infancy_post.eq(1))].copy()
        terms=['infancy_post']+(['later_post'] if frame.later_post.sum() else [])+['male','age','height']
        x=pd.DataFrame({'Intercept':np.ones(len(frame))},index=frame.index)
        for term in terms:
            values=frame[term].astype(float)
            x[term]=(values-values.mean())/values.std(ddof=1) if term in ('age','height') else values
        target_term='infancy_post'
        audit={'observations':len(frame),'children':frame.pid_child.nunique(),'infancy_post':int(frame.infancy_post.sum()),
               'later_post':int(frame.later_post.sum()),'terms':list(x)}
    else:
        for raw,z in [('age_years','age_z'),('height','height_z')]:frame[z]=(frame[raw]-frame[raw].mean())/frame[raw].std(ddof=1)
        terms=[name for name in ['tb_before_measurement','tb_after_measurement'] if frame[name].sum()>0]+['age_z','height_z','male']
        x=pd.DataFrame({'Intercept':np.ones(len(frame))},index=frame.index)
        for term in terms:x[term]=frame[term].astype(float)
        target_term='tb_before_measurement' if frame.tb_before_measurement.sum()>0 else 'tb_after_measurement' if frame.tb_after_measurement.sum()>0 else None
        audit={'observations':len(frame),'children':frame.pid_child.nunique(),'tb_before':int(frame.tb_before_measurement.sum()),
               'tb_after':int(frame.tb_after_measurement.sum()),'terms':list(x),'status':'ready' if target_term else 'no_exposure_contrast'}
    return {'frame':frame,'design':x,'response':frame.log_feno.to_numpy(float),'target':target_term,'audit':audit}


def independent_qr(contract):
    x=contract['design'].to_numpy(float);y=contract['response'];q,r=np.linalg.qr(x,mode='reduced')
    beta=np.linalg.solve(r,q.T@y);residual=y-x@beta;bread=np.linalg.solve(r,np.linalg.solve(r.T,np.eye(x.shape[1])))
    groups=pd.Categorical(contract['frame'].pid_child).codes
    scores=np.zeros((groups.max()+1,x.shape[1]));np.add.at(scores,groups,x*residual[:,None])
    robust=bread@scores.T@scores@bread;scale=residual@residual/(len(y)-x.shape[1])
    return beta,robust,bread*scale,scale


class FenoModelTests(unittest.TestCase):
    def test_both_families_against_independent_qr_child_sandwich(self):
        for target in ('main_12months','main_9years','supplement_6to10weeks','supplement_10years'):
            with self.subTest(target=target):
                contract=fabricated_contract(target);fit=feno.fit_contract(contract,target)
                self.assertTrue(fit.metadata['accepted'],fit.metadata)
                beta,robust,naive,scale=independent_qr(contract)
                np.testing.assert_allclose(fit.coefficients,beta,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(fit.covariance,robust,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(fit.naive_covariance,naive,rtol=1e-10,atol=1e-10)
                self.assertAlmostEqual(fit.metadata['diagnostics']['scale'],scale,places=12)
                self.assertLess(fit.metadata['diagnostics']['score_norm'],1e-6)
                # Repeated within-child observations must contribute cross terms.
                x=contract['design'].to_numpy(float);residual=contract['response']-x@beta
                bread=np.linalg.inv(x.T@x);row_cov=bread@((x*residual[:,None]).T@(x*residual[:,None]))@bread
                self.assertGreater(np.max(abs(robust-row_cov)),1e-4)
                # No HC1 or other small-sample multiplier is silently added.
                n,p=x.shape;self.assertGreater(np.max(abs(fit.covariance.to_numpy()-robust*n/(n-p))),1e-4)

    def test_all_term_transformations_use_robust_se_and_literal_1_96(self):
        fit=feno.fit_contract(fabricated_contract(),'main_12months')
        beta=fit.coefficients.to_numpy();se=np.sqrt(np.diag(fit.covariance))
        np.testing.assert_allclose(fit.estimates.exponentiated_coefficient,np.exp(beta),rtol=1e-14)
        np.testing.assert_allclose(fit.estimates.ci_low,np.exp(beta-1.96*se),rtol=1e-14)
        np.testing.assert_allclose(fit.estimates.ci_high,np.exp(beta+1.96*se),rtol=1e-14)
        np.testing.assert_allclose(fit.estimates.p_value,2*norm.sf(abs(beta/se)),rtol=1e-14)
        self.assertIn('source-standardized SD',fit.metadata['coefficient_transform'])

    def test_selected_target_routes_and_default_controls(self):
        self.assertEqual(len(feno.DEFAULT_TARGETS),11)
        self.assertEqual(feno.fit_spec('main_12months')['scenario'],'nominal12_sex_age_height')
        self.assertEqual(feno.fit_spec('main_12months')['fit_controls']['maxiter'],60)
        self.assertEqual(feno.fit_spec('supplement_12months')['fit_controls']['maxiter'],250)
        c=fabricated_contract('supplement_6to10weeks')
        self.assertEqual(feno.validate_contract(c,'supplement_6to10weeks')['plotted_term'],'tb_after_measurement')
        c=fabricated_contract('supplement_10years')
        self.assertEqual(feno.validate_contract(c,'supplement_10years')['plotted_term'],'tb_before_measurement')
        self.assertEqual(feno.artifact_path('main_9years'),('M03','respiratory','main_feno','samples','FeNO'))

    def test_no_exposure_guard_keeps_three_exact_counts_and_null_rows(self):
        target='supplement_6to10weeks';c=fabricated_contract(target,no_exposure=True)
        fit=feno.fit_contract(c,target)
        self.assertTrue(fit.metadata['accepted']);self.assertEqual(fit.metadata['status'],'non_estimable')
        self.assertEqual(fit.metadata['plotted_term'],'')
        self.assertEqual(fit.counts.observations.tolist(),[0,0,64])
        self.assertEqual(fit.counts.children.tolist(),[0,0,32])
        self.assertTrue(fit.estimates.coefficient.isna().all());self.assertTrue(fit.covariance.isna().all().all())
        self.assertEqual(len(fit.estimates),4)

    def test_singular_design_fails_without_dropping_adjustment(self):
        c=fabricated_contract();c['frame']['height']=c['frame']['age'];c['design']['height']=c['design']['age']
        fit=feno.fit_contract(c,'main_12months')
        self.assertFalse(fit.metadata['accepted']);self.assertEqual(fit.metadata['reason'],'singular_design')
        self.assertEqual(fit.metadata['terms'],list(c['design']))
        self.assertTrue(fit.estimates.p_value.isna().all())

    def test_covariance_rejection_preserves_solver_convergence(self):
        original=feno.GEE.fit
        def invalid_cov(*args,**kwargs):
            result=original(*args,**kwargs);result.cov_robust[0,0]=-1.;return result
        with patch.object(feno.GEE,'fit',invalid_cov):fit=feno.fit_contract(fabricated_contract(),'main_12months')
        self.assertFalse(fit.metadata['accepted']);self.assertEqual(fit.metadata['reason'],'invalid_covariance')
        self.assertTrue(fit.metadata['converged']);self.assertTrue(fit.estimates.exponentiated_coefficient.isna().all())

    def test_nonconverged_and_nonfinite_fits_have_no_inference(self):
        original=feno.GEE.fit
        for alteration,reason in [('converged','optimizer_did_not_converge'),('scale','nonfinite_fit')]:
            def invalid(*args,**kwargs):
                result=original(*args,**kwargs)
                setattr(result,alteration,False if alteration=='converged' else np.nan)
                return result
            with patch.object(feno.GEE,'fit',invalid):fit=feno.fit_contract(fabricated_contract(),'main_12months')
            self.assertFalse(fit.metadata['accepted']);self.assertEqual(fit.metadata['reason'],reason)
            self.assertTrue(fit.estimates.coefficient.isna().all())
            self.assertEqual(fit.metadata['converged'],alteration!='converged')

    def test_naive_positive_definiteness_and_strict_robust_psd_gate(self):
        singular=np.array([[1.,1.],[1.,1.]])
        self.assertTrue(feno._valid_covariance(singular))
        self.assertFalse(feno._valid_covariance(singular,positive_definite=True))
        indefinite=np.array([[1.,1.+1e-10],[1.+1e-10,1.]])
        self.assertFalse(feno._valid_covariance(indefinite))

    def test_logarithm_retains_source_float32_then_expands(self):
        c=fabricated_contract();self.assertEqual(c['frame'].feno.dtype,np.float32)
        correct=np.log(c['frame'].feno).to_numpy(float)
        wrong=np.log(c['frame'].feno.to_numpy(float))
        self.assertGreater(np.max(abs(correct-wrong)),1e-8)
        np.testing.assert_array_equal(c['response'],correct)
        c['response']=wrong
        with self.assertRaisesRegex(ValueError,'log FeNO'):feno.validate_contract(c,'main_12months')

    def test_invalid_ids_dates_design_and_visit_are_rejected(self):
        for column,value in [('pid_child',' '),('pid_child',' padded '),('date',pd.NaT),('visit_label','unexpected'),('feno',0.)]:
            c=fabricated_contract();c['frame'].loc[c['frame'].index[0],column]=value
            with self.subTest(column=column,value=value),self.assertRaises(ValueError):feno.validate_contract(c,'main_12months')
        c=fabricated_contract();c['design'].iloc[0,0]=0
        with self.assertRaises(ValueError):feno.validate_contract(c,'main_12months')

    def test_main_nominal_12_retains_before_birthday_and_supplement_precc_scaling(self):
        c=fabricated_contract();self.assertTrue(c['frame'].age.lt(1).any());feno.validate_contract(c,'main_12months')
        c=fabricated_contract('supplement_12months')
        # Remove a complete-case row after scaling; retain the original z values.
        c['frame']=c['frame'].iloc[1:].copy();c['design']=c['design'].iloc[1:].copy();c['response']=c['response'][1:]
        f=c['frame'];c['audit'].update(observations=len(f),children=f.pid_child.nunique(),tb_before=int(f.tb_before_measurement.sum()),tb_after=int(f.tb_after_measurement.sum()))
        self.assertGreater(abs(f.age_z.mean()),1e-5)
        feno.validate_contract(c,'supplement_12months')

    def test_full_table_inventory_and_input_nonmutation(self):
        contracts={t:fabricated_contract(t) for t in feno.DEFAULT_TARGETS};before=deepcopy(contracts)
        result=feno.fit_selected(contracts)
        self.assertTrue(result['metadata']['accepted'],result['metadata'])
        self.assertEqual(len(result['tables']['estimates']),11);self.assertEqual(len(result['tables']['counts']),33)
        nterms=sum(len(c['design'].columns) for c in contracts.values())
        ncov=sum(2*len(c['design'].columns)**2 for c in contracts.values())
        self.assertEqual(len(result['tables']['coefficients']),nterms);self.assertEqual(len(result['tables']['covariance']),ncov)
        for name,table in result['tables'].items():self.assertEqual(tuple(table.columns),feno.TABLE_COLUMNS[name])
        for target in feno.DEFAULT_TARGETS:
            pd.testing.assert_frame_equal(contracts[target]['frame'],before[target]['frame'])
            pd.testing.assert_frame_equal(contracts[target]['design'],before[target]['design'])
            np.testing.assert_array_equal(contracts[target]['response'],before[target]['response'])
        with self.assertRaises(ValueError):feno.fit_selected({})


if __name__=='__main__':unittest.main()
