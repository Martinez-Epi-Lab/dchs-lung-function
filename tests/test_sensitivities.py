"""Independent fabricated-draw sensitivity summary, selection and attenuation checks."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.analysis import sensitivities as sr, primary_results, wheeze_results
from dchs_lung_function.models import sensitivity as model, spirometry, wheeze
from dchs_lung_function import sensitivity_sampling as sampling
from dchs_lung_function.preparation import joint_inputs
from test_sensitivity_models import fabricated_contract, fabricated_pft
from test_wheeze import fabricated_contract as wheeze_contract
from test_wheeze_results import fabricated_posterior as wheeze_posterior

SETTINGS={'schema_version':1,'chains':4,'draws':3200,'tune':2200,'cores':2,
          'target_accept':.95,'max_treedepth':10,'seed_offset':104729}


def fabricated_posterior(contract=None,target='early_core',*,chains=2,draws=16):
    """Complete arbitrary posterior, with primitive-based deterministic arrays."""
    if contract is None:contract=fabricated_contract(target)
    spec=model.fit_spec(target);n=len(contract['children']);shape=(chains,draws)
    rng=np.random.default_rng(4412)
    result={name:rng.normal(0,.15,shape) for name in spec['posterior_variables']}
    for name in ('mechanics','lci_raw','target_raw'):result[name]=rng.normal(0,.2,(*shape,n))
    for name in ('tau_lci','tau_target','sigma_lci','sigma_ratio'):result[name]=rng.uniform(.5,1.1,shape)
    for name in ('loading_mechanics','sigma_mechanics'):result[name]=rng.uniform(.4,1.2,(*shape,4))
    result['log_baseline_hazard']=rng.normal(-4,.2,(*shape,6 if spec['family']=='window' else 3))
    result['lci_process']=result['tau_lci'][...,None]*result['lci_raw']
    result['target_process']=result['tau_target'][...,None]*result['target_raw']
    # Deliberately correlate loading and shift; product-of-means is incorrect.
    ramp=rng.permutation(np.linspace(-.4,1.,chains*draws)).reshape(shape)
    result['loading_ratio_mechanics']=ramp
    result['shift_mechanics']=ramp*.5-.3
    result['direct_ratio']=ramp*.2-.4
    result['direct_ratio_infancy']=ramp*.1+.1
    result['shift_mechanics_infancy']=-.12+.02*rng.normal(size=shape)
    mid=result['direct_ratio']+ramp*result['shift_mechanics']
    infant=mid+result['direct_ratio_infancy']+ramp*result['shift_mechanics_infancy']
    weights=np.asarray(contract['weights'],float)
    result.update(total_ratio_1to5=mid,total_ratio_infancy=infant,
        total_ratio_lt5_child_weighted=(weights[0]*infant+weights[1]*mid)/sum(weights[:2]),
        total_ratio_infancy_difference=infant-mid)
    if spec['family']=='window':
        late=result['direct_ratio_late']+ramp*result['shift_mechanics_late']
        result.update(total_ratio_late=late,total_ratio_any_child_weighted=(weights[0]*infant+weights[1]*mid+weights[2]*late)/sum(weights),
            total_ratio_late_minus_infancy=late-infant,total_ratio_late_minus_1to5=late-mid)
    return result


def bound_entry(target,estimates,metadata,counts,index):
    return {'estimates':estimates,'metadata':metadata,'summary':{'target':target,'status':'diagnostics_passed',
        **({'covariance_ppc_passed':True} if target not in sr.WHEEZE_TARGETS else {}),
        'diagnostics_passed':True,'counts':counts,'fit_fingerprint':format(index,'064x'),
        'validation_fingerprint':format(100+index,'064x')}}


@lru_cache(maxsize=1)
def _table_inputs():
    new={};primary={};reused_wheeze={}
    for i,target in enumerate(model.DEFAULT_TARGETS,1):
        contract=fabricated_contract(target);posterior=fabricated_posterior(contract,target)
        result=sr.summarize_sensitivity(posterior,contract,target,compute_mcse=False)
        # Selection fixtures supply a declared finite simulation error separately.
        result.estimates['mcse_mean']=.0001
        new[target]=bound_entry(target,result.estimates,result.metadata,model.validate_contract(contract,target),i)
    for i,target in enumerate(('ratio_pre','ratio_post'),11):
        column='fev1_fvc' if target=='ratio_pre' else 'fev1_fvc_post'
        contract=joint_inputs(fabricated_pft(),(column,100.),family='primary',event_end='2007-01-01')
        posterior=fabricated_posterior(contract,'window_'+target)
        result=primary_results.summarize_primary(posterior,contract,target,compute_mcse=False)
        result.estimates['mcse_mean']=.0001
        primary[target]=bound_entry(target,result.estimates,result.metadata,spirometry.validate_contract(contract,target),i)
    for i,target in enumerate(sr.WHEEZE_TARGETS,13):
        contract=wheeze_contract(target);posterior=wheeze_posterior(contract,target)
        result=wheeze_results.summarize_wheeze(posterior,contract,target,ppc_seed=615)
        info=wheeze.validate_contract(contract,target)
        reused_wheeze[target]=bound_entry(target,result.effects,deepcopy(info),info,i)
    return new,primary,reused_wheeze


def table_inputs():return deepcopy(_table_inputs())


class SensitivitySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts={t:fabricated_contract(t) for t in model.DEFAULT_TARGETS}
        cls.posteriors={t:fabricated_posterior(c,t) for t,c in cls.contracts.items()}

    def test_all_ten_models_keep_every_effect_hazard_and_support_cell(self):
        effects=hazards=0
        for target in model.DEFAULT_TARGETS:
            result=sr.summarize_sensitivity(self.posteriors[target],self.contracts[target],target,compute_mcse=False)
            spec=model.fit_spec(target);effects+=len(result.estimates);hazards+=len(result.hazard)
            self.assertEqual(len(result.estimates),8 if target.startswith('window_') else 6)
            self.assertEqual(len(result.hazard),0 if target.startswith('window_') else 2+len(spec['hazard_covariates']))
            for name,columns in sr.FIT_TABLE_COLUMNS.items():self.assertEqual(list(getattr(result,name)),columns)
            self.assertEqual(result.metadata['target'],target);self.assertEqual(result.metadata['joint_children'],24)
            for frame in (result.estimates,result.hazard,result.sample_counts):self.assertNotIn('pid_child',frame)
            self.assertTrue(result.sample_counts.children.between(1,9).any())
        self.assertEqual((effects,hazards),(64,25))

    def test_early_primitive_algebra_native_scale_and_unique_child_weights(self):
        c,p=self.contracts['early_core'],self.posteriors['early_core'];d=sr.derive_contrast_draws(p,c,'early_core')
        scale=c['audit']['outcomes']['target']['scale'];a=p['loading_ratio_mechanics']
        mid=(p['direct_ratio']+a*p['shift_mechanics'])*scale
        infancy=(p['direct_ratio']+p['direct_ratio_infancy']+a*(p['shift_mechanics']+p['shift_mechanics_infancy']))*scale
        np.testing.assert_allclose(d['First TB 1-<5 years'],mid,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(d['First TB <1 year'],infancy,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(d['First TB <5 years, child-weighted'],(2*infancy+3*mid)/5)
        np.testing.assert_allclose(d['Infancy minus age 1-<5 years'],infancy-mid)
        np.testing.assert_array_equal(d['Direct component: first TB 1-<5 years'],p['direct_ratio']*scale)
        self.assertNotAlmostEqual(mid.mean(),scale*(p['direct_ratio'].mean()+a.mean()*p['shift_mechanics'].mean()))
        self.assertNotAlmostEqual(mid.mean(),100*scale*(p['direct_ratio']+a*p['shift_mechanics']).mean())

    def test_window_late_and_any_tb_effects_remain_separate(self):
        c,p=self.contracts['window_ratio_post'],self.posteriors['window_ratio_post'];d=sr.derive_contrast_draws(p,c,'window_ratio_post')
        scale=c['audit']['outcomes']['target']['scale']
        late=(p['direct_ratio_late']+p['loading_ratio_mechanics']*p['shift_mechanics_late'])*scale
        np.testing.assert_allclose(d['TB >=5'],late)
        np.testing.assert_allclose(d['Any TB weighted'],(2*d['TB <1']+3*d['TB 1-<5']+2*late)/7)
        np.testing.assert_allclose(d['Later minus infancy'],late-d['TB <1'])
        np.testing.assert_allclose(d['Later minus 1-<5'],late-d['TB 1-<5'])

    def test_summaries_use_all_draws_linear_intervals_and_strict_sign_probabilities(self):
        c,p=self.contracts['early_core'],self.posteriors['early_core']
        result=sr.summarize_sensitivity(p,c,'early_core',compute_mcse=False)
        for label,draws in sr.derive_contrast_draws(p,c,'early_core').items():
            row=result.estimates.set_index('contrast').loc[label]
            np.testing.assert_allclose(row[['mean','median','lower','upper']].to_numpy(float),[draws.mean(),np.median(draws),*np.quantile(draws,[.025,.975])])
            self.assertEqual(row.probability_below_zero,float((draws<0).mean()))
            self.assertEqual(row.probability_above_zero,float((draws>0).mean()))
            self.assertEqual(row.probability_equal_zero,float((draws==0).mean()))
            self.assertTrue(np.isnan(row.mcse_mean))
        self.assertFalse(result.metadata['mcse_computed'])

    def test_hazard_ratios_are_unscaled_and_actual_covariate_names_preserved(self):
        for target in ('early_core','early_maternal_smoking','early_all_four','window_ratio_pre'):
            posterior=self.posteriors[target];result=sr.derive_hazard_draws(posterior,target)
            for variable,label in model.fit_spec(target)['hazard_variables'].items():
                np.testing.assert_array_equal(result[label],np.exp(posterior[variable]))
            self.assertFalse(any('LRTI' in label for label in result))
        self.assertEqual(sr.derive_hazard_draws(self.posteriors['window_ratio_pre'],'window_ratio_pre'),{})
        bad=deepcopy(self.posteriors['early_core']);bad['hazard_male'][:]=1000
        with self.assertRaises(ValueError):sr.derive_hazard_draws(bad,'early_core')
        bad['hazard_male'][:]=-1000
        with self.assertRaises(ValueError):sr.derive_hazard_draws(bad,'early_core')

    def test_every_stored_deterministic_and_lci_transform_is_checked(self):
        for target in ('early_core','window_ratio_pre'):
            variables={pair[0] if isinstance(pair,(list,tuple)) else pair for pair in model.fit_spec(target)['contrasts'].values()}|{'lci_process'}
            # Direct components are primitives rather than stored derived contrasts.
            variables-={'direct_ratio','direct_ratio_infancy'}
            for name in variables:
                bad=deepcopy(self.posteriors[target]);bad[name]+=.01
                with self.subTest(target=target,name=name),self.assertRaises(ValueError):sr.derive_contrast_draws(bad,self.contracts[target],target)

    def test_complete_inventory_and_every_coordinate_dimension_required(self):
        c,p=self.contracts['early_core'],self.posteriors['early_core']
        for name in p:
            for mode in ('missing','wrong_shape','nonfinite'):
                bad=deepcopy(p)
                if mode=='missing':bad.pop(name)
                elif mode=='wrong_shape':bad[name]=bad[name][...,:-1] if bad[name].ndim>2 else bad[name][:,:-1]
                else:bad[name].flat[0]=np.nan
                with self.subTest(name=name,mode=mode),self.assertRaises(ValueError):sr.summarize_sensitivity(bad,c,'early_core',compute_mcse=False)
        bad=deepcopy(p);bad['unselected_variable']=np.ones((2,16))
        with self.assertRaises(ValueError):sr.validate_posterior(bad,c,'early_core')

    def test_non_numeric_complex_boolean_and_object_arrays_have_controlled_errors(self):
        c,p=self.contracts['early_core'],self.posteriors['early_core']
        for value in (np.full((2,16),'FABRICATED_PRIVATE_RECORD'),np.ones((2,16),complex),np.ones((2,16),bool),np.ones((2,16),object)):
            bad=deepcopy(p);bad['sigma_lci']=value
            with self.subTest(dtype=str(value.dtype)),self.assertRaises(ValueError) as error:sr.validate_posterior(bad,c,'early_core')
            self.assertNotIn('FABRICATED_PRIVATE_RECORD',str(error.exception))

    def test_positive_scales_required_but_unconstrained_target_loading_can_be_negative(self):
        c,p=self.contracts['early_core'],self.posteriors['early_core']
        self.assertTrue((p['loading_ratio_mechanics']<0).any());sr.validate_posterior(p,c,'early_core')
        for name in ('tau_lci','loading_mechanics','sigma_mechanics','sigma_lci','sigma_ratio'):
            for value in (0.,-.1):
                bad=deepcopy(p);bad[name].flat[0]=value
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):sr.validate_posterior(bad,c,'early_core')

    def test_contract_weight_and_residual_scale_contradictions_rejected(self):
        for changed in ('weight','scale','target'):
            c=deepcopy(self.contracts['early_core']);p=self.posteriors['early_core']
            if changed=='weight':c['weights'][0]+=1
            elif changed=='scale':c['audit']['outcomes']['target']['scale']*=100
            else:c['audit']['target']='fev1_fvc_post'
            with self.subTest(changed=changed),self.assertRaises(ValueError):sr.summarize_sensitivity(p,c,'early_core',compute_mcse=False)


class AttenuationTests(unittest.TestCase):
    def test_absolute_posterior_mean_ratio_preserves_negative_and_sign_reversal(self):
        for adjusted,core,expected in [(-2.,-4.,50.),(2.,-4.,50.),(-6.,-4.,-50.),(0.,-4.,100.),(4.,4.,0.)]:
            status='zero_adjusted_mean' if adjusted==0 else 'sign_reversal' if np.sign(adjusted)!=np.sign(core) else 'estimated'
            self.assertEqual(sr.attenuation(adjusted,core),(expected,status))

    def test_exact_zero_core_has_no_epsilon_and_no_posterior_interval(self):
        for adjusted in (0.,1.,-1.):
            result,status=sr.attenuation(adjusted,0.)
            self.assertTrue(np.isnan(result));self.assertEqual(status,'undefined_zero_core')
        self.assertEqual(sr.attenuation(1e-100,2e-100),(50.,'estimated'))

    def test_nonfinite_malformed_boolean_and_overflow_are_rejected_cleanly(self):
        for adjusted,core in [(np.inf,1.),(1.,np.nan),(True,1.),(1.,False),('FABRICATED_PRIVATE_RECORD',1.),(None,1.),([1.],1.),(1e308,1e-308)]:
            with self.subTest(adjusted=type(adjusted).__name__),self.assertRaises(ValueError) as error:sr.attenuation(adjusted,core)
            self.assertNotIn('FABRICATED_PRIVATE_RECORD',str(error.exception))


class SensitivityTableTests(unittest.TestCase):
    def setUp(self):self.new,self.primary,self.wheeze=table_inputs()
    def build(self):return sr.build_tables(self.new,self.primary,self.wheeze)

    def test_exact_table_rows_source_order_units_and_reused_fingerprints(self):
        result=self.build()
        for name,count in [('table_s1',10),('table_s3',24),('table_s4',12)]:
            frame=getattr(result,name);self.assertEqual(len(frame),count);self.assertEqual(list(frame),sr.TABLE_COLUMNS[name]);self.assertNotIn('pid_child',frame)
        self.assertEqual(result.table_s1.target.tolist(),[t for t in sr.S1_TARGETS for _ in range(2)])
        self.assertEqual(result.table_s3.target.tolist(),[t for t in sr.S3_TARGETS for _ in range(2)])
        self.assertEqual(result.table_s4.scenario.tolist(),[t for t in sr.WHEEZE_TARGETS for _ in range(6)])
        for row in result.table_s3.itertuples():
            source=(self.new|self.primary)[row.target]['summary']
            self.assertEqual(row.fit_fingerprint,source['fit_fingerprint']);self.assertEqual(row.validation_fingerprint,source['validation_fingerprint'])
            self.assertEqual(row.unit,'percentage points');self.assertEqual(row.joint_children,24)
        self.assertEqual(result.metadata['reused_fit_count'],4);self.assertFalse(result.metadata['rendered_manuscript'])

    def test_s1_attenuation_uses_same_contrast_means_and_keeps_eight_adjusted_cells(self):
        result=self.build();s1=result.table_s1
        for row in s1.itertuples():
            if row.target=='early_core':self.assertTrue(np.isnan(row.attenuation_percent));self.assertEqual(row.attenuation_status,'reference')
            else:
                core=s1.loc[s1.target.eq('early_core')&s1.contrast.eq(row.contrast),'mean'].iloc[0]
                self.assertAlmostEqual(row.attenuation_percent,100*(1-abs(row.mean)/abs(core)))
                self.assertEqual(row.attenuation_status,'estimated')
        self.assertEqual(s1.attenuation_percent.notna().sum(),8)
        self.assertTrue(s1.attenuation_precision_passed.all())
        self.assertTrue(result.metadata['attenuation_precision_passed'])
        for row in s1.loc[~s1.target.eq('early_core')].itertuples():
            core=s1.loc[s1.target.eq('early_core') & s1.contrast.eq(row.contrast)].iloc[0]
            expected=100*row.mcse_mean/abs(core['mean'])+100*abs(row.mean)*core['mcse_mean']/abs(core['mean'])**2
            self.assertAlmostEqual(row.attenuation_mcse_bound_pp,expected)

    def test_s1_exact_zero_core_retains_all_rows_with_explicit_undefined_status(self):
        self.new['early_core']['estimates'].loc[:,'mean']=0.
        result=self.build();adjusted=result.table_s1[~result.table_s1.target.eq('early_core')]
        self.assertTrue(adjusted.attenuation_percent.isna().all());self.assertTrue(adjusted.attenuation_status.eq('undefined_zero_core').all())
        self.assertFalse(result.metadata['attenuation_precision_passed'])

    def test_mcse_required_and_excess_precision_is_flagged_without_suppressing_estimates(self):
        self.new['early_maternal_hiv']['estimates']['mcse_mean']=1.
        result=self.build()
        self.assertFalse(result.metadata['attenuation_precision_passed'])
        self.assertEqual(len(result.table_s1),10)
        self.assertEqual(result.table_s1.attenuation_percent.notna().sum(),8)
        self.new['early_core']['estimates']['mcse_mean']=np.nan
        with self.assertRaises(ValueError):self.build()

    def test_missing_extra_model_and_reused_fit_selection_are_rejected(self):
        for location in ('new','primary','wheeze'):
            for mode in ('missing','extra'):
                self.new,self.primary,self.wheeze=table_inputs();values=getattr(self,location)
                if mode=='missing':values.pop(next(iter(values)))
                else:values['unselected']=deepcopy(next(iter(values.values())))
                with self.subTest(location=location,mode=mode),self.assertRaises(ValueError):self.build()

    def test_failed_status_malformed_identity_or_target_cannot_be_reused(self):
        for location,target in [('new','early_core'),('primary','ratio_pre'),('wheeze','primary_sex')]:
            for field,bad in [('diagnostics_passed',False),('diagnostics_passed',1),('status','failed'),('fit_fingerprint','bad'),('validation_fingerprint','g'*64),('target','wrong')]:
                self.new,self.primary,self.wheeze=table_inputs();getattr(self,location)[target]['summary'][field]=bad
                with self.subTest(location=location,field=field),self.assertRaises(ValueError):self.build()

    def test_nonpassing_or_missing_covariance_adequacy_cannot_enter_selected_tables(self):
        for location,target in [('new','early_core'),('primary','ratio_pre')]:
            for value in (False,1,None):
                self.new,self.primary,self.wheeze=table_inputs()
                getattr(self,location)[target]['summary']['covariance_ppc_passed']=value
                with self.subTest(location=location,value=value),self.assertRaises(ValueError):self.build()

    def test_duplicate_or_wrong_spirometry_selector_and_units_rejected(self):
        for mode in ('duplicate','missing','unit','interval','nonfinite','target','outcome'):
            self.new,self.primary,self.wheeze=table_inputs();entry=self.new['early_core'];frame=entry['estimates']
            if mode=='duplicate':entry['estimates']=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
            elif mode=='missing':entry['estimates']=frame.iloc[1:]
            elif mode=='unit':frame.loc[0,'unit']='ratio'
            elif mode=='interval':frame.loc[0,'lower']=frame.loc[0,'upper']+1
            elif mode=='nonfinite':frame.loc[0,'mean']=np.nan
            elif mode=='target':frame.loc[0,'target']='early_all_baseline'
            else:frame.loc[0,'outcome']='Post-BD FEV1/FVC'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.build()

    def test_metadata_counts_and_model_semantics_must_match_completion(self):
        mutations=[lambda e:e['metadata'].__setitem__('target','wrong'),lambda e:e['metadata'].__setitem__('joint_children',999),
            lambda e:e['metadata'].__setitem__('observation_window','restricted'),lambda e:e['summary']['counts'].__setitem__('joint_children',-1),
            lambda e:e['summary']['counts']['process_counts']['target'].__setitem__('n_children',True),
            lambda e:e['summary']['counts']['process_counts']['target'].__setitem__('n_observations',1.5)]
        for i,mutate in enumerate(mutations):
            self.new,self.primary,self.wheeze=table_inputs();mutate(self.new['early_core'])
            with self.subTest(i=i),self.assertRaises(ValueError):self.build()

    def test_wheeze_exact_conditional_cells_probabilities_intervals_and_labels(self):
        for mode in ('duplicate','missing','wrong_band','wrong_process','wrong_exposure','rr_substitution','nonfinite','negative','interval','median_outside'):
            self.new,self.primary,self.wheeze=table_inputs();entry=self.wheeze['primary_sex'];frame=entry['estimates']
            if mode=='duplicate':entry['estimates']=pd.concat([frame.iloc[:5],frame.iloc[:1]],ignore_index=True)
            elif mode=='missing':entry['estimates']=frame.iloc[:-1]
            elif mode=='wrong_band':frame.loc[0,'age_band']='0-<1 year'
            elif mode=='wrong_process':frame.loc[0,'process']='wrong'
            elif mode=='wrong_exposure':frame.loc[0,'exposure']='wrong'
            elif mode=='rr_substitution':frame.loc[0,'measure']='standardized risk ratio'
            elif mode=='nonfinite':frame.loc[0,'cri_2_5']=np.nan
            elif mode=='negative':frame.loc[0,'posterior_median']=-1.
            elif mode=='interval':frame.loc[0,'cri_2_5']=frame.loc[0,'cri_97_5']+1.
            else:frame.loc[0,'posterior_median']=frame.loc[0,'cri_97_5']+1.
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.build()

    def test_strict_input_schema_and_free_text_do_not_export_private_records(self):
        for location in ('extra_column','process','exposure','estimand','metadata','counts'):
            self.new,self.primary,self.wheeze=table_inputs();entry=self.wheeze['primary_sex'];secret='FABRICATED_PRIVATE_RECORD'
            if location=='extra_column':entry['estimates']['pid_child']=secret
            elif location=='metadata':entry['metadata']['participant_records']=[secret]
            elif location=='counts':entry['summary']['counts']['joint_children']=secret
            else:entry['estimates'].loc[0,location]=secret
            with self.subTest(location=location):
                with self.assertRaises(ValueError) as error:self.build()
                self.assertNotIn(secret,str(error.exception))

    def test_malformed_nested_inputs_raise_sanitized_value_errors(self):
        for mode in ('entry','summary','metadata','counts','effects'):
            self.new,self.primary,self.wheeze=table_inputs()
            if mode=='entry':self.new['early_core']=None
            elif mode=='summary':self.new['early_core']['summary']=[]
            elif mode=='metadata':self.new['early_core']['metadata']=[]
            elif mode=='counts':self.new['early_core']['summary']['counts']=None
            else:self.wheeze['primary_sex']['estimates']=[]
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.build()


class SensitivitySamplingControlTests(unittest.TestCase):
    def test_frozen_default_protocol_and_extension_preserve_seeds_and_initialization(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/sensitivity_sampling.json').read_text())
        self.assertEqual({k:v for k,v in config.items() if k!='draws_by_target'},SETTINGS)
        self.assertEqual(config['draws_by_target'],{t:12800 for t in sr.S1_TARGETS})
        for target in model.DEFAULT_TARGETS:
            selected=sampling.resolved_sampling(target,config)
            expected_seed=20365542 if target.startswith('early_') else 2026197029 if target=='window_ratio_pre' else 2026197031
            self.assertEqual(selected['random_seed'],expected_seed);self.assertEqual(selected['init'],'adapt_diag')
            self.assertEqual(selected['initvals'],[{'loading_ratio_mechanics':v,'loading_mechanics':[v]*4,'sigma_ratio':.9} for v in [.25,.35,.45,.55]])
            self.assertEqual(selected['draws'],12800 if target in sr.S1_TARGETS else 3200)
            extension=sampling.resolved_sampling(target,{**config,'draws_by_target':{t:25600 for t in sr.S1_TARGETS}})
            self.assertEqual(extension['draws'],25600 if target in sr.S1_TARGETS else 3200)
            self.assertEqual({k:v for k,v in extension.items() if k!='draws'},{k:v for k,v in selected.items() if k!='draws'})
            self.assertTrue(selected['discard_tuned_samples']);self.assertFalse(selected['progressbar'])
            self.assertEqual(selected['nuts_sampler'],'pymc')
        self.assertEqual({k:v for k,v in config.items() if k!='draws_by_target'},SETTINGS)
        self.assertEqual(config['draws_by_target'],{t:12800 for t in sr.S1_TARGETS})

    def test_invalid_controls_boolean_integers_nonfinite_and_extra_keys_rejected(self):
        cases=[{**SETTINGS,k:v} for k,v in [('chains',5),('chains',True),('draws',3199),('draws',True),('tune',2199),('cores',0),('target_accept',np.nan),('max_treedepth',9),('seed_offset',-1)]]
        cases.extend([{**SETTINGS,'unselected':1},{k:v for k,v in SETTINGS.items() if k!='cores'},None])
        for case in cases:
            with self.subTest(case=str(type(case))),self.assertRaises(ValueError):sampling.validate_sampling(case)

    def test_control_copies_do_not_alias_each_other_or_caller(self):
        settings=deepcopy(SETTINGS);one=sampling.resolved_sampling('early_core',settings)
        one['initvals'][0]['loading_mechanics'][0]=999
        two=sampling.resolved_sampling('early_core',settings)
        self.assertEqual(two['initvals'][0]['loading_mechanics'],[.25]*4)
        self.assertEqual(one['initvals'][1]['loading_mechanics'],[.35]*4);self.assertEqual(settings,SETTINGS)
        with self.assertRaises(ValueError):sampling.resolved_sampling('unselected',SETTINGS)


if __name__=='__main__':unittest.main()
