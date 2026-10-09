"""Independent mathematical and adversarial checks on fabricated supporting data."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.preparation import joint_inputs
from dchs_lung_function.supporting_sampling import (
    DEFAULT_TARGETS,build_model,contract_key,expected_posterior_shapes,fit_spec,
    resolved_sampling,validate_contract,validate_sampling,
)
from dchs_lung_function.analysis.supporting_results import (
    derive_supporting_contrast_draws,derive_hazard_draws,summarize_supporting,
)


def fabricated_pft():
    rng=np.random.default_rng(8851);rows=[]
    tb_ages=[.4,.8,1.,2.,5.,6.,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    for child,tb_age in enumerate(tb_ages):
        dob=pd.Timestamp('2000-01-01')+pd.Timedelta(days=child*2)
        dx=dob+pd.Timedelta(days=round(tb_age*365.25)) if np.isfinite(tb_age) else pd.NaT
        for j,age in enumerate([.15,.6,1.5,4.,6.,9.]):
            date=dob+pd.Timedelta(days=round(age*365.25));noise=rng.normal()
            row={'pid_child':f'fabricated_{child:02}','child_dob':dob,'male':float(child%2),
                 'tb_first_dx':dx,'tb_age_first_years':tb_age,'visit_label':f'visit_{j}',
                 'height':55+age*7+child*.2+rng.normal(),
                 'date_of_test':date,'date_of_test_ilf1':date,'date_of_test_ilf2':date,
                 'mmef':1.+.1*age+.2*noise,'mmef_post':1.2+.12*age+.3*noise,
                 'i_lci_2_5':7.+.3*noise,'resistance':9.-.3*age+.8*noise,
                 'compliance':.01+.001*age+.002*noise,'xee':-3.+.03*age+.3*noise,
                 'ree':8.-.3*age+.7*noise,'x6_sa_z':-.2+.04*age+.8*noise,
                 'reactance_6hz':-2.5+.1*age+.6*noise}
            row['mmef_pre_paired']=row['mmef'];row['mmef_post_paired']=row['mmef_post']
            if child==11 and j==5:
                row['mmef_post']=row['mmef_pre_paired']=row['mmef_post_paired']=np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def fabricated_contract(target='mmef_unpaired'):
    s=fit_spec(target)
    return joint_inputs(fabricated_pft(), s['column'] if s['family']=='x6' else (s['column'],1.),
                        family='x6' if s['family']=='x6' else 'early',threshold=s['threshold'])


def normlog(value,mean,sd):
    return -.5*np.log(2*np.pi)-np.log(sd)-.5*((np.asarray(value)-mean)/sd)**2


def assigned_point(model):
    values={}
    for name,p in model.initial_point().items():
        shape=np.shape(p)
        if name.endswith('_log__'):
            values[name[:-6]]=np.full(shape,.8)
        elif name=='log_baseline_hazard':
            values[name]=np.log(np.linspace(.02,.04,np.size(p))).reshape(shape)
        elif shape:
            values[name]=np.linspace(-.3,.3,np.size(p)).reshape(shape)
        else:values[name]=np.asarray(.2)
    overrides={'tau_lci':1.3,'loading_ratio_mechanics':.6,'loading_target_mechanics':.6,
               'shift_mechanics':.1,'shift_mechanics_infancy':-.08,'direct_ratio':.2,
               'direct_ratio_infancy':-.12,'direct_target':.2,'assoc_mechanics':.35,'hazard_male':-.15}
    for name,value in overrides.items():
        if name in values:values[name]=np.asarray(value)
    point={name:(np.log(values[name[:-6]]) if name.endswith('_log__') else values[name]) for name in model.initial_point()}
    return values,point


def independent_factors(contract,target,values):
    s=fit_spec(target);v=values;expected={}
    for category,names in s['priors'].items():
        for name in names:
            sd=.75 if category=='normal_sd0_75' else 1.
            expected[name]=float(np.sum(normlog(v[name],0.,sd)+(np.log(2.) if category=='half_normal_sd1' else 0.)))
    expected['log_baseline_hazard']=float(np.sum(normlog(v['log_baseline_hazard'],s['log_hazard_prior_mean'],1.25)))
    for j,key in enumerate(s['mechanics_indicators']):
        f=contract['frames'][key];current=v['mechanics'][f.child_index]+v['shift_mechanics']*f.post_any
        if s['family']=='mmef':current=current+v['shift_mechanics_infancy']*f.post_infancy
        expected['obs_'+key]=float(np.sum(normlog(f.y_std,v['loading_mechanics'][j]*current,v['sigma_mechanics'][j])))
    f=contract['frames']['target'];current=v['mechanics'][f.child_index]+v['shift_mechanics']*f.post_any
    if s['family']=='mmef':
        current=current+v['shift_mechanics_infancy']*f.post_infancy
        mu=v['loading_ratio_mechanics']*current+v['direct_ratio']*f.post_any+v['direct_ratio_infancy']*f.post_infancy
        expected['obs_fev1_fvc']=float(np.sum(normlog(f.y_std,mu,v['sigma_ratio'])))
        f=contract['frames']['lci'];mu=v['tau_lci']*v['lci_raw'][f.child_index]+v['shift_lci']*f.post_any+v['shift_lci_infancy']*f.post_infancy
        expected['obs_lci']=float(np.sum(normlog(f.y_std,mu,v['sigma_lci'])))
    else:
        mu=v['loading_target_mechanics']*current+v['direct_target']*f.post_any
        expected['obs_target']=float(np.sum(normlog(f.y_std,mu,v['sigma_target'])))
    loghaz=v['log_baseline_hazard'][None,:]+(v['assoc_mechanics']*v['mechanics']+v['hazard_male']*contract['children'].male.to_numpy())[:,None]
    expected['first_tb_loglik']=float(np.sum(contract['event']['event_interval']*loghaz-contract['event']['exposure']*np.exp(loghaz)))
    return expected


class SupportingContractTests(unittest.TestCase):
    def test_selected_inventory_model_specific_samples_and_sampling_protocol(self):
        self.assertEqual(len(DEFAULT_TARGETS),7)
        for target in DEFAULT_TARGETS:
            c=fabricated_contract(target);v=validate_contract(c,target);s=fit_spec(target)
            self.assertEqual(s['contract_key'],contract_key(target));self.assertGreater(v['joint_children'],v['n_exposed_target_children'])
            self.assertEqual(v['tb_events'],2 if s['threshold']==1 else 4)
            json.dumps(v,allow_nan=False);json.dumps(s,allow_nan=False)
        unpaired=validate_contract(fabricated_contract(),'mmef_unpaired')
        paired=validate_contract(fabricated_contract('mmef_pre_paired'),'mmef_pre_paired')
        self.assertEqual(unpaired['process_counts']['target']['n_observations']-paired['process_counts']['target']['n_observations'],1)
        config=json.loads((Path(__file__).resolve().parents[1]/'config/supporting_sampling.json').read_text())
        self.assertEqual(validate_sampling(config),config)
        for target in DEFAULT_TARGETS:
            resolved=resolved_sampling(target,config)
            self.assertEqual(resolved['draws'],config['draws']);self.assertEqual(resolved['tune'],config['tune'])
            self.assertEqual(resolved['random_seed'],fit_spec(target)['sampling']['random_seed']+104729)
        for key,value in [('draws',1200),('tune',1800),('target_accept',.94),('chains',3)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_sampling({**config,key:value})
        with self.assertRaises(ValueError):fit_spec('x6_z_before3')

    def test_invalid_contract_linkage_cutoff_scale_and_design_rejected(self):
        def change_frame(c,name,value):
            frame=c['frames']['target']
            frame[name]=value
        def change_design(c):
            array=c['post_diagnosis_designs']['target'].copy()
            array[0,0]=1
            c['post_diagnosis_designs']['target']=array
        mutations=[lambda c:c['audit'].__setitem__('family','primary'),
                   lambda c:change_frame(c,'child_index',999),
                   lambda c:change_frame(c,'y_std',np.nan),
                   lambda c:c['audit']['outcomes']['target'].__setitem__('scale',-1),
                   lambda c:c['event']['event_interval'].__setitem__((0,0),0),
                   lambda c:c['weights'].__setitem__(0,99),
                   change_design]
        for mutation in mutations:
            c=fabricated_contract();mutation(c)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_contract(c,'mmef_unpaired')
        c=fabricated_contract('x6_z_before1')
        c['frames']['target'].iloc[0,c['frames']['target'].columns.get_loc('age_years')]=2.999
        with self.assertRaises(ValueError):validate_contract(c,'x6_z_before1')
        with self.assertRaises(ValueError):validate_contract(fabricated_contract(),'mmef_pre_paired')

    def test_cutoff_boundary_censored_and_future_observations_retained(self):
        c=fabricated_contract('x6_z_before1');child=c['children'].tb_age_first_years.eq(1.).to_numpy()
        self.assertEqual(c['event']['event'][child].sum(),0)
        np.testing.assert_array_equal(c['event']['stop'][child],[1.])
        c=fabricated_contract();f=c['frames']['target']
        self.assertTrue(f.loc[f.tb_age_first_years.ge(5),'post_any'].eq(0).all())
        self.assertGreater(f.future_tb.sum(),0)
        c['event']['event'][c['children'].tb_age_first_years.eq(5.).to_numpy()]=1
        with self.assertRaises(ValueError):validate_contract(c,'mmef_unpaired')


class SupportingGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples={}
        for target in ['mmef_unpaired','x6_z_before1','x6_native_before5']:
            c=fabricated_contract(target);m=build_model(c,target);values,point=assigned_point(m)
            cls.examples[target]=(c,m,values,point,m.compile_logp(sum=False,jacobian=False))

    def test_every_prior_and_likelihood_against_independent_numpy_equations(self):
        for target,(c,m,v,p,fn) in self.examples.items():
            actual={rv.name:float(np.sum(value)) for rv,value in zip(m.free_RVs+m.observed_RVs+m.potentials,fn(p))}
            expected=independent_factors(c,target,v)
            self.assertEqual(set(actual),set(expected))
            for name,value in expected.items():
                with self.subTest(target=target,factor=name):self.assertAlmostEqual(actual[name],value,places=7)

    def test_hazard_gradient_and_no_lci_path(self):
        for target,(c,m,v,p,fn) in self.examples.items():
            loghaz=v['log_baseline_hazard'][None,:]+(v['assoc_mechanics']*v['mechanics']+v['hazard_male']*c['children'].male.to_numpy())[:,None]
            expected=-v['assoc_mechanics']/.75**2+np.sum((c['event']['event_interval']-c['event']['exposure']*np.exp(loghaz))*v['mechanics'][:,None])
            actual=m.compile_dlogp(vars=[m['assoc_mechanics']],jacobian=False)(p)[0]
            self.assertAlmostEqual(actual,expected,places=11)
        c,m,v,p,fn=self.examples['mmef_unpaired'];p2={**p,'lci_raw':p['lci_raw']+1.2}
        names=[a.name for a in m.free_RVs+m.observed_RVs+m.potentials]
        before=dict(zip(names,fn(p)));after=dict(zip(names,fn(p2)))
        for name in ['obs_fev1_fvc','first_tb_loglik','obs_resistance']:
            np.testing.assert_array_equal(before[name],after[name])
        self.assertNotAlmostEqual(float(np.sum(before['obs_lci'])),float(np.sum(after['obs_lci'])))

    def test_positive_priors_have_correct_jacobians(self):
        for target,(c,m,v,p,fn) in self.examples.items():
            actual=float(m.compile_logp(jacobian=True)(p))-float(m.compile_logp(jacobian=False)(p))
            expected=sum(float(np.sum(value)) for name,value in p.items() if name.endswith('_log__'))
            self.assertAlmostEqual(actual,expected,places=10)

    def test_full_inventory_coordinate_dimensions_and_no_input_mutation(self):
        for target,(c,m,v,p,fn) in self.examples.items():
            before=deepcopy(c);spec=fit_spec(target);shapes=expected_posterior_shapes(c,target,4,1600)
            self.assertEqual(set(shapes),set(spec['posterior_variables']))
            self.assertEqual(len(shapes),22 if spec['family']=='mmef' else 12)
            self.assertEqual(shapes['mechanics'],(4,1600,len(c['children'])))
            self.assertEqual(shapes['log_baseline_hazard'],(4,1600,len(spec['hazard_breaks'])-1))
            self.assertEqual(list(m.coords['hazard_interval']),spec['hazard_labels'])
            for key in c['frames']:pd.testing.assert_frame_equal(c['frames'][key],before['frames'][key])
            np.testing.assert_array_equal(c['event']['exposure'],before['event']['exposure'])

    def test_weighted_contrasts_native_units_hazards_and_result_inventory(self):
        for target,(c,m,v,p,fn) in self.examples.items():
            det=m.compile_fn(m.replace_rvs_by_values(m.deterministics),inputs=m.value_vars,on_unused_input='ignore')(p)
            posterior={name:np.full((4,30),float(value)) for name,value in v.items() if value.ndim==0}
            posterior.update({rv.name:np.full((4,30),float(value)) for rv,value in zip(m.deterministics,det) if np.ndim(value)==0})
            draws=derive_supporting_contrast_draws(posterior,c,target);s=fit_spec(target);scale=c['audit']['outcomes']['target']['scale']
            if s['family']=='mmef':
                middle=.26;infant=.092;w=c['weights'][0]/sum(c['weights'][:2])
                expected=[w*infant+(1-w)*middle,infant,middle,infant-middle,.2,-.12]
                hazards=derive_hazard_draws(posterior,target)
                np.testing.assert_allclose(hazards['Higher pre-TB mechanics factor'],np.exp(.35),rtol=0,atol=1e-14)
            else:expected=[.26,.2,.06]
            for value,e in zip(draws.values(),expected):np.testing.assert_allclose(value,e*scale,rtol=0,atol=1e-14)
            result=summarize_supporting(posterior,c,target,compute_mcse=False)
            self.assertEqual(len(result.estimates),s['result_rows']);self.assertEqual(len(result.hazard),s['hazard_rows'])
            self.assertEqual(set(result.sample_counts.process),set(c['frames']))
            altered=dict(posterior);variable=s['contrasts'][next(iter(s['contrasts']))][0];altered[variable]=altered[variable]+.1
            with self.assertRaises(ValueError):derive_supporting_contrast_draws(altered,c,target)


if __name__=='__main__':unittest.main()
