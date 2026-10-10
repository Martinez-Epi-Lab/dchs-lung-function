"""Independent target persistence changes only the target likelihood path."""
from copy import deepcopy
import unittest
import numpy as np
from dchs_lung_function.models import spirometry, mmef, x6, sensitivity
from test_spirometry import fabricated_contract as primary_contract
from test_supporting_models import fabricated_contract as supporting_contract, assigned_point
from test_sensitivity_models import fabricated_contract as sensitivity_contract


class TargetRandomEffectTests(unittest.TestCase):
    def test_all_29_graphs_have_separate_target_intercept_and_complete_child_dimensions(self):
        total=0
        for module, factory in [(spirometry,primary_contract),(mmef,supporting_contract),
                                (x6,supporting_contract),(sensitivity,sensitivity_contract)]:
            for target in module.TARGETS:
                with self.subTest(target=target):
                    model=module.build_model(factory(target),target)
                    self.assertIn('tau_target',[v.name for v in model.free_RVs])
                    self.assertIn('target_raw',[v.name for v in model.free_RVs])
                    self.assertIn('target_process',[v.name for v in model.deterministics])
                    self.assertEqual(model.named_vars_to_dims['target_raw'],('child',))
                    self.assertEqual(model.named_vars_to_dims['target_process'],('child',))
                    self.assertEqual(module.fit_spec(target)['specification_version'],'2.0.0')
                    total+=1
        self.assertEqual(total,29)

    def test_target_perturbation_cannot_change_shared_measurements_hazard_or_tb_contrasts(self):
        for module,target,factory in [(spirometry,'ratio_pre',primary_contract),
                (mmef,'mmef_unpaired',supporting_contract),(x6,'x6_z_before1',supporting_contract),
                (sensitivity,'early_all_four',sensitivity_contract)]:
            with self.subTest(target=target):
                contract=factory(target); model=module.build_model(contract,target)
                values,point=assigned_point(model)
                variables=model.free_RVs+model.observed_RVs+model.potentials
                fn=model.compile_logp(sum=False,jacobian=False)
                changed=deepcopy(point);changed['target_raw']=changed['target_raw']+.7
                before={v.name:np.sum(x) for v,x in zip(variables,fn(point))}
                after={v.name:np.sum(x) for v,x in zip(variables,fn(changed))}
                affected='obs_target' if module is x6 else 'obs_fev1_fvc'
                self.assertNotEqual(before[affected],after[affected])
                for name in before:
                    if name not in {affected,'target_raw'}:self.assertEqual(before[name],after[name],name)
                deterministics=[v for v in model.deterministics if v.name!='target_process']
                df=model.compile_fn(model.replace_rvs_by_values(deterministics),inputs=model.value_vars,on_unused_input='ignore')
                for a,b in zip(df(point),df(changed)):np.testing.assert_array_equal(a,b)
                target_fn=model.compile_fn(model.replace_rvs_by_values([model['target_process']]),inputs=model.value_vars,on_unused_input='ignore')
                np.testing.assert_allclose(target_fn(point)[0],values['tau_target']*values['target_raw'],rtol=1e-14,atol=0)

    def test_added_prior_and_target_likelihood_gradients_have_independent_analytic_values(self):
        for module,target,factory in [(spirometry,'ratio_pre',primary_contract),
                (mmef,'mmef_post_paired',supporting_contract),(x6,'x6_native_before5',supporting_contract),
                (sensitivity,'early_maternal_smoking',sensitivity_contract)]:
            with self.subTest(target=target):
                contract=factory(target);model=module.build_model(contract,target);v,p=assigned_point(model)
                frame=contract['frames']['target'];idx=frame.child_index.to_numpy(int)
                if module is x6:
                    mu=v['loading_target_mechanics']*(v['mechanics'][idx]+v['shift_mechanics']*frame.post_target_tb)+v['direct_target']*frame.post_target_tb
                    sigma=v['sigma_target']
                else:
                    change=v['shift_mechanics']*frame.post_before5+v['shift_mechanics_infancy']*frame.post_infancy
                    direct=v['direct_ratio']*frame.post_before5+v['direct_ratio_infancy']*frame.post_infancy
                    if module is spirometry:
                        change+=v['shift_mechanics_late']*frame.post_at_after5
                        direct+=v['direct_ratio_late']*frame.post_at_after5
                    mu=v['loading_ratio_mechanics']*(v['mechanics'][idx]+change)+direct
                    sigma=v['sigma_ratio']
                residual=np.asarray(frame.y_std-mu-v['tau_target']*v['target_raw'][idx])
                raw_gradient=-v['target_raw']+np.bincount(idx,weights=v['tau_target']*residual/sigma**2,minlength=len(v['target_raw']))
                # d/d log(tau): HalfNormal prior, target likelihood, plus log-Jacobian.
                logtau_gradient=1-v['tau_target']**2+np.dot(residual,v['tau_target']*v['target_raw'][idx])/sigma**2
                gradient=model.compile_dlogp(vars=[model['target_raw'],model['tau_target']],jacobian=True)(p)
                np.testing.assert_allclose(gradient,np.r_[raw_gradient,logtau_gradient],rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main()
