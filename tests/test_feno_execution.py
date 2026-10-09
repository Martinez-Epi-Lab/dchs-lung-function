"""Synthetic M11 publication, provenance, cache and cumulative-CLI checks.

Invented repeated-child contracts drive the actual GEE once at class setup;
execution tests mock that boundary but use real files, receipts and locks.
"""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
import json
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function import feno_execution as fe, wheeze_execution, pipeline
from dchs_lung_function.models import feno as model
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from test_feno import fabricated_contract


def contracts():
    return {t:fabricated_contract(t) for t in model.DEFAULT_TARGETS}


def row_counts(values):
    widths=[len(c['design'].columns) for c in values.values()]
    return {'estimates':11,'coefficients':sum(widths),'covariance':sum(2*p*p for p in widths),'counts':33}


class FenoDescriptorTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.prep=Path(temp.name);self.contracts=contracts()

    def prepared(self):
        tree={}
        for target,value in self.contracts.items():
            path=model.artifact_path(target);node=tree
            for part in path[:-1]:node=node.setdefault(part,{})
            node[path[-1]]=value
        manifest=ArtifactWriter(self.prep).finish(tree)
        identity={'through':'M03','fixture':'invented'}
        write_json(self.prep/'receipt.json',{'through':'M03','status':'prepared','identity':identity,
            'fingerprint':fe.digest(identity),'artifact_manifest_sha256':file_hash(self.prep/'artifacts.json')})
        return manifest

    def test_all_eleven_contracts_bound_to_their_own_typed_artifact_content(self):
        self.prepared();first=fe.contract_descriptor(self.prep)
        self.assertEqual(set(first['contracts']),set(model.DEFAULT_TARGETS))
        self.contracts['main_12months']['response'][0]+=.1
        self.prepared();second=fe.contract_descriptor(self.prep)
        self.assertNotEqual(first['contracts']['main_12months'],second['contracts']['main_12months'])
        self.assertEqual({t:v for t,v in first['contracts'].items() if t!='main_12months'},
                         {t:v for t,v in second['contracts'].items() if t!='main_12months'})

    def test_missing_selected_target_and_tampered_artifact_are_rejected(self):
        saved=self.contracts.pop('main_12months');self.prepared()
        with self.assertRaises(ValidationError):fe.contract_descriptor(self.prep)
        self.contracts['main_12months']=saved;manifest=self.prepared()
        path=self.prep/next(iter(manifest['files']));path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValidationError):fe.contract_descriptor(self.prep)

    def test_preparation_status_identity_and_manifest_are_required(self):
        self.prepared();path=self.prep/'receipt.json';receipt=json.loads(path.read_text())
        for field,bad in [('through','M02'),('status','failed'),('fingerprint','0'*64),('artifact_manifest_sha256','0'*64),('identity',{})]:
            value=deepcopy(receipt);value[field]=bad;write_json(path,value)
            with self.subTest(field=field),self.assertRaises(ValidationError):fe.contract_descriptor(self.prep)


class FenoExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_fixture=contracts();cls.result_fixture=model.fit_selected(cls.contract_fixture)
        assert cls.result_fixture['metadata']['accepted']
        cls.no_exposure_contracts=deepcopy(cls.contract_fixture)
        cls.no_exposure_contracts['supplement_6to10weeks']=fabricated_contract('supplement_6to10weeks',no_exposure=True)
        cls.no_exposure_result=model.fit_selected(cls.no_exposure_contracts)
        cls.failed_contracts=deepcopy(cls.contract_fixture)
        cls.failed_contracts['main_12months']=fabricated_contract('main_12months',no_exposure=True)
        cls.failed_result=model.fit_selected(cls.failed_contracts)

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name);self.root=self.base/'source';self.prep=self.base/'preparation'
        self.work,self.output=self.base/'private',self.base/'aggregate';self.prep.mkdir()
        for name in fe.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(fe.__file__).parent/name,path)
        (self.root/'requirements-models.lock.txt').write_text('# synthetic lock\n')
        self.contracts=deepcopy(self.contract_fixture);self.result=deepcopy(self.result_fixture)
        self.descriptor={'contracts':{t:'a'*64 for t in model.DEFAULT_TARGETS},'preparation_fingerprint':'b'*64}
        self.path_targets={model.artifact_path(t):t for t in model.DEFAULT_TARGETS}
        for patch in [mock.patch.object(fe,'contract_descriptor',side_effect=lambda *a:deepcopy(self.descriptor)),
                      mock.patch.object(fe,'load_artifacts',side_effect=lambda p,node:self.contracts[self.path_targets[tuple(node)]]),
                      mock.patch.object(fe,'runtime_identity',return_value={'synthetic':True})]:
            patch.start();self.addCleanup(patch.stop)

    def run_fit(self, *, resume=False, compute=None, work=None, output=None):
        with mock.patch.object(model,'fit_selected',side_effect=compute or (lambda *a:deepcopy(self.result))):
            return fe.fit_selected(self.root,self.prep,work or self.work,output or self.output,resume=resume)

    def directory(self,result):return self.output/result['fingerprint']

    def assert_clean(self):
        self.assertEqual(list(self.base.rglob('.staging-*')),[])
        self.assertEqual(list(self.base.rglob('.preparation.lock')),[])

    def test_complete_roundtrip_recomputes_all_eleven_fits_on_resume(self):
        calls=[]
        def compute(*args):calls.append(True);return deepcopy(self.result)
        first=self.run_fit(compute=compute);second=self.run_fit(resume=True,compute=compute)
        self.assertEqual(len(calls),2);self.assertTrue(first['accepted']);self.assertTrue(second['resumed'])
        self.assertEqual(first['rows'],row_counts(self.contracts));self.assertEqual(first['estimable_models'],11)
        self.assertEqual(first['non_estimable_models'],0)
        self.assertEqual({p.name for p in self.directory(first).iterdir()},set(fe.RESULT_FILES)|{'receipt.json'})
        self.assert_clean()

    def test_source_no_exposure_guard_keeps_zero_categories_and_null_inference(self):
        self.contracts=deepcopy(self.no_exposure_contracts);self.result=deepcopy(self.no_exposure_result)
        result=self.run_fit();self.assertTrue(result['accepted']);self.assertEqual(result['non_estimable_models'],1)
        rows=pd.read_csv(self.directory(result)/'counts.csv');rows=rows[rows.target.eq('supplement_6to10weeks')]
        self.assertEqual(rows.observations.tolist(),[0,0,64]);self.assertEqual(rows.children.tolist(),[0,0,32])
        estimates=pd.read_csv(self.directory(result)/'estimates.csv');row=estimates[estimates.target.eq('supplement_6to10weeks')].iloc[0]
        self.assertEqual(row.status,'non_estimable');self.assertEqual(row.reason,'source_no_exposure_contrast')
        self.assertTrue(row[['coefficient','robust_se','geometric_mean_ratio','ci_low','ci_high','percent_difference','p_value']].isna().all())
        self.assertTrue(self.run_fit(resume=True)['accepted'])

    def test_genuine_singular_failure_preserves_every_output_without_acceptance(self):
        self.contracts=deepcopy(self.failed_contracts);self.result=deepcopy(self.failed_result)
        result=self.run_fit();self.assertFalse(result['accepted']);self.assertEqual(result['status'],'fit_failed')
        self.assertEqual(result['rows'],row_counts(self.contracts));self.assertFalse(self.run_fit(resume=True)['accepted'])
        self.assert_clean()

    def test_post_optimizer_failure_preserves_truthful_convergence_and_null_inference(self):
        result=deepcopy(self.result);target=model.DEFAULT_TARGETS[0];meta=result['metadata']['models'][0]
        meta.update(status='fit_failed',accepted=False,reason='invalid_covariance',converged=True)
        result['metadata'].update(accepted=False,status='fit_failed')
        for name,frame in result['tables'].items():
            selected=frame.target.eq(target)
            if name=='counts':continue
            frame.loc[selected,'status']='fit_failed';frame.loc[selected,'accepted']=False
            if 'reason' in frame:frame.loc[selected,'reason']='invalid_covariance'
            if name=='covariance':frame.loc[selected,'value']=np.nan
            elif name=='coefficients':
                frame.loc[selected,['coefficient','robust_se','naive_se','exponentiated_coefficient','ci_low','ci_high','p_value']]=np.nan
            else:frame.loc[selected,['coefficient','robust_se','geometric_mean_ratio','ci_low','ci_high','percent_difference','p_value']]=np.nan
        final=self.run_fit(compute=lambda *a:result)
        self.assertFalse(final['accepted']);self.assertTrue(json.loads((self.directory(final)/'diagnostics.json').read_text())['models'][0]['converged'])

    def test_explicit_resume_is_required_before_computation(self):
        self.run_fit()
        with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('unexpected recomputation'))

    def test_every_required_file_hash_presence_and_directory_inventory_checked(self):
        directory=self.directory(self.run_fit())
        for name in fe.RESULT_FILES:
            path=directory/name;old=path.read_bytes();path.write_bytes(old+b'changed')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.unlink()
            with self.subTest(name=name,missing=True),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.write_bytes(old)
        extra=directory/'untracked.csv';extra.write_text('unexpected\n')
        with self.assertRaises(ValidationError):self.run_fit(resume=True)
        extra.unlink();write_json(directory/'receipt.json',[])
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_repaired_hash_changes_to_all_outputs_are_independently_recomputed(self):
        directory=self.directory(self.run_fit());rp=directory/'receipt.json';old_receipt=rp.read_bytes()
        for name in fe.RESULT_FILES:
            path=directory/name;old=path.read_bytes()
            if name.endswith('.csv'):
                frame=pd.read_csv(path,float_precision='round_trip')
                column=next(c for c in frame.select_dtypes(include=[np.number]) if frame[c].notna().all())
                frame[column]=frame[column].astype(float);frame.loc[0,column]+=.1;frame.to_csv(path,index=False,float_format='%.17g')
            else:
                value=json.loads(path.read_text());value['contradiction']=True;write_json(path,value)
            receipt=json.loads(old_receipt);receipt['files'][name]=file_hash(path)
            if name=='summary.json':receipt['summary']=value
            write_json(rp,receipt)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.write_bytes(old);rp.write_bytes(old_receipt)

    def test_empty_missing_duplicate_unselected_and_reordered_tables_rejected(self):
        for name in fe.TABLES:
            for mode in ('empty','drop','duplicate','wrong_target','reorder'):
                result=deepcopy(self.result);frame=result['tables'][name]
                if mode=='empty':result['tables'][name]=frame.iloc[:0].reset_index(drop=True)
                elif mode=='drop':result['tables'][name]=frame.iloc[1:].reset_index(drop=True)
                elif mode=='duplicate':result['tables'][name]=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
                elif mode=='reorder':result['tables'][name]=frame.iloc[::-1].reset_index(drop=True)
                else:frame.loc[0,'target']='unselected'
                with self.subTest(name=name,mode=mode),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_malformed_mappings_and_acceptance_contradictions_are_controlled(self):
        for change in ('tables_none','metadata_list','accepted_number','status','models','model_status','converged','missing_acceptance','diagnostics_list'):
            result=deepcopy(self.result)
            if change=='tables_none':result['tables']=None
            elif change=='metadata_list':result['metadata']=[]
            elif change=='accepted_number':result['metadata']['accepted']=1
            elif change=='status':result['metadata']['status']='fit_failed'
            elif change=='models':result['metadata']['models'].pop()
            elif change=='model_status':result['metadata']['models'][0]['accepted']=False
            elif change=='converged':result['metadata']['models'][0]['converged']=False
            elif change=='missing_acceptance':del result['metadata']['models'][0]['accepted']
            else:result['metadata']['models'][0]['diagnostics']=[]
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_numeric_metadata_counts_and_variances_are_validated(self):
        for change in ('count_text','count_bool','count_fraction','negative_count','condition','score','scale','rank','clusters','df','iterations','correction','warning_dict'):
            result=deepcopy(self.result);meta=result['metadata']['models'][0]
            if change.startswith('count_'):meta['counts']['observations']={'count_text':'bad','count_bool':True,'count_fraction':2.5}[change]
            elif change=='negative_count':meta['counts']['exposed_children']=-1
            elif change=='correction':meta['covariance_correction']=1.1
            elif change=='warning_dict':meta['warnings']=[{}]
            else:
                key={'condition':'condition_number','score':'score_norm','scale':'scale','rank':'rank','clusters':'n_clusters','df':'df_resid','iterations':'iterations'}[change]
                meta['diagnostics'][key]={'condition':.5,'score':2.,'scale':0.,'rank':1,'clusters':1,'df':1,'iterations':0}[change]
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_finite_wrong_estimates_covariances_and_inference_rejected(self):
        for table,column in [('estimates','coefficient'),('estimates','robust_se'),('estimates','geometric_mean_ratio'),
            ('estimates','percent_difference'),('estimates','ci_low'),('estimates','ci_high'),('estimates','p_value'),
            ('coefficients','coefficient'),('coefficients','robust_se'),('coefficients','naive_se'),
            ('coefficients','exponentiated_coefficient'),('coefficients','ci_low'),('coefficients','p_value'),('covariance','value')]:
            result=deepcopy(self.result);result['tables'][table].loc[0,column]+=.05
            with self.subTest(table=table,column=column),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_consistent_forged_intercept_is_rejected_by_contract_qr_equations(self):
        result=deepcopy(self.result);frame=result['tables']['coefficients'];mask=frame.term.eq('Intercept')
        frame.loc[mask,'coefficient']+=.2;frame.loc[mask,'exponentiated_coefficient']=np.exp(frame.loc[mask,'coefficient'])
        frame.loc[mask,'ci_low']=np.exp(frame.loc[mask,'coefficient']-1.96*frame.loc[mask,'robust_se'])
        frame.loc[mask,'ci_high']=np.exp(frame.loc[mask,'coefficient']+1.96*frame.loc[mask,'robust_se'])
        from scipy.stats import norm
        frame.loc[mask,'p_value']=2*norm.sf(abs(frame.loc[mask,'coefficient']/frame.loc[mask,'robust_se']))
        fe.check_result(result)
        with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_nonestimable_reason_nulls_and_zero_exposure_support_are_required(self):
        self.contracts=deepcopy(self.no_exposure_contracts);self.result=deepcopy(self.no_exposure_result)
        index=list(model.DEFAULT_TARGETS).index('supplement_6to10weeks')
        for change in ('reason','converged','finite_beta','finite_cov','status','counts'):
            result=deepcopy(self.result);meta=result['metadata']['models'][index]
            if change=='reason':meta['reason']='singular_design'
            elif change=='converged':meta['converged']=True
            elif change=='status':meta['status']='estimated'
            elif change=='counts':meta['counts']['exposed_observations']=1
            else:
                name,column=('coefficients','coefficient') if change=='finite_beta' else ('covariance','value')
                frame=result['tables'][name];frame.loc[frame.target.eq('supplement_6to10weeks'),column]=0.
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_source_counts_age_terms_and_labels_bound_to_preparation(self):
        for change in ('age','group_count','family','scenario','nominal_age','contrast','artifact','fit_controls','transform','prepared_response','prepared_terms'):
            result=deepcopy(self.result);meta=result['metadata']['models'][0]
            if change=='age':result['tables']['estimates'].loc[0,'age_median']+=.001
            elif change=='group_count':result['tables']['counts'].loc[0,'children']+=1
            elif change=='family':result['tables']['coefficients'].loc[0,'family']='supplement'
            elif change in ('scenario','contrast'):result['tables']['estimates'].loc[0,change]='wrong'
            elif change=='nominal_age':result['tables']['estimates'].loc[0,'nominal_age_years']=2.
            elif change=='artifact':meta['artifact_path']=['wrong']
            elif change=='fit_controls':meta['fit_controls']['maxiter']+=1
            elif change=='transform':meta['coefficient_transform']='wrong'
            elif change=='prepared_response':self.contracts['main_12months']['response'][0]+=.1
            else:self.contracts['main_12months']['design']=self.contracts['main_12months']['design'].iloc[:,:-1]
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)
            self.contracts=deepcopy(self.contract_fixture)

    def test_private_identifiers_cannot_enter_added_fields_or_allowed_text_slots(self):
        secret='FABRICATED_PRIVATE_RECORD'
        for location in ('table','top_metadata','model_metadata','warning','reason','diagnostic','term','count_group','contrast'):
            result=deepcopy(self.result);meta=result['metadata']['models'][0]
            if location=='table':result['tables']['estimates']['pid_child']=secret
            elif location=='top_metadata':result['metadata']['participant_records']=[secret]
            elif location=='model_metadata':meta['pid_child']=secret
            elif location=='warning':meta['warnings']=[secret]
            elif location=='reason':meta['reason']=secret
            elif location=='diagnostic':meta['diagnostics']['score_norm']=secret
            elif location=='term':result['tables']['coefficients'].loc[0,'term']=secret
            elif location=='count_group':result['tables']['counts'].loc[0,'group']=secret
            else:result['tables']['estimates'].loc[0,'contrast']=secret
            with self.subTest(location=location),self.assertRaises(ValidationError) as error:self.run_fit(compute=lambda *a:result)
            self.assertNotIn(secret,str(error.exception))
        self.assertEqual(list(self.output.rglob('receipt.json')),[])

    def test_symlinked_files_and_publication_directory_are_rejected(self):
        directory=self.directory(self.run_fit())
        for name in (*fe.RESULT_FILES,'receipt.json'):
            path=directory/name;outside=self.base/('outside-'+name);path.rename(outside);path.symlink_to(outside)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.unlink();outside.rename(path)
        elsewhere=self.base/'other-publication';directory.rename(elsewhere);directory.symlink_to(elsewhere,target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_dangling_destination_is_not_replaced_during_resume(self):
        result=self.run_fit();directory=self.directory(result);shutil.rmtree(directory);missing=self.base/'missing'
        directory.symlink_to(missing,target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)
        self.assertTrue(directory.is_symlink());self.assertFalse(missing.exists());self.assert_clean()

    def test_destination_created_during_computation_is_not_replaced(self):
        fingerprint,_=fe.fit_identity(self.root,self.descriptor);destination=self.output/fingerprint
        for mode in ('directory','dangling_symlink'):
            def raced(*args):
                if mode=='directory':destination.mkdir()
                else:destination.symlink_to(self.base/'missing',target_is_directory=True)
                return deepcopy(self.result)
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.run_fit(compute=raced)
            self.assertTrue(destination.is_symlink() if mode=='dangling_symlink' else destination.is_dir())
            if destination.is_symlink():destination.unlink()
            else:destination.rmdir()
            self.assert_clean()

    def test_traversal_receipt_file_claim_is_rejected(self):
        directory=self.directory(self.run_fit());path=directory/'receipt.json';receipt=json.loads(path.read_text())
        receipt['files']['../estimates.csv']=receipt['files'].pop('estimates.csv');write_json(path,receipt)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_stale_import_and_mismatched_source_root_fail_before_computation(self):
        for name in fe.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;old=path.read_bytes();path.write_bytes(old+b'# changed\n')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('wrong source computed'))
            path.write_bytes(old)
        original=fe.file_hash;actual=Path(model.__file__).resolve()
        with mock.patch.object(fe,'file_hash',side_effect=lambda p:'0'*64 if Path(p).resolve()==actual else original(p)):
            with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('stale import computed'))

    def test_changed_source_or_contract_during_fit_blocks_publication(self):
        for source in (True,False):
            path=self.root/'src/dchs_lung_function/models/feno.py';old=path.read_bytes()
            def changed(*args):
                if source:path.write_bytes(old+b'# changed\n')
                else:self.descriptor['contracts']['main_12months']='c'*64
                return deepcopy(self.result)
            with self.subTest(source=source),self.assertRaises(ValidationError):self.run_fit(compute=changed)
            path.write_bytes(old);self.descriptor['contracts']['main_12months']='a'*64
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_clean()

    def test_changed_dependency_before_cached_return_is_rejected(self):
        self.run_fit();original=fe.verify_result
        def changed(*args):
            result=original(*args);self.descriptor['contracts']['main_12months']='c'*64;return result
        with mock.patch.object(fe,'verify_result',side_effect=changed),self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_work_and_output_locks_protect_resume_with_another_work_root(self):
        self.run_fit()
        for locked in (self.work,self.output):
            with fe.preparation_locks(locked,locked):
                with self.subTest(locked=locked),self.assertRaises(ValidationError):self.run_fit(resume=True,compute=lambda *a:self.fail('computed while locked'))
                if locked==self.output:
                    with self.assertRaises(ValidationError):self.run_fit(resume=True,work=self.base/'other',compute=lambda *a:self.fail('computed while locked'))
        self.assert_clean()

    def test_concurrent_different_work_roots_cannot_publish_to_same_output(self):
        entered,release=threading.Event(),threading.Event();errors=[]
        def compute(*args):
            entered.set()
            if not release.wait(10):raise RuntimeError('test synchronization timeout')
            return deepcopy(self.result)
        def worker():
            try:self.run_fit(compute=compute)
            except BaseException as exc:errors.append(exc)
        thread=threading.Thread(target=worker);thread.start()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertRaises(ValidationError):self.run_fit(resume=True,work=self.base/'different')
        finally:release.set();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[]);self.assert_clean()

    def test_producer_stage_allocation_write_and_publication_failures_leave_no_receipt(self):
        with self.assertRaises(RuntimeError):self.run_fit(compute=lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic failure')))
        for owner,name in [(fe.tempfile,'mkdtemp'),(pd.DataFrame,'to_csv'),(fe.os,'replace')]:
            with mock.patch.object(owner,name,side_effect=OSError('synthetic publication failure')):
                with self.subTest(name=name),self.assertRaises(OSError):self.run_fit()
            self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_clean()

    def test_source_preparation_and_overlapping_destinations_are_rejected(self):
        for work,output in [(self.root/'generated',self.output),(self.prep/'generated',self.output),
            (self.work,self.work),(self.work,self.work/'nested'),(self.work,self.base)]:
            with self.subTest(work=work),self.assertRaises(ValidationError):self.run_fit(work=work,output=output)

    def test_symlink_root_cannot_bypass_protected_source_or_preparation_boundaries(self):
        for target in (self.root,self.prep):
            link=self.base/'alias';link.symlink_to(target,target_is_directory=True)
            with self.assertRaises(ValidationError):self.run_fit(work=link/'generated')
            link.unlink()

    def test_contract_lock_and_runtime_changes_generate_new_identities(self):
        first=self.run_fit();fingerprints={first['fingerprint']}
        for change in [lambda:self.descriptor['contracts'].__setitem__('main_12months','c'*64),
                       lambda:(self.root/'requirements-models.lock.txt').write_text('# changed lock\n')]:
            change();result=self.run_fit(resume=True);self.assertNotIn(result['fingerprint'],fingerprints);fingerprints.add(result['fingerprint'])
        with mock.patch.object(fe,'runtime_identity',return_value={'synthetic':'changed'}):
            self.assertNotIn(self.run_fit(resume=True)['fingerprint'],fingerprints)


class FenoBatchAndCliTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name);self.root=self.base/'source';self.work=self.base/'work';self.output=self.base/'outputs'
        self.work.mkdir();self.output.mkdir()
        for name in fe.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(fe.__file__).parent/name,path)
        self.paths=[self.base/name for name in ['paths.json','analysis.json','primary.json','mechanics.json','supporting.json','wheeze.json']]
        write_json(self.paths[0],{'inputs':{},'directories':{'work':str(self.work),'outputs':str(self.output)}})
        for path in self.paths[1:]:write_json(path,{'synthetic':True})
        self.contracts=contracts();self.path_targets={model.artifact_path(t):t for t in model.DEFAULT_TARGETS}
        self.prior={'through':'M10','accepted':True,'status':'accepted','selected_models':2,
            'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
            'mechanics_models_verified':6,'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
            'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,'preparation_fingerprint':'a'*64}
        self.result={'accepted':True,'status':'accepted','selected_models':11,'estimable_models':11,'non_estimable_models':0,
            'rows':row_counts(self.contracts),'fingerprint':'b'*64}

    def batch(self, *, prior=None, result=None, prepared=None, jobs=1):
        with mock.patch.object(wheeze_execution,'run_wheeze',side_effect=prior or (lambda *a,**k:deepcopy(self.prior))):
            with mock.patch.object(fe,'fit_selected',side_effect=result or (lambda *a,**k:deepcopy(self.result))):
                with mock.patch.object(fe,'run_preparation',side_effect=prepared or (lambda *a,**k:{'fingerprint':'a'*64})):
                    with mock.patch.object(fe,'load_artifacts',side_effect=lambda p,node:self.contracts[self.path_targets[tuple(node)]]):
                        return fe.run_feno(self.root,*self.paths,jobs=jobs,resume=True)

    def test_complete_cumulative_inventory_publishes_current_marker_and_retains_limitations(self):
        result=self.batch();self.assertTrue(result['accepted']);self.assertEqual(result['through'],'M11')
        self.assertEqual(result,json.loads((self.work/'feno_current.json').read_text()))
        self.assertEqual(result['wheeze_models_verified'],2);self.assertEqual(result['pretb_models_verified'],17)
        self.assertIn('G08',result['inherited_limitations'][0]);self.assertIn('Four',result['inherited_limitations'][1])

    def test_failed_current_fit_cannot_publish_completion_marker(self):
        self.result.update(accepted=False,status='fit_failed',estimable_models=10)
        self.assertFalse(self.batch()['accepted']);self.assertFalse((self.work/'feno_current.json').exists())

    def test_every_prior_count_status_and_complete_acceptance_is_required(self):
        for field in self.prior:
            if field=='preparation_fingerprint':continue
            saved=self.prior[field];self.prior[field]=False if field=='accepted' else 'wrong' if isinstance(saved,str) else saved-1
            with self.subTest(field=field),self.assertRaises(ValidationError):self.batch(result=lambda *a,**k:self.fail('fit after failed prerequisite'))
            self.prior[field]=saved

    def test_partial_duplicate_or_contradictory_current_inventory_is_rejected(self):
        changes=[('accepted',1),('status','fit_failed'),('selected_models',10),('estimable_models',10),
                 ('estimable_models',True),('non_estimable_models',-1)]
        for name in fe.TABLES:
            counts=deepcopy(self.result['rows']);counts[name]-=1;changes.append(('rows',counts))
        for field,value in changes:
            result={**self.result,field:value}
            with self.subTest(field=field,value=value),self.assertRaises(ValidationError):self.batch(result=lambda *a,**k:result)

    def test_no_exposure_design_widths_determine_actual_inventory(self):
        self.contracts['supplement_6to10weeks']=fabricated_contract('supplement_6to10weeks',no_exposure=True)
        self.result.update(estimable_models=10,non_estimable_models=1,rows=row_counts(self.contracts))
        self.assertTrue(self.batch()['accepted'])

    def test_every_configuration_and_sampling_file_is_bound_until_completion(self):
        for path in self.paths:
            old=path.read_bytes()
            def changed(*a,**k):path.write_bytes(old+b'\n');return deepcopy(self.result)
            with self.subTest(path=path.name),self.assertRaises(ValidationError):self.batch(result=changed)
            path.write_bytes(old)
        with self.assertRaises(ValidationError):self.batch(prepared=lambda *a,**k:{'fingerprint':'changed'})
        self.assertFalse((self.work/'feno_current.json').exists())

    def test_invalid_jobs_and_midrun_source_changes_prevent_completion(self):
        for jobs in [0,4,True,1.5]:
            with self.subTest(jobs=jobs),self.assertRaises(ValidationError):self.batch(jobs=jobs)
        def changed(*a,**k):
            path=self.root/'src/dchs_lung_function/feno_execution.py';path.write_text(path.read_text()+'# changed\n');return deepcopy(self.result)
        with self.assertRaises(ValidationError):self.batch(result=changed)
        self.assertFalse((self.work/'feno_current.json').exists())

    def cli(self, *, extra=None, result=None, error=None):
        stdout,stderr=StringIO(),StringIO();args=['--through','M11','--config',str(self.paths[0]),'--analysis',str(self.paths[1]),'--resume']
        with mock.patch.object(fe,'run_feno',return_value=result or self.result,side_effect=error) as runner:
            with redirect_stdout(stdout),redirect_stderr(stderr):code=pipeline.main(args+(extra or []),root=self.root)
        return code,stdout.getvalue(),stderr.getvalue(),runner

    def test_cli_routes_all_four_prior_sampling_controls(self):
        extra=['--sampling',str(self.paths[2]),'--mechanics-sampling',str(self.paths[3]),
               '--supporting-sampling',str(self.paths[4]),'--wheeze-sampling',str(self.paths[5]),'--jobs','3']
        code,out,err,runner=self.cli(extra=extra)
        self.assertEqual(code,0);self.assertEqual(err,'');self.assertIn('11 models',out)
        self.assertEqual(runner.call_args.args[3:],tuple(map(str,self.paths[2:])))
        self.assertEqual(runner.call_args.kwargs,{'jobs':3,'resume':True})

    def test_cli_returns_failure_and_rejects_partial_selection_without_private_errors(self):
        code,out,_,_=self.cli(result={**self.result,'accepted':False,'status':'fit_failed'})
        self.assertEqual(code,2);self.assertIn('fit_failed',out)
        code,_,err,runner=self.cli(extra=['--targets','main_12months']);self.assertEqual(code,2);runner.assert_not_called()
        self.assertIn('all selected',err)
        for error in [ValueError('FABRICATED_PRIVATE_RECORD'),OSError('FABRICATED_PRIVATE_RECORD'),RuntimeError('FABRICATED_PRIVATE_RECORD')]:
            code,out,err,_=self.cli(error=error);self.assertEqual(code,2);self.assertNotIn('FABRICATED_PRIVATE_RECORD',out+err)

    def test_full_and_render_remain_unavailable(self):
        for option in ['--full','--render']:
            with mock.patch.object(fe,'run_feno') as runner,redirect_stderr(StringIO()):
                self.assertEqual(pipeline.main([option],root=self.root),2)
            runner.assert_not_called()


if __name__=='__main__':unittest.main()
