"""Synthetic M09 execution, cache and cumulative-CLI adversarial checks.

The small Cox fits here use invented records. Files, locks, fingerprints and
receipts are real; the runner's fitting boundary is mocked after class setup.
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
from dchs_lung_function import pretb_execution as pe, supporting_execution, pipeline
from dchs_lung_function.models import pretb_cox as model
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from test_pretb_cox import contract, sample


class PretbDescriptorTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.prep=Path(temporary.name)
        self.contracts={model.contract_key(t):contract(sample(),t) for t in model.DEFAULT_TARGETS}

    def prepared(self):
        manifest=ArtifactWriter(self.prep).finish({'M03':{'pft':{'samples':self.contracts}}})
        identity={'through':'M03','fixture':'invented'}
        write_json(self.prep/'receipt.json',{'through':'M03','status':'prepared',
            'identity':identity,'fingerprint':pe.digest(identity),
            'artifact_manifest_sha256':file_hash(self.prep/'artifacts.json')})
        return manifest

    def test_complete_selection_is_bound_to_verified_typed_artifact_content(self):
        self.prepared();first=pe.contract_descriptor(self.prep)
        self.assertEqual(set(first['contracts']),set(self.contracts))
        changed=self.contracts['pretb_cox/resistance'];changed['frame'].loc[0,'lf_adverse_sd']+=.1
        self.prepared();second=pe.contract_descriptor(self.prep)
        self.assertNotEqual(first['contracts']['pretb_cox/resistance'],second['contracts']['pretb_cox/resistance'])
        self.assertEqual({k:v for k,v in first['contracts'].items() if k!='pretb_cox/resistance'},
                         {k:v for k,v in second['contracts'].items() if k!='pretb_cox/resistance'})

    def test_missing_target_or_tampered_artifact_cannot_be_selected(self):
        del self.contracts['pretb_cox/resistance'];self.prepared()
        with self.assertRaises(ValidationError):pe.contract_descriptor(self.prep)
        self.contracts['pretb_cox/resistance']=contract(sample())
        manifest=self.prepared();path=self.prep/next(iter(manifest['files']))
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValidationError):pe.contract_descriptor(self.prep)

    def test_incomplete_preparation_and_stale_manifest_receipt_rejected(self):
        self.prepared();path=self.prep/'receipt.json';value=json.loads(path.read_text())
        for field,bad in [('through','M02'),('status','failed'),('fingerprint','0'*64),
                          ('artifact_manifest_sha256','0'*64)]:
            changed=deepcopy(value);changed[field]=bad;write_json(path,changed)
            with self.subTest(field=field),self.assertRaises(ValidationError):pe.contract_descriptor(self.prep)


class PretbExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_fixture={}
        base=sample()
        for i,target in enumerate(model.DEFAULT_TARGETS):
            frame=base.copy();frame['lf_adverse_sd']+=.03*i*np.sin(np.arange(len(frame)))
            cls.contract_fixture[model.contract_key(target)]=contract(frame,target)
        cls.result_fixture=model.fit_selected(cls.contract_fixture)
        assert cls.result_fixture['metadata']['accepted']
        low=sample().iloc[:4].copy();low['event']=0;low.iloc[-1,low.columns.get_loc('event')]=1
        cls.low_contracts=deepcopy(cls.contract_fixture)
        cls.low_contracts[model.contract_key(model.DEFAULT_TARGETS[0])]=contract(low)
        cls.low_result=model.fit_selected(cls.low_contracts)
        failed=sample();failed['mbekweni']=failed['male']
        cls.failed_contracts=deepcopy(cls.contract_fixture)
        cls.failed_contracts[model.contract_key(model.DEFAULT_TARGETS[0])]=contract(failed)
        cls.failed_result=model.fit_selected(cls.failed_contracts)

    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name);self.root=self.base/'source';self.prep=self.base/'preparation'
        self.work,self.output=self.base/'private',self.base/'aggregate';self.prep.mkdir()
        for name in pe.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(pe.__file__).parent/name,path)
        (self.root/'requirements-models.lock.txt').write_text('# synthetic lock\n')
        self.contracts=deepcopy(self.contract_fixture);self.result=deepcopy(self.result_fixture)
        self.descriptor={'contracts':{model.contract_key(t):'a'*64 for t in model.DEFAULT_TARGETS},'preparation_fingerprint':'b'*64}
        for patch in [mock.patch.object(pe,'contract_descriptor',side_effect=lambda *a:deepcopy(self.descriptor)),
                      mock.patch.object(pe,'load_artifacts',side_effect=lambda p,node:self.contracts[node[-1]]),
                      mock.patch.object(pe,'runtime_identity',return_value={'synthetic':True})]:
            patch.start();self.addCleanup(patch.stop)

    def run_fit(self, *, resume=False, compute=None, work=None, output=None):
        with mock.patch.object(model,'fit_selected',side_effect=compute or (lambda *a:deepcopy(self.result))):
            return pe.fit_selected(self.root,self.prep,work or self.work,output or self.output,resume=resume)

    def directory(self,result):return self.output/result['fingerprint']

    def assert_clean(self):
        self.assertEqual(list(self.base.rglob('.staging-*')),[])
        self.assertEqual(list(self.base.rglob('.preparation.lock')),[])

    def test_complete_roundtrip_recomputes_every_resume_and_preserves_empty_reasons(self):
        calls=[]
        def compute(*args):calls.append(True);return deepcopy(self.result)
        first=self.run_fit(compute=compute);second=self.run_fit(resume=True,compute=compute)
        self.assertEqual(len(calls),2);self.assertTrue(first['accepted']);self.assertTrue(second['resumed'])
        self.assertEqual(first['rows'],{'estimates':17,'coefficients':51,'covariance':306})
        self.assertEqual(first['estimable_models'],17);self.assertEqual(first['non_estimable_models'],0)
        self.assertEqual({p.name for p in self.directory(first).iterdir()},set(pe.RESULT_FILES)|{'receipt.json'})
        self.assert_clean()

    def test_expected_nonestimability_preserves_exact_small_counts_and_null_inference(self):
        self.contracts=deepcopy(self.low_contracts);self.result=deepcopy(self.low_result)
        result=self.run_fit();self.assertTrue(result['accepted']);self.assertEqual(result['non_estimable_models'],1)
        table=pd.read_csv(self.directory(result)/'estimates.csv')
        row=table.iloc[0]
        self.assertEqual((row.n_children,row.n_intervals,row.n_subsequent_tb),(2,4,1))
        self.assertEqual(row.status,'non_estimable');self.assertEqual(row.reason,'source_minimum_event_rule')
        self.assertTrue(row[['coefficient','robust_se','hr_per_adverse_sd','ci_low','ci_high','p_value','fdr_q_value']].isna().all())
        self.assertTrue(self.run_fit(resume=True)['accepted'])

    def test_genuine_failure_retains_complete_evidence_without_accepted_status(self):
        self.contracts=deepcopy(self.failed_contracts);self.result=deepcopy(self.failed_result)
        result=self.run_fit();self.assertFalse(result['accepted']);self.assertEqual(result['status'],'fit_failed')
        self.assertEqual(result['rows'],{'estimates':17,'coefficients':51,'covariance':306})
        self.assertFalse(self.run_fit(resume=True)['accepted']);self.assert_clean()

    def test_explicit_resume_required_before_recomputation(self):
        self.run_fit()
        with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('unexpected computation'))

    def test_missing_tampered_or_malformed_files_and_extra_artifacts_rejected(self):
        directory=self.directory(self.run_fit())
        for name in pe.RESULT_FILES:
            path=directory/name;original=path.read_bytes();path.write_bytes(original+b'changed')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.unlink()
            with self.subTest(name=name,missing=True),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.write_bytes(original)
        extra=directory/'untracked.csv';extra.write_text('unexpected\n')
        with self.assertRaises(ValidationError):self.run_fit(resume=True)
        extra.unlink();write_json(directory/'receipt.json',[])
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_repaired_hash_attacks_on_all_outputs_are_recomputed(self):
        directory=self.directory(self.run_fit());rp=directory/'receipt.json';original_receipt=rp.read_bytes()
        for name in pe.RESULT_FILES:
            path=directory/name;original=path.read_bytes()
            if name.endswith('.csv'):
                frame=pd.read_csv(path,float_precision='round_trip')
                column=next(c for c in frame.select_dtypes(include=[np.number]) if frame[c].notna().all())
                frame[column]=frame[column].astype(float);frame.loc[0,column]+=.1;frame.to_csv(path,index=False,float_format='%.17g')
            else:
                value=json.loads(path.read_text());value['contradiction']=True;write_json(path,value)
            receipt=json.loads(original_receipt);receipt['files'][name]=file_hash(path)
            if name=='summary.json':receipt['summary']=value
            write_json(rp,receipt)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.write_bytes(original);rp.write_bytes(original_receipt)

    def test_wrong_inventory_duplicates_or_unselected_targets_rejected(self):
        for name in pe.TABLES:
            for mode in ('empty','drop','duplicate','wrong_target'):
                result=deepcopy(self.result);frame=result['tables'][name]
                if mode=='empty':result['tables'][name]=frame.iloc[:0].reset_index(drop=True)
                elif mode=='drop':result['tables'][name]=frame.iloc[1:].reset_index(drop=True)
                elif mode=='duplicate':result['tables'][name]=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
                else:frame.loc[0,'target']='unselected'
                with self.subTest(name=name,mode=mode),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_malformed_mappings_and_status_contradictions_rejected(self):
        for change in ('tables_none','metadata_list','accepted_number','status','models','model_status','optimizer','score'):
            result=deepcopy(self.result)
            if change=='tables_none':result['tables']=None
            elif change=='metadata_list':result['metadata']=[]
            elif change=='accepted_number':result['metadata']['accepted']=1
            elif change=='status':result['metadata']['status']='fit_failed'
            elif change=='models':result['metadata']['models'].pop()
            elif change=='model_status':result['metadata']['models'][0]['accepted']=False
            elif change=='optimizer':result['metadata']['models'][0]['optimizer']['converged']=False
            elif change=='score':result['metadata']['models'][0]['diagnostics']['max_absolute_score']=1.
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_nested_malformed_values_have_controlled_errors(self):
        for change in ('missing_acceptance','missing_count','text_count','risk_none','event_none',
                       'score_dict','covariance_text','coefficient_text','contracts_missing'):
            result=deepcopy(self.result);meta=result['metadata']['models'][0]
            if change=='missing_acceptance':del meta['accepted']
            elif change=='missing_count':del meta['counts']['n_children']
            elif change=='text_count':meta['counts']['person_years']='invalid'
            elif change=='risk_none':meta['risk_sets']=None
            elif change=='event_none':meta['risk_sets']['event_times'][0]['events']=None
            elif change=='score_dict':meta['diagnostics']['score']={'invalid':1}
            elif change=='covariance_text':
                result['tables']['covariance']['value']=result['tables']['covariance']['value'].astype(object)
                result['tables']['covariance'].loc[0,'value']='invalid'
            elif change=='coefficient_text':
                result['tables']['coefficients']['coefficient']=result['tables']['coefficients']['coefficient'].astype(object)
                result['tables']['coefficients'].loc[0,'coefficient']='invalid'
            elif change=='contracts_missing':del self.contracts['pretb_cox/resistance']
            with self.subTest(change=change),self.assertRaises(ValidationError):
                pe.check_result(result,self.contracts)
            self.contracts=deepcopy(self.contract_fixture)

    def test_finite_wrong_coefficients_covariances_ci_p_and_bh_rejected(self):
        for table,column in [('estimates','coefficient'),('estimates','robust_se'),('estimates','hr_per_adverse_sd'),
                             ('estimates','ci_low'),('estimates','p_value'),('estimates','fdr_q_value'),
                             ('coefficients','model_se'),('coefficients','ci_high'),('covariance','value')]:
            result=deepcopy(self.result);frame=result['tables'][table];frame.loc[0,column]+=.05
            with self.subTest(table=table,column=column),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_finite_only_bh_family_and_expected_nonestimability_are_verified(self):
        self.contracts=deepcopy(self.low_contracts);self.result=deepcopy(self.low_result)
        for change in ('q','guard_reason','false_accept','low_count','finite_inference','multiplicity'):
            result=deepcopy(self.result)
            if change=='q':result['tables']['estimates'].loc[0,'fdr_q_value']=.5
            elif change=='guard_reason':result['metadata']['models'][0]['reason']='different_rule'
            elif change=='false_accept':result['metadata']['models'][0]['accepted']=False
            elif change=='low_count':result['metadata']['models'][0]['counts']['n_subsequent_tb']=3
            elif change=='finite_inference':result['tables']['coefficients'].loc[0,'coefficient']=0.
            elif change=='multiplicity':result['metadata']['multiplicity']['finite_tests']=17
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_counts_scales_risk_support_and_specification_bound_to_contract(self):
        for change in ('count','scale','orientation','correction','risk_rows','score_support','prepared_count','prepared_terms'):
            result=deepcopy(self.result);meta=result['metadata']['models'][0]
            if change=='count':meta['counts']['n_intervals']+=1
            elif change=='scale':result['tables']['estimates'].loc[0,'scale']='different'
            elif change=='orientation':meta['specification']['orientation']*=-1
            elif change=='correction':meta['covariance_correction']+=.1
            elif change=='risk_rows':meta['risk_sets']['event_times'][0]['risk_rows']+=1
            elif change=='score_support':meta['risk_sets']['source_score_membership_difference_cells']+=1
            elif change=='prepared_count':self.contracts['pretb_cox/resistance']['frame']=self.contracts['pretb_cox/resistance']['frame'].iloc[:-1]
            elif change=='prepared_terms':self.contracts['pretb_cox/resistance']['terms']=[]
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)
            self.contracts=deepcopy(self.contract_fixture)

    def test_extra_participant_fields_cannot_enter_publication(self):
        for location in ('table','top_metadata','model_metadata'):
            result=deepcopy(self.result)
            if location=='table':result['tables']['estimates']['pid_child']='fabricated_private_identifier'
            elif location=='top_metadata':result['metadata']['participant_records']=['fabricated_private_identifier']
            else:result['metadata']['models'][0]['pid_child']='fabricated_private_identifier'
            with self.subTest(location=location),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)
        self.assertEqual(list(self.output.rglob('receipt.json')),[])

    def test_symlinked_files_and_publication_directory_rejected(self):
        directory=self.directory(self.run_fit())
        for name in (*pe.RESULT_FILES,'receipt.json'):
            path=directory/name;outside=self.base/('outside-'+name);path.rename(outside);path.symlink_to(outside)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.unlink();outside.rename(path)
        elsewhere=self.base/'other-publication';directory.rename(elsewhere);directory.symlink_to(elsewhere,target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_traversal_receipt_file_claim_rejected(self):
        directory=self.directory(self.run_fit());receipt=json.loads((directory/'receipt.json').read_text())
        receipt['files']['../estimates.csv']=receipt['files'].pop('estimates.csv');write_json(directory/'receipt.json',receipt)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_stale_import_and_mismatched_root_rejected_before_computation(self):
        for name in pe.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;old=path.read_bytes();path.write_bytes(old+b'# changed\n')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('wrong source computed'))
            path.write_bytes(old)
        original=pe.file_hash;actual=Path(model.__file__).resolve()
        with mock.patch.object(pe,'file_hash',side_effect=lambda p:'0'*64 if Path(p).resolve()==actual else original(p)):
            with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('stale import computed'))

    def test_source_or_contract_changed_during_fitting_blocks_publication(self):
        for source in (True,False):
            path=self.root/'src/dchs_lung_function/models/pretb_cox.py';old=path.read_bytes()
            def changed(*args):
                if source:path.write_bytes(old+b'# changed\n')
                else:self.descriptor['contracts']['pretb_cox/resistance']='c'*64
                return deepcopy(self.result)
            with self.subTest(source=source),self.assertRaises(ValidationError):self.run_fit(compute=changed)
            path.write_bytes(old);self.descriptor['contracts']['pretb_cox/resistance']='a'*64
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_clean()

    def test_dependency_change_before_cached_return_rejected(self):
        self.run_fit();original=pe.verify_result
        def changed(*args):
            value=original(*args);self.descriptor['contracts']['pretb_cox/resistance']='c'*64;return value
        with mock.patch.object(pe,'verify_result',side_effect=changed),self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_lock_protects_resume_and_alternative_work_root(self):
        self.run_fit()
        with pe.preparation_locks(self.work,self.output):
            with self.assertRaises(ValidationError):self.run_fit(resume=True,compute=lambda *a:self.fail('computed with lock'))
            with self.assertRaises(ValidationError):self.run_fit(resume=True,work=self.base/'other',compute=lambda *a:self.fail('computed with lock'))
        self.assert_clean()

    def test_concurrent_different_work_roots_cannot_publish_same_outputs(self):
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

    def test_failure_or_partial_publication_leaves_no_receipt_and_releases_locks(self):
        with self.assertRaises(RuntimeError):self.run_fit(compute=lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic failure')))
        with mock.patch.object(pe.os,'replace',side_effect=OSError('synthetic publication failure')):
            with self.assertRaises(OSError):self.run_fit()
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_clean()

    def test_source_preparation_and_overlapping_output_roots_rejected(self):
        for work,output in [(self.root/'generated',self.output),(self.prep/'generated',self.output),(self.work,self.work),
                            (self.work,self.work/'nested'),(self.work,self.base)]:
            with self.subTest(work=work),self.assertRaises(ValidationError):self.run_fit(work=work,output=output)

    def test_changed_contract_lock_and_runtime_get_new_identities(self):
        first=self.run_fit();fingerprints={first['fingerprint']}
        for change in [lambda:self.descriptor['contracts'].__setitem__('pretb_cox/resistance','c'*64),
                       lambda:(self.root/'requirements-models.lock.txt').write_text('# changed lock\n')]:
            change();result=self.run_fit(resume=True);self.assertNotIn(result['fingerprint'],fingerprints);fingerprints.add(result['fingerprint'])
        with mock.patch.object(pe,'runtime_identity',return_value={'synthetic':'changed'}):
            self.assertNotIn(self.run_fit(resume=True)['fingerprint'],fingerprints)


class PretbBatchAndCliTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name);self.root=self.base/'source';self.work=self.base/'work';self.output=self.base/'outputs'
        self.work.mkdir();self.output.mkdir()
        for name in pe.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(pe.__file__).parent/name,path)
        self.paths=[self.base/name for name in ['paths.json','analysis.json','primary.json','mechanics.json','supporting.json']]
        write_json(self.paths[0],{'inputs':{},'directories':{'work':str(self.work),'outputs':str(self.output)}})
        for path in self.paths[1:]:write_json(path,{'synthetic':True})
        self.prior={'through':'M08','accepted':True,'status':'accepted','selected_models':7,'deterministic_fits':10,
                    'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
                    'mechanics_models_verified':6,'shared_factor_rows_verified':6,'preparation_fingerprint':'a'*64}
        self.result={'accepted':True,'status':'accepted','selected_models':17,'estimable_models':17,'non_estimable_models':0,
                     'rows':{'estimates':17,'coefficients':51,'covariance':306},'fingerprint':'b'*64}

    def batch(self, *, prior=None, result=None, prepared=None, jobs=1):
        with mock.patch.object(supporting_execution,'run_supporting',side_effect=prior or (lambda *a,**k:deepcopy(self.prior))):
            with mock.patch.object(pe,'fit_selected',side_effect=result or (lambda *a,**k:deepcopy(self.result))):
                with mock.patch.object(pe,'run_preparation',side_effect=prepared or (lambda *a,**k:{'fingerprint':'a'*64})):
                    return pe.run_pretb(self.root,*self.paths,jobs=jobs,resume=True)

    def test_complete_prior_inventory_and_current_marker(self):
        result=self.batch();self.assertTrue(result['accepted']);self.assertEqual(result['through'],'M09')
        self.assertEqual(result,json.loads((self.work/'pretb_current.json').read_text()))
        self.assertEqual(result['supporting_joint_models_verified'],7);self.assertEqual(result['supporting_deterministic_fits_verified'],10)

    def test_failed_current_fit_cannot_publish_completion_marker(self):
        self.result.update(accepted=False,status='fit_failed',estimable_models=16)
        self.assertFalse(self.batch()['accepted']);self.assertFalse((self.work/'pretb_current.json').exists())

    def test_every_prior_module_count_and_status_is_required(self):
        for field in self.prior:
            if field=='preparation_fingerprint':continue
            saved=self.prior[field];self.prior[field]=False if field=='accepted' else 'wrong' if isinstance(saved,str) else saved-1
            with self.subTest(field=field),self.assertRaises(ValidationError):self.batch(result=lambda *a,**k:self.fail('fit after failed prerequisite'))
            self.prior[field]=saved

    def test_incomplete_or_contradictory_current_inventory_rejected(self):
        for field,value in [('accepted',1),('status','fit_failed'),('selected_models',16),('rows',{'estimates':17,'coefficients':50,'covariance':306})]:
            result={**self.result,field:value}
            with self.subTest(field=field),self.assertRaises(ValidationError):self.batch(result=lambda *a,**k:result)

    def test_all_configuration_and_sampling_files_bound_until_completion(self):
        for path in self.paths:
            original=path.read_bytes()
            def changed(*args,**kwargs):path.write_bytes(original+b'\n');return deepcopy(self.result)
            with self.subTest(path=path.name),self.assertRaises(ValidationError):self.batch(result=changed)
            path.write_bytes(original)
        with self.assertRaises(ValidationError):self.batch(prepared=lambda *a,**k:{'fingerprint':'changed'})
        self.assertFalse((self.work/'pretb_current.json').exists())

    def test_invalid_jobs_and_source_changes_prevent_current_marker(self):
        for jobs in [0,4,True,1.5]:
            with self.subTest(jobs=jobs),self.assertRaises(ValidationError):self.batch(jobs=jobs)
        def changed(*args,**kwargs):
            path=self.root/'src/dchs_lung_function/pretb_execution.py';path.write_text(path.read_text()+'# changed\n');return deepcopy(self.result)
        with self.assertRaises(ValidationError):self.batch(result=changed)
        self.assertFalse((self.work/'pretb_current.json').exists())

    def cli(self, *, extra=None, result=None, error=None):
        stdout,stderr=StringIO(),StringIO();args=['--through','M09','--config',str(self.paths[0]),'--analysis',str(self.paths[1]),'--resume']
        with mock.patch.object(pe,'run_pretb',return_value=result or self.result,side_effect=error) as runner:
            with redirect_stdout(stdout),redirect_stderr(stderr):code=pipeline.main(args+(extra or []),root=self.root)
        return code,stdout.getvalue(),stderr.getvalue(),runner

    def test_cli_sampling_options_route_to_preceding_models(self):
        code,out,err,runner=self.cli(extra=['--sampling',str(self.paths[2]),'--mechanics-sampling',str(self.paths[3]),'--supporting-sampling',str(self.paths[4]),'--jobs','3'])
        self.assertEqual(code,0);self.assertEqual(err,'');self.assertIn('17 models',out)
        self.assertEqual(runner.call_args.args[3:],tuple(map(str,self.paths[2:])))
        self.assertEqual(runner.call_args.kwargs,{'jobs':3,'resume':True})

    def test_cli_failure_partial_targets_and_private_errors(self):
        code,out,_,_=self.cli(result={**self.result,'accepted':False,'status':'fit_failed'})
        self.assertEqual(code,2);self.assertIn('fit_failed',out)
        code,_,err,runner=self.cli(extra=['--targets','resistance']);self.assertEqual(code,2);runner.assert_not_called()
        self.assertIn('all selected models',err)
        for error in [ValueError('FABRICATED_PRIVATE_RECORD'),OSError('FABRICATED_PRIVATE_RECORD'),RuntimeError('FABRICATED_PRIVATE_RECORD')]:
            code,out,err,_=self.cli(error=error);self.assertEqual(code,2);self.assertNotIn('FABRICATED_PRIVATE_RECORD',out+err)

    def test_full_and_render_remain_unavailable(self):
        for option in ['--full','--render']:
            with mock.patch.object(pe,'run_pretb') as runner,redirect_stderr(StringIO()):
                self.assertEqual(pipeline.main([option],root=self.root),2)
            runner.assert_not_called()


if __name__=='__main__':unittest.main()
