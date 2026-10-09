"""Synthetic deterministic publication and adversarial evidence checks for M08."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function import supporting_repeated_execution as re
from dchs_lung_function.models import supporting_repeated as model
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from test_supporting_repeated import fabricated_contracts


class SupportingRepeatedExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_fixture=fabricated_contracts()
        cls.result_fixture=model.fit_selected(cls.contract_fixture)
        assert cls.result_fixture['metadata']['accepted']

    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name)
        self.root,self.prep=self.base/'source',self.base/'preparation'
        self.work,self.output=self.base/'private',self.base/'aggregate'
        self.prep.mkdir()
        for name in re.SOURCES:
            path=self.root/'src/dchs_lung_function'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(re.__file__).parent/name,path)
        (self.root/'requirements-models.lock.txt').write_text('# synthetic environment\n')
        self.contracts=deepcopy(self.contract_fixture);self.result=deepcopy(self.result_fixture)
        self.descriptor={'contracts':{key:'a'*64 for key in model.CONTRACT_KEYS},'preparation_fingerprint':'b'*64}
        for patch in [mock.patch.object(re,'contract_descriptor',side_effect=lambda *a:deepcopy(self.descriptor)),
                      mock.patch.object(re,'load_artifacts',side_effect=lambda p,node:self.contracts[node[-1]]),
                      mock.patch.object(re,'runtime_identity',return_value={'synthetic':True})]:
            patch.start();self.addCleanup(patch.stop)

    def run_fit(self, *, resume=False, compute=None, work=None, output=None):
        with mock.patch.object(model,'fit_selected',side_effect=compute or (lambda *a:deepcopy(self.result))):
            return re.fit_selected(self.root,self.prep,work or self.work,output or self.output,resume=resume)

    def directory(self,result):return self.output/result['fingerprint']

    def assert_clean(self):
        self.assertEqual(list(self.base.rglob('.staging-*')),[])
        self.assertEqual(list(self.base.rglob('.preparation.lock')),[])

    def test_complete_roundtrip_recomputes_on_every_resume_and_keeps_empty_text(self):
        calls=[]
        def compute(contracts):calls.append(True);return deepcopy(self.result)
        first=self.run_fit(compute=compute);second=self.run_fit(resume=True,compute=compute)
        self.assertEqual(len(calls),2);self.assertTrue(second['resumed']);self.assertTrue(first['accepted'])
        self.assertEqual(first['fitted_models'],10)
        self.assertEqual(set(p.name for p in self.directory(first).iterdir()),set(re.RESULT_FILES)|{'receipt.json'})
        self.assertEqual(first['rows'],self.result['metadata']['row_counts'])
        counts=pd.read_csv(self.directory(first)/'mmef_descriptive.csv').n_children
        self.assertTrue(((counts>0)&(counts<10)).any())
        self.assertEqual(list(counts),list(self.result['tables']['mmef_descriptive'].n_children))
        self.assert_clean()

    def test_explicit_resume_required_before_recomputation(self):
        self.run_fit()
        with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('unexpected fit'))
        self.assert_clean()

    def test_missing_tampered_or_malformed_saved_files_rejected(self):
        directory=self.directory(self.run_fit())
        for name in re.RESULT_FILES:
            path=directory/name;original=path.read_bytes();path.write_bytes(original+b'corruption')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.unlink()
            with self.subTest(name=name,missing=True),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.write_bytes(original)
        write_json(directory/'receipt.json',[])
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_repaired_hash_attacks_on_every_table_and_summary_are_recomputed(self):
        directory=self.directory(self.run_fit());receipt_path=directory/'receipt.json'
        original_receipt=receipt_path.read_bytes()
        for name in re.RESULT_FILES:
            path=directory/name;original=path.read_bytes()
            if name.endswith('.csv'):
                frame=pd.read_csv(path,float_precision='round_trip')
                numeric=[column for column in frame.select_dtypes(include=[np.number]) if pd.notna(frame.loc[0,column])]
                frame[numeric[-1]]=frame[numeric[-1]].astype(float);frame.loc[0,numeric[-1]]+=1.;frame.to_csv(path,index=False,float_format='%.17g')
            else:
                value=json.loads(path.read_text());value['invented_contradiction']=True;write_json(path,value)
            receipt=json.loads(original_receipt);receipt['files'][name]=file_hash(path)
            if name=='summary.json':receipt['summary']=value
            write_json(receipt_path,receipt)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            path.write_bytes(original);receipt_path.write_bytes(original_receipt)

    def test_accepted_empty_or_missing_table_producer_rejected(self):
        for name in re.TABLES:
            result=deepcopy(self.result);result['tables'][name]=result['tables'][name].iloc[0:0]
            result['metadata']['row_counts'][name]=0
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)
        self.assertEqual(list(self.output.rglob('receipt.json')),[])

    def test_malformed_maps_statuses_and_model_inventory_rejected(self):
        malformed=[[],{'tables':None,'metadata':{}},deepcopy(self.result),deepcopy(self.result),deepcopy(self.result)]
        malformed[2]['metadata']['accepted']=1
        malformed[3]['metadata']['models']=malformed[3]['metadata']['models'][:-1]
        malformed[4]['metadata']['status']='fit_failed'
        for result in malformed:
            with self.subTest(type=type(result).__name__),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_producer_cannot_drop_or_duplicate_coefficients_covariance_or_models(self):
        mutations=[lambda r:r['tables']['coefficients'].drop(index=0,inplace=True),
                   lambda r:r['tables']['covariance'].drop(index=0,inplace=True),
                   lambda r:r['metadata']['models'][0]['terms'].pop(),
                   lambda r:r['metadata']['models'].append(deepcopy(r['metadata']['models'][0]))]
        for mutation in mutations:
            result=deepcopy(self.result);mutation(result)
            result['metadata']['row_counts']={n:len(f) for n,f in result['tables'].items()}
            with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_finite_wrong_coefficients_se_covariance_and_wald_are_rejected(self):
        mutations=[('coefficients','coefficient'),('coefficients','robust_se'),('covariance','covariance'),
                   ('mmef_change','estimate'),('mmef_change','bootstrap_replicates'),
                   ('x6_repeated','ci_low'),('x6_global_tests','interaction_p')]
        for name,column in mutations:
            result=deepcopy(self.result);result['tables'][name][column]=result['tables'][name][column].astype(float);result['tables'][name].loc[0,column]+=.1
            with self.subTest(name=name,column=column),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_prediction_values_and_matching_intervals_reconstructed_from_saved_fit(self):
        for name,field in [('x6_predictions','predicted_mean'),('x6_age_contrasts','difference_tb_minus_reference')]:
            result=deepcopy(self.result)
            result['tables'][name].loc[0,[field,'ci_low','ci_high']]+=.1
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_changed_grid_labels_counts_units_and_precision_evidence_rejected(self):
        def column(result,table,field,value):
            frame=result['tables'][table];frame[field]=value
        mutations=[lambda r:column(r,'x6_predictions','age_years',4.),
                   lambda r:column(r,'mmef_descriptive','group','wrong'),
                   lambda r:column(r,'x6_repeated','unit','wrong'),
                   lambda r:column(r,'mmef_change','n_children',0),
                   lambda r:r['metadata']['mmef_delta_precision'].__setitem__('sample_and_designs_unchanged',False),
                   lambda r:r['metadata']['models'][0].__setitem__('family','wrong'),
                   lambda r:column(r,'coefficients','participant_identifier','forbidden')]
        for mutation in mutations:
            result=deepcopy(self.result);mutation(result)
            with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_failed_numerical_fit_preserves_complete_null_evidence_and_false_status(self):
        result=deepcopy(self.result);meta=result['metadata']['models'][0];key=meta['model_id']
        meta.update(accepted=False,status='fit_failed',failure_reasons=['synthetic numerical failure'])
        result['metadata'].update(accepted=False,status='fit_failed',failure_reasons=[key+'/full: synthetic numerical failure'])
        for name,columns in [('coefficients',['coefficient','robust_se','model_se']),('covariance',['covariance'])]:
            frame=result['tables'][name];frame.loc[frame.model_id.eq(key),columns]=np.nan
        row=result['tables']['mmef_change'];row.loc[0,re.SCHEMAS['mmef_change'][8:24]]=np.nan;row.loc[0,'converged']=False
        first=self.run_fit(compute=lambda *a:result)
        self.assertFalse(first['accepted']);self.assertEqual(first['status'],'fit_failed')
        self.assertEqual(first['rows'],result['metadata']['row_counts'])
        self.assertFalse((self.work/'supporting_current.json').exists())
        self.assertFalse(self.run_fit(resume=True,compute=lambda *a:result)['accepted'])

    def test_failed_status_cannot_retain_coefficient_inference(self):
        result=deepcopy(self.result);meta=result['metadata']['models'][0];meta.update(accepted=False,status='fit_failed',failure_reasons=['failure'])
        result['metadata'].update(accepted=False,status='fit_failed',failure_reasons=[meta['model_id']+'/full: failure'])
        with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:result)

    def test_prepared_term_inventory_and_counts_are_bound_to_result(self):
        self.contracts['mmef/change']['designs']['before5']['terms']=[]
        with self.assertRaises(ValidationError):self.run_fit()
        self.contracts=deepcopy(self.contract_fixture)
        self.contracts['mmef/change']['frame']=self.contracts['mmef/change']['frame'].iloc[:-1]
        with self.assertRaises(ValidationError):self.run_fit()

    def test_stale_import_or_mismatched_root_rejected_before_computation(self):
        for name in re.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;original=path.read_bytes();path.write_bytes(original+b'# changed\n')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('unexpected computation'))
            path.write_bytes(original)
        original_hash=re.file_hash;actual=Path(model.__file__).resolve()
        with mock.patch.object(re,'file_hash',side_effect=lambda p:'0'*64 if Path(p).resolve()==actual else original_hash(p)):
            with self.assertRaises(ValidationError):self.run_fit(compute=lambda *a:self.fail('stale loaded model'))

    def test_source_or_contract_change_during_fitting_prevents_publication(self):
        for source in (True,False):
            original=self.descriptor['contracts']['mmef/change']
            path=self.root/'src/dchs_lung_function/models/supporting_repeated.py';original_source=path.read_bytes()
            def changed(*args):
                if source:path.write_bytes(original_source+b'# changed\n')
                else:self.descriptor['contracts']['mmef/change']='c'*64
                return deepcopy(self.result)
            with self.subTest(source=source),self.assertRaises(ValidationError):self.run_fit(compute=changed)
            path.write_bytes(original_source);self.descriptor['contracts']['mmef/change']=original
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_clean()

    def test_changed_source_before_cached_return_rejected(self):
        self.run_fit();original=re.verify_files
        def changed(*args):
            original(*args);self.descriptor['contracts']['mmef/change']='c'*64
        with mock.patch.object(re,'verify_files',side_effect=changed),self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_locks_protect_resume_and_distinct_work_roots(self):
        self.run_fit()
        with re.preparation_locks(self.work,self.output):
            with self.assertRaises(ValidationError):self.run_fit(resume=True,compute=lambda *a:self.fail('computed during lock'))
            with self.assertRaises(ValidationError):self.run_fit(resume=True,work=self.base/'other',compute=lambda *a:self.fail('computed during lock'))
        self.assert_clean()

    def test_overlap_source_preparation_and_output_rejected(self):
        for work,output in [(self.root/'generated',self.output),(self.prep/'generated',self.output),
                            (self.work,self.work),(self.work,self.work/'child'),(self.work,self.base)]:
            with self.subTest(work=work),self.assertRaises(ValidationError):self.run_fit(work=work,output=output)

    def test_exception_or_partial_publication_leaves_no_receipt_and_releases_locks(self):
        with self.assertRaises(RuntimeError):self.run_fit(compute=lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic')))
        with mock.patch.object(re.os,'replace',side_effect=OSError('synthetic publication failure')):
            with self.assertRaises(OSError):self.run_fit()
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_clean()

    def test_symlink_and_traversal_artifact_claims_rejected(self):
        directory=self.directory(self.run_fit());path=directory/'mmef_change.csv';outside=self.base/'elsewhere.csv'
        path.rename(outside);path.symlink_to(outside)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)
        path.unlink();outside.rename(path)
        receipt=json.loads((directory/'receipt.json').read_text());receipt['files']['../elsewhere.csv']=receipt['files'].pop('mmef_change.csv')
        write_json(directory/'receipt.json',receipt)
        with self.assertRaises(ValidationError):self.run_fit(resume=True)


if __name__=='__main__':unittest.main()
