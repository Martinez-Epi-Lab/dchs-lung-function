"""Synthetic M12 raw/validation, table-publication and cumulative execution tests."""
from copy import deepcopy
from contextlib import redirect_stderr,redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path
import json,shutil,sys,tempfile,threading,unittest
from unittest import mock
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function import sensitivity_execution as pe,diagnostics,pipeline
from dchs_lung_function.analysis import sensitivities
from dchs_lung_function.models import sensitivity as model
from dchs_lung_function.execution import ArtifactWriter,file_hash,write_json
from dchs_lung_function.pipeline import ValidationError
from dchs_lung_function.sensitivity_sampling import resolved_sampling
from test_sensitivity_models import fabricated_contract
from test_sensitivities import fabricated_posterior,table_inputs,SETTINGS
REAL_DIAGNOSTICS=diagnostics.evaluate_diagnostics
REAL_SUMMARIZE=sensitivities.summarize_sensitivity
POSTERIOR_VARIABLES=model.fit_spec('early_core')['posterior_variables']

@lru_cache(maxsize=2)
def _fixture(target):
    contract=fabricated_contract(target)
    posterior=fabricated_posterior(contract,target,chains=4,draws=3200)
    result=REAL_SUMMARIZE(posterior,contract,target)
    return contract,posterior,result,model.validate_contract(contract,target)

def fabricated_result(target):return deepcopy(_fixture(target))

class SensitivityExecutionTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name);self.root,self.prep=self.base/'source',self.base/'preparation';self.prep.mkdir()
        self.work,self.output=self.base/'private-work',self.base/'aggregate-output'
        self.target='early_core';self.settings=deepcopy(SETTINGS)
        for name in (*pe.COMPUTE_SOURCES,*pe.VALIDATION_SOURCES):
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(pe.__file__).parent/name,path)
        (self.root/'requirements-models.lock.txt').write_text('# invented lock\n')
        self.descriptor={'sha256':'a'*64,'preparation_fingerprint':'b'*64}
        self.contract,self.posterior,self.result,self.validated=fabricated_result(self.target)
        self.shapes={name:value.shape for name,value in self.posterior.items()}
        rng=np.random.default_rng(81)
        self.stats={'diverging':np.zeros((4,3200),bool),'reached_max_treedepth':np.zeros((4,3200),bool),
                    'energy':rng.normal(size=(4,3200))}
        self.metadata={'target':self.target,'controls':resolved_sampling(self.target,self.settings),
            'runtime':{'python':'synthetic','packages':{}},'runtime_setup':{'synthetic':True},'elapsed_seconds':.1,
            'realized_chains':4,'realized_draws_per_chain':3200,'realized_tuning_steps':2200,
            'posterior_variables':list(self.posterior),'sample_stat_variables':list(self.stats)}
        self.diagnostic={'passed':True,'status':'passed','reasons':[],'metrics':{'synthetic':True}}
        patches=[mock.patch.object(pe,'contract_descriptor',side_effect=lambda *a:deepcopy(self.descriptor)),
            mock.patch.object(pe,'load_artifacts',side_effect=lambda *a:self.contract),
            mock.patch.object(pe,'validate_contract',side_effect=lambda *a:self.validated),
            mock.patch.object(pe,'configure_runtime',return_value={'synthetic':True}),
            mock.patch.object(pe,'expected_posterior_shapes',side_effect=lambda *a:deepcopy(self.shapes)),
            mock.patch.object(pe,'model_runtime_identity',return_value={'python':'synthetic','packages':{}}),
            mock.patch.object(sensitivities,'summarize_sensitivity',side_effect=lambda *a,**k:deepcopy(self.result)),
            mock.patch.object(diagnostics,'evaluate_diagnostics',side_effect=lambda *a,**k:deepcopy(self.diagnostic))]
        for patch in patches:patch.start();self.addCleanup(patch.stop)

    def sample(self,*args):
        metadata=deepcopy(self.metadata)
        metadata.update(controls=resolved_sampling(self.target,self.settings),target=self.target,runtime=pe.model_runtime_identity())
        return deepcopy(self.posterior),deepcopy(self.stats),metadata

    def run_target(self,*,resume=False,compute=None,work=None,output=None):
        with mock.patch.object(pe,'sample_contract',side_effect=compute or self.sample):
            return pe.sample_one(self.root,self.prep,self.target,self.settings,work or self.work,output or self.output,resume=resume)

    def directories(self,result):
        return self.work/self.target/result['fit_fingerprint'],self.output/self.target/result['validation_fingerprint']

    def assert_no_transient_files(self):
        self.assertEqual(list(self.base.rglob('.staging-*')),[])
        self.assertEqual(list(self.base.rglob('.preparation.lock')),[])

    def test_covariance_failure_retains_raw_fit_and_truthful_diagnostic_status(self):
        self.covariance.update(status='failed',accepted=False,joint_pass=False)
        first=self.run_target()
        self.assertTrue(first['diagnostics_passed'])
        self.assertFalse(first['covariance_ppc_passed'])
        self.assertEqual(first['status'],'model_adequacy_failed')
        raw,validated=self.directories(first)
        before=(raw/'posterior.npz').read_bytes()
        self.assertEqual(json.loads((validated/'covariance_ppc.json').read_text()),self.covariance)
        again=self.run_target(resume=True,compute=lambda *a:self.fail('adequacy failure must not cause an unrequested refit'))
        self.assertEqual(again['status'],'model_adequacy_failed')
        self.assertEqual((raw/'posterior.npz').read_bytes(),before)

    def test_unevaluable_covariance_cannot_pass_completion(self):
        self.covariance.update(status='not_evaluable',accepted=False,joint_pass=False)
        result=self.run_target()
        self.assertTrue(result['diagnostics_passed'])
        self.assertFalse(result['covariance_ppc_passed'])
        self.assertEqual(result['status'],'model_adequacy_failed')

    def test_repaired_hash_covariance_evidence_is_recomputed_on_resume(self):
        result=self.run_target();_raw,validated=self.directories(result)
        path=validated/'covariance_ppc.json'
        evidence=json.loads(path.read_text());evidence['invented_aggregate']=999
        write_json(path,evidence)
        receipt=json.loads((validated/'receipt.json').read_text())
        receipt['files']['covariance_ppc.json']=file_hash(path)
        write_json(validated/'receipt.json',receipt)
        with self.assertRaises(ValidationError):self.run_target(resume=True)

    def test_raw_and_validation_roundtrip_and_resume_skip_sampling(self):
        first = self.run_target()
        raw, validated = self.directories(first)
        self.assertEqual(first["status"], "diagnostics_passed")
        self.assertFalse(first["raw_fit_reused"])
        saved = pe.read_array_archive(raw / "posterior.npz")
        for key in self.posterior:
            np.testing.assert_array_equal(saved[key], self.posterior[key])
        receipt = json.loads((validated / "receipt.json").read_text())
        self.assertEqual(receipt["raw_receipt_sha256"], file_hash(raw / "receipt.json"))
        resumed = self.run_target(resume=True, compute=lambda *a: self.fail("sampled a verified existing fit"))
        self.assertTrue(resumed["raw_fit_reused"])
        self.assertTrue(resumed["validation_reused"])
        self.assert_no_transient_files()

    def test_complete_family_variables_and_child_coordinates_reach_strict_diagnostics(self):
        self.assertEqual(len(self.posterior), 25)
        with mock.patch.object(diagnostics, "evaluate_diagnostics", wraps=REAL_DIAGNOSTICS) as checked:
            result = self.run_target()
            self.assertTrue(result["diagnostics_passed"])
            self.assertEqual(set(checked.call_args.args[0]), set(POSTERIOR_VARIABLES))
            self.assertEqual(checked.call_args.args[2], pe.CRITERIA)
        shutil.rmtree(self.output); shutil.rmtree(self.work)
        self.posterior["mechanics"][0, :, -1] += 20
        with mock.patch.object(diagnostics, "evaluate_diagnostics", wraps=REAL_DIAGNOSTICS):
            self.assertFalse(self.run_target()["diagnostics_passed"])

    def test_missing_mandatory_sampler_statistics_never_pass_validation(self):
        for field in ["diverging", "energy", "reached_max_treedepth"]:
            saved = self.stats.pop(field)
            self.metadata["sample_stat_variables"] = list(self.stats)
            with mock.patch.object(diagnostics, "evaluate_diagnostics", wraps=REAL_DIAGNOSTICS):
                with self.subTest(field=field):
                    result = self.run_target()
                    self.assertFalse(result["diagnostics_passed"])
                    self.assertEqual(result["status"], "diagnostics_failed")
            shutil.rmtree(self.work); shutil.rmtree(self.output)
            self.stats[field] = saved
        self.assert_no_transient_files()

    def test_cached_counts_target_and_result_rows_are_recomputed(self):
        first = self.run_target(); _, directory = self.directories(first)
        path = directory / "summary.json"; original = json.loads(path.read_text())
        receipt_path = directory / "receipt.json"; receipt_original = json.loads(receipt_path.read_text())
        for field, value in [("counts", {}), ("target", "early_maternal_hiv"), ("result_rows", 5)]:
            summary = deepcopy(original); summary[field] = value; write_json(path, summary)
            receipt = deepcopy(receipt_original); receipt["summary"] = summary
            receipt["files"][path.name] = file_hash(path); write_json(receipt_path, receipt)
            with self.subTest(field=field), self.assertRaises(ValidationError): self.run_target(resume=True)
            write_json(path, original); write_json(receipt_path, receipt_original)

    def test_malformed_raw_or_validation_receipt_is_controlled_failure(self):
        first = self.run_target(); raw, validated = self.directories(first)
        for directory in [raw, validated]:
            path = directory / "receipt.json"; original = path.read_bytes(); write_json(path, [])
            with self.subTest(directory=directory.name), self.assertRaises(ValidationError): self.run_target(resume=True)
            path.write_bytes(original)

    def test_malformed_hashed_diagnostics_is_controlled_failure(self):
        first = self.run_target(); _, directory = self.directories(first)
        path = directory / "diagnostics.json"; write_json(path, [])
        receipt_path = directory / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"][path.name] = file_hash(path); write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_cached_posterior_shape_checked_even_with_updated_file_hash(self):
        first = self.run_target(); raw, _ = self.directories(first)
        posterior = deepcopy(self.posterior); posterior["mechanics"] = posterior["mechanics"][:, :, :-1]
        np.savez_compressed(raw / "posterior.npz", **posterior)
        receipt_path = raw / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"]["posterior.npz"] = file_hash(raw / "posterior.npz"); write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_source_changes_during_cached_recomputation_block_return(self):
        self.run_target()
        def changed(*args, **kwargs):
            self.descriptor["sha256"] = "e" * 64
            return self.result
        with mock.patch.object(sensitivities, "summarize_sensitivity", side_effect=changed):
            with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_identical_fit_requires_explicit_resume(self):
        self.run_target()
        with self.assertRaises(ValidationError):
            self.run_target(compute=lambda *a: self.fail("unexpected sampling"))

    def test_reporting_or_diagnostic_changes_revalidate_existing_raw_fit(self):
        first = self.run_target()
        for name in pe.VALIDATION_SOURCES:
            path = self.root / "src/dchs_lung_function" / name
            path.write_text(path.read_text() + "# changed validation\n")
            with self.assertRaises(ValidationError): self.run_target(resume=True)
            # Simulate a fresh interpreter loading the changed reporting source.
            with mock.patch.object(pe, "assert_loaded_sources"):
                result = self.run_target(resume=True, compute=lambda *a: self.fail("validation edit caused refit"))
            self.assertEqual(result["fit_fingerprint"], first["fit_fingerprint"])
            self.assertNotEqual(result["validation_fingerprint"], first["validation_fingerprint"])
            self.assertTrue(result["raw_fit_reused"])
            self.assertFalse(result["validation_reused"])
            first = result

    def test_model_data_sampling_runtime_and_lock_changes_require_new_fit(self):
        first = self.run_target()
        fingerprints = {first["fit_fingerprint"]}
        changes = [lambda: self.descriptor.update(sha256="c" * 64),
                   lambda: self.settings.update(seed_offset=104730),
                   lambda: (self.root / "requirements-models.lock.txt").write_text("# changed environment\n")]
        for change in changes:
            change(); result = self.run_target(resume=True)
            self.assertNotIn(result["fit_fingerprint"], fingerprints)
            self.assertFalse(result["raw_fit_reused"])
            fingerprints.add(result["fit_fingerprint"])
        with mock.patch.object(pe, "model_runtime_identity", return_value={"python": "changed", "packages": {}}):
            result = self.run_target(resume=True)
            self.assertNotIn(result["fit_fingerprint"], fingerprints)

    def test_mismatched_source_tree_and_stale_import_rejected_before_sampling(self):
        for name in (*pe.COMPUTE_SOURCES,*pe.VALIDATION_SOURCES):
            path=self.root/"src/dchs_lung_function"/name; original=path.read_bytes()
            path.write_bytes(original+b"# changed after import\n")
            with self.subTest(name=name),self.assertRaises(ValidationError):
                self.run_target(compute=lambda *a:self.fail("sampled mismatched source tree"))
            path.write_bytes(original)
        original_hash=pe.file_hash
        actual=(Path(pe.__file__).parent/"models/sensitivity.py").resolve()
        def stale(path):
            return "0"*64 if Path(path).resolve()==actual else original_hash(path)
        with mock.patch.object(pe,"file_hash",side_effect=stale), self.assertRaises(ValidationError):
            self.run_target(compute=lambda *a:self.fail("sampled stale imported code"))
        self.assertEqual(list(self.work.rglob("receipt.json")),[])
        self.assert_no_transient_files()

    def test_failed_diagnostics_keep_raw_evidence_and_failed_status(self):
        self.diagnostic = {"passed": False, "status": "failed", "reasons": ["synthetic failure"], "metrics": {}}
        result = self.run_target()
        self.assertEqual(result["status"], "diagnostics_failed")
        self.assertFalse(result["diagnostics_passed"])
        raw, validated = self.directories(result)
        self.assertTrue((raw / "posterior.npz").is_file())
        self.assertEqual(json.loads((raw / "receipt.json").read_text())["status"], "sampled")
        self.assertEqual(json.loads((validated / "summary.json").read_text())["status"], "diagnostics_failed")
        again = self.run_target(resume=True, compute=lambda *a: self.fail("failed fit was sampled again"))
        self.assertFalse(again["diagnostics_passed"])
        self.assert_no_transient_files()

    def test_sampling_failure_has_no_raw_or_validation_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "fabricated"):
            self.run_target(compute=lambda *a: (_ for _ in ()).throw(RuntimeError("fabricated sampling failure")))
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_incomplete_posterior_never_reaches_diagnostics_or_publication(self):
        del self.posterior["mechanics"]
        with mock.patch.object(diagnostics, "evaluate_diagnostics") as checked:
            with self.assertRaises(ValidationError):
                self.run_target()
            checked.assert_not_called()
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_summary_failure_keeps_verified_raw_fit_and_can_be_resumed(self):
        with mock.patch.object(sensitivities, "summarize_sensitivity", side_effect=RuntimeError("fabricated summary failure")):
            with self.assertRaises(RuntimeError):
                self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        recovered = self.run_target(resume=True, compute=lambda *a: self.fail("raw evidence was discarded"))
        self.assertTrue(recovered["raw_fit_reused"])
        self.assert_no_transient_files()

    def test_tampered_raw_arrays_or_sampling_metadata_rejected(self):
        first = self.run_target(); raw, _ = self.directories(first)
        for name in ["posterior.npz", "sample_stats.npz", "sampling.json"]:
            path = raw / name; original = path.read_bytes()
            path.write_bytes(original + b"tampered")
            with self.subTest(name=name), self.assertRaises(ValidationError):
                self.run_target(resume=True)
            path.write_bytes(original)

    def test_raw_receipt_status_identity_and_artifact_inventory_rejected(self):
        first = self.run_target(); raw, _ = self.directories(first)
        path = raw / "receipt.json"; original = json.loads(path.read_text())
        for field, value in [("status", "validated"), ("fingerprint", "wrong"), ("identity", {}), ("files", {})]:
            changed = deepcopy(original); changed[field] = value; write_json(path, changed)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.run_target(resume=True)
            write_json(path, original)

    def test_validation_summary_and_raw_binding_tampering_rejected(self):
        first = self.run_target(); _, output = self.directories(first)
        path = output / "receipt.json"; original = json.loads(path.read_text())
        for field, value in [("summary", {"diagnostics_passed": True}), ("raw_receipt_sha256", "0" * 64), ("files", {})]:
            changed = deepcopy(original); changed[field] = value; write_json(path, changed)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.run_target(resume=True)
            write_json(path, original)
        summary = output / "summary.json"; summary.write_text(summary.read_text() + "\n")
        with self.assertRaises(ValidationError):
            self.run_target(resume=True)

    def test_cached_summary_must_agree_with_verified_diagnostic_result(self):
        self.diagnostic = {"passed": False, "status": "failed", "reasons": ["fabricated"], "metrics": {}}
        first = self.run_target(); _, output = self.directories(first)
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(diagnostics_passed=True, status="diagnostics_passed")
        write_json(summary_path, summary)
        receipt_path = output / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["summary"] = summary
        receipt["files"]["summary.json"] = file_hash(summary_path)
        write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_realized_sampling_metadata_must_match_expected_controls(self):
        def mismatched(*args):
            posterior, stats, metadata = self.sample()
            metadata["controls"]["tune"] = 0
            return posterior, stats, metadata
        with self.assertRaises(ValidationError): self.run_target(compute=mismatched)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])

    def test_cached_metadata_identity_and_realized_sampling_counts_checked(self):
        first = self.run_target(); raw, validated = self.directories(first)
        metadata_path = raw / "sampling.json"; original_metadata = json.loads(metadata_path.read_text())
        receipt_path = raw / "receipt.json"; original_receipt = json.loads(receipt_path.read_text())
        mutations = [lambda m: m["controls"].update(random_seed=0),
                     lambda m: m.update(realized_chains=3),
                     lambda m: m.update(realized_draws_per_chain=100),
                     lambda m: m.update(realized_tuning_steps=0),
                     lambda m: m.update(target="early_maternal_hiv"),
                     lambda m: m.update(runtime={"python": "unexpected"}),
                     lambda m: m.update(posterior_variables=["global"]),
                     lambda m: m.update(sample_stat_variables=["energy"])]
        for mutation in mutations:
            if validated.exists(): shutil.rmtree(validated)
            metadata = deepcopy(original_metadata); mutation(metadata); write_json(metadata_path, metadata)
            receipt = deepcopy(original_receipt); receipt["files"]["sampling.json"] = file_hash(metadata_path)
            write_json(receipt_path, receipt)
            with self.subTest(metadata=metadata), self.assertRaises(ValidationError):
                self.run_target(resume=True)
            write_json(metadata_path, original_metadata); write_json(receipt_path, original_receipt)

    def test_symlink_and_traversal_artifact_locations_rejected(self):
        first = self.run_target(); raw, _ = self.directories(first)
        posterior = raw / "posterior.npz"; outside = self.base / "outside.npz"
        posterior.rename(outside); posterior.symlink_to(outside)
        with self.assertRaises(ValidationError): self.run_target(resume=True)
        posterior.unlink(); outside.rename(posterior)
        receipt_path = raw / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"]["../outside.npz"] = "0" * 64; write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_model_or_contract_changes_during_sampling_reject_publication(self):
        def changed(*args, **kwargs):
            self.descriptor["sha256"] = "d" * 64
            return self.sample()
        with self.assertRaises(ValidationError): self.run_target(compute=changed)
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_change_during_new_summary_rejected_but_raw_preserved(self):
        def changed(*args, **kwargs):
            (self.root / "src/dchs_lung_function/diagnostics.py").write_text("# changed during check\n")
            return self.result
        with mock.patch.object(sensitivities, "summarize_sensitivity", side_effect=changed):
            with self.assertRaises(ValidationError): self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_criteria_mutation_during_run_is_rejected(self):
        def changed(*args, **kwargs):
            pe.CRITERIA["min_tail_ess"] = 800
            return self.result
        with mock.patch.object(pe, "CRITERIA", deepcopy(pe.CRITERIA)):
            with mock.patch.object(sensitivities, "summarize_sensitivity", side_effect=changed):
                with self.assertRaises(ValidationError): self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_change_during_cached_resume_is_rejected(self):
        self.run_target()
        original = pe.verify_files
        def changed(directory, entries, required):
            result = original(directory, entries, required)
            if "diagnostics.json" in required:
                (self.root / "src/dchs_lung_function/diagnostics.py").write_text("# changed during cached validation\n")
            return result
        with mock.patch.object(pe, "verify_files", side_effect=changed):
            with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_validation_publication_failure_preserves_raw_fit(self):
        original = pe.os.replace
        def replace(source, target):
            if self.output.resolve() in Path(target).resolve().parents:
                raise OSError("fabricated publication failure")
            return original(source, target)
        with mock.patch.object(pe.os, "replace", side_effect=replace):
            with self.assertRaises(OSError): self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_source_preparation_and_overlapping_output_roots_rejected(self):
        for work, output in [(self.root / "generated", self.output), (self.prep / "generated", self.output),
                             (self.work, self.work), (self.work, self.work / "child"),
                             (self.work, self.base)]:
            with self.subTest(work=work, output=output), self.assertRaises(ValidationError):
                self.run_target(work=work, output=output)

    def test_resume_and_distinct_work_roots_obey_shared_output_lock(self):
        self.run_target()
        with pe.preparation_locks(self.work / "early_core", self.output / "early_core"):
            with self.assertRaises(ValidationError): self.run_target(resume=True)
            with self.assertRaises(ValidationError):
                self.run_target(work=self.base / "other-work", resume=True)
        self.assert_no_transient_files()

    def test_concurrent_different_work_root_cannot_publish_same_target(self):
        entered, release = threading.Event(), threading.Event()
        failures = []
        def compute(*args):
            entered.set()
            if not release.wait(10): raise RuntimeError("test synchronization timed out")
            return self.sample()
        def worker():
            try: self.run_target(compute=compute)
            except BaseException as exc: failures.append(exc)
        thread = threading.Thread(target=worker); thread.start()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertRaises(ValidationError):
                self.run_target(work=self.base / "other-work", resume=True)
        finally:
            release.set(); thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assert_no_transient_files()

    def test_window_empty_hazard_table_roundtrips_with_its_complete_schema(self):
        self.target='window_ratio_pre';self.contract,self.posterior,self.result,self.validated=fabricated_result(self.target)
        self.shapes={n:a.shape for n,a in self.posterior.items()};self.metadata['posterior_variables']=list(self.posterior)
        result=self.run_target();self.assertEqual(result['result_rows'],{'estimates':8,'hazard':0,'sample_counts':30})
        _,directory=self.directories(result);frame=pd.read_csv(directory/'hazard.csv')
        self.assertEqual(list(frame),sensitivities.HAZARD_COLUMNS);self.assertEqual(len(frame),0)
        self.assertTrue(self.run_target(resume=True)['diagnostics_passed'])

    def test_result_schema_and_private_metadata_additions_are_rejected(self):
        for location in ('table','metadata','label','counts'):
            result=deepcopy(self.result)
            if location=='table':result.estimates['pid_child']='FABRICATED_PRIVATE_RECORD'
            elif location=='metadata':result.metadata['participant_records']=['FABRICATED_PRIVATE_RECORD']
            elif location=='label':result.estimates.loc[0,'contrast']='FABRICATED_PRIVATE_RECORD'
            else:result.sample_counts.loc[0,'children']+=1
            with self.subTest(location=location),self.assertRaises(ValidationError):pe.validate_results(result,self.contract,self.target,self.posterior)

    def test_independent_primitive_checks_reject_mutated_producer_numbers(self):
        for name,column in [('estimates','mean'),('estimates','lower'),('hazard','upper'),('sample_counts','measurements')]:
            result=deepcopy(self.result);frame=getattr(result,name)
            frame[column]=frame[column].astype(float);frame.loc[0,column]+=.1
            with mock.patch.object(sensitivities,'summarize_sensitivity',return_value=result):
                with self.subTest(name=name,column=column),self.assertRaises(ValidationError):pe.validate_results(result,self.contract,self.target,self.posterior)

    def test_all_repaired_cached_table_values_are_recomputed(self):
        result=self.run_target();_,directory=self.directories(result);receipt_path=directory/'receipt.json';old_receipt=receipt_path.read_bytes()
        for name in pe.TABLES:
            path=directory/(name+'.csv');old=path.read_bytes();frame=pd.read_csv(path,float_precision='round_trip')
            column='children' if name=='sample_counts' else 'mean';frame[column]=frame[column].astype(float);frame.loc[0,column]+=.01
            frame.to_csv(path,index=False,float_format='%.17g');receipt=json.loads(old_receipt);receipt['files'][path.name]=file_hash(path);write_json(receipt_path,receipt)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_target(resume=True)
            path.write_bytes(old);receipt_path.write_bytes(old_receipt)

    def test_repaired_summary_extra_record_fields_and_boolean_counts_rejected(self):
        result=self.run_target();_,directory=self.directories(result);path=directory/'summary.json';old=path.read_bytes()
        rp=directory/'receipt.json';old_receipt=rp.read_bytes()
        for mode in ('extra','bool_count','free_text'):
            value=json.loads(old)
            if mode=='extra':value['participant_records']=['FABRICATED_PRIVATE_RECORD']
            elif mode=='bool_count':value['counts']['weights'][0]=True
            else:value['scientific_reference_comparison']='FABRICATED_PRIVATE_RECORD'
            write_json(path,value);receipt=json.loads(old_receipt);receipt['summary']=value;receipt['files'][path.name]=file_hash(path);write_json(rp,receipt)
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.run_target(resume=True)
            path.write_bytes(old);rp.write_bytes(old_receipt)

    def test_publication_receipts_and_directories_are_nonsymlink_exact_inventories(self):
        result=self.run_target();raw,validation=self.directories(result)
        for directory in (raw,validation):
            for name in ('receipt.json',next(n for n in json.loads((directory/'receipt.json').read_text())['files'])):
                path=directory/name;outside=self.base/('external-'+directory.name+name);path.rename(outside);path.symlink_to(outside)
                with self.subTest(name=name,directory=directory.name),self.assertRaises(ValidationError):self.run_target(resume=True)
                path.unlink();outside.rename(path)
            extra=directory/'untracked.csv';extra.write_text('untracked\n')
            with self.assertRaises(ValidationError):self.run_target(resume=True)
            extra.unlink()
            outside=self.base/('directory-'+directory.name);directory.rename(outside);directory.symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValidationError):self.run_target(resume=True)
            directory.unlink();outside.rename(directory)

    def test_target_and_compiler_symlinks_rejected_before_runtime_setup(self):
        for mode in ('work','output','compiler'):
            self.work.mkdir(exist_ok=True);self.output.mkdir(exist_ok=True)
            if mode=='compiler':
                (self.work/self.target).mkdir();link=self.work/self.target/'compiler_cache'
            else:link=(self.work if mode=='work' else self.output)/self.target
            link.symlink_to(self.base/'missing',target_is_directory=True)
            with mock.patch.object(pe,'configure_runtime') as configure:
                with self.subTest(mode=mode),self.assertRaises(ValidationError):self.run_target()
                configure.assert_not_called()
            link.unlink()
            if mode=='compiler':(self.work/self.target).rmdir()

    def test_destination_races_reject_dangling_links_and_preserve_raw_evidence(self):
        def raced_raw(*args):
            fp,_=pe.sampling_identity(self.root,self.descriptor,self.target,self.settings)
            (self.work/self.target/fp).symlink_to(self.base/'missing',target_is_directory=True)
            return self.sample()
        with self.assertRaises(ValidationError):self.run_target(compute=raced_raw)
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.assert_no_transient_files()
        fp,_=pe.sampling_identity(self.root,self.descriptor,self.target,self.settings)
        (self.work/self.target/fp).unlink()
        def raced_validation(*args,**kwargs):
            vfp,_=pe.validation_identity(self.root,fp,pe.CRITERIA);destination=self.output/self.target/vfp
            if not destination.is_symlink():destination.symlink_to(self.base/'missing',target_is_directory=True)
            return deepcopy(self.result)
        with mock.patch.object(sensitivities,'summarize_sensitivity',side_effect=raced_validation):
            with self.assertRaises(ValidationError):self.run_target()
        self.assertEqual(len(list(self.work.rglob('posterior.npz'))),1);self.assert_no_transient_files()


class SensitivityDescriptorTests(unittest.TestCase):
    def test_typed_selected_contract_and_receipt_integrity_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep=Path(tmp);target='early_core';contract=fabricated_contract(target)
            def publish():
                manifest=ArtifactWriter(prep).finish({'M03':{'pft':{'samples':{model.contract_key(target):contract}}}})
                identity={'through':'M03','fixture':'invented'}
                write_json(prep/'receipt.json',{'through':'M03','status':'prepared','identity':identity,'fingerprint':pe.digest(identity),
                    'artifact_manifest_sha256':file_hash(prep/'artifacts.json')})
                return manifest
            publish();first=pe.contract_descriptor(prep,target)
            contract['weights'][0]+=1;manifest=publish();second=pe.contract_descriptor(prep,target)
            self.assertNotEqual(first['sha256'],second['sha256'])
            with self.assertRaises(ValidationError):pe.contract_descriptor(prep,'unknown')
            with self.assertRaises(ValidationError):pe.contract_descriptor(prep,'early_maternal_hiv')
            path=prep/next(iter(manifest['files']));path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValidationError):pe.contract_descriptor(prep,target)


class SensitivityBatchAndCliTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        from dchs_lung_function import feno_execution,sensitivity_tables_execution
        self.feno=feno_execution;self.tables=sensitivity_tables_execution
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name).resolve();self.root=self.base/'source';self.work=self.base/'work';self.output=self.base/'outputs'
        self.work.mkdir();self.output.mkdir()
        for name in (*pe.COMPUTE_SOURCES,*pe.VALIDATION_SOURCES):
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(pe.__file__).parent/name,path)
        self.paths=[self.base/n for n in ['paths.json','analysis.json','primary.json','mechanics.json','supporting.json','wheeze.json','sensitivity.json']]
        write_json(self.paths[0],{'inputs':{},'directories':{'work':str(self.work),'outputs':str(self.output)}})
        write_json(self.paths[1],{'synthetic':True})
        for path in self.paths[2:]:write_json(path,SETTINGS)
        self.prior={'through':'M11','accepted':True,'status':'accepted','selected_models':11,
            'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
            'mechanics_models_verified':6,'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
            'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,'wheeze_models_verified':2,
            'preparation_fingerprint':'a'*64}
        new,_,_=table_inputs();self.results={}
        for target,entry in new.items():
            spec=model.fit_spec(target)
            self.results[target]={**entry['summary'],'covariance_ppc_passed':True,'result_rows':{'estimates':spec['result_rows'],'hazard':spec['hazard_rows'],'sample_counts':30},
                'scientific_reference_comparison':'requires separate independent validation','raw_fit_reused':True,'validation_reused':True}
        self.table_summary={'fingerprint':'d'*64,'accepted':True,'status':'accepted','rows':{'table_s1':10,'table_s3':24,'table_s4':12},
            'new_fit_count':10,'reused_fit_count':4,'source_fits_verified':14,'reused_primary_models':2,'reused_wheeze_models':2,
            'scientific_reference_comparison':'requires separate independent validation','resumed':True}

    def batch(self,*,prior=None,fit=None,table=None,prepared=None,jobs=1):
        with mock.patch.object(self.feno,'run_feno',side_effect=prior or (lambda *a,**k:deepcopy(self.prior))):
            with mock.patch.object(pe,'sample_one',side_effect=fit or (lambda r,p,t,*a,**k:deepcopy(self.results[t]))):
                with mock.patch.object(self.tables,'publish_tables',side_effect=table or (lambda *a,**k:deepcopy(self.table_summary))):
                    with mock.patch.object(pe,'run_preparation',side_effect=prepared or (lambda *a,**k:{'fingerprint':'a'*64})):
                        return pe.run_sensitivities(self.root,*self.paths,jobs=jobs,resume=True)

    def test_all_ten_new_four_reused_and_prior_modules_required_for_marker(self):
        result=self.batch();self.assertTrue(result['accepted']);self.assertEqual(result['through'],'M12')
        self.assertEqual(result['selected_models'],10);self.assertEqual(result['reused_models'],4)
        self.assertEqual([r['target'] for r in result['sensitivity_models']],list(model.DEFAULT_TARGETS))
        self.assertEqual(result,json.loads((self.work/'sensitivity_current.json').read_text()))
        self.assertEqual(result['feno_models_verified'],11);self.assertIn('G08',result['inherited_limitations'][0])

    def test_inadequate_model_never_reaches_table_publication(self):
        self.results['early_core'].update(covariance_ppc_passed=False,status='model_adequacy_failed')
        result=self.batch(table=lambda *a,**k:self.fail('failed adequacy reached table publication'))
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'model_adequacy_failed')
        self.assertIsNone(result['sensitivity_tables'])
        self.assertFalse((self.work/'sensitivity_current.json').exists())

    def test_failed_diagnostic_retains_failure_and_never_invokes_table_publication(self):
        self.results['early_core'].update(diagnostics_passed=False,status='diagnostics_failed')
        result=self.batch(table=lambda *a,**k:self.fail('failed diagnostics reached table publication'))
        self.assertFalse(result['accepted']);self.assertIsNone(result['sensitivity_tables'])
        self.assertFalse((self.work/'sensitivity_current.json').exists())

    def test_every_prior_count_status_and_typed_acceptance_gate_is_required(self):
        for field in self.prior:
            if field=='preparation_fingerprint':continue
            saved=self.prior[field];self.prior[field]=False if field=='accepted' else 'wrong' if isinstance(saved,str) else saved-1
            with self.subTest(field=field),self.assertRaises(ValidationError):self.batch(fit=lambda *a,**k:self.fail('fit after failed prerequisite'))
            self.prior[field]=saved
        self.prior['wheeze_models_verified']=True
        with self.assertRaises(ValidationError):self.batch()

    def test_duplicate_missing_or_malformed_new_target_returns_rejected(self):
        for mode in ('duplicate','none','unknown','list'):
            def bad(root,prep,target,*args,**kwargs):
                result=deepcopy(self.results[target])
                if target=='early_maternal_hiv':
                    if mode=='duplicate':result['target']='early_core'
                    elif mode=='unknown':result['target']='unknown'
                    elif mode=='none':return None
                    else:return []
                return result
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.batch(fit=bad)

    def test_new_fit_complete_row_inventories_status_and_privacy_schema_required(self):
        for mode in ('bool_accept','wrong_status','missing_effect','missing_hazard','wrong_counts','extra_field','bool_reuse'):
            def bad(root,prep,target,*args,**kwargs):
                result=deepcopy(self.results[target])
                if target=='early_core':
                    if mode=='bool_accept':result['diagnostics_passed']=1
                    elif mode=='wrong_status':result['status']='diagnostics_failed'
                    elif mode=='missing_effect':result['result_rows']['estimates']-=1
                    elif mode=='missing_hazard':result['result_rows']['hazard']-=1
                    elif mode=='wrong_counts':result['result_rows']['sample_counts']=29
                    elif mode=='extra_field':result['participant_records']=['FABRICATED_PRIVATE_RECORD']
                    else:result['raw_fit_reused']=1
                return result
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.batch(fit=bad)

    def test_table_completion_requires_all_fourteen_bindings_and_exact_rows(self):
        changes=[('accepted',False),('accepted',1),('status','failed'),('new_fit_count',9),('reused_fit_count',3),
            ('source_fits_verified',13),('reused_primary_models',1),('reused_wheeze_models',1),
            ('rows',{'table_s1':10,'table_s3':24,'table_s4':11}),('participant_records',['FABRICATED_PRIVATE_RECORD'])]
        for key,value in changes:
            summary={**self.table_summary,key:value}
            with self.subTest(key=key),self.assertRaises(ValidationError):self.batch(table=lambda *a,**k:summary)
        self.assertFalse((self.work/'sensitivity_current.json').exists())

    def test_all_seven_configuration_files_and_preparation_bound_until_completion(self):
        for path in self.paths:
            old=path.read_bytes()
            def changed(*a,**k):path.write_bytes(old+b'\n');return deepcopy(self.table_summary)
            with self.subTest(path=path.name),self.assertRaises(ValidationError):self.batch(table=changed)
            path.write_bytes(old)
        with self.assertRaises(ValidationError):self.batch(prepared=lambda *a,**k:{'fingerprint':'changed'})
        self.assertFalse((self.work/'sensitivity_current.json').exists())

    def test_invalid_jobs_and_generated_sampling_file_are_rejected(self):
        for jobs in (0,4,True,1.5):
            with self.subTest(jobs=jobs),self.assertRaises(ValidationError):self.batch(jobs=jobs)
        self.paths[-1]=self.output/'sampling.json';write_json(self.paths[-1],SETTINGS)
        with self.assertRaises(ValidationError):self.batch()

    def test_late_source_change_blocks_completion(self):
        def changed(*a,**k):
            path=self.root/'src/dchs_lung_function/analysis/sensitivities.py';path.write_text(path.read_text()+'# changed\n')
            return deepcopy(self.table_summary)
        with self.assertRaises(ValidationError):self.batch(table=changed)
        self.assertFalse((self.work/'sensitivity_current.json').exists())

    def cli(self,*,extra=None,result=None,error=None):
        stdout,stderr=StringIO(),StringIO();args=['--through','M12','--config',str(self.paths[0]),'--analysis',str(self.paths[1]),'--resume']
        default={'selected_models':10,'reused_models':4,'accepted':True,'status':'accepted'}
        with mock.patch.object(pe,'run_sensitivities',return_value=result or default,side_effect=error) as runner:
            with redirect_stdout(stdout),redirect_stderr(stderr):code=pipeline.main(args+(extra or []),root=self.root)
        return code,stdout.getvalue(),stderr.getvalue(),runner

    def test_cli_routes_all_five_sampling_configurations(self):
        flags=['--sampling','--mechanics-sampling','--supporting-sampling','--wheeze-sampling','--sensitivity-sampling']
        args=[part for flag,path in zip(flags,self.paths[2:]) for part in (flag,str(path))]+['--jobs','3']
        code,out,err,runner=self.cli(extra=args);self.assertEqual(code,0);self.assertEqual(err,'')
        self.assertIn('10 new models and 4 reused fits',out)
        self.assertEqual(runner.call_args.args[3:],tuple(map(str,self.paths[2:])))
        self.assertEqual(runner.call_args.kwargs,{'jobs':3,'resume':True})

    def test_cli_failure_partial_targets_and_private_errors_are_safe(self):
        code,out,_,_=self.cli(result={'selected_models':10,'reused_models':4,'accepted':False,'status':'diagnostics_failed'})
        self.assertEqual(code,2);self.assertIn('diagnostics_failed',out)
        code,_,_,runner=self.cli(extra=['--targets','early_core']);self.assertEqual(code,2);runner.assert_not_called()
        for error in (ValueError('FABRICATED_PRIVATE_RECORD'),OSError('FABRICATED_PRIVATE_RECORD'),RuntimeError('FABRICATED_PRIVATE_RECORD')):
            code,out,err,_=self.cli(error=error);self.assertEqual(code,2);self.assertNotIn('FABRICATED_PRIVATE_RECORD',out+err)

    def test_sensitivity_controls_are_unavailable_for_earlier_or_full_render_modes(self):
        for arguments in (['--full'],['--render'],['--through','M11','--sensitivity-sampling',str(self.paths[-1])]):
            with mock.patch.object(pe,'run_sensitivities') as runner,redirect_stderr(StringIO()):
                self.assertEqual(pipeline.main(arguments,root=self.root),2)
            runner.assert_not_called()


class SensitivityTableExecutionTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        from dchs_lung_function import sensitivity_tables_execution as tables,model_runtime
        from types import SimpleNamespace
        self.te=tables;self.runtime=model_runtime
        self.original_verified_fit=tables._verified_fit
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name).resolve();self.root=self.base/'source';self.prep=self.base/'preparation';self.prep.mkdir()
        self.work,self.output=self.base/'work',self.base/'outputs';self.work.mkdir();self.output.mkdir()
        self.directories={'work':self.work,'outputs':self.output}
        for name in tables.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(tables.__file__).parent/name,path)
        (self.root/'requirements-models.lock.txt').write_text('# invented lock\n')
        new,primary,wheeze=table_inputs();self.entries={'sensitivity':new,'primary':primary,'wheeze':wheeze}
        self.new_results=[]
        for target,entry in new.items():
            spec=model.fit_spec(target)
            self.new_results.append({**entry['summary'],'result_rows':{'estimates':spec['result_rows'],'hazard':spec['hazard_rows'],'sample_counts':30},
                'scientific_reference_comparison':'requires separate independent validation','raw_fit_reused':True,'validation_reused':True})
        self.proofs={};self.descriptors={};self.calls=[]
        for family,entries in self.entries.items():
            for target,entry in entries.items():
                source=self.base/(family+'_'+target+'.json');write_json(source,{'target':target,'fabricated':True})
                contract_sha=tables.digest({'family':family,'target':target});self.descriptors[(family,target)]=contract_sha
                binding={'family':family,'target':target,'contract_sha256':contract_sha,
                    'fit_fingerprint':entry['summary']['fit_fingerprint'],'validation_fingerprint':entry['summary']['validation_fingerprint'],
                    'raw_receipt_sha256':'e'*64,'validation_receipt_sha256':'f'*64}
                self.proofs[(family,target)]={'entry':entry,'binding':binding,'input_files':{source:file_hash(source)}}
        def verified(family,root,prep,directories,settings,target,supplied=None):
            self.calls.append((family,target));return deepcopy(self.proofs[(family,target)])
        self.verified=verified
        def api(family):
            return SimpleNamespace(CRITERIA={},
                contract_descriptor=lambda prep,target:{'sha256':self.descriptors[(family,target)]},
                sampling_identity=lambda root,d,target,settings:(self.entries[family][target]['summary']['fit_fingerprint'],{}),
                validation_identity=lambda root,fp,criteria:(next(e['summary']['validation_fingerprint'] for e in self.entries[family].values() if e['summary']['fit_fingerprint']==fp),{}))
        for patch in [mock.patch.object(tables,'_verified_fit',side_effect=verified),
                      mock.patch.object(tables,'_family_api',side_effect=api),
                      mock.patch.object(model_runtime,'model_runtime_identity',return_value={'synthetic':True})]:
            patch.start();self.addCleanup(patch.stop)

    def publish(self,*,resume=False,work=None,output=None):
        dirs={'work':work or self.work,'outputs':output or self.output}
        return self.te.publish_tables(self.root,self.prep,dirs,self.new_results,SETTINGS,SETTINGS,SETTINGS,resume=resume)
    def directory(self,result):return self.output/'sensitivity_tables'/result['fingerprint']
    def clean(self):
        self.assertEqual(list(self.base.rglob('.staging-*')),[]);self.assertEqual(list(self.base.rglob('.preparation.lock')),[])

    def test_precision_failure_blocks_real_table_publication(self):
        self.proofs[('sensitivity','early_maternal_hiv')]['entry']['estimates']['mcse_mean']=1.
        with self.assertRaisesRegex(ValidationError,'precision'):self.publish()
        self.assertEqual(list(self.output.rglob('receipt.json')),[])
        self.assertFalse((self.work/'sensitivity_current.json').exists())
        self.clean()

    def test_complete_publication_revalidates_all_fourteen_inputs_on_each_resume(self):
        first=self.publish();second=self.publish(resume=True)
        self.assertFalse(first['resumed']);self.assertTrue(second['resumed']);self.assertEqual(len(self.calls),28)
        self.assertEqual(set(self.calls),set(self.proofs));self.assertEqual(second['source_fits_verified'],14)
        self.assertEqual(second['rows'],{'table_s1':10,'table_s3':24,'table_s4':12})
        self.assertEqual({p.name for p in self.directory(first).iterdir()},set(self.te.RESULT_FILES)|{'receipt.json'})
        for name in self.te.TABLES:self.assertNotIn('pid_child',pd.read_csv(self.directory(first)/(name+'.csv')).columns)
        self.clean()

    def test_explicit_resume_required_for_existing_publication(self):
        self.publish()
        with self.assertRaises(ValidationError):self.publish()
        self.clean()

    def test_incomplete_duplicate_and_unselected_new_fit_completions_rejected(self):
        saved=deepcopy(self.new_results)
        for mode in ('missing','duplicate','unknown','none'):
            self.new_results=deepcopy(saved)
            if mode=='missing':self.new_results.pop()
            elif mode=='duplicate':self.new_results[-1]=deepcopy(self.new_results[0])
            elif mode=='unknown':self.new_results[0]['target']='unknown'
            else:self.new_results[0]=None
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.publish()
        self.assertEqual(self.calls,[])

    def test_every_missing_corrupt_and_untracked_result_file_rejected(self):
        directory=self.directory(self.publish())
        for name in self.te.RESULT_FILES:
            path=directory/name;old=path.read_bytes();path.write_bytes(old+b'changed')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.publish(resume=True)
            path.unlink()
            with self.subTest(name=name,missing=True),self.assertRaises(ValidationError):self.publish(resume=True)
            path.write_bytes(old)
        extra=directory/'private.csv';extra.write_text('FABRICATED_PRIVATE_RECORD\n')
        with self.assertRaises(ValidationError):self.publish(resume=True)
        extra.unlink();write_json(directory/'receipt.json',[])
        with self.assertRaises(ValidationError):self.publish(resume=True)

    def test_repaired_hash_changes_to_all_tables_metadata_and_summary_rejected(self):
        directory=self.directory(self.publish());rp=directory/'receipt.json';old_receipt=rp.read_bytes()
        for name in self.te.RESULT_FILES:
            path=directory/name;old=path.read_bytes()
            if name.endswith('.csv'):
                frame=pd.read_csv(path,float_precision='round_trip');column='posterior_median' if name=='table_s4.csv' else 'mean'
                frame.loc[0,column]+=.0001;frame.to_csv(path,index=False,float_format='%.17g')
            else:
                value=json.loads(old);value['participant_records']=['FABRICATED_PRIVATE_RECORD'];write_json(path,value)
            receipt=json.loads(old_receipt);receipt['files'][name]=file_hash(path)
            if name=='summary.json':receipt['summary']=value
            write_json(rp,receipt)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.publish(resume=True)
            path.write_bytes(old);rp.write_bytes(old_receipt)

    def test_exact_csv_bytes_catch_changes_even_with_equal_rounded_numbers(self):
        directory=self.directory(self.publish());path=directory/'table_s1.csv';path.write_bytes(path.read_bytes()+b'\n')
        rp=directory/'receipt.json';receipt=json.loads(rp.read_text());receipt['files'][path.name]=file_hash(path);write_json(rp,receipt)
        with self.assertRaises(ValidationError):self.publish(resume=True)

    def test_malformed_typed_or_traversal_receipt_is_rejected(self):
        directory=self.directory(self.publish());path=directory/'receipt.json';old=path.read_bytes()
        for change in ('schema_bool','summary_bool','identity_bool','traversal'):
            receipt=json.loads(old)
            if change=='schema_bool':receipt['schema_version']=True
            elif change=='summary_bool':receipt['summary']['new_fit_count']=True
            elif change=='identity_bool':receipt['identity']['schema_version']=True
            else:receipt['files']['../table_s1.csv']=receipt['files'].pop('table_s1.csv')
            write_json(path,receipt)
            with self.subTest(change=change),self.assertRaises(ValidationError):self.publish(resume=True)
            path.write_bytes(old)

    def test_symlinked_artifacts_destinations_and_children_are_rejected(self):
        directory=self.directory(self.publish())
        for name in (*self.te.RESULT_FILES,'receipt.json'):
            path=directory/name;outside=self.base/('outside-'+name);path.rename(outside);path.symlink_to(outside)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.publish(resume=True)
            path.unlink();outside.rename(path)
        outside=self.base/'outside-dir';directory.rename(outside);directory.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValidationError):self.publish(resume=True)
        directory.unlink();outside.rename(directory)
        shutil.rmtree(directory);directory.symlink_to(self.base/'missing',target_is_directory=True)
        with self.assertRaises(ValidationError):self.publish(resume=True)
        directory.unlink()
        parent=self.output/'sensitivity_tables';parent.rmdir();parent.symlink_to(self.base/'missing',target_is_directory=True)
        with self.assertRaises(ValidationError):self.publish()

    def test_changed_upstream_file_or_contract_before_new_or_cached_commit_rejected(self):
        for cached in (False,True):
            if cached:self.publish()
            original=self.te.check_tables
            def changed(result,expected):
                original(result,expected)
                path=next(iter(self.proofs[('primary','ratio_pre')]['input_files']));path.write_bytes(path.read_bytes()+b'changed')
            with mock.patch.object(self.te,'check_tables',side_effect=changed):
                with self.subTest(cached=cached),self.assertRaises(ValidationError):self.publish(resume=cached)
            path=next(iter(self.proofs[('primary','ratio_pre')]['input_files']));write_json(path,{'target':'ratio_pre','fabricated':True})
            self.clean()
        def changed(*args):
            self.descriptors[('wheeze','primary_sex')]='0'*64
        with mock.patch.object(self.te,'check_tables',side_effect=changed),self.assertRaises(ValidationError):self.publish(resume=True)

    def test_changed_source_root_or_loaded_module_prevents_publication(self):
        for name in self.te.SOURCES:
            path=self.root/'src/dchs_lung_function'/name;old=path.read_bytes();path.write_bytes(old+b'# changed\n')
            with self.subTest(name=name),self.assertRaises(ValidationError):self.publish()
            path.write_bytes(old)
        original=self.te.file_hash;actual=Path(self.te.__file__).resolve()
        with mock.patch.object(self.te,'file_hash',side_effect=lambda p:'0'*64 if Path(p).resolve()==actual else original(p)):
            with self.assertRaises(ValidationError):self.publish()
        self.assertEqual(self.calls,[])

    def test_late_source_runtime_and_lockfile_changes_block_publication(self):
        original=self.te.verify_result
        for mode in ('source','lock'):
            path=self.root/('src/dchs_lung_function/analysis/sensitivities.py' if mode=='source' else 'requirements-models.lock.txt');old=path.read_bytes()
            def changed(*args):original(*args);path.write_bytes(old+b'# changed\n')
            with mock.patch.object(self.te,'verify_result',side_effect=changed):
                with self.subTest(mode=mode),self.assertRaises(ValidationError):self.publish()
            path.write_bytes(old)
        def changed_runtime(*args):
            original(*args);self.runtime.model_runtime_identity.return_value={'synthetic':False}
        with mock.patch.object(self.te,'verify_result',side_effect=changed_runtime),self.assertRaises(ValidationError):self.publish()
        self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.clean()

    def test_mutated_producer_tables_and_metadata_fail_exact_comparison(self):
        expected=sensitivities.build_tables(*[self.entries[f] for f in ('sensitivity','primary','wheeze')])
        for mode in ('number','count','private_column','metadata'):
            result=deepcopy(expected)
            if mode=='number':result.table_s1.loc[0,'mean']+=.1
            elif mode=='count':result.table_s3=result.table_s3.iloc[:-1]
            elif mode=='private_column':result.table_s4['pid_child']='FABRICATED_PRIVATE_RECORD'
            else:result.metadata['participant_records']=['FABRICATED_PRIVATE_RECORD']
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.te.check_tables(result,expected)
        result=deepcopy(expected);result.metadata['new_fit_count']=True
        with self.assertRaises(ValidationError):self.te.check_tables(result,result)

    def test_private_output_locks_cover_resume_and_independent_work_roots(self):
        self.publish();work=self.work/'sensitivity_table_validation';out=self.output/'sensitivity_tables'
        with self.te.preparation_locks(work,out):
            with self.assertRaises(ValidationError):self.publish(resume=True)
            with self.assertRaises(ValidationError):self.publish(resume=True,work=self.base/'other-work')
        self.clean()

    def test_partial_writes_and_publication_failure_leave_no_completion(self):
        for owner,name in [(self.te.tempfile,'mkdtemp'),(self.te,'write_json'),(self.te.os,'replace')]:
            with mock.patch.object(owner,name,side_effect=OSError('fabricated publication failure')):
                with self.subTest(name=name),self.assertRaises(OSError):self.publish()
            self.assertEqual(list(self.output.rglob('receipt.json')),[]);self.clean()

    def test_new_destination_created_during_verification_is_preserved(self):
        original=self.te.verify_result;created=[]
        def raced(directory,fp,*args):
            original(directory,fp,*args)
            destination=self.output/'sensitivity_tables'/fp;destination.mkdir();created.append(destination)
        with mock.patch.object(self.te,'verify_result',side_effect=raced),self.assertRaises(ValidationError):self.publish()
        self.assertTrue(created[0].is_dir());self.assertEqual(list(created[0].iterdir()),[]);self.clean()

    def test_protected_and_overlapping_roots_and_invalid_resume_are_rejected(self):
        for work,output in [(self.root/'work',self.output),(self.prep/'work',self.output),(self.work,self.work),(self.work,self.work/'nested')]:
            with self.subTest(work=work),self.assertRaises(ValidationError):self.publish(work=work,output=output)
        with self.assertRaises(ValidationError):self.publish(resume=1)

    def test_preparation_can_share_work_root_but_not_generated_output_paths(self):
        self.prep=self.work/'preparation_runs'/'fabricated-preparation';self.prep.mkdir(parents=True)
        first=self.publish();self.assertTrue(first['accepted']);self.assertTrue(self.publish(resume=True)['resumed'])
        for folder in (self.work/'sensitivity_table_validation',self.output/'sensitivity_tables'):
            self.prep=folder/'fabricated-preparation'
            with self.subTest(folder=folder),self.assertRaises(ValidationError):self.publish(resume=True)
        self.clean()

    def test_upstream_missing_fit_is_rejected_without_any_sampling_fallback(self):
        with mock.patch.object(self.te,'_verified_fit',side_effect=ValidationError('source fit unavailable')):
            with mock.patch.object(pe,'sample_contract') as sampler:
                with self.assertRaises(ValidationError):self.publish()
                sampler.assert_not_called()
        self.assertEqual(list(self.output.rglob('receipt.json')),[])

    def test_upstream_verifier_requires_existing_raw_and_validation_receipts(self):
        from types import SimpleNamespace
        original=self.original_verified_fit
        for family in ('sensitivity','primary','wheeze'):
            target=next(iter(self.entries[family]));identity={'synthetic':family};fp=self.te.digest(identity)
            api=SimpleNamespace(assert_loaded_sources=lambda root:None,CRITERIA={},
                contract_descriptor=lambda prep,target:{'sha256':'a'*64},
                sampling_identity=lambda *args:(fp,identity),validation_identity=lambda *args:(fp,identity))
            with mock.patch.object(self.te,'_family_api',return_value=api),mock.patch.object(pe,'sample_contract') as sampler:
                with self.subTest(family=family),self.assertRaisesRegex(ValidationError,'existing source fits'):
                    original(family,self.root,self.prep,self.directories,SETTINGS,target)
                sampler.assert_not_called()
