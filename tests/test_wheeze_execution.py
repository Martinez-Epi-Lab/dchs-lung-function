"""Synthetic M10 runner and cumulative-CLI adversarial checks.

No study fitting is performed. Sampler/reporting boundaries are mocked after
building valid invented fixtures; filesystem, fingerprints and locks are real.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
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
from dchs_lung_function import wheeze_execution as pe, diagnostics, pretb_execution, pipeline
from dchs_lung_function.analysis import wheeze_results
from dchs_lung_function.models.wheeze import POSTERIOR_VARIABLES, validate_contract
from dchs_lung_function.execution import ArtifactWriter,file_hash,write_json
from dchs_lung_function.pipeline import ValidationError
from dchs_lung_function.wheeze_sampling import resolved_sampling,validate_sampling
from test_wheeze import fabricated_contract
from test_wheeze_results import fabricated_posterior

SETTINGS={'schema_version':1,'chains':4,'draws':1600,'tune':2200,'cores':2,
          'target_accept':.95,'max_treedepth':10,'seed_offset':104729}
REAL_DIAGNOSTICS=diagnostics.evaluate_diagnostics
REAL_SUMMARIZE=wheeze_results.summarize_wheeze


@lru_cache(maxsize=2)
def _fixture(target):
    contract=fabricated_contract(target)
    posterior=fabricated_posterior(contract,target,chains=4,draws=1600)
    rng=np.random.default_rng(951)
    for name,center in [('sigma_scheduled',.4),('sigma_episode',.3)]:
        posterior[name]=rng.normal(center,.01,(4,1600))
    for ratio,log in wheeze_results.TRANSFORMS.items():posterior[ratio]=np.exp(posterior[log])
    result=REAL_SUMMARIZE(posterior,contract,target,
        ppc_seed=resolved_sampling(target,SETTINGS)['random_seed']+77)
    return contract,posterior,result,validate_contract(contract,target)


def fabricated_result(target):return deepcopy(_fixture(target))


class WheezeExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name)
        self.root,self.prep=self.base/'source',self.base/'preparation';self.prep.mkdir()
        self.work,self.output=self.base/'private-work',self.base/'aggregate-output'
        self.target='primary_sex';self.settings=deepcopy(SETTINGS)
        for name in (*pe.COMPUTE_SOURCES,*pe.VALIDATION_SOURCES):
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(pe.__file__).parent/name,path)
        (self.root/'requirements-models.lock.txt').write_text('# invented lock\n')
        self.descriptor={'sha256':'a'*64,'preparation_fingerprint':'b'*64}
        self.contract,self.posterior,self.result,self.validated=fabricated_result(self.target)
        self.shapes={name:value.shape for name,value in self.posterior.items()}
        rng=np.random.default_rng(81)
        self.stats={'diverging':np.zeros((4,1600),bool),'reached_max_treedepth':np.zeros((4,1600),bool),
                    'energy':rng.normal(size=(4,1600))}
        self.metadata={'target':self.target,'controls':resolved_sampling(self.target,self.settings),
            'runtime':{'python':'synthetic','packages':{}},'runtime_setup':{'synthetic':True},'elapsed_seconds':.1,
            'realized_chains':4,'realized_draws_per_chain':1600,'realized_tuning_steps':2200,
            'posterior_variables':list(self.posterior),'sample_stat_variables':list(self.stats)}
        self.diagnostic={'passed':True,'status':'passed','reasons':[],'metrics':{'synthetic':True}}
        patches=[mock.patch.object(pe,'contract_descriptor',side_effect=lambda *a:deepcopy(self.descriptor)),
                 mock.patch.object(pe,'load_artifacts',side_effect=lambda *a:self.contract),
                 mock.patch.object(pe,'validate_contract',side_effect=lambda *a:self.validated),
                 mock.patch.object(pe,'configure_runtime',return_value={'synthetic':True}),
                 mock.patch.object(pe,'expected_posterior_shapes',side_effect=lambda *a:deepcopy(self.shapes)),
                 mock.patch.object(pe,'model_runtime_identity',return_value={'python':'synthetic','packages':{}}),
                 mock.patch.object(wheeze_results,'summarize_wheeze',side_effect=self.summary),
                 mock.patch.object(diagnostics,'evaluate_diagnostics',side_effect=lambda *a,**k:deepcopy(self.diagnostic))]
        for patch in patches:patch.start();self.addCleanup(patch.stop)

    def summary(self,*args,**kwargs):
        self.result.metadata['posterior_predictive']['seed']=kwargs['ppc_seed']
        return self.result

    def sample(self,*args):
        metadata=deepcopy(self.metadata);metadata.update(controls=resolved_sampling(self.target,self.settings),
            target=self.target,runtime=pe.model_runtime_identity())
        return deepcopy(self.posterior),deepcopy(self.stats),metadata

    def run_target(self,*,resume=False,compute=None,work=None,output=None):
        with mock.patch.object(pe,'sample_contract',side_effect=compute or self.sample):
            return pe.sample_one(self.root,self.prep,self.target,self.settings,work or self.work,output or self.output,resume=resume)

    def directories(self,result):
        return self.work/self.target/result['fit_fingerprint'],self.output/self.target/result['validation_fingerprint']

    def assert_no_transient_files(self):
        self.assertEqual(list(self.base.rglob('.staging-*')),[])
        self.assertEqual(list(self.base.rglob('.preparation.lock')),[])


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
        self.assertEqual(len(self.posterior), 22)
        with mock.patch.object(diagnostics, "evaluate_diagnostics", wraps=REAL_DIAGNOSTICS) as checked:
            result = self.run_target()
            self.assertTrue(result["diagnostics_passed"])
            self.assertEqual(set(checked.call_args.args[0]), set(POSTERIOR_VARIABLES))
            self.assertEqual(checked.call_args.args[2], pe.CRITERIA)
        shutil.rmtree(self.output); shutil.rmtree(self.work)
        self.posterior["latent_propensity"][0, :, -1] += 20
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
        for field, value in [("counts", {}), ("target", "secondary_sex_smoking_birthweight"), ("result_rows", 5)]:
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
        posterior = deepcopy(self.posterior); posterior["latent_propensity"] = posterior["latent_propensity"][:, :, :-1]
        np.savez_compressed(raw / "posterior.npz", **posterior)
        receipt_path = raw / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"]["posterior.npz"] = file_hash(raw / "posterior.npz"); write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_source_changes_during_cached_recomputation_block_return(self):
        self.run_target()
        def changed(*args, **kwargs):
            self.descriptor["sha256"] = "e" * 64
            return self.result
        with mock.patch.object(wheeze_results, "summarize_wheeze", side_effect=changed):
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
        actual=(Path(pe.__file__).parent/"models/wheeze.py").resolve()
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
        del self.posterior["latent_propensity"]
        with mock.patch.object(diagnostics, "evaluate_diagnostics") as checked:
            with self.assertRaises(ValidationError):
                self.run_target()
            checked.assert_not_called()
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_summary_failure_keeps_verified_raw_fit_and_can_be_resumed(self):
        with mock.patch.object(wheeze_results, "summarize_wheeze", side_effect=RuntimeError("fabricated summary failure")):
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
                     lambda m: m.update(target="secondary_sex_smoking_birthweight"),
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
        with mock.patch.object(wheeze_results, "summarize_wheeze", side_effect=changed):
            with self.assertRaises(ValidationError): self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_criteria_mutation_during_run_is_rejected(self):
        def changed(*args, **kwargs):
            pe.CRITERIA["min_tail_ess"] = 800
            return self.result
        with mock.patch.object(pe, "CRITERIA", deepcopy(pe.CRITERIA)):
            with mock.patch.object(wheeze_results, "summarize_wheeze", side_effect=changed):
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
        with pe.preparation_locks(self.work / "primary_sex", self.output / "primary_sex"):
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

    def test_secondary_empty_standardized_rr_retains_exact_schema(self):
        self.target='secondary_sex_smoking_birthweight'
        self.contract,self.posterior,self.result,self.validated=fabricated_result(self.target)
        self.shapes={n:a.shape for n,a in self.posterior.items()}
        first=self.run_target();_,directory=self.directories(first)
        self.assertEqual(first['result_rows'],{'effects':6,'contrasts':6,'global_tests':2,
            'support_counts':22,'posterior_predictive_cells':22,'standardized_rr':0})
        table=pd.read_csv(directory/'standardized_rr.csv')
        self.assertEqual(list(table),wheeze_results.TABLE_COLUMNS['standardized_rr']);self.assertEqual(len(table),0)
        self.assertTrue(self.run_target(resume=True)['validation_reused'])

    def test_declared_table_schemas_rows_and_cell_order_required(self):
        original=deepcopy(self.result)
        for name in pe.TABLES:
            for change in ('missing','duplicate','extra_field','wrong_scenario','reordered'):
                self.result=deepcopy(original);table=getattr(self.result,name)
                if change=='missing':setattr(self.result,name,table.iloc[1:].reset_index(drop=True))
                elif change=='duplicate':setattr(self.result,name,pd.concat([table,table.iloc[:1]],ignore_index=True))
                elif change=='extra_field':table['pid_child']='fabricated-00'
                elif change=='wrong_scenario':table.loc[0,'scenario']='wrong'
                else:setattr(self.result,name,table.iloc[::-1].reset_index(drop=True))
                with self.subTest(table=name,change=change),self.assertRaises(ValidationError):
                    pe.validate_results(self.result,self.target,self.validated,self.posterior,self.metadata['controls']['random_seed']+77)
        self.result=original

    def test_nested_metadata_and_false_quadrature_acceptance_rejected(self):
        for section in ('posterior','posterior_predictive','standardization'):
            result=deepcopy(self.result);result.metadata[section]['pid_child']='fabricated-00'
            with self.subTest(section=section),self.assertRaises(ValidationError):
                pe.validate_results(result,self.target,self.validated,self.posterior,self.metadata['controls']['random_seed']+77)
        for field,value in [('convergence_limit',1.),('latent_nodes',2),('convergence_nodes',2),
                            ('maximum_draw_level_RR_quadrature_change_80_vs_120',1.)]:
            result=deepcopy(self.result);result.metadata['standardization'][field]=value
            with self.subTest(field=field),self.assertRaises(ValidationError):
                pe.validate_results(result,self.target,self.validated,self.posterior,self.metadata['controls']['random_seed']+77)

    def test_effect_contrast_and_global_numbers_must_match_raw_draws(self):
        for name,column in [('effects','posterior_mean'),('contrasts','posterior_mean'),
                            ('global_tests','posterior_wald_chi_square')]:
            result=deepcopy(self.result);getattr(result,name).loc[0,column]+=.2
            with self.subTest(table=name),self.assertRaises(ValidationError):
                pe.validate_results(result,self.target,self.validated,self.posterior,self.metadata['controls']['random_seed']+77)

    def test_all_cached_table_values_are_recomputed_after_hash_repair(self):
        first=self.run_target();_,directory=self.directories(first)
        for table in pe.TABLES:
            path=directory/(table+'.csv');original=path.read_bytes()
            rp=directory/'receipt.json';receipt_bytes=rp.read_bytes()
            frame=pd.read_csv(path,float_precision='round_trip')
            column=next(c for c in frame.select_dtypes(include=[np.number]) if frame[c].notna().any())
            frame[column]=frame[column].astype(float);index=frame.index[frame[column].notna()][0]
            frame.loc[index,column]+=.1;frame.to_csv(path,index=False,float_format='%.17g')
            receipt=json.loads(receipt_bytes);receipt['files'][path.name]=file_hash(path);write_json(rp,receipt)
            with self.subTest(table=table),self.assertRaises(ValidationError):self.run_target(resume=True)
            path.write_bytes(original);rp.write_bytes(receipt_bytes)

    def test_predictive_seed_and_all_numeric_replicates_are_bound(self):
        with mock.patch.object(wheeze_results,'summarize_wheeze',side_effect=self.summary) as summarize:
            first=self.run_target()
            self.assertTrue(summarize.call_args_list)
            self.assertTrue(all(c.kwargs['ppc_seed']==self.metadata['controls']['random_seed']+77 for c in summarize.call_args_list))
        _,directory=self.directories(first);path=directory/'ppc_replicates.npz';original=path.read_bytes()
        rp=directory/'receipt.json';receipt_bytes=rp.read_bytes()
        for name in self.result.ppc_replicates:
            values=deepcopy(self.result.ppc_replicates);values[name].flat[0]+=1
            np.savez_compressed(path,**values);receipt=json.loads(receipt_bytes)
            receipt['files'][path.name]=file_hash(path);write_json(rp,receipt)
            with self.subTest(variable=name),self.assertRaises(ValidationError):self.run_target(resume=True)
            path.write_bytes(original);rp.write_bytes(receipt_bytes)

    def test_ppc_inventory_shape_counts_and_zero_denominator_semantics(self):
        seed=self.metadata['controls']['random_seed']+77
        for change in ('extra','missing','object','wrong_shape','wrong_index','negative_count','fractional_count','wrong_statistics','nonempty_nan'):
            result=deepcopy(self.result);r=result.ppc_replicates
            if change=='extra':r['pid_child']=np.array(['fabricated-00'])
            elif change=='missing':del r['conditional_variances']
            elif change=='object':r['counts']=r['counts'].astype(object)
            elif change=='wrong_shape':r['counts']=r['counts'][:,:-1]
            elif change=='wrong_index':r['draw_indices'][1]=r['draw_indices'][0]
            elif change=='negative_count':r['counts'][0,0]=-1
            elif change=='fractional_count':r['counts']=r['counts'].astype(float);r['counts'][0,0]=.5
            elif change=='wrong_statistics':r['statistics'][0,np.flatnonzero(result.posterior_predictive_cells.observed_denominator>0)[0]]+=.1
            else:r['conditional_variances'][0,np.flatnonzero(result.posterior_predictive_cells.observed_denominator>0)[0]]=np.nan
            with self.subTest(change=change),self.assertRaises(ValidationError):pe.validate_results(result,self.target,self.validated,self.posterior,seed)

    def test_zero_denominator_predictive_cell_keeps_zero_counts_and_missing_rates(self):
        # Construct a valid empty scheduled cell at the aggregate API boundary.
        result=deepcopy(self.result);validated=deepcopy(self.validated)
        ppc=result.posterior_predictive_cells;first=ppc.iloc[0]
        mask=(result.support_counts.process.eq(first.process)&result.support_counts.age_band.eq(first.age_band)
              &result.support_counts.exposure_group.eq('post-TB after infancy TB'))
        self.assertEqual(int(mask.sum()),1)
        support=result.support_counts.loc[mask].iloc[0]
        validated['counts']['scheduled_observations']-=int(support.observations_or_intervals)
        validated['counts']['scheduled_positive']-=int(support.events_or_positive)
        result.metadata['counts']=deepcopy(validated['counts'])
        result.support_counts.loc[mask,['children','observations_or_intervals','events_or_positive']]=0
        ppc.loc[0,['observed_denominator','observed_events']]=0
        ppc.loc[0,['observed_statistic','replicated_mean','replicated_2_5','replicated_97_5']]=np.nan
        result.ppc_replicates['counts'][:,0]=0
        for name in ('statistics','conditional_means','conditional_variances'):result.ppc_replicates[name][:,0]=np.nan
        result.metadata['posterior_predictive']['conditional_mean_checks'][0].update(
            empty=True,simulation_mean_error=None,conditional_simulation_mcse=None,within_four_conditional_mcse=None)
        seed=self.metadata['controls']['random_seed']+77
        pe.validate_results(result,self.target,validated,self.posterior,seed)
        result.ppc_replicates['conditional_means'][0,0]=0.
        with self.assertRaises(ValidationError):pe.validate_results(result,self.target,validated,self.posterior,seed)

    def test_receipt_directories_and_extra_untracked_files_rejected(self):
        first=self.run_target();raw,validated=self.directories(first)
        for directory in (raw,validated):
            path=directory/'receipt.json';outside=self.base/('external-'+directory.name+'.json')
            path.rename(outside);path.symlink_to(outside)
            with self.subTest(kind='receipt',directory=directory),self.assertRaises(ValidationError):self.run_target(resume=True)
            path.unlink();outside.rename(path)
            extra=directory/'untracked.csv';extra.write_text('fabricated-only\n')
            with self.subTest(kind='extra',directory=directory),self.assertRaises(ValidationError):self.run_target(resume=True)
            extra.unlink();outside=self.base/('outside-'+directory.name);directory.rename(outside);directory.symlink_to(outside,target_is_directory=True)
            with self.subTest(kind='directory',directory=directory),self.assertRaises(ValidationError):self.run_target(resume=True)
            directory.unlink();outside.rename(directory)

    def test_target_and_compiler_symlinks_rejected_before_runtime_setup(self):
        for location in ('work','output','cache'):
            self.work.mkdir(exist_ok=True);self.output.mkdir(exist_ok=True)
            link=(self.work if location!='output' else self.output)/self.target
            if location=='cache':link.mkdir(exist_ok=True);link=link/'compiler_cache'
            link.symlink_to(self.prep,target_is_directory=True)
            with mock.patch.object(pe,'configure_runtime') as configure:
                with self.subTest(location=location),self.assertRaises(ValidationError):self.run_target()
                configure.assert_not_called()
            link.unlink();shutil.rmtree(self.work);shutil.rmtree(self.output)


class WheezeProtocolTests(unittest.TestCase):
    def test_recorded_sampling_minima_initialization_and_seeds(self):
        for target,seed in [('primary_sex',2026091133),('secondary_sex_smoking_birthweight',2026092133)]:
            controls=resolved_sampling(target,SETTINGS)
            self.assertEqual(controls['random_seed'],seed+104729)
            self.assertEqual(controls['init'],'jitter+adapt_diag');self.assertIsNone(controls['initvals'])
            self.assertEqual((controls['chains'],controls['draws'],controls['tune']),(4,1600,2200))
        for changes in [{'chains':3},{'chains':5},{'draws':1599},{'tune':2199},{'target_accept':.94},
                        {'max_treedepth':9},{'seed_offset':-1},{'schema_version':True},{'unknown':1}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_sampling({**SETTINGS,**changes})

    def test_selected_contract_bytes_only_and_complete_preparation_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def write(value,other):
                ArtifactWriter(folder).finish({'M03':{'respiratory':{'wheeze':{
                    'primary_sex':{'matrix':np.array([value])},'secondary_sex_smoking_birthweight':{'matrix':np.array([other])}}}}})
                identity={'fabricated':True}
                write_json(folder/'receipt.json',{'through':'M03','status':'prepared','identity':identity,
                    'fingerprint':pe.digest(identity),'artifact_manifest_sha256':file_hash(folder/'artifacts.json')})
            write(1,2);first=pe.contract_descriptor(folder,'primary_sex')
            write(1,3);self.assertEqual(pe.contract_descriptor(folder,'primary_sex'),first)
            write(4,3);self.assertNotEqual(pe.contract_descriptor(folder,'primary_sex')['sha256'],first['sha256'])
            with self.assertRaises(ValidationError):pe.contract_descriptor(folder,'unknown')
            (folder/'artifact_00001.npy').write_bytes(b'changed')
            with self.assertRaises(ValidationError):pe.contract_descriptor(folder,'primary_sex')

    def test_complete_variable_and_child_coordinate_inventory(self):
        expected={'global':(4,1600),'child':(4,1600,12)}
        good={n:np.zeros(s) for n,s in expected.items()};pe.validate_posterior_inventory(good,expected)
        for bad in [{'global':good['global']},{**good,'other':good['global']},{**good,'child':good['child'][:,:,:11]},
                    {**good,'global':np.zeros((1600,4))}]:
            with self.assertRaises(ValidationError):pe.validate_posterior_inventory(bad,expected)


class WheezeBatchAndCliTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name);self.root=self.base/'source'
        self.work,self.output=self.base/'work',self.base/'outputs';self.work.mkdir();self.output.mkdir()
        for name in (*pe.COMPUTE_SOURCES,*pe.VALIDATION_SOURCES):
            path=self.root/'src/dchs_lung_function'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(pe.__file__).parent/name,path)
        self.paths=[self.base/n for n in ('paths.json','analysis.json','primary.json','mechanics.json','supporting.json','wheeze.json')]
        write_json(self.paths[0],{'inputs':{},'directories':{'work':str(self.work),'outputs':str(self.output)}})
        write_json(self.paths[1],{'invented':True})
        for path in self.paths[2:]:write_json(path,SETTINGS)
        self.prior={'through':'M09','accepted':True,'status':'accepted','selected_models':17,
            'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
            'mechanics_models_verified':6,'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
            'supporting_deterministic_fits_verified':10,'preparation_fingerprint':'a'*64}

    def computed(self,root,prep,target,*args,**kwargs):
        return {'target':target,'fit_fingerprint':'b'*64,'validation_fingerprint':'c'*64,
            'diagnostics_passed':True,'status':'diagnostics_passed',
            'result_rows':{'effects':6,'contrasts':6,'global_tests':2,'support_counts':22,
                'posterior_predictive_cells':22,'standardized_rr':3 if target=='primary_sex' else 0}}

    def batch(self,*,computed=None,prior=None,prepared=None,jobs=1):
        with mock.patch.object(pretb_execution,'run_pretb',side_effect=prior or (lambda *a,**k:deepcopy(self.prior))):
            with mock.patch.object(pe,'sample_one',side_effect=computed or self.computed):
                with mock.patch.object(pe,'run_preparation',side_effect=prepared or (lambda *a,**k:{'fingerprint':'a'*64})):
                    return pe.run_wheeze(self.root,*self.paths,jobs=jobs,resume=True)

    def test_both_targets_and_all_prior_models_required_for_marker(self):
        result=self.batch();self.assertTrue(result['accepted']);self.assertEqual(result['through'],'M10')
        self.assertEqual(result['selected_models'],2);self.assertEqual(result['pretb_models_verified'],17)
        self.assertEqual([r['target'] for r in result['wheeze_models']],list(pe.TARGETS))
        self.assertEqual(json.loads((self.work/'wheeze_current.json').read_text()),result)
        self.assertTrue(any('G08' in s for s in result['inherited_limitations']))

    def test_failed_diagnostics_cannot_write_completion(self):
        def failed(*args,**kwargs):
            value=self.computed(*args,**kwargs)
            if value['target']=='primary_sex':value.update(diagnostics_passed=False,status='diagnostics_failed')
            return value
        result=self.batch(computed=failed);self.assertFalse(result['accepted'])
        self.assertFalse((self.work/'wheeze_current.json').exists())

    def test_each_prior_count_and_acceptance_required_before_sampling(self):
        for field,good in self.prior.items():
            if field=='preparation_fingerprint':continue
            self.prior[field]=False if field=='accepted' else 'wrong' if isinstance(good,str) else good-1
            with self.subTest(field=field),self.assertRaises(ValidationError):
                self.batch(computed=lambda *a,**k:self.fail('sampled after incomplete prerequisites'))
            self.prior[field]=good

    def test_duplicate_missing_malformed_target_returns_rejected(self):
        for target in ('primary_sex','unknown',None,[]):
            def invalid(*args,**kwargs):
                value=self.computed(*args,**kwargs);value['target']=target;return value
            with self.subTest(target=target),self.assertRaises(ValidationError):self.batch(computed=invalid)
        with self.assertRaises(ValidationError):self.batch(computed=lambda *a,**k:None)

    def test_exact_all_table_counts_status_and_bool_acceptance_required(self):
        for table in pe.TABLES:
            def invalid(*args,**kwargs):
                value=self.computed(*args,**kwargs);value['result_rows'][table]+=1;return value
            with self.subTest(table=table),self.assertRaises(ValidationError):self.batch(computed=invalid)
        for field,bad in [('result_rows',None),('result_rows',[]),('diagnostics_passed',1),('status','accepted')]:
            def invalid(*args,**kwargs):
                value=self.computed(*args,**kwargs);value[field]=bad;return value
            with self.subTest(field=field),self.assertRaises(ValidationError):self.batch(computed=invalid)
        self.assertFalse((self.work/'wheeze_current.json').exists())

    def test_all_six_config_files_and_preparation_bound_until_completion(self):
        for path in self.paths:
            original=path.read_bytes()
            def changed(*args,**kwargs):path.write_bytes(original+b'\n');return self.computed(*args,**kwargs)
            with self.subTest(path=path.name),self.assertRaises(ValidationError):self.batch(computed=changed)
            path.write_bytes(original)
        with self.assertRaises(ValidationError):self.batch(prepared=lambda *a,**k:{'fingerprint':'changed'})
        self.assertFalse((self.work/'wheeze_current.json').exists())

    def test_invalid_jobs_generated_sampling_path_and_late_source_change_rejected(self):
        for jobs in (0,4,True,1.5):
            with self.subTest(jobs=jobs),self.assertRaises(ValidationError):self.batch(jobs=jobs)
        original=self.paths[-1];self.paths[-1]=self.work/'wheeze.json';write_json(self.paths[-1],SETTINGS)
        with self.assertRaises(ValidationError):self.batch()
        self.paths[-1]=original
        def changed(*args,**kwargs):
            p=self.root/'src/dchs_lung_function/wheeze_execution.py';p.write_text(p.read_text()+'# changed\n')
            return self.computed(*args,**kwargs)
        with self.assertRaises(ValidationError):self.batch(computed=changed)

    def cli(self,extra=None,result=None,error=None):
        out,err=StringIO(),StringIO()
        args=['--through','M10','--config',str(self.paths[0]),'--analysis',str(self.paths[1]),'--resume']
        with mock.patch.object(pe,'run_wheeze',return_value=result or {'accepted':True,'status':'accepted','selected_models':2},side_effect=error) as call:
            with redirect_stdout(out),redirect_stderr(err):code=pipeline.main(args+(extra or []),root=self.root)
        return code,out.getvalue(),err.getvalue(),call

    def test_all_four_sampling_controls_route_to_cumulative_modules(self):
        flags=['--sampling','--mechanics-sampling','--supporting-sampling','--wheeze-sampling']
        args=[v for flag,path in zip(flags,self.paths[2:]) for v in (flag,str(path))]+['--jobs','3']
        code,out,err,call=self.cli(args);self.assertEqual(code,0);self.assertIn('2 models',out);self.assertEqual(err,'')
        self.assertEqual(call.call_args.args[3:],tuple(map(str,self.paths[2:])))
        self.assertEqual(call.call_args.kwargs,{'jobs':3,'resume':True})

    def test_cli_failures_partial_targets_and_sanitized_exceptions(self):
        code,out,_,_=self.cli(result={'accepted':False,'status':'diagnostics_failed','selected_models':2})
        self.assertEqual(code,2);self.assertIn('diagnostics_failed',out)
        code,_,_,call=self.cli(['--targets','primary_sex']);self.assertEqual(code,2);call.assert_not_called()
        for error in (ValueError('fabricated_private_record'),OSError('fabricated_private_record'),RuntimeError('fabricated_private_record')):
            code,out,err,_=self.cli(error=error);self.assertEqual(code,2);self.assertNotIn('fabricated_private_record',out+err)
        for args in (['--full'],['--render'],['--through','M09','--config',str(self.paths[0]),'--analysis',str(self.paths[1]),'--wheeze-sampling',str(self.paths[-1])]):
            with redirect_stderr(StringIO()),mock.patch.object(pe,'run_wheeze') as call:
                self.assertEqual(pipeline.main(args,root=self.root),2);call.assert_not_called()


if __name__=='__main__':unittest.main()
