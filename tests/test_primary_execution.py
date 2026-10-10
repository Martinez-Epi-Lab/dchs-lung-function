"""Synthetic adversarial checks for raw fits, revalidation, and publication.

No study data or model sampling are used. Scientific computation boundaries are
mocked; source fingerprints, filesystem operations, locks and receipts are real.
"""
from copy import deepcopy
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function import primary_execution as pe
from dchs_lung_function import diagnostics
from dchs_lung_function.analysis import primary_results
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from dchs_lung_function.sampling import resolved_sampling, validate_sampling


SETTINGS = {"schema_version": 1, "chains": 4, "draws": 1200, "tune": 1600,
            "cores": 2, "target_accept": .95, "max_treedepth": 10, "seed_offset": 104729}


class PrimaryProtocolTests(unittest.TestCase):
    def test_sampling_controls_preserve_initialization_and_independent_seed(self):
        selected = resolved_sampling("ratio_pre", SETTINGS)
        self.assertEqual(selected["random_seed"], 2026092308 + 104729)
        self.assertEqual(selected["init"], "adapt_diag")
        self.assertEqual([v["loading_ratio_mechanics"] for v in selected["initvals"]], [.25, .35, .45, .55])
        self.assertTrue(selected["discard_tuned_samples"])
        self.assertEqual((selected["chains"], selected["draws"], selected["tune"]), (4, 1200, 1600))

    def test_malformed_or_relaxed_sampling_controls_rejected(self):
        changes = [{"chains": 3}, {"draws": 1199}, {"tune": 1599}, {"cores": 0},
                   {"target_accept": .949}, {"target_accept": np.nan}, {"target_accept": True},
                   {"max_treedepth": 9}, {"seed_offset": -1}, {"chains": 4.0},
                   {"schema_version": True}, {"unexpected": 1}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_sampling({**SETTINGS, **change})

    def test_posterior_inventory_requires_every_variable_and_coordinate(self):
        expected = {"global": (4, 1200), "latent": (4, 1200, 5)}
        good = {name: np.zeros(shape) for name, shape in expected.items()}
        pe.validate_posterior_inventory(good, expected)
        for malformed in [{"global": good["global"]}, {**good, "unexpected": np.zeros((4, 1200))},
                          {**good, "latent": np.zeros((4, 1200, 4))},
                          {**good, "global": np.zeros((1200, 4))}]:
            with self.subTest(keys=list(malformed)), self.assertRaises(ValidationError):
                pe.validate_posterior_inventory(malformed, expected)

    def test_contract_descriptor_selects_verified_target_and_excludes_other_content(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            identity = {"synthetic": True}
            def write_contract(target_value, other_value):
                ArtifactWriter(folder).finish({"M03": {"pft": {"samples": {
                    "primary/ratio_pre": {"matrix": np.array([target_value])},
                    "primary/ratio_post": {"matrix": np.array([other_value])}}}}})
                write_json(folder / "receipt.json", {"through": "M03", "status": "prepared",
                    "identity": identity, "fingerprint": pe.digest(identity),
                    "artifact_manifest_sha256": file_hash(folder / "artifacts.json")})
            write_contract(1., 2.)
            first = pe.contract_descriptor(folder, "ratio_pre")
            write_contract(1., 3.)
            self.assertEqual(pe.contract_descriptor(folder, "ratio_pre"), first)
            write_contract(4., 3.)
            self.assertNotEqual(pe.contract_descriptor(folder, "ratio_pre")["sha256"], first["sha256"])
            (folder / "artifact_00001.npy").write_bytes(b"changed")
            with self.assertRaises(ValidationError):
                pe.contract_descriptor(folder, "ratio_pre")


class PrimaryExecutionTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.prep = self.base / "source", self.base / "preparation"
        self.work, self.output = self.base / "private-work", self.base / "aggregate-output"
        self.prep.mkdir()
        for name in (*pe.COMPUTE_SOURCES, *pe.VALIDATION_SOURCES):
            path = self.root / "src/dchs_lung_function" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fabricated source identity\n")
        (self.root / "requirements-models.lock.txt").write_text("# fabricated environment lock\n")
        self.descriptor = {"sha256": "a" * 64, "preparation_fingerprint": "b" * 64}
        self.settings = deepcopy(SETTINGS)
        rng = np.random.default_rng(81)
        self.posterior = {"global": rng.normal(size=(4, 1200)), "latent": rng.normal(size=(4, 1200, 5))}
        self.stats = {"diverging": np.zeros((4, 1200), dtype=bool),
                      "reached_max_treedepth": np.zeros((4, 1200), dtype=bool),
                      "energy": rng.normal(size=(4, 1200))}
        self.metadata = {"target": "ratio_pre", "controls": resolved_sampling("ratio_pre", self.settings),
                         "runtime": {"python": "synthetic", "packages": {}},
                         "runtime_setup": {"synthetic": True}, "elapsed_seconds": .1,
                         "realized_chains": 4, "realized_draws_per_chain": 1200,
                         "realized_tuning_steps": 1600,
                         "posterior_variables": list(self.posterior), "sample_stat_variables": list(self.stats)}
        self.result = SimpleNamespace(
            estimates=pd.DataFrame({"contrast": ["fabricated"], "mean": [.2]}),
            shared_factor=pd.DataFrame({"mean": [.3]}), sample_counts=pd.DataFrame({"n": [5]}),
            metadata={"synthetic": True})
        self.diagnostic = {"passed": True, "status": "passed", "reasons": [], "metrics": {"synthetic": True}}
        patches = [
            mock.patch.object(pe, "contract_descriptor", side_effect=lambda *a: deepcopy(self.descriptor)),
            mock.patch.object(pe, "load_artifacts", return_value={"synthetic": True}),
            mock.patch.object(pe, "validate_contract", return_value={"joint_children": 5}),
            mock.patch.object(pe, "configure_runtime", return_value={"synthetic": True}),
            mock.patch.object(pe, "expected_posterior_shapes", side_effect=lambda *a: {"global": (4, 1200), "latent": (4, 1200, 5)}),
            mock.patch.object(pe, "model_runtime_identity", return_value={"python": "synthetic", "packages": {}}),
            mock.patch.object(primary_results, "summarize_primary", side_effect=lambda *a, **k: self.result),
            mock.patch.object(diagnostics, "evaluate_diagnostics", side_effect=lambda *a, **k: deepcopy(self.diagnostic)),
        ]
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)

    def sample(self, *args):
        metadata = deepcopy(self.metadata)
        metadata["controls"] = resolved_sampling("ratio_pre", self.settings)
        metadata["runtime"] = pe.model_runtime_identity()
        return deepcopy(self.posterior), deepcopy(self.stats), metadata

    def run_target(self, *, resume=False, compute=None, work=None, output=None):
        with mock.patch.object(pe, "sample_contract", side_effect=compute or self.sample):
            return pe.sample_one(self.root, self.prep, "ratio_pre", self.settings,
                                 work or self.work, output or self.output, resume=resume)

    def directories(self, result):
        return self.work / "ratio_pre" / result["fit_fingerprint"], self.output / "ratio_pre" / result["validation_fingerprint"]

    def assert_no_transient_files(self):
        self.assertEqual(list(self.base.rglob(".staging-*")), [])
        self.assertEqual(list(self.base.rglob(".preparation.lock")), [])

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

    def test_identical_fit_requires_explicit_resume(self):
        self.run_target()
        with self.assertRaises(ValidationError):
            self.run_target(compute=lambda *a: self.fail("unexpected sampling"))

    def test_reporting_or_diagnostic_changes_revalidate_existing_raw_fit(self):
        first = self.run_target()
        for name in pe.VALIDATION_SOURCES:
            path = self.root / "src/dchs_lung_function" / name
            path.write_text(path.read_text() + "# changed validation\n")
            result = self.run_target(resume=True, compute=lambda *a: self.fail("validation edit caused refit"))
            self.assertEqual(result["fit_fingerprint"], first["fit_fingerprint"])
            self.assertNotEqual(result["validation_fingerprint"], first["validation_fingerprint"])
            self.assertTrue(result["raw_fit_reused"])
            self.assertFalse(result["validation_reused"])
            first = result

    def test_model_data_sampling_runtime_and_lock_changes_require_new_fit(self):
        first = self.run_target()
        fingerprints = {first["fit_fingerprint"]}
        changes = [lambda: (self.root / "src/dchs_lung_function/models/spirometry.py").write_text("# changed model\n"),
                   lambda: self.descriptor.update(sha256="c" * 64),
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
        del self.posterior["latent"]
        with mock.patch.object(diagnostics, "evaluate_diagnostics") as checked:
            with self.assertRaises(ValidationError):
                self.run_target()
            checked.assert_not_called()
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_summary_failure_keeps_verified_raw_fit_and_can_be_resumed(self):
        with mock.patch.object(primary_results, "summarize_primary", side_effect=RuntimeError("fabricated summary failure")):
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
                     lambda m: m.update(target="ratio_post"),
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
        def changed(*args):
            self.descriptor["sha256"] = "d" * 64
            return self.sample()
        with self.assertRaises(ValidationError): self.run_target(compute=changed)
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_change_during_new_summary_rejected_but_raw_preserved(self):
        def changed(*args):
            (self.root / "src/dchs_lung_function/diagnostics.py").write_text("# changed during check\n")
            return self.result
        with mock.patch.object(primary_results, "summarize_primary", side_effect=changed):
            with self.assertRaises(ValidationError): self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_criteria_mutation_during_run_is_rejected(self):
        def changed(*args):
            pe.CRITERIA["min_tail_ess"] = 800
            return self.result
        with mock.patch.object(pe, "CRITERIA", deepcopy(pe.CRITERIA)):
            with mock.patch.object(primary_results, "summarize_primary", side_effect=changed):
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
        with pe.preparation_locks(self.work / "ratio_pre", self.output / "ratio_pre"):
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


class PrimaryBatchTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.work, self.output = self.base / "work", self.base / "outputs"
        self.work.mkdir(); self.output.mkdir()
        self.config = self.base / "paths.local.json"
        self.analysis = self.base / "analysis.local.json"
        self.sampling = self.base / "sampling.local.json"
        write_json(self.config, {"inputs": {}, "directories": {"work": str(self.work), "outputs": str(self.output)}})
        write_json(self.analysis, {"synthetic": True}); write_json(self.sampling, SETTINGS)
        self.prepared = {"fingerprint": "a" * 64}

    def computed(self, root, prep, target, settings, work, output, **kwargs):
        return {"target": target, "diagnostics_passed": True, "covariance_ppc_passed": True, "status": "diagnostics_passed",
                "fit_fingerprint": "b" * 64, "validation_fingerprint": "c" * 64}

    def run_batch(self, *, targets=None, jobs=1, computed=None, preparation=None):
        with mock.patch.object(pe, "run_preparation", side_effect=preparation or (lambda *a, **k: deepcopy(self.prepared))):
            with mock.patch.object(pe, "sample_one", side_effect=computed or self.computed):
                return pe.run_primary(self.base / "source", self.config, self.analysis, self.sampling,
                                      targets=targets, jobs=jobs, resume=True)

    def test_converged_but_inadequate_primary_batch_never_publishes_current_marker(self):
        def inadequate(*args,**kwargs):
            result=self.computed(*args,**kwargs)
            if result['target']=='ratio_pre':result.update(covariance_ppc_passed=False,status='model_adequacy_failed')
            return result
        result=self.run_batch(computed=inadequate)
        self.assertTrue(result['diagnostics_passed']);self.assertFalse(result['covariance_ppc_passed'])
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'model_adequacy_failed')
        self.assertEqual(list(self.work.glob('primary*current.json')),[])

    def test_complete_and_selected_batches_publish_distinct_explicit_markers(self):
        all_models = self.run_batch()
        self.assertEqual(all_models["selected_models"], 12)
        self.assertTrue(all_models["all_twelve_models"])
        self.assertTrue((self.work / "primary_current.json").is_file())
        self.assertEqual(all_models["scientific_reference_comparison"], "requires separate independent validation")
        selected = ["ratio_post", "ratio_pre"]
        subset = self.run_batch(targets=selected)
        self.assertEqual([m["target"] for m in subset["models"]], selected)
        self.assertFalse(subset["all_twelve_models"])
        self.assertTrue((self.work / "primary_selection_current.json").is_file())
        self.assertTrue(json.loads((self.work / "primary_current.json").read_text())["all_twelve_models"])

    def test_failed_model_never_publishes_current_batch_marker(self):
        def failed(*args, **kwargs):
            result = self.computed(*args, **kwargs)
            if result["target"] == "ratio_pre": result.update(diagnostics_passed=False, status="diagnostics_failed")
            return result
        result = self.run_batch(computed=failed)
        self.assertFalse(result["diagnostics_passed"])
        self.assertEqual(list(self.work.glob("primary*current.json")), [])

    def test_sampling_file_change_during_batch_rejects_completion(self):
        def changed(*args, **kwargs):
            self.sampling.write_text(self.sampling.read_text() + "\n")
            return self.computed(*args, **kwargs)
        with self.assertRaises(ValidationError): self.run_batch(targets=["ratio_pre"], computed=changed)
        self.assertEqual(list(self.work.glob("primary*current.json")), [])

    def test_changed_preparation_at_batch_completion_rejects_marker(self):
        values = iter([{"fingerprint": "a" * 64}, {"fingerprint": "d" * 64}])
        with self.assertRaises(ValidationError):
            self.run_batch(targets=["ratio_pre"], preparation=lambda *a, **k: next(values))
        self.assertEqual(list(self.work.glob("primary*current.json")), [])

    def test_invalid_target_selection_concurrency_and_unsafe_settings_path(self):
        for targets, jobs in [([], 1), (["unknown"], 1), (["ratio_pre", "ratio_pre"], 1),
                              (None, 0), (None, 4), (None, True)]:
            with self.subTest(targets=targets, jobs=jobs), self.assertRaises(ValidationError):
                self.run_batch(targets=targets, jobs=jobs)
        self.sampling = self.output / "sampling.local.json"; write_json(self.sampling, SETTINGS)
        with self.assertRaises(ValidationError): self.run_batch(targets=["ratio_pre"])


if __name__ == "__main__":
    unittest.main()
