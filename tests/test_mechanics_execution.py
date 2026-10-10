"""Synthetic adversarial checks for raw fits, revalidation, and publication.

No study data or model sampling are used. Scientific computation boundaries are
mocked; source fingerprints, filesystem operations, locks and receipts are real.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function import mechanics_execution as pe
from dchs_lung_function import diagnostics
from dchs_lung_function import shared_factor_execution as factor_execution
from dchs_lung_function import trajectory_execution, pipeline
from dchs_lung_function.analysis import mechanics_results
from dchs_lung_function.models.mechanics import POSTERIOR_VARIABLES
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from dchs_lung_function.mechanics_sampling import resolved_sampling
from dchs_lung_function.sampling import validate_sampling


SETTINGS = {"schema_version": 1, "chains": 4, "draws": 1200, "tune": 1600,
            "cores": 2, "target_accept": .95, "max_treedepth": 10, "seed_offset": 104729}
REAL_DIAGNOSTICS = diagnostics.evaluate_diagnostics


class MechanicsProtocolTests(unittest.TestCase):
    def test_sampling_controls_preserve_initialization_and_independent_seed(self):
        selected = resolved_sampling("R", SETTINGS)
        self.assertEqual(selected["random_seed"], 2026092612 + 104729)
        self.assertEqual(selected["init"], "jitter+adapt_diag")
        self.assertIsNone(selected["initvals"])
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
                    "mechanics/R": {"matrix": np.array([target_value])},
                    "mechanics/C": {"matrix": np.array([other_value])}}}}})
                write_json(folder / "receipt.json", {"through": "M03", "status": "prepared",
                    "identity": identity, "fingerprint": pe.digest(identity),
                    "artifact_manifest_sha256": file_hash(folder / "artifacts.json")})
            write_contract(1., 2.)
            first = pe.contract_descriptor(folder, "R")
            write_contract(1., 3.)
            self.assertEqual(pe.contract_descriptor(folder, "R"), first)
            write_contract(4., 3.)
            self.assertNotEqual(pe.contract_descriptor(folder, "R")["sha256"], first["sha256"])
            (folder / "artifact_00001.npy").write_bytes(b"changed")
            with self.assertRaises(ValidationError):
                pe.contract_descriptor(folder, "R")


class MechanicsExecutionTests(unittest.TestCase):
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
        self.shapes = {name: (4, 1200, 5) if name in {"child_z", "shared_outcome"} else (4, 1200, 6) if name == "log_baseline_hazard" else (4, 1200) for name in POSTERIOR_VARIABLES}
        self.posterior = {name:rng.normal(size=shape) for name,shape in self.shapes.items()}
        self.stats = {"diverging": np.zeros((4, 1200), dtype=bool),
                      "reached_max_treedepth": np.zeros((4, 1200), dtype=bool),
                      "energy": rng.normal(size=(4, 1200))}
        self.metadata = {"target": "R", "controls": resolved_sampling("R", self.settings),
                         "runtime": {"python": "synthetic", "packages": {}},
                         "runtime_setup": {"synthetic": True}, "elapsed_seconds": .1,
                         "realized_chains": 4, "realized_draws_per_chain": 1200,
                         "realized_tuning_steps": 1600,
                         "posterior_variables": list(self.posterior), "sample_stat_variables": list(self.stats)}
        self.result = SimpleNamespace(
            estimates=pd.DataFrame({"contrast": [f"fabricated-{i}" for i in range(8)], "mean": [.2]*8}),
            hazard=pd.DataFrame({"mean": [.3]}), sample_counts=pd.DataFrame({"n": [5]}),
            metadata={"synthetic": True})
        self.diagnostic = {"passed": True, "status": "passed", "reasons": [], "metrics": {"synthetic": True}}
        patches = [
            mock.patch.object(pe, "contract_descriptor", side_effect=lambda *a: deepcopy(self.descriptor)),
            mock.patch.object(pe, "load_artifacts", return_value={"synthetic": True}),
            mock.patch.object(pe, "validate_contract", return_value={"joint_children": 5}),
            mock.patch.object(pe, "configure_runtime", return_value={"synthetic": True}),
            mock.patch.object(pe, "expected_posterior_shapes", side_effect=lambda *a: deepcopy(self.shapes)),
            mock.patch.object(pe, "model_runtime_identity", return_value={"python": "synthetic", "packages": {}}),
            mock.patch.object(mechanics_results, "summarize_mechanics", side_effect=lambda *a, **k: self.result),
            mock.patch.object(diagnostics, "evaluate_diagnostics", side_effect=lambda *a, **k: deepcopy(self.diagnostic)),
        ]
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)

    def sample(self, *args):
        metadata = deepcopy(self.metadata)
        metadata["controls"] = resolved_sampling("R", self.settings)
        metadata["runtime"] = pe.model_runtime_identity()
        return deepcopy(self.posterior), deepcopy(self.stats), metadata

    def run_target(self, *, resume=False, compute=None, work=None, output=None):
        with mock.patch.object(pe, "sample_contract", side_effect=compute or self.sample):
            return pe.sample_one(self.root, self.prep, "R", self.settings,
                                 work or self.work, output or self.output, resume=resume)

    def directories(self, result):
        return self.work / "R" / result["fit_fingerprint"], self.output / "R" / result["validation_fingerprint"]

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

    def test_all_sixteen_variables_and_child_coordinates_reach_strict_diagnostics(self):
        self.assertEqual(len(self.posterior), 16)
        with mock.patch.object(diagnostics, "evaluate_diagnostics", wraps=REAL_DIAGNOSTICS) as checked:
            result = self.run_target()
            self.assertTrue(result["diagnostics_passed"])
            self.assertEqual(set(checked.call_args.args[0]), set(POSTERIOR_VARIABLES))
            self.assertEqual(checked.call_args.args[2], pe.CRITERIA)
        shutil.rmtree(self.output); shutil.rmtree(self.work)
        self.posterior["child_z"][0, :, -1] += 20
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

    def test_fresh_validation_requires_all_eight_contrasts(self):
        self.result.estimates = self.result.estimates.iloc[:-1].copy()
        with self.assertRaises(ValidationError): self.run_target()
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)

    def test_cached_aggregate_tables_are_recomputed_even_with_updated_hashes(self):
        first = self.run_target(); _, directory = self.directories(first)
        for name in ["estimates.csv", "hazard.csv", "sample_counts.csv"]:
            path = directory / name; original = path.read_bytes()
            receipt_path = directory / "receipt.json"; original_receipt = receipt_path.read_bytes()
            frame = pd.read_csv(path); frame.iloc[0, -1] += 1
            frame.to_csv(path, index=False, float_format="%.17g")
            receipt = json.loads(receipt_path.read_text()); receipt["files"][name] = file_hash(path)
            write_json(receipt_path, receipt)
            with self.subTest(name=name), self.assertRaises(ValidationError): self.run_target(resume=True)
            path.write_bytes(original); receipt_path.write_bytes(original_receipt)

    def test_cached_counts_target_and_result_rows_are_recomputed(self):
        first = self.run_target(); _, directory = self.directories(first)
        path = directory / "summary.json"; original = json.loads(path.read_text())
        receipt_path = directory / "receipt.json"; receipt_original = json.loads(receipt_path.read_text())
        for field, value in [("counts", {}), ("target", "C"), ("result_rows", 7)]:
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
        posterior = deepcopy(self.posterior); posterior["shared_outcome"] = posterior["shared_outcome"][:, :, :-1]
        np.savez_compressed(raw / "posterior.npz", **posterior)
        receipt_path = raw / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"]["posterior.npz"] = file_hash(raw / "posterior.npz"); write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): self.run_target(resume=True)

    def test_source_changes_during_cached_recomputation_block_return(self):
        self.run_target()
        def changed(*args, **kwargs):
            self.descriptor["sha256"] = "e" * 64
            return self.result
        with mock.patch.object(mechanics_results, "summarize_mechanics", side_effect=changed):
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
            result = self.run_target(resume=True, compute=lambda *a: self.fail("validation edit caused refit"))
            self.assertEqual(result["fit_fingerprint"], first["fit_fingerprint"])
            self.assertNotEqual(result["validation_fingerprint"], first["validation_fingerprint"])
            self.assertTrue(result["raw_fit_reused"])
            self.assertFalse(result["validation_reused"])
            first = result

    def test_model_data_sampling_runtime_and_lock_changes_require_new_fit(self):
        first = self.run_target()
        fingerprints = {first["fit_fingerprint"]}
        changes = [lambda: (self.root / "src/dchs_lung_function/models/mechanics.py").write_text("# changed model\n"),
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
        del self.posterior["child_z"]
        with mock.patch.object(diagnostics, "evaluate_diagnostics") as checked:
            with self.assertRaises(ValidationError):
                self.run_target()
            checked.assert_not_called()
        self.assertEqual(list(self.base.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_summary_failure_keeps_verified_raw_fit_and_can_be_resumed(self):
        with mock.patch.object(mechanics_results, "summarize_mechanics", side_effect=RuntimeError("fabricated summary failure")):
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
                     lambda m: m.update(target="C"),
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
        with mock.patch.object(mechanics_results, "summarize_mechanics", side_effect=changed):
            with self.assertRaises(ValidationError): self.run_target()
        self.assertEqual(len(list(self.work.rglob("posterior.npz"))), 1)
        self.assertEqual(list(self.output.rglob("receipt.json")), [])
        self.assert_no_transient_files()

    def test_validation_criteria_mutation_during_run_is_rejected(self):
        def changed(*args):
            pe.CRITERIA["min_tail_ess"] = 800
            return self.result
        with mock.patch.object(pe, "CRITERIA", deepcopy(pe.CRITERIA)):
            with mock.patch.object(mechanics_results, "summarize_mechanics", side_effect=changed):
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
        with pe.preparation_locks(self.work / "R", self.output / "R"):
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




class MechanicsBatchAndCliTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name); self.work = self.base / "work"; self.output = self.base / "outputs"
        self.work.mkdir(); self.output.mkdir()
        self.config = self.base / "paths.local.json"; self.analysis = self.base / "analysis.local.json"
        self.primary_sampling = self.base / "primary.local.json"; self.mechanics_sampling = self.base / "mechanics.local.json"
        write_json(self.config, {"inputs":{}, "directories":{"work":str(self.work), "outputs":str(self.output)}})
        write_json(self.analysis, {"synthetic":True})
        write_json(self.primary_sampling, SETTINGS); write_json(self.mechanics_sampling, SETTINGS)
        self.prior = {"accepted":True, "analyses":4, "fitted_models":8,
                      "primary_models_verified":12, "binary_models_verified":28, "preparation_fingerprint":"a"*64}
        self.factor = {"accepted":True, "result_rows":6, "fingerprint":"d"*64, "new_fits":0}

    def computed(self, root, prep, target, settings, work, output, **kwargs):
        return {"target":target, "diagnostics_passed":True, "covariance_ppc_passed":True, "status":"diagnostics_passed",
                "fit_fingerprint":"b"*64, "validation_fingerprint":"c"*64}

    def batch(self, *, computed=None, preparation=None, factor=None, prior=None, jobs=1):
        with mock.patch.object(trajectory_execution, "run_trajectories", side_effect=prior or (lambda *a, **k:deepcopy(self.prior))):
            with mock.patch.object(pe, "sample_one", side_effect=computed or self.computed):
                with mock.patch.object(factor_execution, "summarize_selected_shared_factors", side_effect=factor or (lambda *a, **k:deepcopy(self.factor))):
                    with mock.patch.object(pe, "run_preparation", side_effect=preparation or (lambda *a, **k:{"fingerprint":"a"*64})):
                        return pe.run_mechanics(self.base/"source", self.config, self.analysis,
                            self.primary_sampling, self.mechanics_sampling, jobs=jobs, resume=True)

    def test_complete_six_model_and_six_factor_batch_publishes_explicit_marker(self):
        result = self.batch()
        self.assertEqual(result["selected_models"], 6); self.assertEqual(result["shared_factor_rows"], 6)
        self.assertEqual([x["target"] for x in result["mechanics_models"]], list(pe.TARGETS))
        self.assertTrue(result["accepted"])
        self.assertEqual(json.loads((self.work/"mechanics_current.json").read_text()), result)

    def test_failed_mechanics_or_factor_cannot_publish_current_marker(self):
        def failed(*args, **kwargs):
            value = self.computed(*args, **kwargs)
            if value["target"] == "R": value.update(diagnostics_passed=False,status="diagnostics_failed")
            return value
        self.assertFalse(self.batch(computed=failed)["accepted"])
        self.factor["accepted"] = False; self.assertFalse(self.batch()["accepted"])
        self.assertFalse((self.work/"mechanics_current.json").exists())

    def test_incomplete_preceding_trajectory_batch_is_rejected_before_sampling(self):
        for field, value in [("accepted",False), ("analyses",3), ("fitted_models",7)]:
            original = self.prior[field]; self.prior[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.batch(computed=lambda *a, **k:self.fail("sampled after failed prerequisite"))
            self.prior[field] = original
        self.assertFalse((self.work/"mechanics_current.json").exists())

    def test_changed_primary_or_mechanics_sampling_and_preparation_block_completion(self):
        for path in [self.primary_sampling, self.mechanics_sampling]:
            def changed(*args, **kwargs):
                path.write_text(path.read_text()+"\n")
                return self.computed(*args, **kwargs)
            with self.subTest(path=path.name), self.assertRaises(ValidationError): self.batch(computed=changed)
        with self.assertRaises(ValidationError): self.batch(preparation=lambda *a, **k:{"fingerprint":"changed"})
        self.assertFalse((self.work/"mechanics_current.json").exists())

    def test_invalid_concurrency_or_generated_sampling_path_rejected(self):
        for jobs in [0, 4, True, 1.5]:
            with self.subTest(jobs=jobs), self.assertRaises(ValidationError): self.batch(jobs=jobs)
        self.mechanics_sampling = self.work/"mechanics.json"; write_json(self.mechanics_sampling, SETTINGS)
        with self.assertRaises(ValidationError): self.batch()

    def cli(self, *, extra=None, result=None, error=None):
        stdout, stderr = StringIO(), StringIO()
        result = result or {"selected_models":6, "shared_factor_rows":6, "accepted":True, "status":"accepted"}
        args = ["--through","M07","--config",str(self.config),"--analysis",str(self.analysis),"--resume"]
        with mock.patch.object(pe, "run_mechanics", return_value=result, side_effect=error) as runner:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = pipeline.main(args+(extra or []), root=self.base/"source")
        return code, stdout.getvalue(), stderr.getvalue(), runner

    def test_cli_exact_counts_and_both_explicit_sampling_controls(self):
        code, output, error, runner = self.cli(extra=["--sampling",str(self.primary_sampling),
             "--mechanics-sampling",str(self.mechanics_sampling),"--jobs","3"])
        self.assertEqual(code,0); self.assertEqual(error,"")
        self.assertIn("6 models and 6 shared-factor rows",output)
        self.assertEqual(runner.call_args.args[3:5],(str(self.primary_sampling),str(self.mechanics_sampling)))
        self.assertEqual(runner.call_args.kwargs,{"jobs":3,"resume":True})

    def test_cli_failure_status_partial_targets_and_private_error_text(self):
        result={"selected_models":6,"shared_factor_rows":6,"accepted":False,"status":"diagnostics_failed"}
        code, output, _, _ = self.cli(result=result); self.assertEqual(code,2); self.assertIn("diagnostics_failed",output)
        code, _, error, runner = self.cli(extra=["--targets","ratio_pre"])
        self.assertEqual(code,2); self.assertIn("all selected models",error); runner.assert_not_called()
        for error in [ValueError("FABRICATED_PRIVATE_SENTINEL"), RuntimeError("FABRICATED_PRIVATE_SENTINEL"), OSError("FABRICATED_PRIVATE_SENTINEL")]:
            code, output, errors, _ = self.cli(error=error)
            self.assertEqual(code,2); self.assertNotIn("FABRICATED_PRIVATE_SENTINEL",output+errors)

    def test_full_render_and_misplaced_mechanics_sampling_remain_explicit_errors(self):
        for option in ["--full","--render"]:
            with mock.patch.object(pe,"run_mechanics") as runner, redirect_stderr(StringIO()):
                self.assertEqual(pipeline.main([option],root=self.base/"source"),2)
            runner.assert_not_called()
        with redirect_stderr(StringIO()):
            self.assertEqual(pipeline.main(["--through","M06","--config",str(self.config),
                "--analysis",str(self.analysis),"--mechanics-sampling",str(self.mechanics_sampling)],root=self.base/"source"),2)


class SharedFactorExecutionTests(unittest.TestCase):
    def setUp(self):
        # Runner boundary fixture: numerical PPC is tested independently.
        self.covariance={'status':'passed','accepted':True,'joint_pass':True}
        ppc_patch=mock.patch.object(pe,'evaluate_covariance',side_effect=lambda *a,**k:deepcopy(self.covariance))
        ppc_patch.start();self.addCleanup(ppc_patch.stop)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.prep = self.base/"source", self.base/"preparation"
        self.work, self.output = self.base/"private", self.base/"aggregate"
        self.prep.mkdir()
        for name in factor_execution.SOURCES:
            path = self.root/"src/dchs_lung_function"/name
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text("# synthetic source\n")
        (self.root/"requirements-models.lock.txt").write_text("# synthetic lock\n")
        self.input_file = self.prep/"verified-evidence.json"; write_json(self.input_file,{"synthetic":True})
        self.input_hash = file_hash(self.input_file)
        self.descriptor = {"sha256":"a"*64,"preparation_fingerprint":"b"*64}
        for patch in [mock.patch.object(factor_execution.primary,"contract_descriptor",side_effect=lambda *a:deepcopy(self.descriptor)),
                      mock.patch.object(factor_execution,"model_runtime_identity",return_value={"synthetic":True})]:
            patch.start();self.addCleanup(patch.stop)

    def verified(self, root, prep, work, output, settings, target):
        return {"posterior":{"synthetic":True},"contract":{"synthetic":True},
                "binding":{"target":target,"fit_fingerprint":"c"*64,"validation_fingerprint":"d"*64,
                    "raw_receipt_sha256":"e"*64,"validation_receipt_sha256":"f"*64,"contract_sha256":self.descriptor["sha256"]},
                "input_files":{self.input_file:self.input_hash}}

    def derived(self, posterior, contract, target):
        return SimpleNamespace(estimates=pd.DataFrame({"target":[target],"mean":[-.4],"n_outcome_children":[3]}),
                               metadata={"target":target,"new_fit_required":False})

    def run_factor(self,*,resume=False,derive=None,verify=None,work=None):
        with mock.patch.object(factor_execution,"_verified_primary",side_effect=verify or self.verified):
            with mock.patch.object(factor_execution,"summarize_shared_factor",side_effect=derive or self.derived):
                return factor_execution.summarize_selected_shared_factors(self.root,self.prep,work or self.work,
                    self.output,SETTINGS,resume=resume)

    def destination(self,result):
        return self.output/"shared_factor"/result["fingerprint"]

    def assert_clean(self):
        self.assertEqual(list(self.base.rglob(".staging-*")),[])
        self.assertEqual(list(self.base.rglob(".preparation.lock")),[])

    def test_six_selected_factor_rows_roundtrip_small_counts_and_no_sampling(self):
        with mock.patch.object(factor_execution.primary,"sample_one",side_effect=AssertionError("unexpected sampler")):
            result=self.run_factor();directory=self.destination(result)
            self.assertTrue(result["accepted"]);self.assertEqual(result["new_fits"],0)
            table=pd.read_csv(directory/"estimates.csv")
            self.assertEqual(list(table.target),list(factor_execution.DEFAULT_TARGETS))
            self.assertEqual(list(table.n_outcome_children),[3]*6)
            self.assertTrue(self.run_factor(resume=True)["resumed"])
        self.assert_clean()

    def test_exactly_one_correct_row_and_metadata_per_target_required(self):
        for kind in ["none","duplicate","wrong_target","metadata_target","requires_fit"]:
            def malformed(posterior,contract,target):
                result=self.derived(posterior,contract,target)
                if kind=="none":result.estimates=result.estimates.iloc[0:0]
                if kind=="duplicate":result.estimates=pd.concat([result.estimates]*2,ignore_index=True)
                if kind=="wrong_target":result.estimates["target"]="wrong"
                if kind=="metadata_target":result.metadata["target"]="wrong"
                if kind=="requires_fit":result.metadata["new_fit_required"]=True
                return result
            with self.subTest(kind=kind),self.assertRaises(ValidationError):self.run_factor(derive=malformed)
        self.assertEqual(list(self.output.rglob("receipt.json")),[])

    def test_missing_primary_prerequisite_never_calls_sampling(self):
        primary=factor_execution.primary
        with mock.patch.object(primary,"sampling_identity",return_value=("c"*64,{"sampling":SETTINGS})), \
             mock.patch.object(primary,"validation_identity",return_value=("d"*64,{})), \
             mock.patch.object(primary,"sample_one",side_effect=AssertionError("unexpected sampling")) as sampler:
            with self.assertRaises(ValidationError):
                factor_execution._verified_primary(self.root,self.prep,self.work,self.output,SETTINGS,"ratio_pre_z")
            sampler.assert_not_called()

    def test_primary_helper_verifies_existing_complete_arrays_controls_and_diagnostics(self):
        primary=factor_execution.primary;target="ratio_pre_z";fit_fp="c"*64;validation_fp="d"*64
        controls=primary.resolved_sampling(target,SETTINGS)
        fit_identity={"target":target,"sampling":controls,"runtime":{"synthetic":True}}
        validation_identity={"synthetic":True}
        raw=self.work/"primary_fits"/target/fit_fp;validated=self.output/"primary_models"/target/validation_fp
        raw.mkdir(parents=True);validated.mkdir(parents=True)
        rng=np.random.default_rng(107)
        posterior={"loading_ratio_mechanics":rng.normal(size=(4,1200)),"child_z":rng.normal(size=(4,1200,3))}
        shapes={key:value.shape for key,value in posterior.items()}
        stats={"energy":rng.normal(size=(4,1200)),"diverging":np.zeros((4,1200),bool),
               "reached_max_treedepth":np.zeros((4,1200),bool)}
        diagnostic=REAL_DIAGNOSTICS(posterior,stats,primary.CRITERIA)
        self.assertTrue(diagnostic["passed"])
        metadata={"target":target,"controls":controls,"runtime":fit_identity["runtime"],
                  "realized_chains":4,"realized_draws_per_chain":1200,"realized_tuning_steps":1600,
                  "posterior_variables":list(posterior),"sample_stat_variables":list(stats)}
        summary={"target":target,"fit_fingerprint":fit_fp,"validation_fingerprint":validation_fp,
                 "diagnostics_passed":True,"covariance_ppc_passed":True,"status":"diagnostics_passed"}

        def publish():
            np.savez_compressed(raw/"posterior.npz",**posterior)
            np.savez_compressed(raw/"sample_stats.npz",**stats)
            write_json(raw/"sampling.json",metadata)
            write_json(raw/"receipt.json",{"status":"sampled","identity":fit_identity,"fingerprint":fit_fp,
                "files":{name:file_hash(raw/name) for name in ["posterior.npz","sample_stats.npz","sampling.json"]}})
            write_json(validated/"diagnostics.json",diagnostic);write_json(validated/"summary.json",summary)
            write_json(validated/"covariance_ppc.json",self.covariance)
            write_json(validated/"result_metadata.json",{"synthetic":True})
            for name in ["estimates.csv","shared_factor.csv","sample_counts.csv"]:(validated/name).write_text("value\n1\n")
            names=list(primary.RESULT_FILES)
            write_json(validated/"receipt.json",{"identity":validation_identity,"fingerprint":validation_fp,"summary":summary,
                "raw_receipt_sha256":file_hash(raw/"receipt.json"),"files":{name:file_hash(validated/name) for name in names}})

        def verify():
            return factor_execution._verified_primary(self.root,self.prep,self.work,self.output,SETTINGS,target)

        with mock.patch.object(primary,"sampling_identity",return_value=(fit_fp,fit_identity)), \
             mock.patch.object(primary,"validation_identity",return_value=(validation_fp,validation_identity)), \
             mock.patch.object(primary,"expected_posterior_shapes",return_value=shapes), \
             mock.patch.object(factor_execution,"evaluate_covariance",side_effect=lambda *a,**k:deepcopy(self.covariance)), \
             mock.patch.object(factor_execution,"configure_runtime",return_value={}), \
             mock.patch.object(factor_execution,"load_artifacts",return_value={"synthetic":True}), \
             mock.patch.object(primary,"sample_one",side_effect=AssertionError("unexpected sampler")) as sampler:
            publish();result=verify()
            self.assertEqual(result["binding"]["target"],target)
            np.testing.assert_array_equal(result["posterior"]["child_z"],posterior["child_z"])
            metadata["realized_tuning_steps"]=0;publish()
            with self.assertRaises(ValidationError):verify()
            metadata["realized_tuning_steps"]=1600
            saved=posterior.pop("child_z");metadata["posterior_variables"]=list(posterior);publish()
            with self.assertRaises(ValidationError):verify()
            posterior["child_z"]=saved;metadata["posterior_variables"]=list(posterior)
            stats["diverging"][0,0]=True;publish()
            with self.assertRaises(ValidationError):verify()
            stats["diverging"][0,0]=False
            summary["diagnostics_passed"]=False;publish()
            with self.assertRaises(ValidationError):verify()
            sampler.assert_not_called()

    def test_missing_or_changed_factor_files_and_malformed_receipt_rejected(self):
        directory=self.destination(self.run_factor())
        for name in factor_execution.RESULT_FILES:
            path=directory/name;original=path.read_bytes();path.write_bytes(original+b"tampered")
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_factor(resume=True)
            path.unlink()
            with self.subTest(name=name,missing=True),self.assertRaises(ValidationError):self.run_factor(resume=True)
            path.write_bytes(original)
        write_json(directory/"receipt.json",[])
        with self.assertRaises(ValidationError):self.run_factor(resume=True)

    def test_cached_factor_values_and_metadata_recomputed_even_with_matching_hashes(self):
        for name in ["estimates.csv","metadata.json","summary.json"]:
            directory=self.destination(self.run_factor());path=directory/name
            if name.endswith("csv"):
                frame=pd.read_csv(path);frame.loc[0,"mean"]+=1;frame.to_csv(path,index=False)
            else:
                value=json.loads(path.read_text());value["synthetic_contradiction"]=True;write_json(path,value)
            receipt=json.loads((directory/"receipt.json").read_text());receipt["files"][name]=file_hash(path)
            if name=="summary.json":receipt["summary"]=value
            write_json(directory/"receipt.json",receipt)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_factor(resume=True)
            shutil.rmtree(directory)

    def test_source_or_primary_evidence_changed_before_publication_rejected(self):
        def changed(posterior,contract,target):
            self.input_file.write_text("changed evidence")
            return self.derived(posterior,contract,target)
        with self.assertRaises(ValidationError):self.run_factor(derive=changed)
        self.assertEqual(list(self.output.rglob("receipt.json")),[]);self.assert_clean()

    def test_source_or_contract_change_during_cached_read_rejected(self):
        original=factor_execution.primary.verify_files
        for change_source in [True,False]:
            self.run_factor()
            def changed(*args):
                result=original(*args)
                if change_source:(self.root/"src/dchs_lung_function/analysis/shared_factor.py").write_text("# changed after identity\n")
                else:self.descriptor["sha256"]="z"*64
                return result
            with mock.patch.object(factor_execution.primary,"verify_files",side_effect=changed):
                with self.subTest(source=change_source),self.assertRaises(ValidationError):self.run_factor(resume=True)

    def test_lock_cleanup_publication_failure_and_shared_output_lock(self):
        with mock.patch.object(factor_execution.os,"replace",side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):self.run_factor()
        self.assert_clean();self.assertEqual(list(self.output.rglob("receipt.json")),[])
        self.run_factor()
        with factor_execution.preparation_locks(self.work/"shared_factor",self.output/"shared_factor"):
            with self.assertRaises(ValidationError):self.run_factor(resume=True)
            with self.assertRaises(ValidationError):self.run_factor(work=self.base/"other-private",resume=True)
        self.assert_clean()

    def test_loaded_output_roots_protect_source_and_preparation(self):
        for work in [self.root,self.prep,self.output]:
            with self.subTest(work=work),self.assertRaises(ValidationError):self.run_factor(work=work)


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
