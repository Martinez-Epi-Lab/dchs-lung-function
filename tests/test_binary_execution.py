"""Synthetic adversarial tests for M05 publication, cache integrity and CLI.

Scientific fitting is mocked. File fingerprints, receipts, locks, CSV/JSON
validation, publication, and command dispatch are exercised without study data.
"""
from contextlib import redirect_stderr, redirect_stdout
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function import binary_execution as be
from dchs_lung_function import primary_execution, pipeline
from dchs_lung_function.models import binary_spirometry as model
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError

PRIVATE_SENTINEL = "FABRICATED_PRIVATE_RECORD_SENTINEL"


def synthetic_fit(contract, design):
    stage, outcome = contract["audit"]["stage"], contract["audit"]["outcome"]
    spec = model.fit_spec(stage, outcome, design)
    terms = ["Intercept", *spec["exposure_terms"]]
    rows = []
    for term in spec["reported_terms"]:
        rows.append({"stage": stage, "outcome": outcome, "design_key": design, "term": term,
                     "estimable": True, "status": "estimable", "reason_code": None, "reason": "",
                     "beta": .1, "se": .2, "rr": float(np.exp(.1)),
                     "lower": float(np.exp(.1 - 1.96 * .2)), "upper": float(np.exp(.1 + 1.96 * .2)),
                     "p_value": .617075077451974, "n_observations": 20, "n_children": 5, "events": 4,
                     "exposed_observations": 3, "exposed_children": 2, "exposed_events": 1,
                     "reference_observations": 10, "reference_children": 3, "reference_events": 2})
    metadata = {**spec, "accepted": True, "status": "success", "converged": True,
                "fit_status": "converged", "fit_failure_reasons": [], "expected_nonestimable_reason": None,
                "n_observations": 20, "n_children": 5, "events": 4, "terms": terms, "dropped_terms": [],
                "estimable_contrasts": len(rows), "non_estimable_contrasts": 0,
                "coefficients_finite": True, "covariance_finite": True}
    return model.BinaryFit(pd.DataFrame(rows), pd.Series([.1] * len(terms), index=terms),
                           pd.DataFrame(np.eye(len(terms)) * .04, index=terms, columns=terms), metadata)


def synthetic_patterns(contract):
    rows = []
    for state in ["no_recorded_tb", "future_tb", "post_infancy", "post_age1_before5", "post_at_after5"]:
        n = 0 if state == "future_tb" else 3
        for pattern in ["Normal", "Obstructive", "Possible restriction", "Mixed"]:
            events = n if pattern == "Normal" else 0
            rows.append({"stage": contract["audit"]["stage"], "exposure_state": state, "pattern": pattern,
                         "observations": n, "children": min(n, 2), "event_observations": events,
                         "event_children": min(events, 2), "prevalence": events / n if n else None})
    return pd.DataFrame(rows)


def make_nonestimable(fit, *, failed=False):
    table = fit.estimates.copy()
    table["estimable"] = False; table["status"] = "non_estimable"
    table["reason_code"] = "numerical_fit_failure" if failed else "zero_exposed_events"
    table["reason"] = "numerical fit failure" if failed else "zero exposed events"
    if not failed: table["exposed_events"] = 0
    for field in ["beta", "se", "rr", "lower", "upper", "p_value"]: table[field] = np.nan
    metadata = deepcopy(fit.metadata)
    metadata.update(accepted=not failed, status="fit_failed" if failed else "non_estimable",
                    estimable_contrasts=0, non_estimable_contrasts=len(table))
    if failed:
        metadata.update(converged=False, fit_status="not_converged", fit_failure_reasons=["Synthetic numerical failure."])
    return model.BinaryFit(table, fit.coefficients, fit.covariance, metadata)


class BinaryContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.keys = [f"binary/{s}/{o}" for s, o in dict.fromkeys((s, o) for s, o, _ in model.DEFAULT_FIT_SELECTION)]

    def write_preparation(self, *, changed_key=None, unrelated=1., omit=None):
        samples = {key: {"matrix": np.array([2. if key == changed_key else 1.])}
                   for key in self.keys if key != omit}
        samples["primary/unrelated"] = {"matrix": np.array([unrelated])}
        ArtifactWriter(self.directory).finish({"M03": {"pft": {"samples": samples}}})
        identity = {"synthetic": True}
        write_json(self.directory / "receipt.json", {"through": "M03", "status": "prepared",
            "identity": identity, "fingerprint": be._digest(identity),
            "artifact_manifest_sha256": file_hash(self.directory / "artifacts.json")})

    def test_all_selected_contracts_are_bound_without_unrelated_model_content(self):
        self.assertEqual(len(model.DEFAULT_FIT_SELECTION), 28)
        self.assertEqual(len(self.keys), 10)
        self.write_preparation(); first = be.binary_contracts(self.directory)
        self.assertEqual(set(first["contracts"]), set(self.keys))
        self.write_preparation(unrelated=9.)
        self.assertEqual(be.binary_contracts(self.directory), first)
        self.write_preparation(changed_key=self.keys[-1])
        changed = be.binary_contracts(self.directory)
        self.assertNotEqual(changed["contracts"][self.keys[-1]], first["contracts"][self.keys[-1]])
        self.assertEqual(changed["contracts"][self.keys[0]], first["contracts"][self.keys[0]])

    def test_incomplete_unverified_or_corrupt_preparation_is_rejected(self):
        self.write_preparation(omit=self.keys[-1])
        with self.assertRaises(ValidationError): be.binary_contracts(self.directory)
        self.write_preparation()
        receipt_path = self.directory / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["status"] = "incomplete"; write_json(receipt_path, receipt)
        with self.assertRaises(ValidationError): be.binary_contracts(self.directory)
        self.write_preparation()
        (self.directory / "artifact_00001.npy").write_bytes(b"corrupt")
        with self.assertRaises(ValidationError): be.binary_contracts(self.directory)


class BinaryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.prep = self.base / "source", self.base / "preparation"
        self.work, self.output = self.base / "private-work", self.base / "aggregate-output"
        self.prep.mkdir()
        for name in be.SOURCES:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fabricated implementation fingerprint\n")
        (self.root / "requirements-preparation.lock.txt").write_text("# fabricated runtime lock\n")
        self.descriptor = {"contracts": {f"binary/{s}/{o}": "a" * 64 for s, o, _ in model.DEFAULT_FIT_SELECTION},
                           "preparation_fingerprint": "b" * 64}
        for patch in [mock.patch.object(be, "binary_contracts", side_effect=lambda *a: deepcopy(self.descriptor)),
                      mock.patch.object(be, "load_artifacts", side_effect=self.load_contract),
                      mock.patch.object(be, "runtime_identity", return_value={"python": "synthetic", "packages": {}}),
                      mock.patch.object(model, "pattern_counts", side_effect=synthetic_patterns)]:
            patch.start(); self.addCleanup(patch.stop)

    def load_contract(self, prep, keys):
        _, stage, outcome = keys[-1].split("/")
        return {"audit": {"family": "binary", "stage": stage, "outcome": outcome},
                "frame": pd.DataFrame({"pid_child": [PRIVATE_SENTINEL]})}

    def run_fit(self, *, resume=False, compute=None, root=None, work=None, output=None):
        with mock.patch.object(model, "fit_binary", side_effect=compute or synthetic_fit):
            return be.fit_prepared_binary(root or self.root, self.prep, work or self.work,
                                          output or self.output, resume=resume)

    def destination(self, summary):
        return self.output / summary["fingerprint"]

    def assert_clean(self):
        self.assertEqual(list(self.base.rglob(".staging-*")), [])
        self.assertEqual(list(self.base.rglob(".preparation.lock")), [])

    def edit_verified_json(self, destination, filename, mutate):
        path = destination / filename; document = json.loads(path.read_text())
        mutate(document); write_json(path, document)
        receipt_path = destination / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"][filename] = file_hash(path)
        if filename == "summary.json": receipt["summary"] = document
        write_json(receipt_path, receipt)

    def edit_verified_csv(self, destination, filename, mutate):
        path = destination / filename
        frame = mutate(pd.read_csv(path)); frame.to_csv(path, index=False, float_format="%.17g")
        receipt_path = destination / "receipt.json"; receipt = json.loads(receipt_path.read_text())
        receipt["files"][filename] = file_hash(path); write_json(receipt_path, receipt)

    def test_all_selected_models_publish_six_aggregate_files_and_resume(self):
        summary = self.run_fit(); destination = self.destination(summary)
        self.assertEqual((summary["models"], summary["estimate_rows"]), (28, 46))
        self.assertTrue(summary["accepted"])
        self.assertEqual({p.name for p in destination.iterdir()}, {*be.RESULT_FILES, "receipt.json"})
        self.assertEqual(len(pd.read_csv(destination / "pattern_counts.csv")), 40)
        resumed = self.run_fit(resume=True, compute=lambda *a: self.fail("cached run refitted"))
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["fingerprint"], summary["fingerprint"])
        for file in destination.iterdir():
            self.assertNotIn(PRIVATE_SENTINEL, file.read_text())
            self.assertNotIn('"pid_child":', file.read_text())
        self.assert_clean()

    def test_expected_boundary_keeps_null_cells_and_small_counts_without_failure(self):
        summary = self.run_fit(compute=lambda c, d: make_nonestimable(synthetic_fit(c, d)))
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["non_estimable_rows"], 46)
        table = pd.read_csv(self.destination(summary) / "estimates.csv")
        self.assertTrue(table.rr.isna().all())
        self.assertEqual(set(table.exposed_events), {0})
        self.assertEqual(set(table.exposed_observations), {3})
        patterns = pd.read_csv(self.destination(summary) / "pattern_counts.csv")
        self.assertIn(0, patterns.event_observations.tolist())
        self.assertIn(3, patterns.event_observations.tolist())
        shutil.rmtree(self.destination(summary))
        def constant_outcome(contract, design):
            fit = make_nonestimable(synthetic_fit(contract, design))
            metadata = deepcopy(fit.metadata)
            metadata.update(expected_nonestimable_reason="constant_outcome", converged=False,
                            fit_status="not_fitted", coefficients_finite=False, covariance_finite=False, events=0)
            table = fit.estimates.copy(); table["events"] = 0; table["reference_events"] = 0
            coefficients = fit.coefficients * np.nan; covariance = fit.covariance * np.nan
            return model.BinaryFit(table, coefficients, covariance, metadata)
        structural = self.run_fit(compute=constant_outcome)
        self.assertTrue(structural["accepted"])
        self.assertTrue(pd.read_csv(self.destination(structural) / "coefficients.csv").coefficient.isna().all())

    def test_numerical_failure_retains_complete_failed_result_and_resume(self):
        summary = self.run_fit(compute=lambda c, d: make_nonestimable(synthetic_fit(c, d), failed=True))
        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["status"], "fit_failed")
        self.assertTrue((self.destination(summary) / "diagnostics.json").is_file())
        again = self.run_fit(resume=True, compute=lambda *a: self.fail("failed result evidence refitted"))
        self.assertFalse(again["accepted"])
        self.assert_clean()

    def test_existing_run_requires_explicit_resume(self):
        self.run_fit()
        with self.assertRaises(ValidationError): self.run_fit()

    def test_source_contract_runtime_and_lock_changes_invalidate_identity(self):
        seen = {self.run_fit()["fingerprint"]}
        changes = [lambda: (self.root / be.SOURCES[0]).write_text("# different model\n"),
                   lambda: self.descriptor["contracts"].update({"binary/pre/ratio_LLN": "c" * 64}),
                   lambda: (self.root / "requirements-preparation.lock.txt").write_text("# different lock\n")]
        for change in changes:
            change(); result = self.run_fit(resume=True)
            self.assertFalse(result["resumed"]); self.assertNotIn(result["fingerprint"], seen)
            seen.add(result["fingerprint"])
        with mock.patch.object(be, "runtime_identity", return_value={"python": "changed", "packages": {}}):
            self.assertNotIn(self.run_fit(resume=True)["fingerprint"], seen)

    def test_identical_copied_source_has_same_identity_and_reuses_results(self):
        first = self.run_fit()
        copied = self.base / "copied-source"; shutil.copytree(self.root, copied)
        result = self.run_fit(root=copied, resume=True, compute=lambda *a: self.fail("identical copy refitted"))
        self.assertTrue(result["resumed"])
        self.assertEqual(first["fingerprint"], result["fingerprint"])

    def test_each_missing_or_tampered_output_is_rejected(self):
        destination = self.destination(self.run_fit())
        for filename in be.RESULT_FILES:
            path = destination / filename; original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.subTest(filename=filename, kind="changed"), self.assertRaises(ValidationError):
                self.run_fit(resume=True)
            path.unlink()
            with self.subTest(filename=filename, kind="missing"), self.assertRaises(ValidationError):
                self.run_fit(resume=True)
            path.write_bytes(original)

    def test_symlink_and_receipt_inventory_identity_or_status_rejected(self):
        destination = self.destination(self.run_fit())
        path = destination / "estimates.csv"; outside = self.base / "elsewhere.csv"
        path.rename(outside); path.symlink_to(outside)
        with self.assertRaises(ValidationError): self.run_fit(resume=True)
        path.unlink(); outside.rename(path)
        receipt_path = destination / "receipt.json"; original = json.loads(receipt_path.read_text())
        for field, value in [("files", {}), ("fingerprint", "wrong"), ("identity", {}), ("status", "fit_failed")]:
            changed = deepcopy(original); changed[field] = value; write_json(receipt_path, changed)
            with self.subTest(field=field), self.assertRaises(ValidationError): self.run_fit(resume=True)
            write_json(receipt_path, original)

    def test_summary_and_diagnostic_acceptance_cannot_contradict(self):
        destination = self.destination(self.run_fit())
        self.edit_verified_json(destination, "summary.json", lambda x: x.update(accepted=False, status="fit_failed"))
        with self.assertRaises(ValidationError): self.run_fit(resume=True)

    def test_contradictory_fit_metadata_fails_even_with_matching_hashes(self):
        for mutation in [lambda m: m.update(status="fit_failed"),
                         lambda m: m.update(fit_failure_reasons=["failure"]),
                         lambda m: m.update(converged=False),
                         lambda m: m.update(estimable_contrasts=0)]:
            destination = self.destination(self.run_fit())
            self.edit_verified_json(destination, "diagnostics.json", lambda x: mutation(x["models"][0]["metadata"]))
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError): self.run_fit(resume=True)
            shutil.rmtree(destination)

    def test_missing_model_duplicate_model_or_invalid_labels_rejected(self):
        mutations = [lambda x: x["models"].pop(),
                     lambda x: x["models"].__setitem__(1, deepcopy(x["models"][0])),
                     lambda x: x["models"][0].update(stage="post")]
        for mutation in mutations:
            destination = self.destination(self.run_fit())
            self.edit_verified_json(destination, "diagnostics.json", mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError): self.run_fit(resume=True)
            shutil.rmtree(destination)

    def test_malformed_diagnostic_objects_raise_validation_error(self):
        for value in [None, "invalid", {"model_id": "pre/ratio_LLN/before5", "metadata": None}]:
            destination = self.destination(self.run_fit())
            self.edit_verified_json(destination, "diagnostics.json", lambda x: x["models"].__setitem__(0, value))
            with self.subTest(value=value), self.assertRaises(ValidationError): self.run_fit(resume=True)
            shutil.rmtree(destination)

    def test_missing_contrasts_coefficients_covariance_and_patterns_rejected(self):
        for filename in ["estimates.csv", "coefficients.csv", "covariance.csv", "pattern_counts.csv"]:
            destination = self.destination(self.run_fit())
            self.edit_verified_csv(destination, filename, lambda frame: frame.iloc[1:])
            with self.subTest(filename=filename), self.assertRaises(ValidationError): self.run_fit(resume=True)
            shutil.rmtree(destination)

    def test_nonfinite_coefficients_or_covariance_cannot_claim_finite_accepted_fit(self):
        for filename, column in [("coefficients.csv", "coefficient"), ("covariance.csv", "covariance")]:
            destination = self.destination(self.run_fit())
            def changed(frame):
                frame.loc[0, column] = np.nan
                return frame
            self.edit_verified_csv(destination, filename, changed)
            with self.subTest(filename=filename), self.assertRaises(ValidationError): self.run_fit(resume=True)
            shutil.rmtree(destination)

    def test_pattern_prevalence_and_count_contradictions_rejected(self):
        for field, value in [("event_observations", 99), ("prevalence", .5)]:
            destination = self.destination(self.run_fit())
            def changed(frame):
                frame.loc[0, field] = value
                return frame
            self.edit_verified_csv(destination, "pattern_counts.csv", changed)
            with self.subTest(field=field), self.assertRaises(ValidationError): self.run_fit(resume=True)
            shutil.rmtree(destination)

    def test_missing_or_mislabelled_fresh_contrast_blocks_publication(self):
        for missing in [True, False]:
            def changed(contract, design):
                fit = synthetic_fit(contract, design)
                table = fit.estimates.iloc[0:0].copy() if missing else fit.estimates.assign(stage="invalid")
                return model.BinaryFit(table, fit.coefficients, fit.covariance, fit.metadata)
            with self.subTest(missing=missing), self.assertRaises(ValidationError): self.run_fit(compute=changed)
            self.assertEqual(list(self.output.rglob("receipt.json")), [])
            self.assert_clean()

    def test_contract_stage_or_outcome_mismatch_is_rejected_before_fitting(self):
        for field in ["stage", "outcome"]:
            def changed(prep, keys):
                contract = self.load_contract(prep, keys)
                contract["audit"][field] = "invalid"
                return contract
            with mock.patch.object(be, "load_artifacts", side_effect=changed):
                with self.subTest(field=field), self.assertRaises(ValidationError):
                    self.run_fit(compute=lambda *a: self.fail("invalid contract fitted"))
            self.assertEqual(list(self.output.rglob("receipt.json")), [])
            self.assert_clean()

    def test_changed_contract_or_source_during_fitting_blocks_publication(self):
        def changed(contract, design):
            self.descriptor["contracts"]["binary/pre/ratio_LLN"] = "d" * 64
            return synthetic_fit(contract, design)
        with self.assertRaises(ValidationError): self.run_fit(compute=changed)
        self.assertEqual(list(self.output.rglob("receipt.json")), []); self.assert_clean()

    def test_changed_contract_or_source_during_cached_return_rejected(self):
        original = be._verify_result
        for changed_source in [False, True]:
            self.run_fit()
            def changed(*args):
                result = original(*args)
                if changed_source: (self.root / be.SOURCES[0]).write_text("# changed during resume\n")
                else: self.descriptor["contracts"]["binary/pre/ratio_LLN"] = "e" * 64
                return result
            with mock.patch.object(be, "_verify_result", side_effect=changed):
                with self.assertRaises(ValidationError): self.run_fit(resume=True)

    def test_computation_allocation_or_publication_failure_cleans_staging_and_locks(self):
        with self.assertRaises(RuntimeError):
            self.run_fit(compute=lambda *a: (_ for _ in ()).throw(RuntimeError("fabricated compute error")))
        self.assert_clean()
        with mock.patch.object(be.tempfile, "mkdtemp", side_effect=OSError("fabricated allocation error")):
            with self.assertRaises(OSError): self.run_fit()
        self.assert_clean()
        with mock.patch.object(be.os, "replace", side_effect=OSError("fabricated publication error")):
            with self.assertRaises(OSError): self.run_fit()
        self.assert_clean(); self.assertEqual(list(self.output.rglob("receipt.json")), [])

    def test_source_preparation_and_nested_generated_roots_rejected(self):
        for work, output in [(self.root / "work", self.output), (self.prep / "work", self.output),
                             (self.work, self.work), (self.work, self.work / "nested"), (self.work, self.base)]:
            with self.subTest(work=work, output=output), self.assertRaises(ValidationError):
                self.run_fit(work=work, output=output)

    def test_resume_and_separate_work_roots_respect_shared_output_lock(self):
        self.run_fit()
        with be.preparation_locks(self.work, self.output):
            with self.assertRaises(ValidationError): self.run_fit(resume=True)
            with self.assertRaises(ValidationError): self.run_fit(work=self.base / "other-work", resume=True)
        self.assert_clean()

    def test_concurrent_writer_cannot_bypass_output_lock(self):
        entered, released = threading.Event(), threading.Event(); failures = []
        def blocked(contract, design):
            entered.set()
            if not released.wait(10): raise RuntimeError("test synchronization timeout")
            return synthetic_fit(contract, design)
        def worker():
            try: self.run_fit(compute=blocked)
            except BaseException as exc: failures.append(exc)
        thread = threading.Thread(target=worker); thread.start()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertRaises(ValidationError): self.run_fit(work=self.base / "other-work", resume=True)
        finally:
            released.set(); thread.join(10)
        self.assertFalse(thread.is_alive()); self.assertEqual(failures, []); self.assert_clean()


class BinaryCumulativeAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name); self.work = self.base / "work"; self.work.mkdir()
        self.config = self.base / "paths.local.json"; self.analysis = self.base / "analysis.local.json"
        self.sampling = self.base / "sampling.local.json"
        write_json(self.config, {"inputs": {}, "directories": {"work": str(self.work), "outputs": str(self.base / "outputs")}})
        write_json(self.analysis, {"synthetic": True}); write_json(self.sampling, {"synthetic": True})
        self.primary = {"all_twelve_models": True, "diagnostics_passed": True, "preparation_fingerprint": "a" * 64}
        self.binary = {"through": "M05", "models": 28, "accepted": True, "status": "accepted",
                       "estimable_rows": 45, "non_estimable_rows": 1, "fingerprint": "b" * 64}

    def run_cumulative(self, *, primary=None, binary=None, preparation=None):
        with mock.patch.object(primary_execution, "run_primary", side_effect=primary or (lambda *a, **k: deepcopy(self.primary))):
            with mock.patch.object(be, "fit_prepared_binary", side_effect=binary or (lambda *a, **k: deepcopy(self.binary))):
                with mock.patch.object(be, "run_preparation", side_effect=preparation or (lambda *a, **k: {"fingerprint": self.primary["preparation_fingerprint"]})):
                    return be.run_binary(self.base / "source", self.config, self.analysis, self.sampling, resume=True)

    def test_cumulative_m05_requires_all_twelve_accepted_primary_models(self):
        for field in ["all_twelve_models", "diagnostics_passed"]:
            original = self.primary[field]; self.primary[field] = False
            with mock.patch.object(be, "fit_prepared_binary") as fitting:
                with self.assertRaises(ValidationError): self.run_cumulative()
                fitting.assert_not_called()
            self.primary[field] = original
        self.assertFalse((self.work / "binary_current.json").exists())

    def test_success_publishes_pointer_and_failure_does_not(self):
        result = self.run_cumulative()
        self.assertEqual(result["primary_models_verified"], 12)
        self.assertEqual(json.loads((self.work / "binary_current.json").read_text())["fingerprint"], self.binary["fingerprint"])
        (self.work / "binary_current.json").unlink()
        self.binary.update(accepted=False, status="fit_failed")
        self.assertFalse(self.run_cumulative()["accepted"])
        self.assertFalse((self.work / "binary_current.json").exists())

    def test_changed_sampling_or_preparation_dependencies_block_cumulative_marker(self):
        def changed(*args, **kwargs):
            self.sampling.write_text(self.sampling.read_text() + "\n")
            return deepcopy(self.binary)
        with self.assertRaises(ValidationError): self.run_cumulative(binary=changed)
        with self.assertRaises(ValidationError):
            self.run_cumulative(preparation=lambda *a, **k: {"fingerprint": "different"})
        self.assertFalse((self.work / "binary_current.json").exists())

    def cli(self, extra=None, result=None, error=None):
        output, errors = StringIO(), StringIO()
        args = ["--through", "M05", "--config", str(self.config), "--analysis", str(self.analysis), "--resume"]
        with mock.patch.object(be, "run_binary", return_value=result or self.binary, side_effect=error) as runner:
            with redirect_stdout(output), redirect_stderr(errors):
                status = pipeline.main(args + (extra or []), root=self.base / "source")
        return status, output.getvalue(), errors.getvalue(), runner

    def test_cli_success_and_failed_fit_use_exact_aggregate_counts_and_exit_status(self):
        status, output, errors, runner = self.cli()
        self.assertEqual(status, 0); self.assertIn("28 models", output); self.assertIn("1 explicitly non-estimable", output)
        self.assertEqual(errors, ""); self.assertEqual(runner.call_args.kwargs["resume"], True)
        failed = {**self.binary, "accepted": False, "status": "fit_failed"}
        status, output, _, _ = self.cli(result=failed)
        self.assertEqual(status, 2); self.assertIn("fit_failed", output)

    def test_cli_rejects_partial_m04_targets_for_cumulative_m05(self):
        status, _, errors, runner = self.cli(extra=["--targets", "ratio_pre"])
        self.assertEqual(status, 2); self.assertIn("all twelve", errors); runner.assert_not_called()

    def test_cli_passes_explicit_sampling_and_jobs_to_primary_prerequisite(self):
        status, _, _, runner = self.cli(extra=["--sampling", str(self.sampling), "--jobs", "3"])
        self.assertEqual(status, 0); self.assertEqual(runner.call_args.kwargs["jobs"], 3)
        self.assertEqual(runner.call_args.args[3], str(self.sampling))

    def test_cli_sanitizes_third_party_exception_text(self):
        for error in [ValueError(PRIVATE_SENTINEL), RuntimeError(PRIVATE_SENTINEL), OSError(PRIVATE_SENTINEL)]:
            status, output, errors, _ = self.cli(error=error)
            self.assertEqual(status, 2)
            self.assertNotIn(PRIVATE_SENTINEL, output + errors)


if __name__ == "__main__":
    unittest.main()
