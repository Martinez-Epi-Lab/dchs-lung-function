"""Synthetic adversarial checks for typed artifacts and preparation publication."""
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function import execution as ex
from dchs_lung_function.pipeline import ValidationError


@dataclass
class SyntheticBundle:
    frame: pd.DataFrame
    array: np.ndarray


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, value):
        return ex.ArtifactWriter(self.directory).finish(value)

    def test_typed_frames_arrays_and_dataclass_roundtrip_exactly(self):
        frame = pd.DataFrame({
            "integer": pd.array([1, None, 9], dtype="Int64"),
            "text": pd.array(["a", None, "μ"], dtype="string"),
            "flag": pd.array([True, None, False], dtype="boolean"),
            "category": pd.Categorical(["yes", "no", None], categories=["no", "yes"], ordered=True),
            "date": pd.to_datetime(["2001-01-01", None, "2001-01-03"]),
            "number": [1.25, np.nan, -np.inf],
        }, index=pd.Index([5, 5, 2], name="synthetic_index"))
        arrays = {"float": np.array([[1.0, np.nan], [np.inf, -3.0]], dtype="float64"),
                  "integer": np.arange(6, dtype="int16").reshape(2, 3),
                  "empty": np.empty((0, 3), dtype="float32"),
                  "date": np.array(["2001-01-01", "NaT"], dtype="datetime64[ns]"),
                  "object": np.array(["synthetic", None, pd.NA], dtype=object)}
        payload = {"bundle": SyntheticBundle(frame, arrays["integer"]), "arrays": arrays,
                   "series": frame["integer"], "tuple": ("x", 3, None),
                   "index": pd.Index([2, 4, 4], name="position")}
        self.write(payload)
        result = ex.load_artifacts(self.directory)
        pd.testing.assert_frame_equal(result["bundle"]["frame"], frame, check_exact=True)
        pd.testing.assert_series_equal(result["series"], frame["integer"], check_exact=True)
        pd.testing.assert_index_equal(result["index"], payload["index"], exact=True)
        for key, value in arrays.items():
            self.assertEqual(result["arrays"][key].dtype, value.dtype)
            if key == "object":
                self.assertEqual(result["arrays"][key][0], "synthetic")
                self.assertIsNone(result["arrays"][key][1])
                self.assertIs(result["arrays"][key][2], pd.NA)
            else:
                np.testing.assert_array_equal(result["arrays"][key], value)
        self.assertEqual(result["tuple"], ("x", 3, None))

    def test_changed_and_missing_artifacts_rejected(self):
        manifest = self.write(np.array([1.0, 2.0]))
        path = self.directory / next(iter(manifest["files"]))
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with self.assertRaises(ValidationError):
            ex.load_artifacts(self.directory)
        path.unlink()
        with self.assertRaises(ValidationError):
            ex.verify_artifacts(self.directory)

    def test_manifest_path_traversal_and_unverified_tree_file_rejected(self):
        self.write(np.array([1.0]))
        manifest_path = self.directory / "artifacts.json"
        original = json.loads(manifest_path.read_text())
        changed = json.loads(json.dumps(original))
        changed["files"]["../outside.npy"] = "0" * 64
        ex.write_json(manifest_path, changed)
        with self.assertRaises(ValidationError):
            ex.verify_artifacts(self.directory)
        original["tree"]["file"] = "unregistered.npy"
        ex.write_json(manifest_path, original)
        with self.assertRaises(ValidationError):
            ex.load_artifacts(self.directory)

    def test_artifact_symlink_rejected_even_when_digest_matches(self):
        manifest = self.write(np.array([1.0]))
        path = self.directory / next(iter(manifest["files"]))
        outside = self.directory / "outside.npy"
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaises(ValidationError):
            ex.load_artifacts(self.directory)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        (self.root / "src").mkdir(parents=True)
        (self.root / "config").mkdir()
        (self.root / "src/example.py").write_text("VALUE = 1\n")
        (self.root / "run.py").write_text("# synthetic runner\n")
        (self.root / "requirements-preparation.lock.txt").write_text("# synthetic lock\n")
        ex.write_json(self.root / "config/input_contract.json", {
            "schema_version": 1, "inputs": [{"id": "baseline", "required": True}]})
        self.input_path = self.base / "synthetic-input.txt"
        self.input_path.write_text("fabricated source only\n")
        self.font = self.base / "synthetic-font.ttf"
        self.font.write_bytes(b"synthetic placeholder for file-level preflight")
        self.configuration = {"schema_version": 1,
            "inputs": {"baseline": {"path": str(self.input_path)}},
            "directories": {"work": str(self.base / "private-work"), "outputs": str(self.base / "private-output")},
            "fonts": {"arial": str(self.font), "arial_bold": str(self.font)}}
        self.settings = {"schema_version": 1, "event_end": "2020-12-31", "respiratory_end": "2021-12-31"}
        self.config_path = self.base / "paths.local.json"
        self.analysis_path = self.base / "analysis.local.json"
        self.save_configuration()
        ex.write_json(self.analysis_path, self.settings)
        self.runtime_patch = mock.patch.object(ex, "runtime_identity", return_value={"python": "synthetic", "packages": {}})
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def save_configuration(self):
        ex.write_json(self.config_path, self.configuration)

    def output(self):
        return ({"synthetic": pd.DataFrame({"count": [1, 2]}), "matrix": np.eye(2)},
                {"through": "M01", "synthetic": True, "n": 2},
                {"counts": pd.DataFrame({"category": ["a", "b"], "n": [1, 2]})})

    def run_preparation(self, *, resume=False, config_path=None, compute=None):
        with mock.patch.object(ex, "compute_preparation", side_effect=compute or (lambda *args: self.output())):
            return ex.run_preparation(self.root, config_path or self.config_path, self.analysis_path, "M01", resume=resume)

    def run_dirs(self, result):
        fingerprint = result["fingerprint"]
        return (Path(self.configuration["directories"]["work"]) / "preparation_runs" / fingerprint,
                Path(self.configuration["directories"]["outputs"]) / "preparation_runs" / fingerprint)

    def assert_no_completed_or_staged_run(self):
        self.assertFalse(list(self.base.rglob("receipt.json")))
        self.assertFalse(list(self.base.rglob(".staging-*")))
        self.assertFalse(list(self.base.rglob(".preparation.lock")))
        self.assertFalse(list(self.base.rglob("preparation_current.json")))

    def test_completed_run_roundtrip_and_resume_skip_recomputation(self):
        result = self.run_preparation()
        work, out = self.run_dirs(result)
        restored = ex.load_artifacts(work)
        pd.testing.assert_frame_equal(restored["synthetic"], self.output()[0]["synthetic"])
        receipt = json.loads((work / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "prepared")
        self.assertEqual(receipt["aggregate_files"]["summary.json"], ex.file_hash(out / "summary.json"))
        with mock.patch.object(ex, "compute_preparation", side_effect=AssertionError("Cached run recomputed")):
            again = ex.run_preparation(self.root, self.config_path, self.analysis_path, "M01", resume=True)
        self.assertTrue(again["resumed"])
        self.assertEqual(again["fingerprint"], result["fingerprint"])

    def test_input_code_configuration_and_analysis_changes_invalidate_cache(self):
        first = self.run_preparation()
        seen = {first["fingerprint"]}
        changes = [lambda: self.input_path.write_text("changed fabricated source\n"),
                   lambda: (self.root / "src/example.py").write_text("VALUE = 2\n"),
                   lambda: self.configuration["inputs"]["baseline"].update(sha256=ex.file_hash(self.input_path)),
                   lambda: self.settings.update(event_end="2020-12-30")]
        for change in changes:
            change()
            self.save_configuration()
            ex.write_json(self.analysis_path, self.settings)
            result = self.run_preparation(resume=True)
            self.assertFalse(result["resumed"])
            self.assertNotIn(result["fingerprint"], seen)
            seen.add(result["fingerprint"])

    def test_configured_hash_mismatch_rejected_before_computation(self):
        self.configuration["inputs"]["baseline"]["sha256"] = "0" * 64
        self.save_configuration()
        with mock.patch.object(ex, "compute_preparation") as compute:
            with self.assertRaises(ValidationError):
                ex.run_preparation(self.root, self.config_path, self.analysis_path, "M01")
            compute.assert_not_called()

    def test_failed_computation_leaves_no_completed_receipt_or_lock(self):
        def fail(*args):
            raise RuntimeError("deliberate synthetic failure")
        with self.assertRaises(RuntimeError):
            self.run_preparation(compute=fail)
        self.assert_no_completed_or_staged_run()

    def test_second_staging_allocation_failure_cleans_first_stage_and_locks(self):
        original = ex.tempfile.mkdtemp
        count = 0
        def allocate(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("synthetic staging failure")
            return original(*args, **kwargs)
        with mock.patch.object(ex.tempfile, "mkdtemp", side_effect=allocate):
            with self.assertRaises(OSError):
                self.run_preparation()
        self.assert_no_completed_or_staged_run()

    def test_partial_publication_is_rolled_back(self):
        original = ex.os.replace
        failed = False
        def replace(source, target):
            nonlocal failed
            source, target = Path(source), Path(target)
            if not failed and source.name.startswith(".staging-") and "private-work" in target.parts:
                failed = True
                raise OSError("synthetic publication failure")
            return original(source, target)
        with mock.patch.object(ex.os, "replace", side_effect=replace):
            with self.assertRaises(OSError):
                self.run_preparation()
        self.assertTrue(failed)
        self.assert_no_completed_or_staged_run()
        out = Path(self.configuration["directories"]["outputs"]) / "preparation_runs"
        self.assertEqual(list(out.iterdir()), [])

    def test_pointer_publication_failure_rolls_back_pair(self):
        original = ex.os.replace
        def replace(source, target):
            if Path(target).name == "preparation_current.json":
                raise OSError("synthetic pointer failure")
            return original(source, target)
        with mock.patch.object(ex.os, "replace", side_effect=replace):
            with self.assertRaises(OSError):
                self.run_preparation()
        self.assert_no_completed_or_staged_run()

    def test_files_changed_during_computation_are_rejected(self):
        def change(*args):
            self.analysis_path.write_text(self.analysis_path.read_text() + "\n")
            return self.output()
        with self.assertRaises(ValidationError):
            self.run_preparation(compute=change)
        self.assert_no_completed_or_staged_run()
        def change_config(*args):
            self.config_path.write_text(self.config_path.read_text() + "\n")
            return self.output()
        with self.assertRaises(ValidationError):
            self.run_preparation(compute=change_config)
        self.assert_no_completed_or_staged_run()

    def test_changed_source_during_computation_is_rejected(self):
        def change(*args):
            (self.root / "src/example.py").write_text("VALUE = 3\n")
            return self.output()
        with self.assertRaises(ValidationError):
            self.run_preparation(compute=change)
        self.assert_no_completed_or_staged_run()

    def test_resume_rejects_modified_aggregate_and_artifact_manifest(self):
        result = self.run_preparation()
        work, out = self.run_dirs(result)
        aggregate = out / "counts.csv"
        original = aggregate.read_bytes()
        aggregate.write_bytes(original + b"tamper")
        with self.assertRaises(ValidationError):
            self.run_preparation(resume=True)
        aggregate.write_bytes(original)
        manifest = work / "artifacts.json"
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaises(ValidationError):
            self.run_preparation(resume=True)

    def test_resume_requires_completed_receipt_and_output_inventory(self):
        result = self.run_preparation()
        work, _out = self.run_dirs(result)
        receipt_path = work / "receipt.json"
        original = json.loads(receipt_path.read_text())
        changed = {**original, "status": "failed"}
        ex.write_json(receipt_path, changed)
        with self.assertRaises(ValidationError):
            self.run_preparation(resume=True)
        changed = {**original, "aggregate_files": {}}
        ex.write_json(receipt_path, changed)
        with self.assertRaises(ValidationError):
            self.run_preparation(resume=True)

    def test_resume_rejects_receipt_summary_disagreeing_with_verified_output(self):
        result = self.run_preparation()
        work, _out = self.run_dirs(result)
        receipt_path = work / "receipt.json"
        changed = json.loads(receipt_path.read_text())
        changed["summary"]["n"] = 999
        ex.write_json(receipt_path, changed)
        with self.assertRaises(ValidationError):
            self.run_preparation(resume=True)

    def test_generated_directories_cannot_overlap_source_inputs_or_analysis(self):
        choices = [self.root, self.input_path.parent, self.base / "private-work/child"]
        for path in choices:
            with self.subTest(path_kind=path.name):
                self.configuration["directories"]["outputs"] = str(path)
                self.save_configuration()
                with self.assertRaises(ValidationError):
                    self.run_preparation()
        self.configuration["directories"]["outputs"] = str(self.base / "private-output")
        self.configuration["directories"]["work"] = str(self.base / "settings-work")
        self.analysis_path = self.base / "settings-work/analysis.local.json"
        self.analysis_path.parent.mkdir()
        ex.write_json(self.analysis_path, self.settings)
        self.save_configuration()
        with self.assertRaises(ValidationError):
            self.run_preparation()

    def _assert_concurrent_call_rejected(self, *, second_configuration=None, resume=False, first_resume=False):
        entered, release = threading.Event(), threading.Event()
        failures = []
        original_verify = ex.verify_artifacts
        def blocking_compute(*args):
            if threading.current_thread().name == "synthetic-owner":
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test synchronization timed out")
            return self.output()
        def blocking_verify(*args):
            result = original_verify(*args)
            if first_resume and threading.current_thread().name == "synthetic-owner":
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test synchronization timed out")
            return result
        def owner():
            try:
                ex.run_preparation(self.root, self.config_path, self.analysis_path, "M01", resume=first_resume)
            except BaseException as error:
                failures.append(type(error).__name__)
        with mock.patch.object(ex, "compute_preparation", side_effect=blocking_compute), mock.patch.object(ex, "verify_artifacts", side_effect=blocking_verify):
            thread = threading.Thread(target=owner, name="synthetic-owner")
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(ValidationError):
                    ex.run_preparation(self.root, second_configuration or self.config_path, self.analysis_path, "M01", resume=resume)
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])

    def test_work_lock_rejects_concurrent_writer(self):
        self._assert_concurrent_call_rejected()

    def test_output_lock_covers_independent_work_roots(self):
        other = json.loads(json.dumps(self.configuration))
        other["directories"]["work"] = str(self.base / "different-private-work")
        path = self.base / "other-paths.local.json"
        ex.write_json(path, other)
        self._assert_concurrent_call_rejected(second_configuration=path)

    def test_resume_obeys_locks_against_another_resume(self):
        self.run_preparation()
        self._assert_concurrent_call_rejected(resume=True, first_resume=True)


if __name__ == "__main__":
    unittest.main()
