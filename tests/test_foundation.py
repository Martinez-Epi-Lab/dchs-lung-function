"""Synthetic checks for workflow contracts and read-only configuration validation."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dchs_lung_function.pipeline import ValidationError, main, validate_config, validate_plan


def synthetic_plan():
    modules = []
    for number in range(1, 16):
        module_id = "M{:02d}".format(number)
        modules.append({"id": module_id, "title": "Synthetic module", "stage": 2, "status": "planned",
                        "depends_on": [] if number == 1 else ["M{:02d}".format(number - 1)],
                        "model_families": ["MODEL"] if number == 4 else [],
                        "preparation_families": ["PREPARATION"] if number == 2 else [],
                        "output_ids": ["TABLE"] if number in (2, 4, 13) else [],
                        "code_targets": ["src/example.py"],
                        "steps": [{"id": module_id + ".1", "title": "Prepare", "work": "Compute", "acceptance": "Check"}],
                        "acceptance": ["Check"]})
    return {"schema_version": 1, "model_families": ["MODEL"], "preparation_families": ["PREPARATION"],
            "targets": [{"id": "TABLE", "label": "Table", "kind": "table", "model_ids": ["MODEL"],
                         "preparation_ids": ["PREPARATION"], "producer_modules": ["M02", "M04"], "render_module": "M13"}],
            "modules": modules,
            "gaps": [{"id": "G01", "title": "Input", "owners": ["M01"], "resolution": "Validate", "status": "open"}]}


class PlanTests(unittest.TestCase):
    def test_valid_complete_plan(self):
        self.assertEqual(validate_plan(synthetic_plan())["modules"], 15)

    def test_dependency_cycle_rejected(self):
        plan = synthetic_plan()
        plan["modules"][0]["depends_on"] = ["M04"]
        with self.assertRaisesRegex(ValidationError, "cycle"):
            validate_plan(plan)

    def test_forward_dependency_rejected(self):
        plan = synthetic_plan()
        plan["modules"][0]["depends_on"] = ["M15"]
        for module in plan["modules"][1:]:
            module["depends_on"] = []
        with self.assertRaisesRegex(ValidationError, "precede"):
            validate_plan(plan)

    def test_duplicate_model_ownership_rejected(self):
        plan = synthetic_plan()
        plan["modules"][4]["model_families"] = ["MODEL"]
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            validate_plan(plan)

    def test_prose_only_model_cannot_disappear(self):
        plan = synthetic_plan()
        plan["model_families"].append("PROSE_ONLY")
        plan["modules"][5]["model_families"] = ["PROSE_ONLY"]
        with self.assertRaisesRegex(ValidationError, "prose-only"):
            validate_plan(plan)

    def test_target_requires_family_owner_as_producer(self):
        plan = synthetic_plan()
        plan["targets"][0]["producer_modules"] = ["M02"]
        with self.assertRaisesRegex(ValidationError, "owning producer"):
            validate_plan(plan)

    def test_figure_requires_figure_renderer(self):
        plan = synthetic_plan()
        plan["targets"][0]["kind"] = "figure"
        with self.assertRaisesRegex(ValidationError, "M14"):
            validate_plan(plan)

    def test_malformed_collections_fail_gracefully(self):
        for key in ("modules", "targets", "gaps", "model_families", "preparation_families"):
            for malformed in (None, {}, "wrong", []):
                with self.subTest(key=key, value=malformed):
                    plan = synthetic_plan()
                    plan[key] = malformed
                    with self.assertRaises(ValidationError):
                        validate_plan(plan)

    def test_unknown_gap_owner_rejected(self):
        plan = synthetic_plan()
        plan["gaps"][0]["owners"] = ["M99"]
        with self.assertRaisesRegex(ValidationError, "unknown reference"):
            validate_plan(plan)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input_path = self.directory / "synthetic.txt"
        self.input_path.write_bytes(b"synthetic fixture only\n")
        for font in ("regular.ttf", "bold.ttf"):
            (self.directory / font).write_bytes(b"synthetic font placeholder")
        self.config_path = self.directory / "paths.local.json"
        self.contract = {"schema_version": 1, "inputs": [{"id": "study", "required": True}]}
        self.config = {"schema_version": 1, "inputs": {"study": {"path": "synthetic.txt"}},
                       "directories": {"work": "scratch/work", "outputs": "scratch/results"},
                       "fonts": {"arial": "regular.ttf", "arial_bold": "bold.ttf"}}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def check(self, config=None):
        return validate_config(config if config is not None else self.config, self.contract,
                               self.config_path, self.directory / "repository" / "src")

    def test_relative_paths_and_no_writes(self):
        before = sorted(str(path) for path in self.directory.rglob("*"))
        self.assertEqual(self.check()["inputs"], 1)
        self.assertEqual(before, sorted(str(path) for path in self.directory.rglob("*")))

    def test_missing_required_input_rejected(self):
        self.config["inputs"] = {}
        with self.assertRaisesRegex(ValidationError, "required input"):
            self.check()

    def test_unreadable_input_rejected_without_path(self):
        self.config["inputs"]["study"]["path"] = "private-participant-location.txt"
        with self.assertRaises(ValidationError) as caught:
            self.check()
        self.assertNotIn("private-participant", str(caught.exception))
        self.assertNotIn(str(self.directory), str(caught.exception))

    def test_hash_mismatch_rejected(self):
        self.config["inputs"]["study"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValidationError, "SHA256"):
            self.check()

    def test_matching_hash_passes(self):
        self.config["inputs"]["study"]["sha256"] = hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        self.assertEqual(self.check()["hashes_checked"], 1)

    def test_output_cannot_contain_input(self):
        self.config["directories"]["work"] = "."
        with self.assertRaises(ValidationError):
            self.check()

    def test_output_cannot_overlap_source(self):
        self.config["directories"]["work"] = "repository/src/generated"
        with self.assertRaisesRegex(ValidationError, "source directory"):
            self.check()

    def test_output_cannot_be_nested_inside_input_file(self):
        self.config["directories"]["work"] = "synthetic.txt/nested"
        with self.assertRaisesRegex(ValidationError, "inside a file"):
            self.check()

    def test_output_directories_cannot_overlap(self):
        self.config["directories"]["outputs"] = "scratch/work/results"
        with self.assertRaisesRegex(ValidationError, "nonoverlapping"):
            self.check()

    def test_optional_input_may_be_absent_but_not_invalid_if_provided(self):
        self.contract["inputs"].append({"id": "optional", "required": False})
        self.check()
        self.config["inputs"]["optional"] = {"path": "missing.txt"}
        with self.assertRaises(ValidationError):
            self.check()

    def test_malformed_config_fails_gracefully(self):
        for key in ("inputs", "fonts", "directories"):
            malformed = deepcopy(self.config)
            malformed[key] = []
            with self.subTest(key=key), self.assertRaises(ValidationError):
                self.check(malformed)

    def test_unknown_root_field_rejected(self):
        self.config["inputz"] = {}
        with self.assertRaisesRegex(ValidationError, "must contain only"):
            self.check()

    def test_misspelled_root_field_rejected(self):
        self.config["directory"] = self.config.pop("directories")
        with self.assertRaisesRegex(ValidationError, "must contain only"):
            self.check()

    def test_cli_protects_entire_repository(self):
        repository = self.directory / "repository"
        (repository / "config").mkdir(parents=True)
        (repository / "config" / "input_contract.json").write_text(json.dumps(self.contract), encoding="utf-8")
        for relative in ("repository/docs/generated", "repository/tests/generated", "."):
            self.config["directories"]["work"] = relative
            self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
            error = io.StringIO()
            before = sorted(str(path) for path in self.directory.rglob("*"))
            with self.subTest(location=relative), contextlib.redirect_stderr(error):
                self.assertEqual(main(["--check-config", str(self.config_path)], root=repository), 2)
            self.assertIn("source directory", error.getvalue())
            self.assertNotIn(str(self.directory), error.getvalue())
            self.assertEqual(before, sorted(str(path) for path in self.directory.rglob("*")))


class CommandTests(unittest.TestCase):
    def test_full_and_render_require_explicit_configs_before_loading_plan(self):
        for command in ("--full", "--render"):
            error = io.StringIO()
            with self.subTest(command=command), contextlib.redirect_stderr(error):
                self.assertEqual(main([command], root=Path("/nonexistent")), 2)
            self.assertIn(command + " requires", error.getvalue())
            self.assertNotIn("/nonexistent", error.getvalue())

    def test_invalid_config_error_does_not_disclose_path(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            status = main(["--check-config", "/private/identifying-location/missing.json"])
        self.assertEqual(status, 2)
        self.assertNotIn("identifying-location", error.getvalue())


if __name__ == "__main__":
    unittest.main()
