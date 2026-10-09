"""Synthetic adversarial M06 runner checks; no study records or model fitting."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
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
from scipy.stats import chi2, norm

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from dchs_lung_function import trajectory_execution as te
from dchs_lung_function import binary_execution, pipeline
from dchs_lung_function.models import spirometry_trajectories as model
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError

PRIVATE_SENTINEL = "FABRICATED_PRIVATE_RECORD_SENTINEL"


def synthetic_fit(contract):
    outcome, cutoff = contract["audit"]["outcome"], contract["audit"]["threshold"]
    spec = model.fit_spec(outcome, cutoff)
    counts = dict(n_observations=60, n_children=20, n_exposed_observations=3,
                  n_exposed_children=2, n_reference_observations=40, n_reference_children=15,
                  n_future_observations=7, n_future_children=2,
                  n_later_observations=10, n_later_children=1)
    fits = []
    for terms in (model.FULL_TERMS, model.REDUCED_TERMS):
        beta = pd.Series(0., index=terms); beta["Intercept"] = .5; beta["target"] = .1
        if len(terms) == 12: beta.iloc[9:] = [.01, .02, .03]
        covariance = pd.DataFrame(np.eye(len(terms)) * .04, index=terms, columns=terms)
        metadata = dict(accepted=True, status="accepted", failure_reasons=[], terms=list(terms),
                        n_observations=60, n_children=20, n_parameters=len(terms),
                        sigma2=1., tau2=.25, icc=.2, objective=100., condition_number=2.,
                        finite_sample_correction=20/19*59/(60-len(terms)),
                        optimizer={"grid_candidates":91, "positive_grid_points":90, "golden_steps":60,
                                   "grid_winner_index":46, "boundary":False})
        metadata["lambda"] = .25
        fits.append(model.RandomInterceptFit(beta, covariance, covariance.copy(), metadata))
    full, reduced = fits
    base = {k: spec[k] for k in ("outcome", "outcome_label", "unit")}
    base["tb_age_cutoff_years"] = cutoff
    interaction = float(full.coefficients.iloc[9:].to_numpy() @ full.coefficients.iloc[9:].to_numpy() / .04)
    overall = interaction + .1**2/.04
    global_row = {**base, **counts, "interaction_wald_chi2":interaction, "interaction_df":3,
                  "interaction_p":chi2.sf(interaction, 3), "overall_trajectory_wald_chi2":overall,
                  "overall_trajectory_df":4, "overall_trajectory_p":chi2.sf(overall, 4),
                  "parallel_trajectory_difference":.1, "parallel_trajectory_robust_se":.2,
                  "parallel_trajectory_ci_low":.1-1.96*.2, "parallel_trajectory_ci_high":.1+1.96*.2,
                  "parallel_trajectory_p":2*norm.sf(.5), "icc":.2, "condition_number":2.,
                  "random_intercept_variance":.25, "residual_variance":1.}
    _, design_info = model.spline_basis(np.linspace(5., 10., 60))
    height = dict(coefficients=[100., 5., 0., 0., 0.], mean=100., sd=10., male_mean=.4, age_center=7.5)
    standardization = {"height":height, "spline":model._spline_state(design_info)}

    def vector(age, exposed):
        basis = model.basis_at(design_info, [age])[0]
        return np.array([1., float(exposed), 0., 0., (age-7.5)/2., .4, *basis, *(basis*exposed)])

    def inference(v, name):
        estimate = float(v @ full.coefficients.to_numpy())
        se = float(np.sqrt(v @ full.covariance.to_numpy() @ v))
        return {name:estimate, "robust_se":se, "ci_low":estimate-1.96*se, "ci_high":estimate+1.96*se}

    predictions = [{**base, "age_years":float(age), "group":group,
                    **inference(vector(age, exposed), "predicted_mean")}
                   for age in np.linspace(5., 10., 101)
                   for group, exposed in [("No recorded TB", False), (f"First TB <{cutoff} years", True)]]
    contrasts = [{**base, "age_years":float(age),
                  **inference(vector(age, True)-vector(age, False), "difference_tb_minus_reference")}
                 for age in range(5, 11)]
    metadata = {**spec, "accepted":True, "status":"accepted", "failure_reasons":[],
                "counts":counts, "standardization":standardization}
    return model.TrajectoryFit(full, reduced, pd.DataFrame([global_row]), pd.DataFrame(predictions),
                               pd.DataFrame(contrasts), metadata)


def failed_fit(contract):
    result = synthetic_fit(contract)
    metadata = {**result.metadata, "accepted":False, "status":"fit_failed",
                "failure_reasons":["Synthetic full-fit numerical failure."]}
    fitted = result.full
    part = {**fitted.metadata, "accepted":False, "status":"fit_failed",
            "failure_reasons":["Synthetic full-fit numerical failure."]}
    for key in ("lambda", "sigma2", "tau2", "icc", "objective", "condition_number", "finite_sample_correction", "optimizer"):
        part[key] = None
    full = replace(fitted, coefficients=fitted.coefficients*np.nan, covariance=fitted.covariance*np.nan,
                   model_covariance=fitted.model_covariance*np.nan, metadata=part)
    tests = result.global_tests.copy()
    for key in tests:
        if key not in {"outcome", "outcome_label", "unit", "tb_age_cutoff_years", *metadata["counts"]}:
            tests[key] = np.nan
    predictions = result.predictions.copy(); contrasts = result.age_contrasts.copy()
    for frame, estimate in [(predictions, "predicted_mean"), (contrasts, "difference_tb_minus_reference")]:
        frame[[estimate, "robust_se", "ci_low", "ci_high"]] = np.nan
    return replace(result, full=full, global_tests=tests, predictions=predictions,
                   age_contrasts=contrasts, metadata=metadata)


class TrajectoryContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.keys = [f"trajectory/{outcome}/{cutoff}" for outcome, cutoff in model.DEFAULT_FIT_SELECTION]

    def prepare(self, *, changed=None, unrelated=1., omit=None):
        samples = {key:{"matrix":np.array([2. if key == changed else 1.])} for key in self.keys if key != omit}
        samples["unrelated"] = {"matrix":np.array([unrelated])}
        ArtifactWriter(self.directory).finish({"M03":{"pft":{"samples":samples}}})
        identity = {"synthetic":True}
        write_json(self.directory/"receipt.json", {"through":"M03", "status":"prepared", "identity":identity,
            "fingerprint":te._digest(identity), "artifact_manifest_sha256":file_hash(self.directory/"artifacts.json")})

    def test_four_selected_contracts_bound_independently_of_unrelated_models(self):
        self.prepare(); first = te.trajectory_contracts(self.directory)
        self.assertEqual(len(first["contracts"]), 4)
        self.prepare(unrelated=9.); self.assertEqual(te.trajectory_contracts(self.directory), first)
        self.prepare(changed=self.keys[-1]); second = te.trajectory_contracts(self.directory)
        self.assertNotEqual(first["contracts"][self.keys[-1]], second["contracts"][self.keys[-1]])
        self.assertEqual(first["contracts"][self.keys[0]], second["contracts"][self.keys[0]])

    def test_incomplete_or_corrupt_preparation_rejected(self):
        self.prepare(omit=self.keys[-1])
        with self.assertRaises(ValidationError): te.trajectory_contracts(self.directory)
        self.prepare(); (self.directory/"artifact_00001.npy").write_bytes(b"corrupt")
        with self.assertRaises(ValidationError): te.trajectory_contracts(self.directory)
        self.prepare(); receipt = json.loads((self.directory/"receipt.json").read_text())
        receipt["through"] = "M02"; write_json(self.directory/"receipt.json", receipt)
        with self.assertRaises(ValidationError): te.trajectory_contracts(self.directory)


class TrajectoryRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.prep = self.base/"source", self.base/"preparation"
        self.work, self.output = self.base/"private-work", self.base/"aggregate-output"
        self.prep.mkdir()
        for name in te.SOURCES:
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY/name, path)
        (self.root/"requirements-preparation.lock.txt").write_text("# synthetic runtime lock\n")
        self.descriptor = {"contracts":{f"trajectory/{o}/{t}":"a"*64 for o,t in model.DEFAULT_FIT_SELECTION},
                           "preparation_fingerprint":"b"*64}
        for patch in [mock.patch.object(te, "trajectory_contracts", side_effect=lambda *a: deepcopy(self.descriptor)),
                      mock.patch.object(te, "load_artifacts", side_effect=self.load_contract),
                      mock.patch.object(te, "runtime_identity", return_value={"python":"synthetic", "packages":{}})]:
            patch.start(); self.addCleanup(patch.stop)

    def load_contract(self, prep, keys):
        _, outcome, cutoff = keys[-1].split("/")
        return {"audit":{"family":"trajectory", "outcome":outcome, "threshold":int(cutoff)},
                "frame":pd.DataFrame({"pid_child":[PRIVATE_SENTINEL]})}

    def run_fit(self, *, resume=False, compute=None, root=None, work=None, output=None):
        with mock.patch.object(model, "fit_trajectory", side_effect=compute or synthetic_fit):
            return te.fit_prepared_trajectories(root or self.root, self.prep, work or self.work,
                                                output or self.output, resume=resume)

    def destination(self, result):
        return self.output/result["fingerprint"]

    def assert_clean(self):
        self.assertEqual(list(self.base.rglob(".staging-*")), [])
        self.assertEqual(list(self.base.rglob(".preparation.lock")), [])

    def edit_json(self, directory, name, change):
        path = directory/name; value = json.loads(path.read_text()); change(value); write_json(path,value)
        receipt = json.loads((directory/"receipt.json").read_text()); receipt["files"][name] = file_hash(path)
        if name == "summary.json": receipt["summary"] = value
        write_json(directory/"receipt.json", receipt)

    def edit_csv(self, directory, name, change):
        path = directory/name; frame = pd.read_csv(path, float_precision="round_trip")
        frame = change(frame); frame.to_csv(path, index=False, float_format="%.17g")
        receipt = json.loads((directory/"receipt.json").read_text()); receipt["files"][name] = file_hash(path)
        write_json(directory/"receipt.json", receipt)

    def reject_csv_mutations(self, name, mutations):
        for field, value in mutations:
            directory = self.destination(self.run_fit())
            def change(frame):
                frame[field] = frame[field].astype(object)
                frame.loc[0, field] = value
                return frame
            self.edit_csv(directory, name, change)
            with self.subTest(file=name, field=field, value=value), self.assertRaises(ValidationError):
                self.run_fit(resume=True)
            shutil.rmtree(directory)

    def test_complete_outputs_exact_small_counts_grid_and_resume(self):
        result = self.run_fit(); directory = self.destination(result)
        self.assertEqual((result["analyses"],result["fitted_models"],result["prediction_rows"],result["age_contrast_rows"]),(4,8,808,24))
        self.assertTrue(result["accepted"])
        self.assertEqual({p.name for p in directory.iterdir()}, {*te.RESULT_FILES,"receipt.json"})
        self.assertEqual(len(pd.read_csv(directory/"coefficients.csv")),84)
        self.assertEqual(len(pd.read_csv(directory/"covariance.csv")),1800)
        tests = pd.read_csv(directory/"global_tests.csv")
        self.assertEqual(set(tests.n_exposed_observations),{3}); self.assertEqual(set(tests.n_exposed_children),{2})
        saved = pd.read_csv(directory/"predictions.csv",float_precision="round_trip")
        np.testing.assert_array_equal(saved.age_years[:202].to_numpy()[::2],np.linspace(5.,10.,101))
        resumed = self.run_fit(resume=True,compute=lambda *a:self.fail("cached fit recomputed"))
        self.assertTrue(resumed["resumed"]); self.assertEqual(result["fingerprint"],resumed["fingerprint"])
        for path in directory.iterdir():
            self.assertNotIn(PRIVATE_SENTINEL,path.read_text()); self.assertNotIn('"pid_child":',path.read_text())
        self.assert_clean()

    def test_failed_fit_retains_complete_null_inference_without_acceptance(self):
        result = self.run_fit(compute=failed_fit); self.assertFalse(result["accepted"])
        self.assertEqual(result["status"],"fit_failed")
        directory = self.destination(result)
        self.assertTrue(pd.read_csv(directory/"predictions.csv").predicted_mean.isna().all())
        self.assertFalse(self.run_fit(resume=True,compute=lambda *a:self.fail("failed evidence refitted"))["accepted"])
        self.assert_clean()

    def test_failed_model_cannot_retain_reportable_inference(self):
        def contradictory(contract):
            failed = failed_fit(contract)
            return replace(failed,predictions=synthetic_fit(contract).predictions)
        with self.assertRaises(ValidationError): self.run_fit(compute=contradictory)
        self.assertEqual(list(self.output.rglob("receipt.json")),[])

    def test_identical_run_requires_explicit_resume(self):
        self.run_fit()
        with self.assertRaises(ValidationError): self.run_fit()

    def test_source_contract_lock_runtime_changes_invalidate_identity(self):
        seen = {self.run_fit()["fingerprint"]}
        for change in [lambda:(self.root/"run.py").write_text("# changed entrypoint\n"),
                       lambda:self.descriptor["contracts"].update({"trajectory/fev1_fvc/1":"c"*64}),
                       lambda:(self.root/"requirements-preparation.lock.txt").write_text("# changed lock\n")]:
            change(); result = self.run_fit(resume=True)
            self.assertFalse(result["resumed"]); self.assertNotIn(result["fingerprint"],seen); seen.add(result["fingerprint"])
        with mock.patch.object(te,"runtime_identity",return_value={"python":"changed"}):
            self.assertNotIn(self.run_fit(resume=True)["fingerprint"],seen)

    def test_identical_copied_source_reuses_accepted_artifacts(self):
        first = self.run_fit(); copied = self.base/"copy"; shutil.copytree(self.root,copied)
        second = self.run_fit(root=copied,resume=True,compute=lambda *a:self.fail("copy refitted"))
        self.assertTrue(second["resumed"]); self.assertEqual(first["fingerprint"],second["fingerprint"])

    def test_changed_imported_model_source_is_rejected(self):
        (self.root/te.SOURCES[0]).write_text("# mismatched imported model\n")
        with self.assertRaises(ValidationError): self.run_fit(compute=lambda *a:self.fail("wrong implementation fitted"))

    def test_each_missing_or_tampered_output_is_rejected(self):
        directory = self.destination(self.run_fit())
        for name in te.RESULT_FILES:
            path = directory/name; original = path.read_bytes(); path.write_bytes(original+b"tampered")
            with self.subTest(file=name,kind="changed"),self.assertRaises(ValidationError): self.run_fit(resume=True)
            path.unlink()
            with self.subTest(file=name,kind="missing"),self.assertRaises(ValidationError): self.run_fit(resume=True)
            path.write_bytes(original)

    def test_symlinks_receipt_identity_status_and_inventory_are_verified(self):
        directory = self.destination(self.run_fit()); path = directory/"predictions.csv"; other = self.base/"other.csv"
        path.rename(other); path.symlink_to(other)
        with self.assertRaises(ValidationError): self.run_fit(resume=True)
        path.unlink(); other.rename(path)
        receipt_path = directory/"receipt.json"; original = json.loads(receipt_path.read_text())
        for field,value in [("fingerprint","wrong"),("identity",{}),("files",{}),("status","fit_failed")]:
            changed=deepcopy(original); changed[field]=value; write_json(receipt_path,changed)
            with self.subTest(field=field),self.assertRaises(ValidationError):self.run_fit(resume=True)
            write_json(receipt_path,original)

    def test_summary_cannot_contradict_diagnostics(self):
        directory=self.destination(self.run_fit())
        self.edit_json(directory,"summary.json",lambda x:x.update(accepted=False,status="fit_failed"))
        with self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_malformed_missing_duplicate_or_mislabelled_diagnostics_rejected(self):
        changes=[lambda x:x["models"].pop(),lambda x:x["models"].__setitem__(1,deepcopy(x["models"][0])),
                 lambda x:x["models"].__setitem__(0,None),lambda x:x["models"][0].update(threshold=99),
                 lambda x:x["models"][0].update(metadata=None),lambda x:x["models"][0].update(fits=[])]
        for change in changes:
            directory=self.destination(self.run_fit());self.edit_json(directory,"diagnostics.json",change)
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(resume=True)
            shutil.rmtree(directory)

    def test_accepted_status_and_component_metadata_must_be_consistent(self):
        changes=[lambda m:m["metadata"].update(failure_reasons=["failure"]),
                 lambda m:m["fits"]["full"].update(status="fit_failed"),
                 lambda m:m["fits"]["reduced"].update(accepted=False,status="fit_failed",failure_reasons=["failure"]),
                 lambda m:m["fits"]["full"].update(terms=[]),
                 lambda m:m["fits"]["full"].update(n_parameters=11),
                 lambda m:m["metadata"]["counts"].update(n_exposed_observations=4)]
        for change in changes:
            directory=self.destination(self.run_fit());self.edit_json(directory,"diagnostics.json",lambda x:change(x["models"][0]))
            with self.subTest(change=change),self.assertRaises(ValidationError):self.run_fit(resume=True)
            shutil.rmtree(directory)

    def test_variance_parameter_count_and_correction_metadata_checked(self):
        for field,value in [("sigma2",0),("tau2",-.1),("lambda",.5),("icc",.4),("condition_number",.5),
                            ("finite_sample_correction",1),("n_observations",59),("n_children",2.5),("objective",None)]:
            directory=self.destination(self.run_fit())
            self.edit_json(directory,"diagnostics.json",lambda x:x["models"][0]["fits"]["full"].update({field:value}))
            with self.subTest(field=field),self.assertRaises(ValidationError):self.run_fit(resume=True)
            shutil.rmtree(directory)

    def test_global_statistics_samples_df_and_intervals_checked(self):
        self.reject_csv_mutations("global_tests.csv",[("n_observations",59.5),("n_children",61),
            ("n_exposed_observations",-1),("n_exposed_children",4),("n_reference_children",19),
            ("interaction_wald_chi2",-.1),("overall_trajectory_wald_chi2",9),("interaction_df",2),
            ("overall_trajectory_df",3),("interaction_p",.5),("parallel_trajectory_robust_se",.3),
            ("parallel_trajectory_difference",.2),("parallel_trajectory_ci_low",-99),
            ("parallel_trajectory_p",.5),("icc",.1),("residual_variance",2),
            ("random_intercept_variance",0),("condition_number",5)])

    def test_semantic_outcome_and_unit_labels_checked(self):
        for name in ["global_tests.csv","predictions.csv","age_contrasts.csv"]:
            self.reject_csv_mutations(name,[("outcome","fev1_fvc_gli_z"),("unit","reference SD"),
                                           ("outcome_label","wrong label"),("tb_age_cutoff_years",5)])

    def test_missing_or_reordered_rows_rejected_across_all_tables(self):
        for name in ["global_tests.csv","predictions.csv","age_contrasts.csv","coefficients.csv","covariance.csv"]:
            directory=self.destination(self.run_fit());self.edit_csv(directory,name,lambda x:x.iloc[1:])
            with self.subTest(file=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            shutil.rmtree(directory)
        self.reject_csv_mutations("predictions.csv",[("age_years",5.01),("group","unknown")])

    def test_accepted_coefficients_covariance_and_intervals_must_be_valid(self):
        self.reject_csv_mutations("coefficients.csv",[("coefficient",np.nan)])
        self.reject_csv_mutations("covariance.csv",[("covariance",np.nan),("covariance",-.04)])
        for name in ["predictions.csv","age_contrasts.csv"]:
            self.reject_csv_mutations(name,[("robust_se",-.1),("ci_low",99),("ci_high",np.nan)])

    def test_standardization_and_prediction_values_match_full_fitted_model(self):
        directory=self.destination(self.run_fit())
        self.edit_json(directory,"diagnostics.json",lambda x:x["models"][0]["metadata"]["standardization"]["height"].update(sd=-1))
        with self.assertRaises(ValidationError):self.run_fit(resume=True)
        shutil.rmtree(directory)
        for name,estimate in [("predictions.csv","predicted_mean"),("age_contrasts.csv","difference_tb_minus_reference")]:
            directory=self.destination(self.run_fit())
            def changed(frame):
                for column in [estimate,"ci_low","ci_high"]:frame.loc[0,column]+=10
                return frame
            self.edit_csv(directory,name,changed)
            with self.subTest(file=name),self.assertRaises(ValidationError):self.run_fit(resume=True)
            shutil.rmtree(directory)

    def test_contract_outcome_or_cutoff_mismatch_fails_before_fitting(self):
        original=self.load_contract
        for field,value in [("outcome","wrong"),("threshold",99)]:
            def changed(*args):
                c=original(*args);c["audit"][field]=value;return c
            with mock.patch.object(te,"load_artifacts",side_effect=changed):
                with self.subTest(field=field),self.assertRaises(ValidationError):
                    self.run_fit(compute=lambda *a:self.fail("wrong contract fitted"))
        self.assert_clean()

    def test_source_and_contract_changes_during_fit_or_resume_block_acceptance(self):
        for source in [False,True]:
            def changed(contract):
                if source:(self.root/"run.py").write_text("# changed while fitting\n")
                else:self.descriptor["contracts"]["trajectory/fev1_fvc/1"]="d"*64
                return synthetic_fit(contract)
            with self.subTest(source=source),self.assertRaises(ValidationError):self.run_fit(compute=changed)
            self.assertEqual(list(self.output.rglob("receipt.json")),[]);self.assert_clean()
        original=te._verify_result
        for source in [False,True]:
            self.run_fit()
            def changed(*args):
                result=original(*args)
                if source:(self.root/"run.py").write_text("# changed while resuming\n")
                else:self.descriptor["contracts"]["trajectory/fev1_fvc/1"]="e"*64
                return result
            with mock.patch.object(te,"_verify_result",side_effect=changed):
                with self.subTest(source=source),self.assertRaises(ValidationError):self.run_fit(resume=True)

    def test_computation_allocation_publication_failures_clean_staging_and_locks(self):
        with self.assertRaises(RuntimeError):self.run_fit(compute=lambda *a:(_ for _ in ()).throw(RuntimeError("synthetic failure")))
        self.assert_clean()
        with mock.patch.object(te.tempfile,"mkdtemp",side_effect=OSError("synthetic allocation failure")):
            with self.assertRaises(OSError):self.run_fit()
        self.assert_clean()
        with mock.patch.object(te.os,"replace",side_effect=OSError("synthetic publication failure")):
            with self.assertRaises(OSError):self.run_fit()
        self.assert_clean();self.assertEqual(list(self.output.rglob("receipt.json")),[])

    def test_source_preparation_nested_and_shared_roots_rejected(self):
        for work,output in [(self.root/"work",self.output),(self.prep/"work",self.output),
                            (self.work,self.work),(self.work,self.work/"nested"),(self.work,self.base)]:
            with self.subTest(work=work,output=output),self.assertRaises(ValidationError):self.run_fit(work=work,output=output)

    def test_resume_and_different_work_roots_obey_output_lock(self):
        self.run_fit()
        with te.preparation_locks(self.work,self.output):
            with self.assertRaises(ValidationError):self.run_fit(resume=True)
            with self.assertRaises(ValidationError):self.run_fit(work=self.base/"other-work",resume=True)
        self.assert_clean()

    def test_concurrent_writer_cannot_bypass_output_lock(self):
        entered,released=threading.Event(),threading.Event();failures=[]
        def blocked(contract):
            entered.set()
            if not released.wait(10):raise RuntimeError("synthetic synchronization timeout")
            return synthetic_fit(contract)
        def worker():
            try:self.run_fit(compute=blocked)
            except BaseException as error:failures.append(error)
        thread=threading.Thread(target=worker);thread.start()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertRaises(ValidationError):self.run_fit(work=self.base/"other-work",resume=True)
        finally:released.set();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertEqual(failures,[]);self.assert_clean()


class TrajectoryCumulativeAndCliTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name);self.work=self.base/"work";self.work.mkdir()
        self.config=self.base/"paths.local.json";self.analysis=self.base/"analysis.local.json";self.sampling=self.base/"sampling.local.json"
        write_json(self.config,{"inputs":{},"directories":{"work":str(self.work),"outputs":str(self.base/"outputs")}})
        write_json(self.analysis,{"synthetic":True});write_json(self.sampling,{"synthetic":True})
        self.binary={"accepted":True,"models":28,"primary_models_verified":12,"preparation_fingerprint":"a"*64}
        self.trajectory={"through":"M06","analyses":4,"fitted_models":8,"prediction_rows":808,
                         "age_contrast_rows":24,"accepted":True,"status":"accepted","fingerprint":"b"*64}

    def cumulative(self,*,binary=None,fit=None,preparation=None):
        with mock.patch.object(binary_execution,"run_binary",side_effect=binary or (lambda *a,**k:deepcopy(self.binary))):
            with mock.patch.object(te,"fit_prepared_trajectories",side_effect=fit or (lambda *a,**k:deepcopy(self.trajectory))):
                with mock.patch.object(te,"run_preparation",side_effect=preparation or (lambda *a,**k:{"fingerprint":"a"*64})):
                    return te.run_trajectories(self.base/"source",self.config,self.analysis,self.sampling,resume=True)

    def test_all_twelve_primary_and_twenty_eight_binary_models_required(self):
        for field,value in [("accepted",False),("models",27),("primary_models_verified",11)]:
            original=self.binary[field];self.binary[field]=value
            with self.subTest(field=field),self.assertRaises(ValidationError):
                self.cumulative(fit=lambda *a,**k:self.fail("invalid prerequisite fitted"))
            self.binary[field]=original
        self.assertFalse((self.work/"trajectory_current.json").exists())

    def test_success_publishes_pointer_and_failed_fit_does_not(self):
        result=self.cumulative();self.assertEqual(result["primary_models_verified"],12);self.assertEqual(result["binary_models_verified"],28)
        pointer=self.work/"trajectory_current.json";self.assertEqual(json.loads(pointer.read_text())["fingerprint"],"b"*64)
        pointer.unlink();self.trajectory.update(accepted=False,status="fit_failed")
        self.assertFalse(self.cumulative()["accepted"]);self.assertFalse(pointer.exists())

    def test_changed_sampling_and_m03_prevent_cumulative_completion(self):
        def changed(*args,**kwargs):
            self.sampling.write_text(self.sampling.read_text()+"\n");return deepcopy(self.trajectory)
        with self.assertRaises(ValidationError):self.cumulative(fit=changed)
        with self.assertRaises(ValidationError):self.cumulative(preparation=lambda *a,**k:{"fingerprint":"changed"})
        self.assertFalse((self.work/"trajectory_current.json").exists())

    def cli(self,*,extra=None,result=None,error=None):
        stdout,stderr=StringIO(),StringIO()
        args=["--through","M06","--config",str(self.config),"--analysis",str(self.analysis),"--resume"]
        with mock.patch.object(te,"run_trajectories",return_value=result or self.trajectory,side_effect=error) as runner:
            with redirect_stdout(stdout),redirect_stderr(stderr):
                code=pipeline.main(args+(extra or []),root=self.base/"source")
        return code,stdout.getvalue(),stderr.getvalue(),runner

    def test_cli_exact_counts_status_and_forwarded_options(self):
        code,output,errors,runner=self.cli(extra=["--sampling",str(self.sampling),"--jobs","3"])
        self.assertEqual(code,0);self.assertIn("4 analyses",output);self.assertIn("8 fitted models",output);self.assertIn("808 prediction rows",output)
        self.assertEqual(errors,"");self.assertEqual(runner.call_args.kwargs,{"jobs":3,"resume":True})
        self.assertEqual(runner.call_args.args[3],str(self.sampling))
        code,output,_,_=self.cli(result={**self.trajectory,"accepted":False,"status":"fit_failed"})
        self.assertEqual(code,2);self.assertIn("fit_failed",output)

    def test_cli_partial_targets_rejected_and_errors_sanitized(self):
        code,_,errors,runner=self.cli(extra=["--targets","ratio_pre"])
        self.assertEqual(code,2);self.assertIn("all primary",errors);runner.assert_not_called()
        for error in [ValueError(PRIVATE_SENTINEL),RuntimeError(PRIVATE_SENTINEL),OSError(PRIVATE_SENTINEL)]:
            code,output,errors,_=self.cli(error=error)
            self.assertEqual(code,2);self.assertNotIn(PRIVATE_SENTINEL,output+errors)

    def test_full_and_render_missing_configs_fail_without_invoking_models(self):
        for option in ["--full","--render"]:
            output=StringIO()
            with mock.patch.object(te,"run_trajectories") as runner,redirect_stderr(output):
                self.assertEqual(pipeline.main([option],root=self.base/"source"),2)
            self.assertIn(option+" requires",output.getvalue());runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
