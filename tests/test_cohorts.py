"""Population-boundary tests use synthetic data only."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function.cohorts import (
    BASELINE_DESCRIPTIVE_FIELDS, DESCRIPTIVE_OUTCOMES, PRIMARY_OUTCOMES,
    CohortError, build_cohorts, build_flow_summary,
)


def synthetic_cohort_inputs():
    dob = pd.Timestamp("2010-01-01")
    rows = []
    for child, dx_days, test_days in [("early", 180, 190), ("later", 1827, 1827),
                                       ("missing_height", None, 2000), ("lci_only", None, 2000)]:
        row = dict(pid_child=child, child_dob=dob, tb_first_dx=pd.NaT if dx_days is None else dob + pd.Timedelta(days=dx_days),
                   tb_age_first_years=np.nan if dx_days is None else dx_days / 365.25,
                   sex=0, site=1, ethnicity=1, gestation_delivery=39., preterm=0.,
                   birth_weight=3000., hiv_exposed=0., hiv_infected=0., height=100.,
                   date_of_test=dob + pd.Timedelta(days=test_days),
                   date_of_test_ilf1=dob + pd.Timedelta(days=test_days),
                   date_of_test_ilf2=dob + pd.Timedelta(days=test_days))
        for column, _ in list(DESCRIPTIVE_OUTCOMES.values()) + list(PRIMARY_OUTCOMES.values()):
            row[column] = np.nan
        row["i_lci_2_5"] = 7.
        if child != "lci_only":
            row["fev1_fvc"] = .8
            row["fev1_fvc_gli_z"] = -.5
        if child == "missing_height":
            row["height"] = np.nan
        rows.append(row)
    pft = pd.DataFrame(rows)
    baseline = pd.DataFrame({"pid_child": [*pft.pid_child, "baseline_only"], "etq1": pd.NaT})
    for field in BASELINE_DESCRIPTIVE_FIELDS:
        baseline[field] = 0.
    baseline["pid_maternal"] = ["m1", "m2", "m3", "m4", "m4"]
    return pft, baseline


class CohortTests(unittest.TestCase):
    def test_descriptive_union_and_primary_later_tb_are_distinct(self):
        p, b = synthetic_cohort_inputs()
        result = build_cohorts(p, b)
        self.assertEqual(set(result.descriptive_children.pid_child), {"early"})
        self.assertEqual(set(result.joint_children.pid_child), set(p.pid_child))
        primary = result.outcome_membership["primary:ratio_pre"]
        self.assertEqual(set(primary.pid_child), {"early", "later"})
        self.assertTrue(primary.loc[primary.pid_child.eq("later"), "post_tb"].iloc[0])
        self.assertEqual(result.audit["baseline_children_without_pft_records"], 1)
        # LCI-only children remain in the linked joint population.
        self.assertIn("lci_only", set(result.outcome_membership["primary:lci"].pid_child))
        self.assertEqual(result.ledger.loc[result.ledger.step.eq("nonmissing_height"), "records_excluded"].max(), 1)

    def test_measurement_before_diagnosis_stays_in_descriptive_union(self):
        p, b = synthetic_cohort_inputs()
        p.loc[p.pid_child.eq("later"), "date_of_test"] -= pd.Timedelta(days=1)
        result = build_cohorts(p, b)
        self.assertEqual(set(result.descriptive_children.pid_child), {"early", "later"})
        self.assertEqual(result.outcome_membership["descriptive:ratio_pre_z"].exposure_group.tolist(), ["infancy", "pre_or_no_tb"])

    def test_exit_boundary_and_administrative_restriction(self):
        p, b = synthetic_cohort_inputs()
        b.loc[b.pid_child.eq("early"), "etq1"] = p.loc[0, "date_of_test"]
        # On the exit date remains eligible; the following day is excluded.
        same_day = build_cohorts(p, b)
        self.assertIn("early", same_day.outcome_membership["primary:ratio_pre"].pid_child.tolist())
        p.loc[0, "date_of_test"] += pd.Timedelta(days=1)
        after = build_cohorts(p, b)
        self.assertNotIn("early", after.outcome_membership["primary:ratio_pre"].pid_child.tolist())
        # Cutoff only restricts the designated population, not primary.
        p, b = synthetic_cohort_inputs()
        result = build_cohorts(p, b, ascertainment_end=p.loc[1, "date_of_test"])
        self.assertNotIn("lci_only", result.outcome_membership["restricted:lci"].pid_child.tolist())
        self.assertIn("lci_only", result.outcome_membership["primary:lci"].pid_child.tolist())

    def test_unsupported_antenatal_counts_not_invented(self):
        p, b = synthetic_cohort_inputs()
        flow = build_flow_summary(build_cohorts(p, b), b).set_index("node")
        self.assertTrue(pd.isna(flow.loc["antenatal_enrollment", "count"]))
        self.assertEqual(flow.loc["antenatal_enrollment", "status"], "not_reconstructable_from_supplied_inputs")
        self.assertEqual(flow.loc["women_with_liveborn_children", "count"], 4)
        self.assertEqual(flow.loc["live_birth_sets_of_2", "count"], 1)

    def test_rejects_duplicate_baseline_without_disclosing_ids(self):
        p, b = synthetic_cohort_inputs()
        with self.assertRaises(CohortError) as caught:
            build_cohorts(p, pd.concat([b, b.iloc[[0]]]))
        self.assertNotIn("early", str(caught.exception))

    def test_child_attributes_cannot_silently_conflict(self):
        p, b = synthetic_cohort_inputs()
        duplicate = p.iloc[[0]].copy()
        duplicate["birth_weight"] = 3300.
        with self.assertRaisesRegex(CohortError, "inconsistent"):
            build_cohorts(pd.concat([p, duplicate], ignore_index=True), b)

    def test_impossible_dates_and_unknown_sex_fail(self):
        for edit in ("date", "sex", "age"):
            p, b = synthetic_cohort_inputs()
            if edit == "date":
                p.loc[0, "date_of_test"] = p.loc[0, "child_dob"] - pd.Timedelta(days=1)
            elif edit == "sex":
                p.loc[0, "sex"] = 2
            else:
                p.loc[0, "tb_age_first_years"] = 3.
            with self.subTest(edit=edit), self.assertRaises(CohortError):
                build_cohorts(p, b)

    def test_repeated_source_records_are_retained_not_deduplicated(self):
        p, b = synthetic_cohort_inputs()
        result = build_cohorts(pd.concat([p, p.iloc[[0]]], ignore_index=True), b)
        frame = result.outcome_membership["primary:ratio_pre"]
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.pid_child.nunique(), 2)
        self.assertTrue(frame.source_row.is_unique)


if __name__ == "__main__":
    unittest.main()
