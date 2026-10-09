"""Numerical summaries, coding and empirical availability boundary checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function.cohorts import CohortError
from dchs_lung_function.descriptive import build_modality_schedule, build_table1


def sample_children():
    return pd.DataFrame({
        "pid_child": ["s1", "s2", "s3", "s4"], "tb_age_first_years": [np.nan, .5, 2., 6.],
        "sex": [0., 1., np.nan, 0.], "site": [0., 1., 1., 0.], "ethnicity": [0., 1., 1., 0.],
        "gestation_delivery": [38., 40., 36., np.nan], "preterm": [0., 0., 1., 0.],
        "birth_weight": [3200., 3000., 2200., np.nan], "lowbirthweight": [0., 0., 1., 0.],
        "hiv_exposed": [0., 1., 0., 0.], "hiv_infected": [0., 0., 0., 0.],
        "maternal_age_birth": [20., 30., 25., 25.], "education": [0., 1., 2., np.nan],
        "Prenatal_smoking_composite": [1., 0., 0., 0.], "Prenatal_alcohol_composite": [np.nan, 1., 0., 0.],
        "parity": [0., 1., 3., 6.], "crowd": [1., 2., 3., 1.], "household_income": [0., 1., 2., 0.],
        "smoke_hh": [0., 1., 1., 0.],
    })


class DescriptiveTests(unittest.TestCase):
    def test_small_counts_and_explicit_inherited_recode_policy(self):
        result = build_table1(sample_children())
        table = result.table.set_index("characteristic")
        self.assertEqual(table.loc["Male sex", "overall"], "2/4 (50.0)")
        self.assertEqual(result.audit["inherited_missing_to_reference_recodes"]["male"], 1)
        self.assertEqual(table.loc["Alcohol use during pregnancy", "overall"], "1/3 (33.3)")
        source = result.missingness.query("variable == 'male' and group == 'overall'").iloc[0]
        self.assertEqual(source.n_source_missing, 1)
        self.assertEqual(source.n_missing_recoded_to_reference, 1)
        self.assertEqual(table.loc["HIV exposure at birth", "tb_lt5"], "1/2 (50.0)")
        self.assertEqual(table.loc["Child HIV infection", "tb_lt1"], "0/1 (0.0)")
        self.assertEqual(table.loc["No education", "overall"], "1/3 (33.3)")
        self.assertEqual(result.audit["group_counts"], {"overall": 4, "no_tb": 2, "tb_lt5": 2, "tb_lt1": 1})

    def test_missing_ethnicity_follows_selected_field_rule_and_remains_audited(self):
        children = sample_children()
        children.loc[3, "ethnicity"] = np.nan
        result = build_table1(children)
        table = result.table.set_index("characteristic")
        self.assertEqual(table.loc["Black African ethnicity", "overall"], "2/4 (50.0)")
        self.assertEqual(table.loc["Black African ethnicity", "no_tb"], "0/2 (0.0)")
        row = result.missingness.query("variable == 'black_african' and group == 'overall'").iloc[0]
        self.assertEqual(row.n_source_missing, 1)
        self.assertEqual(row.n_missing_recoded_to_reference, 1)
        self.assertTrue(pd.isna(children.loc[3, "ethnicity"]))

    def test_parity_uses_linear_quantiles_and_not_mean_sd(self):
        result = build_table1(sample_children())
        row = result.table.set_index("characteristic").loc["Parity, median (IQR)"]
        self.assertEqual(row["overall"], "2 [0.75–3.75]")
        self.assertEqual(row["tb_lt5"], "2 [1.5–2.5]")
        stats = result.statistics.query("variable == 'parity' and group == 'overall'").iloc[0]
        self.assertEqual(stats.n_nonmissing, 4)
        self.assertEqual(stats.q3, 3.75)
        self.assertTrue(pd.isna(stats["mean"]))

    def test_mean_sd_uses_sample_sd_and_exact_nonmissing_count(self):
        result = build_table1(sample_children())
        row = result.statistics.query("variable == 'gestational_age_weeks' and group == 'overall'").iloc[0]
        self.assertEqual(row["mean"], 38.)
        self.assertEqual(row.sd, 2.)
        self.assertEqual(row.n_nonmissing, 3)
        self.assertEqual(row.n_missing, 1)

    def test_missing_parity_and_empty_groups_are_explicit(self):
        children = sample_children()
        children["parity"] = np.nan
        children["tb_age_first_years"] = np.nan
        result = build_table1(children)
        self.assertEqual(result.table.set_index("characteristic").loc["Parity, median (IQR)", "overall"], "—")
        stats = result.statistics.query("variable == 'parity' and group == 'overall'").iloc[0]
        self.assertEqual(stats.n_missing, 4)
        self.assertEqual(stats.n_nonmissing, 0)

    def test_invalid_category_and_baseline_sex_codes_rejected(self):
        for field, invalid in [("sex", 2), ("education", 99), ("crowd", 0), ("parity", -1)]:
            children = sample_children()
            children.loc[0, field] = invalid
            with self.subTest(field=field), self.assertRaises(CohortError):
                build_table1(children)

    def test_schedule_requires_explicit_numeric_visit_mapping(self):
        data = pd.DataFrame({"pid_child": ["x", "y", "z"], "visit": [4, 4, 9],
                             "value": [1., 2., np.nan], "date": pd.to_datetime(["2020-01-01", None, "2021-01-01"])})
        specs = {"example": {"columns": ["value"], "date": "date"}}
        with self.assertRaisesRegex(CohortError, "explicit source label"):
            build_modality_schedule(data, modalities=specs)
        result = build_modality_schedule(data, modalities=specs, visit_labels={4: "one", 9: "two", 12: "three"}).set_index("visit")
        self.assertEqual(result.loc["one", "dated_measurements"], 1)
        self.assertEqual(result.loc["one", "missing_date_with_value"], 1)
        self.assertTrue(result.loc["one", "represented"])
        self.assertFalse(result.loc["two", "represented"])
        self.assertEqual(result.loc["three", "source_records"], 0)

    def test_schedule_uses_any_selected_measure_but_requires_date(self):
        data = pd.DataFrame({"pid_child": ["x", "x", "y"], "visit_label": ["one"] * 3,
                             "r": [np.nan, 1., np.nan], "x": [-1., np.nan, 2.],
                             "date": pd.to_datetime(["2020-01-01", "2020-01-01", None])})
        result = build_modality_schedule(data, modalities={"example": {"columns": ["r", "x"], "date": "date"}}).iloc[0]
        self.assertEqual(result.dated_measurements, 2)
        self.assertEqual(result.children_with_measurement, 1)
        self.assertEqual(result.records_with_numeric_value, 3)


if __name__ == "__main__":
    unittest.main()
