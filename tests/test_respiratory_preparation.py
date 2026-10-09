"""Fabricated respiratory records test timing, episode and FeNO contracts."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function import respiratory_preparation as prep


def feno_fixture():
    """Return wholly synthetic repeated tests and child-level follow-up."""
    visits = ["6-10 weeks", *prep.FENO_VISITS]
    nominal_days = [45, 355, 730, 2922, 3287, 3652]
    children = []
    records = []
    for j in range(12):
        child = f"synthetic_{j}"
        dob = pd.Timestamp("2010-01-01") + pd.Timedelta(days=3 * j)
        diagnosis_days = {0: 180, 1: 330, 2: 550, 3: 4018}.get(j)
        tb_date = dob + pd.Timedelta(days=diagnosis_days) if diagnosis_days else pd.NaT
        children.append({"pid_child": child, "dob": dob,
                         "censor_date": pd.Timestamp("2025-01-01"),
                         "tb_date": tb_date,
                         "tb_age": diagnosis_days / prep.YEAR_DAYS if diagnosis_days else np.nan,
                         "male": float(j % 2)})
        for v, (visit, days) in enumerate(zip(visits, nominal_days)):
            records.append({"pid_child": child, "child_dob": dob,
                            "date_of_test_ilf1": dob + pd.Timedelta(days=days + j * j),
                            "feno_result": float(10 + j + v),
                            "height": 70.0 + v * 9 + j + (j % 3) * 2 + j * j * .2,
                            "tb_first_dx": tb_date, "male": float(j % 2),
                            "visit_label": visit})
    return pd.DataFrame(records), pd.DataFrame(children)


class RespiratoryPreparationTests(unittest.TestCase):
    def test_episode_chain_uses_consecutive_contacts_and_strict_gap(self):
        origin = pd.Timestamp("2020-01-01")
        contacts = pd.DataFrame({
            "pid_child": ["synthetic_a"] * 5,
            "date": [origin + pd.Timedelta(days=d) for d in [0, 20, 40, 70, 101]],
            "age": [1., 1.1, 1.2, 1.3, 1.4],
            "episode_source": ["dedicated wheeze visit"] * 5,
        })
        episodes, audit = prep.deduplicate_episodes(contacts, 30)
        self.assertEqual(episodes.date.tolist(), [origin, origin + pd.Timedelta(days=101)])
        self.assertEqual(audit["contacts_combined"], 3)
        self.assertEqual(audit["deduplicated_episodes"], 2)

    def test_effect_bands_include_each_left_boundary(self):
        ages = np.array([0., .999999, 1., 2.999999, 3., 4.999999, 5., 10.4999])
        np.testing.assert_array_equal(prep.effect_band(ages), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_scheduled_same_day_tb_is_post_and_censor_is_exclusive(self):
        diagnosis = pd.Timestamp("2020-01-01")
        children = pd.DataFrame({"pid_child": ["synthetic_a"], "tb_event": [1],
                                 "tb_date": [diagnosis], "tb_age": [1.],
                                 "censor_date": [diagnosis + pd.Timedelta(days=1)]})
        scheduled = pd.DataFrame({"pid_child": ["synthetic_a"] * 3,
                                  "date": [diagnosis + pd.Timedelta(days=d) for d in [-1, 0, 1]],
                                  "age": [.99, 1., 1.01], "wheeze": [1, 1, 1]})
        sources = prep.Sources(children, scheduled, pd.DataFrame(), {})
        retained, audit = prep.scheduled_for_scenario(sources, children, 0)
        self.assertEqual(retained.post_tb.tolist(), [0, 1])
        self.assertEqual(audit["records"], 2)
        self.assertEqual(audit["records_excluded_in_post_tb_lag"], 0)

    def test_episode_and_survival_arrays_conserve_events_and_time(self):
        diagnosis = pd.Timestamp("2020-01-01")
        children = pd.DataFrame({"pid_child": ["synthetic_a"], "censor_age": [2.],
                                 "tb_age": [1.], "tb_event": [1], "tb_date": [diagnosis]})
        episodes = pd.DataFrame({"pid_child": ["synthetic_a"] * 3,
                                 "date": [diagnosis + pd.Timedelta(days=d) for d in [-1, 0, 1]],
                                 "age": [.99, 1., 1.01]})
        intervals, retained, audit = prep.build_episode_intervals(children, episodes, 0)
        self.assertEqual(int(intervals.events.sum()), len(retained))
        self.assertEqual(int(intervals.loc[intervals.post_tb.eq(1), "events"].sum()), 2)
        self.assertAlmostEqual(intervals.person_years.sum(), 2.)
        arrays = prep.build_survival_arrays(children)
        self.assertEqual(int(arrays["event_interval"].sum()), 1)
        self.assertEqual(arrays["event_interval"][0, 3], 1)
        self.assertAlmostEqual(arrays["exposure"].sum(), 1.)
        self.assertAlmostEqual(arrays["full_exposure"].sum(), 2.)

    def test_supplement_no_exposure_is_explicitly_nonestimable(self):
        pft, _ = feno_fixture()
        pft["tb_first_dx"] = pd.NaT
        samples = prep.prepare_supplementary_feno(pft)["samples"]
        self.assertEqual(len(samples), 6)
        for item in samples.values():
            self.assertIsNone(item["target"])
            self.assertEqual(item["audit"]["status"], "no_exposure_contrast")
            self.assertEqual(item["audit"]["observations"], 12)
            self.assertEqual(item["audit"]["tb_before"], 0)
            self.assertEqual(item["audit"]["tb_after"], 0)
            self.assertNotIn("tb_before_measurement", item["design"].columns)
            self.assertNotIn("tb_after_measurement", item["design"].columns)

    def test_main_and_supplement_scale_at_different_complete_case_steps(self):
        pft, children = feno_fixture()
        missing = pft.pid_child.eq("synthetic_11")
        pft.loc[missing, "male"] = np.nan
        pft.loc[missing, "height"] = 300.
        children.loc[children.pid_child.eq("synthetic_11"), "male"] = np.nan
        main = prep.prepare_main_feno(pft, children)["samples"]["24 months"]
        supplement = prep.prepare_supplementary_feno(pft)["samples"]["24 months"]
        self.assertEqual(len(main["frame"]), 11)
        self.assertEqual(len(supplement["frame"]), 11)
        self.assertAlmostEqual(main["design"].height.mean(), 0.)
        self.assertAlmostEqual(main["design"].height.std(ddof=1), 1.)
        visit = pft.loc[pft.visit_label.eq("24 months")]
        expected = (visit.height - visit.height.mean()) / visit.height.std(ddof=1)
        expected_by_child = pd.Series(expected.to_numpy(), index=visit.pid_child)
        wanted = supplement["frame"].pid_child.map(expected_by_child).to_numpy()
        np.testing.assert_allclose(supplement["design"].height_z, wanted)
        self.assertGreater(abs(supplement["design"].height_z.mean()), .1)

    def test_nominal_twelve_months_retains_early_post_tb_attendance(self):
        pft, children = feno_fixture()
        nominal = pft.visit_label.eq("12 months")
        for child, days in [("synthetic_0", 200), ("synthetic_1", 250), ("synthetic_4", 300)]:
            mask = nominal & pft.pid_child.eq(child)
            pft.loc[mask, "date_of_test_ilf1"] = pft.loc[mask, "child_dob"] + pd.Timedelta(days=days)
        sample = prep.prepare_main_feno(pft, children)["samples"]["12 months"]["frame"]
        by_child = sample.set_index("pid_child")
        self.assertIn("synthetic_0", by_child.index)
        self.assertLess(by_child.loc["synthetic_0", "age"], 1.)
        self.assertEqual(by_child.loc["synthetic_0", "infancy_post"], 1)
        self.assertNotIn("synthetic_1", by_child.index)
        self.assertIn("synthetic_4", by_child.index)
        self.assertEqual(by_child.loc["synthetic_4", "infancy_post"], 0)

    def test_main_and_supplement_keep_different_future_tb_comparators(self):
        pft, children = feno_fixture()
        main = prep.prepare_main_feno(pft, children)["samples"]["24 months"]
        supplement = prep.prepare_supplementary_feno(pft)["samples"]["24 months"]
        main_child = main["frame"].set_index("pid_child").loc["synthetic_3"]
        supplement_child = supplement["frame"].set_index("pid_child").loc["synthetic_3"]
        self.assertEqual(main_child.infancy_post, 0)
        self.assertEqual(main_child.later_post, 0)
        self.assertEqual(supplement_child.tb_after_measurement, 1)
        self.assertNotIn("tb_after_measurement", main["design"].columns)
        self.assertIn("tb_after_measurement", supplement["design"].columns)

    def test_duplicate_dedicated_feno_visit_has_priority_and_nonpositive_excluded(self):
        pft, children = feno_fixture()
        original = pft.loc[pft.pid_child.eq("synthetic_6") & pft.visit_label.eq("FeNO")].iloc[0]
        duplicate = original.copy()
        duplicate["visit_label"] = "96 months"
        duplicate["feno_result"] = original.feno_result + 50
        pft = pd.concat([pft, duplicate.to_frame().T], ignore_index=True)
        zero = pft.pid_child.eq("synthetic_5") & pft.visit_label.eq("12 months")
        pft.loc[zero, "feno_result"] = 0.
        result = prep.prepare_main_feno(pft, children)
        self.assertEqual(result["audit"]["duplicate_rows_removed"], 1)
        self.assertEqual(result["audit"]["conflicting_duplicate_groups"], 1)
        chosen = result["samples"]["FeNO"]["frame"].set_index("pid_child").loc["synthetic_6"]
        self.assertEqual(chosen.feno, original.feno_result)
        self.assertNotIn("synthetic_5", set(result["samples"]["12 months"]["frame"].pid_child))


if __name__ == "__main__":
    unittest.main()
