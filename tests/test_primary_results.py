"""Synthetic algebra, weighting, scale and manuscript-selection checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dchs_lung_function.analysis.primary_results import (
    CONTRAST_VARIABLES, TARGETS, PrimaryResultsError,
    derive_primary_contrast_draws, summarize_primary,
)


def fixture(scale=2.0):
    # Infancy has more measurements but fewer distinct children than age1–<5.
    frame = pd.DataFrame({
        "pid_child": ["infant"] * 4 + ["one", "two", "later", "reference"],
        "tb_age_first_years": [.5] * 4 + [2., 3., 6., np.nan],
        "post_any": [1] * 7 + [0],
    })
    prepared = {
        "frames": {"target": frame}, "weights": np.array([1, 2, 1]),
        "children": frame.drop_duplicates("pid_child"),
        "event": {"event": np.array([1, 1, 1, 1, 0])},
        "audit": {"target": "fev1_fvc", "joint_children": 5, "tb_events": 4, "weights": [1, 2, 1],
                  "outcomes": {"target": {"scale": scale, "n_children": 5, "n_observations": 8}}},
    }
    x = np.array([[0., 1., 2., 3.], [4., 5., 6., 7.]])
    posterior = {
        "direct_ratio": x - 4., "direct_ratio_infancy": np.ones_like(x),
        "direct_ratio_late": x / 2., "loading_ratio_mechanics": x + 1.,
        "shift_mechanics": .1 * x, "shift_mechanics_infancy": np.full_like(x, .25),
        "shift_mechanics_late": np.full_like(x, -.5),
    }
    return posterior, prepared


class PrimaryResultsTests(unittest.TestCase):
    def test_components_combined_per_draw_before_summarizing(self):
        posterior, prepared = fixture()
        draws = derive_primary_contrast_draws(posterior, prepared)
        expected = np.array([[-5.5, -2.6, .7, 4.4], [8.5, 13., 17.9, 23.2]])
        np.testing.assert_allclose(draws["TB <1"], expected, atol=1e-12)
        # The product of posterior means would be a different estimand.
        effect = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False).estimates.set_index("contrast")
        self.assertAlmostEqual(effect.loc["TB <1", "mean"], expected.mean())
        self.assertNotAlmostEqual(effect.loc["TB <1", "mean"], 2 * (
            posterior["direct_ratio"].mean() + posterior["direct_ratio_infancy"].mean() +
            posterior["loading_ratio_mechanics"].mean() *
            (posterior["shift_mechanics"].mean() + posterior["shift_mechanics_infancy"].mean())))

    def test_child_weights_not_measurement_weights(self):
        posterior, prepared = fixture()
        d = derive_primary_contrast_draws(posterior, prepared)
        np.testing.assert_allclose(d["TB <5 weighted"], (d["TB <1"] + 2 * d["TB 1-<5"]) / 3)
        np.testing.assert_allclose(d["Any TB weighted"], (d["TB <1"] + 2 * d["TB 1-<5"] + d["TB >=5"]) / 4)
        result = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)
        row = result.estimates.set_index("contrast").loc["TB <5 weighted"]
        self.assertEqual(row.n_post_tb_infancy_children, 1)
        self.assertEqual(row.n_post_tb_children_in_contrast, 3)
        self.assertAlmostEqual(row.weight_infancy, 1 / 3)
        # A measurement-weighted infancy share would be4/6; do not use it.
        self.assertNotAlmostEqual(row.weight_infancy, 4 / 6)

    def test_later_term_is_its_own_direct_and_mechanics_path(self):
        posterior, prepared = fixture()
        d = derive_primary_contrast_draws(posterior, prepared)
        expected = 2 * (posterior["direct_ratio_late"] +
                        posterior["loading_ratio_mechanics"] * posterior["shift_mechanics_late"])
        np.testing.assert_array_equal(d["TB >=5"], expected)
        np.testing.assert_allclose(d["Later minus infancy"], d["TB >=5"] - d["TB <1"])
        np.testing.assert_allclose(d["Infancy minus 1-<5"], d["TB <1"] - d["TB 1-<5"])

    def test_scale_applied_exactly_once_and_factor_oriented_poorer(self):
        posterior, prepared = fixture(scale=7.)
        result = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)
        middle = posterior["direct_ratio"] + posterior["loading_ratio_mechanics"] * posterior["shift_mechanics"]
        self.assertAlmostEqual(result.estimates.set_index("contrast").loc["TB 1-<5", "mean"], middle.mean() * 7)
        self.assertAlmostEqual(result.shared_factor.iloc[0]["mean"], -posterior["loading_ratio_mechanics"].mean() * 7)
        self.assertEqual(result.metadata["unit"], "percentage points")
        self.assertFalse(result.shared_factor.iloc[0].figure_s5)

    def test_linear_equal_tailed_interval_and_strict_zero_probabilities(self):
        posterior, prepared = fixture()
        for key in posterior:
            posterior[key] = np.zeros((2, 4))
        posterior["direct_ratio_late"] = np.array([[-4., -1., 0., 0.], [0., 1., 2., 10.]])
        row = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False).estimates.set_index("contrast").loc["TB >=5"]
        # Linear interpolation of eight ordered draws, after multiplying by2.
        self.assertAlmostEqual(row.lower, -6.95)
        self.assertAlmostEqual(row.upper, 17.2)
        self.assertEqual(row["median"], 0.)
        self.assertEqual(row.probability_below_zero, .25)
        self.assertEqual(row.probability_above_zero, .375)
        self.assertEqual(row.probability_equal_zero, .375)
        self.assertTrue(pd.isna(row.mcse_mean))

    def test_optional_deterministic_checks_reject_stale_saved_contrasts(self):
        posterior, prepared = fixture()
        d = derive_primary_contrast_draws(posterior, prepared)
        for label, name in CONTRAST_VARIABLES.items():
            posterior[name] = d[label] / 2
        result = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)
        self.assertEqual(len(result.metadata["deterministic_variables_crosschecked"]), 8)
        posterior["total_ratio_lt5_child_weighted"] += .01
        with self.assertRaisesRegex(PrimaryResultsError, "disagrees"):
            summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)

    def test_wrong_weights_and_sample_audit_rejected(self):
        for field in ["weights", "count", "scale", "age"]:
            posterior, prepared = fixture()
            if field == "weights":
                prepared["weights"] = [4, 2, 1]
            elif field == "count":
                prepared["audit"]["outcomes"]["target"]["n_observations"] = 7
            elif field == "scale":
                prepared["audit"]["outcomes"]["target"]["scale"] = 0
            else:
                prepared["frames"]["target"].loc[0, "tb_age_first_years"] = 3.
            with self.subTest(field=field), self.assertRaises(PrimaryResultsError):
                derive_primary_contrast_draws(posterior, prepared)

    def test_bad_draw_shapes_and_nonfinite_values_rejected(self):
        for malformed in [np.ones(8), np.ones((2, 4, 1)), np.ones((2, 5)), np.full((2, 4), np.nan),
                          np.ones((2, 4), dtype=complex) * (1 + 2j)]:
            posterior, prepared = fixture()
            posterior["direct_ratio"] = malformed
            with self.subTest(shape=malformed.shape), self.assertRaises(PrimaryResultsError):
                derive_primary_contrast_draws(posterior, prepared)

    def test_invalid_nontarget_process_membership_is_not_silently_omitted(self):
        for field, value in [("pid_child", None), ("tb_age_first_years", np.nan),
                             ("tb_age_first_years", -1.), ("post_any", 2)]:
            posterior, prepared = fixture()
            prepared["frames"]["compliance"] = prepared["frames"]["target"].copy()
            prepared["frames"]["compliance"].loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(PrimaryResultsError):
                summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)

    def test_no_early_post_tb_support_cannot_define_weighted_contrast(self):
        posterior, prepared = fixture()
        prepared["frames"]["target"]["tb_age_first_years"] = 6.
        prepared["weights"] = prepared["audit"]["weights"] = [0, 0, 4]
        with self.assertRaisesRegex(PrimaryResultsError, "at least one"):
            derive_primary_contrast_draws(posterior, prepared)

    def test_zero_support_group_retains_counts_and_parameter_estimate(self):
        posterior, prepared = fixture()
        prepared["frames"]["target"].loc[prepared["frames"]["target"].pid_child.eq("later"), "post_any"] = 0
        prepared["weights"] = prepared["audit"]["weights"] = [1, 2, 0]
        result = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)
        row = result.estimates.set_index("contrast").loc["TB >=5"]
        self.assertEqual(row.n_post_tb_children_in_contrast, 0)
        self.assertFalse(row.all_contrast_groups_have_post_tb_support)
        self.assertTrue(np.isfinite(row["mean"]))

    def test_twelve_target_units_and_manuscript_selections(self):
        frames = []
        shared = []
        self.assertEqual(len(TARGETS), 12)
        for target, spec in TARGETS.items():
            posterior, prepared = fixture()
            prepared["audit"]["target"] = spec["column"]
            result = summarize_primary(posterior, prepared, target, compute_mcse=False)
            frames.append(result.estimates)
            shared.append(result.shared_factor)
            self.assertEqual(result.estimates.unit.unique().tolist(), [spec["unit"]])
        all_rows = pd.concat(frames)
        self.assertEqual(int(all_rows.table_2.sum()), 24)
        self.assertEqual(int(all_rows.table_s6.sum()), 60)
        self.assertEqual(int(all_rows.figure_2_continuous.sum()), 8)
        self.assertEqual(int(pd.concat(shared).figure_s5.sum()), 6)

    def test_target_mismatch_rejected_and_no_subject_keys_returned(self):
        posterior, prepared = fixture()
        with self.assertRaisesRegex(PrimaryResultsError, "does not match"):
            summarize_primary(posterior, prepared, "fvc_post_z", compute_mcse=False)
        result = summarize_primary(posterior, prepared, "ratio_pre", compute_mcse=False)
        for frame in [result.estimates, result.shared_factor, result.sample_counts]:
            self.assertNotIn("pid_child", frame.columns)
        self.assertEqual(result.metadata["joint_children"], 5)
        self.assertEqual(result.metadata["first_tb_events"], 4)


if __name__ == "__main__":
    unittest.main()
