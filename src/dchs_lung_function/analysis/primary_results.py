"""Full-precision aggregate summaries for primary spirometry joint models.

Contrasts are derived draw by draw from model parameters, using the target
outcome's residual scale and its post-diagnosis child counts. No manuscript
values, archive paths, participant records, or fitting operations are stored
or written here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


CONTRAST_VARIABLES = {
    "TB <1": "total_ratio_infancy",
    "TB 1-<5": "total_ratio_1to5",
    "TB >=5": "total_ratio_late",
    "TB <5 weighted": "total_ratio_lt5_child_weighted",
    "Any TB weighted": "total_ratio_any_child_weighted",
    "Infancy minus 1-<5": "total_ratio_infancy_difference",
    "Later minus infancy": "total_ratio_late_minus_infancy",
    "Later minus 1-<5": "total_ratio_late_minus_1to5",
}
PRIMITIVES = (
    "direct_ratio", "direct_ratio_infancy", "direct_ratio_late",
    "loading_ratio_mechanics", "shift_mechanics", "shift_mechanics_infancy",
    "shift_mechanics_late",
)
TARGETS = {
    f"{name}_{stage}{suffix}": {
        "outcome": label, "stage": stage_label, "unit": unit,
        "scale_type": scale_type, "column": column + post_suffix + z_suffix,
    }
    for stage, post_suffix, stage_label in [("pre", "", "Pre-bronchodilator"),
                                            ("post", "_post", "Post-bronchodilator")]
    for name, column, label in [("ratio", "fev1_fvc", "FEV1/FVC"),
                                ("fev1", "fev1", "FEV1"), ("fvc", "fvc", "FVC")]
    for suffix, z_suffix, unit, scale_type in (
        [("", "", "percentage points", "native"), ("_z", "_gli_z", "GLI z-score", "z")]
        if name == "ratio" else
        [("_native", "", "L", "native"), ("_z", "_gli_z", "GLI z-score", "z")]
    )
}


class PrimaryResultsError(ValueError):
    """The posterior or prepared sample cannot support the declared summaries."""


@dataclass
class PrimaryResults:
    estimates: pd.DataFrame
    shared_factor: pd.DataFrame
    sample_counts: pd.DataFrame
    metadata: dict


def _posterior(posterior):
    return posterior.posterior if hasattr(posterior, "posterior") else posterior


def _scalar_draws(posterior, variable):
    try:
        value = posterior[variable]
    except (KeyError, TypeError):
        raise PrimaryResultsError(f"Posterior lacks required scalar parameter {variable}.") from None
    if hasattr(value, "dims"):
        if set(value.dims) != {"chain", "draw"}:
            raise PrimaryResultsError(f"Posterior parameter {variable} must have only chain and draw dimensions.")
        value = value.transpose("chain", "draw")
    if np.iscomplexobj(value):
        raise PrimaryResultsError(f"Posterior parameter {variable} must be real-valued.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise PrimaryResultsError(f"Posterior parameter {variable} must be numeric.") from None
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise PrimaryResultsError(f"Posterior parameter {variable} requires finite, nonempty chain-by-draw values.")
    return array


def _count_vector(value, label):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise PrimaryResultsError(f"{label} must contain three nonnegative integer counts.") from None
    if array.shape != (3,) or not np.isfinite(array).all() or (array < 0).any() or (array != np.floor(array)).any():
        raise PrimaryResultsError(f"{label} must contain three nonnegative integer counts.")
    return array.astype(np.int64)


def _prepared_summary(prepared):
    try:
        frame = prepared["frames"]["target"]
        audit = prepared["audit"]["outcomes"]["target"]
        scale = float(audit["scale"])
        given_weights = _count_vector(prepared["weights"], "Prepared contrast weights")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, PrimaryResultsError):
            raise
        raise PrimaryResultsError("Prepared primary target requires frames, residual scale, and contrast weights.") from None
    if not np.isfinite(scale) or scale <= 0:
        raise PrimaryResultsError("Target residual scale must be finite and positive.")
    required = ["pid_child", "post_any", "tb_age_first_years"]
    if not isinstance(frame, pd.DataFrame) or not set(required).issubset(frame.columns):
        raise PrimaryResultsError("Prepared target frame lacks post-diagnosis child membership fields.")
    if frame.pid_child.isna().any() or frame.pid_child.astype("string").str.strip().eq("").any():
        raise PrimaryResultsError("Prepared target membership requires complete child keys.")
    if not frame.post_any.isin([0, 1]).all():
        raise PrimaryResultsError("Prepared target post-diagnosis indicator must be binary and nonmissing.")
    post = frame.loc[frame.post_any.eq(1), ["pid_child", "tb_age_first_years"]].copy()
    try:
        post["tb_age_first_years"] = pd.to_numeric(post.tb_age_first_years, errors="raise")
    except (TypeError, ValueError):
        raise PrimaryResultsError("Post-diagnosis children require numeric diagnosis ages.") from None
    if not np.isfinite(post.tb_age_first_years.to_numpy(float)).all() or post.tb_age_first_years.lt(0).any():
        raise PrimaryResultsError("Post-diagnosis children require finite nonnegative diagnosis ages.")
    if post.groupby("pid_child", observed=True).tb_age_first_years.nunique().gt(1).any():
        raise PrimaryResultsError("A post-diagnosis child has inconsistent diagnosis ages.")
    post_children = post.drop_duplicates("pid_child")
    age = post_children.tb_age_first_years
    weights = np.array([age.lt(1).sum(), (age.ge(1) & age.lt(5)).sum(), age.ge(5).sum()], dtype=np.int64)
    if not np.array_equal(weights, given_weights):
        raise PrimaryResultsError("Prepared weights do not equal distinct post-diagnosis target-child counts.")
    if "weights" in prepared.get("audit", {}) and not np.array_equal(weights, _count_vector(prepared["audit"]["weights"], "Audited contrast weights")):
        raise PrimaryResultsError("Audited and recomputed contrast weights disagree.")
    if int(weights[:2].sum()) == 0:
        raise PrimaryResultsError("The pooled before-five contrast requires at least one post-diagnosis target child before age five.")
    for key, observed in [("n_observations", len(frame)), ("n_children", frame.pid_child.nunique())]:
        if key in audit and audit[key] != observed:
            raise PrimaryResultsError("Target sample counts disagree with its preparation audit.")
    return scale, weights, frame


def _standardized_draws(posterior, prepared):
    posterior = _posterior(posterior)
    scale, weights, frame = _prepared_summary(prepared)
    d = {name: _scalar_draws(posterior, name) for name in PRIMITIVES}
    if len({array.shape for array in d.values()}) != 1:
        raise PrimaryResultsError("All scalar posterior parameters must share chain and draw dimensions.")
    middle = d["direct_ratio"] + d["loading_ratio_mechanics"] * d["shift_mechanics"]
    infancy = (d["direct_ratio"] + d["direct_ratio_infancy"] +
               d["loading_ratio_mechanics"] * (d["shift_mechanics"] + d["shift_mechanics_infancy"]))
    later = d["direct_ratio_late"] + d["loading_ratio_mechanics"] * d["shift_mechanics_late"]
    weight_infant = weights[0] / weights[:2].sum()
    contrasts = {
        "TB <1": infancy,
        "TB 1-<5": middle,
        "TB >=5": later,
        "TB <5 weighted": weight_infant * infancy + (1.0 - weight_infant) * middle,
        "Any TB weighted": (weights[0] * infancy + weights[1] * middle + weights[2] * later) / weights.sum(),
        "Infancy minus 1-<5": infancy - middle,
        "Later minus infancy": later - infancy,
        "Later minus 1-<5": later - middle,
    }
    if any(not np.isfinite(draws * scale).all() for draws in contrasts.values()):
        raise PrimaryResultsError("Derived posterior contrasts must remain finite after rescaling.")
    checked = []
    for contrast, variable in CONTRAST_VARIABLES.items():
        if variable in posterior:
            stored = _scalar_draws(posterior, variable)
            if stored.shape != contrasts[contrast].shape or not np.allclose(stored, contrasts[contrast], rtol=1e-11, atol=1e-12):
                raise PrimaryResultsError(f"Stored deterministic parameter {variable} disagrees with its declared contrast.")
            checked.append(variable)
    return contrasts, d, scale, weights, frame, checked


def derive_primary_contrast_draws(posterior, prepared) -> dict[str, np.ndarray]:
    """Return eight global contrast arrays on the reported outcome scale.

    Arrays retain their chain-by-draw shape for independent MCSE/ESS/R-hat
    calculations. The scale already includes percentage-point orientation for
    an unstandardized FEV1/FVC target; no second factor of 100 is applied.
    """
    contrasts, _, scale, _, _, _ = _standardized_draws(posterior, prepared)
    return {label: draws * scale for label, draws in contrasts.items()}


def _summarize(draws, compute_mcse):
    flat = draws.reshape(-1)
    low, median, high = np.quantile(flat, [.025, .5, .975], method="linear")
    result = {"mean": float(flat.mean()), "median": float(median), "lower": float(low), "upper": float(high),
              "probability_below_zero": float(np.mean(flat < 0)),
              "probability_above_zero": float(np.mean(flat > 0)),
              "probability_equal_zero": float(np.mean(flat == 0)), "mcse_mean": np.nan}
    if compute_mcse:
        try:
            import arviz as az
        except ImportError:
            raise PrimaryResultsError("Mean MCSE requires the configured ArviZ dependency.") from None
        result["mcse_mean"] = float(az.mcse(az.from_dict(posterior={"effect": draws}), method="mean").effect)
        if not np.isfinite(result["mcse_mean"]) or result["mcse_mean"] < 0:
            raise PrimaryResultsError("Mean MCSE is not finite and nonnegative; inspect the posterior draw history.")
    return result


def _sample_counts(prepared):
    rows = []
    for process, frame in prepared["frames"].items():
        required = {"pid_child", "post_any", "tb_age_first_years"}
        if not isinstance(frame, pd.DataFrame) or not required.issubset(frame):
            raise PrimaryResultsError("Each prepared process requires child keys, diagnosis ages, and post-diagnosis indicators.")
        if frame.pid_child.isna().any() or frame.pid_child.astype("string").str.strip().eq("").any():
            raise PrimaryResultsError("Every prepared process requires complete child keys.")
        if not frame.post_any.isin([0, 1]).all():
            raise PrimaryResultsError("A process post-diagnosis indicator must be binary and nonmissing.")
        age = pd.to_numeric(frame.tb_age_first_years, errors="coerce")
        post_age = age.loc[frame.post_any.eq(1)]
        if not np.isfinite(post_age.to_numpy(float)).all() or post_age.lt(0).any():
            raise PrimaryResultsError("Every post-diagnosis process measurement requires a finite nonnegative diagnosis age.")
        masks = {
            "all": pd.Series(True, index=frame.index),
            "pre_or_no_tb": frame.post_any.eq(0),
            "post_tb_infancy": frame.post_any.eq(1) & age.lt(1),
            "post_tb_age_one_to_under_five": frame.post_any.eq(1) & age.ge(1) & age.lt(5),
            "post_tb_age_five_or_later": frame.post_any.eq(1) & age.ge(5),
        }
        for group, mask in masks.items():
            rows.append({"process": process, "group": group, "measurements": int(mask.sum()),
                         "children": int(frame.loc[mask, "pid_child"].nunique())})
    return pd.DataFrame(rows)


def summarize_primary(posterior, prepared, target: str, *, compute_mcse: bool = True) -> PrimaryResults:
    """Summarize one of the twelve primary outcomes without rounding.

    Diagnostic acceptance belongs to the fitting workflow. This function
    reports estimates and MCSE but never marks a fit as converged. Setting
    compute_mcse=False leaves MCSE missing and is intended for pure algebra
    tests, not accepted analysis output.
    """
    if target not in TARGETS:
        raise PrimaryResultsError("Unknown primary spirometry target.")
    info = TARGETS[target]
    audit_target = prepared.get("audit", {}).get("target")
    if audit_target is not None and audit_target != info["column"]:
        raise PrimaryResultsError("Primary target name does not match the prepared outcome column.")
    contrasts, primitives, scale, weights, frame, checked = _standardized_draws(posterior, prepared)
    relative_weights = {
        "TB <1": np.array([1., 0., 0.]), "TB 1-<5": np.array([0., 1., 0.]),
        "TB >=5": np.array([0., 0., 1.]),
        "TB <5 weighted": np.array([weights[0], weights[1], 0.]) / weights[:2].sum(),
        "Any TB weighted": weights / weights.sum(),
        "Infancy minus 1-<5": np.array([1., -1., 0.]),
        "Later minus infancy": np.array([-1., 0., 1.]),
        "Later minus 1-<5": np.array([0., -1., 1.]),
    }
    rows = []
    for contrast, draws in contrasts.items():
        coefficients = relative_weights[contrast]
        selected_groups = np.abs(coefficients) > 0
        rows.append({
            "target": target, "outcome": info["outcome"], "stage": info["stage"], "unit": info["unit"],
            "scale_type": info["scale_type"], "contrast": contrast,
            "posterior_variable": CONTRAST_VARIABLES[contrast], **_summarize(draws * scale, compute_mcse),
            "n_outcome_children": int(frame.pid_child.nunique()), "n_outcome_measurements": len(frame),
            "n_post_tb_infancy_children": int(weights[0]), "n_post_tb_age_one_to_under_five_children": int(weights[1]),
            "n_post_tb_age_five_or_later_children": int(weights[2]),
            "n_post_tb_children_in_contrast": int(weights[selected_groups].sum()),
            "weight_infancy": float(coefficients[0]), "weight_age_one_to_under_five": float(coefficients[1]),
            "weight_age_five_or_later": float(coefficients[2]),
            "all_contrast_groups_have_post_tb_support": bool((weights[selected_groups] > 0).all()),
            "residual_scale": scale,
            "table_2": contrast in {"TB <5 weighted", "TB <1"},
            "table_s6": contrast in list(CONTRAST_VARIABLES)[:5],
            "figure_2_continuous": target.startswith("ratio_") and contrast in {"TB <5 weighted", "TB <1"},
        })
    factor_draws = -primitives["loading_ratio_mechanics"] * scale
    shared = pd.DataFrame([{
        "target": target, "outcome": info["outcome"], "stage": info["stage"], "unit": info["unit"],
        "contrast": "One model-defined SD poorer combined respiratory mechanics",
        **_summarize(factor_draws, compute_mcse), "residual_scale": scale,
        "n_outcome_children": int(frame.pid_child.nunique()), "n_outcome_measurements": len(frame),
        "figure_s5": info["scale_type"] == "z",
    }])
    sample_counts = _sample_counts(prepared)
    shape = next(iter(contrasts.values())).shape
    children = prepared.get("children")
    n_joint = len(children) if isinstance(children, pd.DataFrame) else prepared.get("audit", {}).get("joint_children")
    events = prepared.get("event", {}).get("event")
    n_events = int(np.asarray(events).sum()) if events is not None else prepared.get("audit", {}).get("tb_events")
    metadata = {
        "target": target, **info, "chains": shape[0], "draws_per_chain": shape[1], "posterior_draws": int(np.prod(shape)),
        "interval": "equal-tailed 95% credible interval", "quantile_method": "linear",
        "table_point_estimator": "posterior mean", "joint_children": n_joint, "first_tb_events": n_events,
        "contrast_weight_rule": "distinct children with post-diagnosis measurements of this target outcome",
        "post_diagnosis_child_counts": weights.tolist(), "residual_scale": scale,
        "scale_rule": "multiply standardized posterior contrasts once by target nuisance residual SD",
        "shared_factor_rule": "negative target mechanics loading times target residual SD",
        "deterministic_variables_crosschecked": checked, "mcse_computed": bool(compute_mcse),
        "diagnostic_acceptance": "not evaluated by result summarizer", "small_cells_suppressed": False,
    }
    return PrimaryResults(pd.DataFrame(rows), shared, sample_counts, metadata)
