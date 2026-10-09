"""Construct explicit DCHS populations and aggregate inclusion ledgers.

Membership frames contain private records and are returned in memory only. This
module does not write files, load archived results, or fit statistical models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

ID = "pid_child"

# Definitions are variable/date specifications, not expected study results.
DESCRIPTIVE_OUTCOMES = {
    "ratio_pre_z": ("fev1_fvc_gli_z", "date_of_test"),
    "resistance": ("resistance", "date_of_test_ilf2"),
    "compliance": ("compliance", "date_of_test_ilf2"),
    "ree": ("ree", "date_of_test_ilf2"),
    "rei": ("rei", "date_of_test_ilf2"),
    "xee": ("xee", "date_of_test_ilf2"),
    "xei": ("xei", "date_of_test_ilf2"),
    "x6": ("x6_sa_z", "date_of_test_ilf2"),
}
PRIMARY_OUTCOMES = {
    **{name: (column, "date_of_test") for name, column in [
        ("ratio_pre", "fev1_fvc"), ("ratio_pre_z", "fev1_fvc_gli_z"),
        ("ratio_post", "fev1_fvc_post"), ("ratio_post_z", "fev1_fvc_post_gli_z"),
        ("fev1_pre_native", "fev1"), ("fev1_pre_z", "fev1_gli_z"),
        ("fvc_pre_native", "fvc"), ("fvc_pre_z", "fvc_gli_z"),
        ("fev1_post_native", "fev1_post"), ("fev1_post_z", "fev1_post_gli_z"),
        ("fvc_post_native", "fvc_post"), ("fvc_post_z", "fvc_post_gli_z"),
    ]},
    **{name: (name, "date_of_test_ilf2") for name in ["resistance", "compliance", "ree", "rei", "xee", "xei"]},
    "lci": ("i_lci_2_5", "date_of_test_ilf1"),
}
BASELINE_DESCRIPTIVE_FIELDS = [
    "maternal_age_birth", "education", "Prenatal_smoking_composite",
    "Prenatal_alcohol_composite", "parity", "crowd", "household_income",
    "smoke_hh", "lowbirthweight",
]
PFT_CHILD_FIELDS = [
    "child_dob", "tb_first_dx", "tb_age_first_years", "sex", "site", "ethnicity",
    "gestation_delivery", "preterm", "birth_weight", "hiv_exposed", "hiv_infected",
]


class CohortError(ValueError):
    """An input cannot support a declared population safely."""


@dataclass
class CohortResult:
    joint_children: pd.DataFrame
    descriptive_children: pd.DataFrame
    outcome_membership: Mapping[str, pd.DataFrame]
    ledger: pd.DataFrame
    audit: dict


def require_columns(frame: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise CohortError(f"{label} lacks required fields: {', '.join(missing)}.")


def numeric(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if (series.notna() & values.isna()).any() or np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
        raise CohortError(f"{label} contains an invalid numeric value.")
    return values


def _dates(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if (series.notna() & values.isna()).any():
        raise CohortError(f"{label} contains an invalid date.")
    return values


def _identifiers(frame: pd.DataFrame, label: str, unique=False) -> pd.DataFrame:
    require_columns(frame, [ID], label)
    result = frame.copy()
    ids = result[ID].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise CohortError(f"{label} contains missing child identifiers.")
    if unique and ids.duplicated().any():
        raise CohortError(f"{label} must contain one record per child.")
    result[ID] = ids
    return result


def _child_records(pft: pd.DataFrame) -> pd.DataFrame:
    # Do not silently select a row when child-level attributes conflict.
    for column in PFT_CHILD_FIELDS:
        if pft.groupby(ID, observed=True)[column].nunique(dropna=False).gt(1).any():
            raise CohortError(f"Pulmonary-function child field {column} is inconsistent across records.")
    return pft.sort_values(ID, kind="stable").drop_duplicates(ID).copy().reset_index(drop=True)


def _selection(pft, column, date_column, population, outcome, *, exclude_later=False,
               restrict_exit=False, ascertainment_end=None):
    require_columns(pft, [column, date_column], "Pulmonary function")
    value = numeric(pft[column], column)
    date = _dates(pft[date_column], date_column)
    height = numeric(pft["height"], "height")
    sex = numeric(pft["sex"], "pulmonary-function sex")
    if not sex.dropna().isin([0, 1]).all():
        raise CohortError("Pulmonary-function sex must use the declared 0/1 source codes.")
    eligible = pd.Series(True, index=pft.index)
    ledger = []

    def apply(step, keep):
        nonlocal eligible
        before = int(eligible.sum())
        eligible &= keep.fillna(False)
        ledger.append({"population": population, "outcome": outcome, "step": step,
                       "records_before": before, "records_excluded": before - int(eligible.sum()),
                       "records_retained": int(eligible.sum()),
                       "children_retained": int(pft.loc[eligible, ID].nunique())})

    apply("nonmissing_outcome", value.notna())
    apply("dated_measurement", date.notna())
    apply("nonmissing_height", height.notna())
    apply("nonmissing_sex", sex.notna())
    if (eligible & date.lt(pft["child_dob"])).any():
        raise CohortError("An eligible measurement predates birth.")
    post = pft["tb_first_dx"].notna() & date.ge(pft["tb_first_dx"])
    if exclude_later:
        apply("exclude_post_diagnosis_at_age_five_or_later", ~(post & pft["tb_age_first_years"].ge(5)))
    if restrict_exit:
        apply("on_or_before_study_exit", pft["exit_date"].isna() | date.le(pft["exit_date"]))
    if ascertainment_end is not None:
        apply("on_or_before_ascertainment_end", date.le(ascertainment_end))
    result = pft.loc[eligible, [ID, "source_row", "child_dob", "tb_first_dx", "tb_age_first_years"]].copy()
    result["test_date"] = date[eligible]
    result["value"] = value[eligible]
    result["post_tb"] = post[eligible]
    age = result["tb_age_first_years"]
    result["exposure_group"] = np.select(
        [result["post_tb"] & age.lt(1), result["post_tb"] & age.ge(1) & age.lt(5),
         result["post_tb"] & age.ge(5)], ["infancy", "age_one_to_under_five", "age_five_or_later"],
        default="pre_or_no_tb")
    return result.reset_index(drop=True), ledger


def build_cohorts(pft: pd.DataFrame, baseline: pd.DataFrame, *, ascertainment_end=None) -> CohortResult:
    """Derive descriptive, full joint, primary and restricted outcome populations.

    The descriptive union uses the selected early-TB eligibility definition:
    outcome/date/height/sex observed, excluding observations after first TB at
    age five or later. Primary available-case populations retain later-TB
    measurements and restrict observations to formal study exit. These are
    deliberately separate population definitions, even when counts coincide.
    """
    require_columns(pft, [*PFT_CHILD_FIELDS, "height"], "Pulmonary function")
    require_columns(baseline, [ID, "etq1", *BASELINE_DESCRIPTIVE_FIELDS], "Baseline")
    p = _identifiers(pft, "Pulmonary function").reset_index(drop=True)
    b = _identifiers(baseline, "Baseline", unique=True)
    if not p[ID].isin(b[ID]).all():
        raise CohortError("Pulmonary-function children lack baseline linkage.")
    p["source_row"] = np.arange(len(p), dtype=int)
    p["child_dob"] = _dates(p["child_dob"], "child_dob")
    p["tb_first_dx"] = _dates(p["tb_first_dx"], "tb_first_dx")
    p["tb_age_first_years"] = numeric(p["tb_age_first_years"], "tb_age_first_years")
    if p["child_dob"].isna().any():
        raise CohortError("Pulmonary-function children require a date of birth.")
    if p["tb_first_dx"].notna().ne(p["tb_age_first_years"].notna()).any():
        raise CohortError("Diagnosis date and age availability disagree.")
    if (p["tb_first_dx"].notna() & p["tb_first_dx"].lt(p["child_dob"])).any():
        raise CohortError("A diagnosis predates birth.")
    derived_age = (p["tb_first_dx"] - p["child_dob"]).dt.days / 365.25
    if not np.allclose(derived_age, p["tb_age_first_years"], equal_nan=True, atol=1e-6, rtol=0):
        raise CohortError("Diagnosis age does not agree with exact dates.")
    base_index = b.set_index(ID)
    p["exit_date"] = p[ID].map(_dates(base_index["etq1"], "etq1"))
    if (p["exit_date"].notna() & p["exit_date"].lt(p["child_dob"])).any():
        raise CohortError("A study exit predates birth.")
    if ascertainment_end is not None:
        try:
            ascertainment_end = pd.Timestamp(ascertainment_end)
        except (TypeError, ValueError):
            raise CohortError("Ascertainment end must be a valid date.") from None
        if pd.isna(ascertainment_end):
            raise CohortError("Ascertainment end must be a valid date.")
        censor = p["exit_date"].fillna(ascertainment_end).clip(upper=ascertainment_end)
        if (p["tb_first_dx"].notna() & p["tb_first_dx"].gt(censor)).any():
            raise CohortError("A diagnosis falls after its allowed follow-up end.")
    children = _child_records(p)
    children["child_index"] = np.arange(len(children), dtype=int)
    for field in BASELINE_DESCRIPTIVE_FIELDS:
        children[field] = children[ID].map(base_index[field])
    children["tb_diagnosis_group"] = np.select(
        [children.tb_age_first_years.lt(1), children.tb_age_first_years.ge(1) & children.tb_age_first_years.lt(5),
         children.tb_age_first_years.ge(5)], ["infancy", "age_one_to_under_five", "age_five_or_later"],
        default="no_recorded_tb")
    memberships, ledger = {}, []
    for population, specs in [("descriptive", DESCRIPTIVE_OUTCOMES), ("primary", PRIMARY_OUTCOMES)]:
        for outcome, (column, date) in specs.items():
            result, rows = _selection(p, column, date, population, outcome,
                                      exclude_later=population == "descriptive", restrict_exit=population == "primary")
            memberships[f"{population}:{outcome}"] = result
            ledger.extend(rows)
    if ascertainment_end is not None:
        for outcome, (column, date) in PRIMARY_OUTCOMES.items():
            result, rows = _selection(p, column, date, "restricted", outcome,
                                      restrict_exit=True, ascertainment_end=ascertainment_end)
            memberships[f"restricted:{outcome}"] = result
            ledger.extend(rows)
    descriptive_ids = set().union(*(set(memberships[f"descriptive:{key}"][ID]) for key in DESCRIPTIVE_OUTCOMES))
    descriptive = children.loc[children[ID].isin(descriptive_ids)].copy().reset_index(drop=True)
    outcome_counts = []
    for name, frame in memberships.items():
        population, outcome = name.split(":", 1)
        outcome_counts.append({"population": population, "outcome": outcome,
                               "measurements": len(frame), "children": int(frame[ID].nunique()),
                               "post_tb_measurements": int(frame.post_tb.sum()),
                               "post_tb_children": int(frame.loc[frame.post_tb, ID].nunique())})
    groups = descriptive["tb_diagnosis_group"].value_counts().to_dict()
    audit = {"baseline_children": len(b), "pft_source_records": len(p), "joint_children": len(children),
             "descriptive_children": len(descriptive), "baseline_children_without_pft_records": len(b) - len(children),
             "joint_children_without_descriptive_measurements": len(children) - len(descriptive),
             "descriptive_diagnosis_groups": {k: int(groups.get(k, 0)) for k in
                 ["no_recorded_tb", "infancy", "age_one_to_under_five", "age_five_or_later"]},
             "outcome_counts": outcome_counts, "small_cells_suppressed": False}
    return CohortResult(children, descriptive, memberships, pd.DataFrame(ledger), audit)


def build_flow_summary(cohorts: CohortResult, baseline: pd.DataFrame) -> pd.DataFrame:
    """Derive child-cohort flow; mark absent antenatal recruitment evidence explicitly."""
    b = _identifiers(baseline, "Baseline", unique=True)
    if len(b) != cohorts.audit["baseline_children"] or not cohorts.joint_children[ID].isin(b[ID]).all():
        raise CohortError("Flow baseline does not match the cohort construction input.")
    rows = []
    def add(key, value, unit="children", status="derived", evidence="baseline and cohort membership"):
        rows.append({"node": key, "count": value, "unit": unit, "status": status, "evidence": evidence})
    for key in ["antenatal_enrollment", "antenatal_loss_to_followup", "pregnancy_loss", "excluded_before_child_cohort"]:
        add(key, None, "women", "not_reconstructable_from_supplied_inputs", "Antenatal enrollment/disposition source required")
    add("liveborn_children_enrolled", len(b))
    if "pid_maternal" in b and b.pid_maternal.notna().all() and b.pid_maternal.astype("string").str.strip().ne("").all():
        sizes = b.groupby(b.pid_maternal.astype("string").str.strip(), observed=True).size()
        add("women_with_liveborn_children", len(sizes), "women", evidence="distinct maternal identifiers in child baseline")
        for plurality, count in sizes.value_counts().sort_index().items():
            add(f"live_birth_sets_of_{int(plurality)}", int(count), "women", evidence="children per maternal identifier in child baseline")
    else:
        add("women_with_liveborn_children", None, "women", "not_reconstructable_from_supplied_inputs", "Complete maternal linkage required")
    add("children_with_pft_records", cohorts.audit["joint_children"])
    add("children_without_pft_records", cohorts.audit["baseline_children_without_pft_records"])
    add("combined_eligible_lung_function", len(cohorts.descriptive_children))
    add("pft_children_without_descriptive_measurements", cohorts.audit["joint_children_without_descriptive_measurements"])
    g = cohorts.audit["descriptive_diagnosis_groups"]
    for key, value in g.items():
        add(key, value)
    add("no_tb_before_five", g["no_recorded_tb"] + g["age_five_or_later"])
    add("tb_before_five", g["infancy"] + g["age_one_to_under_five"])
    for row in cohorts.audit["outcome_counts"]:
        if row["population"] in {"primary", "descriptive"}:
            add(f"{row['population']}:{row['outcome']}", row["children"], evidence="outcome-specific available-case membership")
    result = pd.DataFrame(rows)
    result["count"] = pd.array(result["count"], dtype="Int64")
    return result
