"""Aggregate baseline summaries and empirical modality-by-visit availability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .cohorts import CohortError, ID, numeric, require_columns


@dataclass
class DescriptiveResult:
    table: pd.DataFrame
    statistics: pd.DataFrame
    missingness: pd.DataFrame
    definitions: pd.DataFrame
    audit: dict


GROUPS = ["overall", "no_tb", "tb_lt5", "tb_lt1"]
MODALITIES = {
    "multiple_breath_washout": {"columns": ["i_lci_2_5"], "date": "date_of_test_ilf1"},
    "mechanics_within_breath": {"columns": ["resistance", "compliance", "ree", "rei", "xee", "xei"], "date": "date_of_test_ilf2"},
    "frequency_domain_oscillometry": {"columns": ["resistance_6hz", "reactance_6hz", "resistance_8hz", "reactance_8hz", "resistance_10hz", "reactance_10hz", "ax_mean", "r6_r20"], "date": "date_of_test_ilf2"},
    "spirometry": {"columns": ["fev1", "fvc", "fev1_fvc", "mmef"], "date": "date_of_test"},
}


def _indicator(source, positive, allowed, label, *, missing_as_reference=False):
    s = numeric(source, label)
    if not s.dropna().isin(allowed).all():
        raise CohortError(f"{label} contains values outside its declared coding.")
    indicator = s.eq(positive).astype(float)
    return indicator if missing_as_reference else indicator.where(s.notna())


def _smd(a, b, binary):
    a, b = a.dropna(), b.dropna()
    if not len(a) or not len(b) or (not binary and (len(a) < 2 or len(b) < 2)):
        return np.nan
    pa, pb = float(a.mean()), float(b.mean())
    denominator = np.sqrt(((pa * (1 - pa) + pb * (1 - pb)) if binary else (a.var(ddof=1) + b.var(ddof=1))) / 2)
    if denominator == 0:
        return 0.0 if pa == pb else np.nan
    return abs(pa - pb) / denominator


def build_table1(cohort: pd.DataFrame) -> DescriptiveResult:
    """Compute Table 1, including exact nonmissing denominators and parity IQR.

    PFT source sex uses 0=male, 1=female. Child/birth fields come from the linked
    pulmonary-function source; maternal/household fields come from baseline.
    Quantiles use linear interpolation. Sample standard deviations use ddof=1.
    Selected inherited equality-based child/birth recodes classify missing raw
    codes in the reference category, as in the manuscript producer. Original
    missing and recoded counts are reported explicitly; direct baseline
    indicators and categorical variables retain missing denominators.
    """
    fields = [ID, "tb_age_first_years", "sex", "site", "ethnicity", "gestation_delivery",
              "preterm", "birth_weight", "hiv_exposed", "hiv_infected", "maternal_age_birth",
              "education", "Prenatal_smoking_composite", "Prenatal_alcohol_composite", "parity",
              "crowd", "household_income", "smoke_hh", "lowbirthweight"]
    require_columns(cohort, fields, "Descriptive cohort")
    c = cohort.copy()
    if c[ID].isna().any() or c[ID].astype("string").str.strip().eq("").any() or c[ID].astype("string").str.strip().duplicated().any():
        raise CohortError("Descriptive cohort must contain one identified record per child.")
    age = numeric(c.tb_age_first_years, "tb_age_first_years")
    if age.dropna().lt(0).any():
        raise CohortError("Diagnosis age cannot be negative.")
    masks = {"overall": pd.Series(True, index=c.index), "no_tb": ~age.lt(5),
             "tb_lt5": age.lt(5), "tb_lt1": age.lt(1)}
    conversions = {
        "male": ("sex", 0, [0, 1]), "mbekweni": ("site", 1, [0, 1]),
        "black_african": ("ethnicity", 1, [0, 1]), "preterm_bin": ("preterm", 1, [0, 1]),
        "hiv_exposed_bin": ("hiv_exposed", 1, [0, 1]), "hiv_infected_bin": ("hiv_infected", 1, [0, 1]),
        "low_birthweight_bin": ("lowbirthweight", 1, [0, 1]),
        "prenatal_smoking_bin": ("Prenatal_smoking_composite", 1, [0, 1]),
        "prenatal_alcohol_bin": ("Prenatal_alcohol_composite", 1, [0, 1]),
        "household_smoker_bin": ("smoke_hh", 1, [0, 1]),
    }
    inherited_recodes = {"male", "mbekweni", "black_african", "preterm_bin",
                         "hiv_exposed_bin", "hiv_infected_bin", "low_birthweight_bin"}
    original_missing = {}
    for dest, (source, positive, allowed) in conversions.items():
        original_missing[dest] = c[source].isna()
        c[dest] = _indicator(c[source], positive, allowed, source,
                             missing_as_reference=dest in inherited_recodes)
    c["gestational_age_weeks"] = numeric(c.gestation_delivery, "gestation_delivery")
    c["birthweight_g"] = numeric(c.birth_weight, "birth_weight")
    for name in ["maternal_age_birth", "education", "parity", "crowd", "household_income"]:
        c[name] = numeric(c[name], name)
    if c.parity.dropna().lt(0).any():
        raise CohortError("Parity cannot be negative.")
    rows, stats, missing = [], [], []

    def section(label, kind="section"):
        rows.append({"row_type": kind, "characteristic": label, **dict.fromkeys(GROUPS, ""), "smd": np.nan})

    def add(variable, label, kind="binary", decimals=1, source=None):
        values = c[variable] if source is None else source
        row = {"row_type": kind, "characteristic": label}
        for group in GROUPS:
            selected = values[masks[group]]
            valid = selected.dropna()
            n = len(valid)
            m = {"variable": variable, "group": group, "n_group": int(masks[group].sum()),
                 "n_nonmissing": n, "n_missing": int(selected.isna().sum())}
            raw_missing = original_missing.get(variable, values.isna())
            m["n_source_missing"] = int(raw_missing[masks[group]].sum())
            m["n_missing_recoded_to_reference"] = m["n_source_missing"] if variable in inherited_recodes else 0
            missing.append(m)
            s = {**m, "statistic": kind, "numerator": np.nan, "denominator": n,
                 "percent": np.nan, "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan}
            if not n:
                row[group] = "—"
            elif kind in {"binary", "category_level"}:
                count = int(valid.sum())
                s.update(numerator=count, percent=100.0 * count / n)
                row[group] = f"{count}/{n} ({s['percent']:.1f})"
            elif kind == "median_iqr":
                q1, med, q3 = valid.quantile([.25, .5, .75], interpolation="linear").to_numpy(float)
                s.update(median=med, q1=q1, q3=q3)
                row[group] = f"{med:g} [{q1:g}–{q3:g}]"
            else:
                mean, sd = float(valid.mean()), float(valid.std(ddof=1))
                s.update(mean=mean, sd=sd)
                sd_text = f"{sd:.{decimals}f}" if np.isfinite(sd) else "—"
                row[group] = f"{mean:.{decimals}f} ({sd_text})"
            stats.append(s)
        row["smd"] = _smd(values[masks["tb_lt5"]], values[masks["no_tb"]], kind in {"binary", "category_level"})
        rows.append(row)

    def category(variable, heading, levels):
        if not c[variable].dropna().isin([code for code, _ in levels]).all():
            raise CohortError(f"{variable} contains values outside its declared coding.")
        section(heading, "category_heading")
        for code, label in levels:
            indicator = c[variable].eq(code).astype(float).where(c[variable].notna())
            add(f"{variable}:{code}", label, "category_level", source=indicator)

    section("Child characteristics at birth")
    add("male", "Male sex")
    add("mbekweni", "Recruitment site: Mbekweni")
    add("black_african", "Black African ethnicity")
    add("gestational_age_weeks", "Mean gestational age, weeks (SD)", "continuous")
    add("preterm_bin", "Preterm birth (<37 weeks)")
    add("birthweight_g", "Mean birthweight, grams (SD)", "continuous", 0)
    add("low_birthweight_bin", "Low birthweight (<2500 g)")
    add("hiv_exposed_bin", "HIV exposure at birth")
    add("hiv_infected_bin", "Child HIV infection")
    section("Maternal and pregnancy characteristics")
    add("maternal_age_birth", "Mean maternal age at childbirth, years (SD)", "continuous")
    category("education", "Maternal education", [(0, "No education"), (1, "Primary school only"),
              (2, "Some secondary school"), (3, "Completed secondary school")])
    add("prenatal_smoking_bin", "Smoking during pregnancy")
    add("prenatal_alcohol_bin", "Alcohol use during pregnancy")
    add("parity", "Parity, median (IQR)", "median_iqr")
    section("Household characteristics at enrollment")
    category("crowd", "Number of people in household", [(1, "2 or fewer"), (2, "3–4"), (3, "5 or more")])
    category("household_income", "Monthly household income, South African rand",
             [(0, "<1,000"), (1, "1,000–5,000"), (2, ">5,000")])
    add("household_smoker_bin", "Any smoker in household")
    table = pd.DataFrame(rows)
    table["smd"] = table.smd.round(2)
    definitions = pd.DataFrame([
        {"item": "No TB before five", "definition": "No recorded diagnosis before age five; includes diagnoses at older ages."},
        {"item": "Infancy TB", "definition": "First recorded TB before age one; nested within first TB before five."},
        {"item": "Binary/category summaries", "definition": "Numerator / denominator (%) of defined indicators. Inherited equality-based child/birth recodes put missing source codes in the reference category; explicit recoding counts are audited. Direct baseline and categorical missing values are excluded."},
        {"item": "Mean/SD", "definition": "Arithmetic mean and sample SD (ddof=1); nonmissing n in statistics and missingness outputs."},
        {"item": "Parity", "definition": "Median [25th–75th percentile]; linear quantile interpolation; exact nonmissing n in audit."},
        {"item": "SMD", "definition": "Absolute standardized mean difference for first TB before five versus no TB before five; no hypothesis test."},
    ])
    audit = {"group_counts": {group: int(mask.sum()) for group, mask in masks.items()},
             "rows": len(table), "statistic_cells": len(stats), "small_cells_suppressed": False,
             "participant_records_exported": False,
             "inherited_missing_to_reference_recodes": {variable: int(original_missing[variable].sum())
                                                         for variable in sorted(inherited_recodes)}}
    return DescriptiveResult(table, pd.DataFrame(stats), pd.DataFrame(missing), definitions, audit)


def build_modality_schedule(pft: pd.DataFrame, *, visit_labels: Mapping | None = None,
                            modalities: Mapping | None = None) -> pd.DataFrame:
    """Count actual modality presence per nominal visit, independently of protocol.

    Presence requires a numeric recorded outcome and a valid modality-specific
    measurement date. Numeric visit codes require their explicit source label
    mapping; they are never guessed from code order or from exact testing age.
    Counts are source records, not unique tests inferred from duplicate rows.
    """
    specs = MODALITIES if modalities is None else modalities
    require_columns(pft, [ID], "Pulmonary function")
    if pft[ID].isna().any() or pft[ID].astype("string").str.strip().eq("").any():
        raise CohortError("Modality schedule requires complete child identifiers.")
    if "visit_label" in pft:
        labels = pft.visit_label.astype("string")
        all_visits = list(dict.fromkeys(labels.dropna().tolist()))
    else:
        require_columns(pft, ["visit"], "Pulmonary function")
        if visit_labels is not None:
            labels = pft.visit.map(visit_labels).astype("string")
            all_visits = list(dict.fromkeys(str(x) for x in visit_labels.values()))
        elif isinstance(pft.visit.dtype, pd.CategoricalDtype) and all(isinstance(x, str) for x in pft.visit.cat.categories):
            labels = pft.visit.astype("string")
            all_visits = list(pft.visit.cat.categories)
        elif pft.visit.dropna().map(lambda x: isinstance(x, str)).all():
            labels = pft.visit.astype("string")
            all_visits = list(dict.fromkeys(labels.dropna().tolist()))
        else:
            raise CohortError("Numeric visit codes require an explicit source label mapping.")
    if labels.isna().any() or labels.str.strip().eq("").any():
        raise CohortError("Every modality-schedule record must have a mapped nominal visit.")
    ids = pft[ID].astype("string").str.strip()
    rows = []
    for modality, spec in specs.items():
        columns, date_column = spec["columns"], spec["date"]
        if not columns:
            raise CohortError("Each modality requires at least one measurement field.")
        require_columns(pft, [*columns, date_column], f"Modality {modality}")
        values = pd.concat([numeric(pft[column], column) for column in columns], axis=1)
        dates = pd.to_datetime(pft[date_column], errors="coerce")
        if (pft[date_column].notna() & dates.isna()).any():
            raise CohortError(f"{date_column} contains an invalid date.")
        has_value = values.notna().any(axis=1)
        present = has_value & dates.notna()
        for visit in all_visits:
            mask = labels.eq(visit)
            eligible = mask & present
            rows.append({"modality": modality, "visit": visit, "source_records": int(mask.sum()),
                         "records_with_numeric_value": int((mask & has_value).sum()),
                         "dated_measurements": int(eligible.sum()),
                         "children_with_measurement": int(ids[eligible].nunique()),
                         "missing_date_with_value": int((mask & has_value & dates.isna()).sum()),
                         "represented": bool(eligible.any()), "date_field": date_column,
                         "value_fields": ";".join(columns)})
    return pd.DataFrame(rows)
