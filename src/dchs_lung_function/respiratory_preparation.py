"""Prepare wheeze processes and FeNO model samples without fitting models."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import patsy
from .inputs import normalize_dates, normalize_identifiers

ID = "pid_child"
YEAR_DAYS = 365.25
MAX_AGE = 10.5
EFFECT_BREAKS = np.array([0., 1., 3., 5., MAX_AGE])
EFFECT_LABELS = ["0-<1 year", "1-<3 years", "3-<5 years", "5-10.5 years"]
BASE_BREAKS = np.array([0., .25, .5, 1., 2., 3., 5., 8., MAX_AGE])
BASE_LABELS = ["0-<0.25", "0.25-<0.5", "0.5-<1", "1-<2", "2-<3", "3-<5", "5-<8", "8-10.5"]
SCENARIOS = {
    "primary_sex": {"covariates": ["male"], "dedup_days": 30, "lag_days": 0},
    "secondary_sex_smoking_birthweight": {"covariates": ["male", "prenatal_smoking", "birthweight_z"], "dedup_days": 30, "lag_days": 0},
}
FENO_VISITS = ["12 months", "24 months", "96 months", "FeNO", "10 years"]

def clean_id(series):
    return normalize_identifiers(series)

def parse_date(series):
    return normalize_dates(series, "respiratory_date")

def numeric(series):
    try:
        values = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError):
        raise ValueError("Respiratory numeric field has an invalid value.") from None
    if np.isinf(values.to_numpy(float, na_value=np.nan)).any():
        raise ValueError("Respiratory numeric field contains infinity.")
    return values

def effect_band(age):
    return np.searchsorted(EFFECT_BREAKS[1:], np.asarray(age, float), side="right").clip(0, 3)

def base_band(age):
    return np.searchsorted(BASE_BREAKS[1:], np.asarray(age, float), side="right").clip(0, len(BASE_LABELS)-1)

def reshape_wheeze(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pieces = []
    for slot in range(1, 37):
        cols = [
            ID,
            f"redcap_event_name{slot}",
            f"visitcat{slot}",
            f"date{slot}",
            f"ageyears{slot}",
            f"wheeze_current{slot}",
        ]
        part = wide[cols].copy()
        part.columns = [ID, "visit", "visit_category", "date", "source_age", "wheeze"]
        part["slot"] = slot
        pieces.append(part)
    long = pd.concat(pieces, ignore_index=True)
    long[ID] = clean_id(long[ID])
    long["visit"] = long["visit"].astype("string").fillna("").str.strip()
    long["date"] = parse_date(long["date"])
    for col in ["visit_category", "source_age", "wheeze"]:
        long[col] = numeric(long[col])
    populated = (
        long["visit"].ne("")
        | long["date"].notna()
        | long["visit_category"].notna()
        | long["wheeze"].notna()
    )
    long = long.loc[populated].copy()
    valid_categories = set(long["visit_category"].dropna().unique())
    if not valid_categories.issubset({1.0, 2.0, 3.0}):
        raise ValueError(f"Unexpected visit categories: {valid_categories}")
    if not set(long["wheeze"].dropna().unique()).issubset({0.0, 1.0}):
        raise ValueError("Current wheeze is not binary")
    dup = long.loc[long["date"].notna()].duplicated([ID, "date"])
    if dup.any():
        raise ValueError(f"Found {int(dup.sum())} duplicate child-date records")
    cat3 = long["visit_category"].eq(3)
    if long.loc[cat3, "date"].isna().any() or not long.loc[cat3, "wheeze"].eq(1).all():
        raise ValueError("Dedicated wheeze visits must be dated and wheeze-positive")
    return long, {
        "populated_records": int(len(long)),
        "dated_records": int(long["date"].notna().sum()),
        "scheduled_records": int(long["visit_category"].eq(1).sum()),
        "pneumonia_or_sick_records": int(long["visit_category"].eq(2).sum()),
        "dedicated_wheeze_records": int(cat3.sum()),
        "dedicated_wheeze_all_positive": True,
        "duplicate_child_date_records": 0,
    }

@dataclass
class Sources:
    cohort: pd.DataFrame
    scheduled: pd.DataFrame
    candidate_episodes: pd.DataFrame
    source_audit: dict

def deduplicate_episodes(candidate: pd.DataFrame, days: int) -> tuple[pd.DataFrame, dict]:
    """Collapse chains of contacts separated by no more than the washout."""
    frame = candidate.sort_values([ID, "date"]).copy()
    gap = frame.groupby(ID, observed=True)["date"].diff().dt.days
    frame["new_episode"] = gap.isna() | gap.gt(days)
    frame["episode_number"] = frame.groupby(ID, observed=True)["new_episode"].cumsum()
    first = frame.groupby([ID, "episode_number"], observed=True, as_index=False).first()
    cluster_sizes = frame.groupby([ID, "episode_number"], observed=True).size()
    # Audit the alternative "last kept/index contact" algorithm.  It can split
    # a chain such as day 0, 20, 40 at a 30-day window, whereas the primary
    # consecutive-gap definition treats it as one continuing illness episode.
    anchor_kept = []
    for _, group in frame.groupby(ID, observed=True):
        last_kept = None
        for index, row in group.iterrows():
            if last_kept is None or (row["date"] - last_kept).days > days:
                anchor_kept.append(index)
                last_kept = row["date"]
    anchor = frame.loc[anchor_kept]
    chain_by_child = first.groupby(ID, observed=True).size()
    anchor_by_child = anchor.groupby(ID, observed=True).size()
    comparison_ids = set(chain_by_child.index) | set(anchor_by_child.index)
    audit = {
        "dedup_window_days": days,
        "primary_deduplication_rule": "new episode only when gap from immediately preceding qualifying contact is greater than the window (consecutive-gap clustering)",
        "candidate_contacts": int(len(frame)),
        "deduplicated_episodes": int(len(first)),
        "contacts_combined": int(len(frame) - len(first)),
        "children_with_episode": int(first[ID].nunique()),
        "clusters_with_multiple_contacts": int(cluster_sizes.gt(1).sum()),
        "maximum_contacts_in_cluster": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "consecutive_gaps_le_window": int(gap.le(days).sum()),
        "minimum_positive_gap_days": int(gap[gap.gt(0)].min()) if gap.gt(0).any() else None,
        "alternative_last_kept_anchor_episodes": int(len(anchor)),
        "alternative_minus_primary_episodes": int(len(anchor) - len(first)),
        "children_with_different_counts_under_alternative": int(sum(
            int(chain_by_child.get(child, 0)) != int(anchor_by_child.get(child, 0))
            for child in comparison_ids
        )),
    }
    return first[[ID, "date", "age", "episode_source"]].copy(), audit

def scheduled_for_scenario(sources: Sources, children: pd.DataFrame, lag_days: int) -> tuple[pd.DataFrame, dict]:
    frame = sources.scheduled.loc[sources.scheduled[ID].isin(set(children[ID]))].copy()
    child_info = children.set_index(ID)
    for column in ["censor_date", "tb_date", "tb_event", "tb_age"]:
        frame[column] = frame[ID].map(child_info[column])
    frame = frame.loc[frame["date"].lt(frame["censor_date"])].copy()
    lag_end = frame["tb_date"] + pd.to_timedelta(lag_days, unit="D")
    in_lag = (
        frame["tb_event"].eq(1)
        & frame["date"].ge(frame["tb_date"])
        & frame["date"].lt(lag_end)
    ) if lag_days else pd.Series(False, index=frame.index)
    removed_positive = int(frame.loc[in_lag, "wheeze"].sum())
    frame = frame.loc[~in_lag].copy()
    post_start = frame["tb_date"] + pd.to_timedelta(lag_days, unit="D")
    frame["post_tb"] = (
        frame["tb_event"].eq(1) & frame["date"].ge(post_start)
    ).astype(int)
    frame["effect_band"] = effect_band(frame["age"].to_numpy(float)).astype(int)
    audit = {
        "records": int(len(frame)),
        "children": int(frame[ID].nunique()),
        "positive_records": int(frame["wheeze"].sum()),
        "lag_window_days": lag_days,
        "records_excluded_in_post_tb_lag": int(in_lag.sum()),
        "positive_records_excluded_in_post_tb_lag": removed_positive,
        "post_tb_records": int(frame["post_tb"].sum()),
        "post_tb_positive_records": int(frame.loc[frame["post_tb"].eq(1), "wheeze"].sum()),
        "children_with_post_tb_records": int(frame.loc[frame["post_tb"].eq(1), ID].nunique()),
    }
    return frame.sort_values([ID, "date"]).reset_index(drop=True), audit

def build_episode_intervals(
    children: pd.DataFrame,
    episodes: pd.DataFrame,
    lag_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split counting-process time at exact TB, lag end, and age breakpoints."""
    episodes = episodes.loc[episodes[ID].isin(set(children[ID]))].copy()
    info = children.set_index(ID)
    episodes = episodes.join(info[["tb_event", "tb_date"]], on=ID)
    lag_end_date = episodes["tb_date"] + pd.to_timedelta(lag_days, unit="D")
    event_in_lag = (
        episodes["tb_event"].eq(1)
        & episodes["date"].ge(episodes["tb_date"])
        & episodes["date"].lt(lag_end_date)
    ) if lag_days else pd.Series(False, index=episodes.index)
    excluded_events = episodes.loc[event_in_lag].copy()
    episodes = episodes.loc[~event_in_lag].copy()

    rows = []
    for child_idx, row in children.reset_index(drop=True).iterrows():
        stop = float(row["censor_age"])
        if not np.isfinite(stop) or stop <= 0:
            continue
        tb_age = float(row["tb_age"]) if int(row["tb_event"]) else math.inf
        post_start_age = tb_age + lag_days / YEAR_DAYS if np.isfinite(tb_age) else math.inf
        boundaries = [0.0, stop]
        boundaries.extend(float(x) for x in BASE_BREAKS[1:-1] if 0 < x < stop)
        if 0 < tb_age < stop:
            boundaries.append(tb_age)
        if lag_days and 0 < post_start_age < stop:
            boundaries.append(post_start_age)
        bounds = np.unique(np.asarray(boundaries, dtype=float))
        child_events = episodes.loc[episodes[ID].eq(row[ID]), "age"].to_numpy(float)
        for left, right in zip(bounds[:-1], bounds[1:]):
            if right <= left:
                continue
            mid = (left + right) / 2.0
            in_omitted_lag = lag_days > 0 and tb_age <= mid < post_start_age
            if in_omitted_lag:
                continue
            count = int(((child_events >= left) & (child_events < right)).sum())
            post = int(np.isfinite(post_start_age) and left >= post_start_age - 1e-10)
            rows.append({
                "child_index": child_idx,
                ID: row[ID],
                "start": left,
                "stop": right,
                "person_years": right - left,
                "events": count,
                "post_tb": post,
                "effect_band": int(effect_band(mid)),
                "base_band": int(base_band(mid)),
            })
    intervals = pd.DataFrame(rows)
    if int(intervals["events"].sum()) != len(episodes):
        raise AssertionError("Episode interval counts do not reconcile to retained episodes")
    lag_py = 0.0
    if lag_days:
        for _, row in children.loc[children["tb_event"].eq(1)].iterrows():
            lag_py += max(0.0, min(float(row["censor_age"]), float(row["tb_age"]) + lag_days / YEAR_DAYS) - float(row["tb_age"]))
    audit = {
        "retained_deduplicated_episodes": int(len(episodes)),
        "children_with_retained_episode": int(episodes[ID].nunique()),
        "episode_process_rows": int(len(intervals)),
        "person_years": float(intervals["person_years"].sum()),
        "post_tb_person_years": float(intervals.loc[intervals["post_tb"].eq(1), "person_years"].sum()),
        "post_tb_events": int(intervals.loc[intervals["post_tb"].eq(1), "events"].sum()),
        "children_with_post_tb_person_time": int(intervals.loc[intervals["post_tb"].eq(1), ID].nunique()),
        "events_excluded_in_post_tb_lag": int(len(excluded_events)),
        "children_with_event_excluded_in_lag": int(excluded_events[ID].nunique()),
        "person_years_excluded_in_post_tb_lag": lag_py,
    }
    return intervals, episodes, audit

def build_survival_arrays(children: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(children)
    m = len(BASE_LABELS)
    exposure = np.zeros((n, m), dtype=float)
    full_exposure = np.zeros((n, m), dtype=float)
    event_interval = np.zeros((n, m), dtype=float)
    for i, row in children.reset_index(drop=True).iterrows():
        censor = float(row["censor_age"])
        stop = float(row["tb_age"]) if int(row["tb_event"]) else censor
        for j, (left, right) in enumerate(zip(BASE_BREAKS[:-1], BASE_BREAKS[1:])):
            exposure[i, j] = max(0.0, min(stop, right) - left)
            full_exposure[i, j] = max(0.0, min(censor, right) - left)
            if int(row["tb_event"]) and left <= stop < right:
                event_interval[i, j] = 1.0
    if int(event_interval.sum()) != int(children["tb_event"].sum()):
        raise AssertionError("TB event indicators do not reconcile")
    return {
        "exposure": exposure,
        "full_exposure": full_exposure,
        "event_interval": event_interval,
    }

def make_spline(age: np.ndarray) -> tuple[np.ndarray, dict]:
    formula = "0 + cr(age, df=5, lower_bound=0, upper_bound=10.5, constraints='center')"
    raw = np.asarray(patsy.dmatrix(formula, {"age": age}, return_type="dataframe"), dtype=float)
    center = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return (raw - center) / scale, {
        "formula": formula,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "columns": int(raw.shape[1]),
    }

@dataclass
class Prepared:
    scenario: str
    covariates: list[str]
    children: pd.DataFrame
    scheduled: pd.DataFrame
    episodes: pd.DataFrame
    episode_intervals: pd.DataFrame
    spline: np.ndarray
    survival: dict[str, np.ndarray]
    audit: dict

def prepare_scenario(sources: Sources, scenario: str) -> Prepared:
    spec = SCENARIOS[scenario]
    covariates = list(spec["covariates"])
    children = sources.cohort.dropna(subset=covariates).copy().sort_values(ID).reset_index(drop=True)
    admin_override = spec.get("admin_censor_exclusive")
    if admin_override is not None:
        children["censor_date"] = children["censor_date"].where(
            children["censor_date"].le(admin_override), admin_override
        )
        children["censor_age"] = (children["censor_date"] - children["dob"]).dt.days / YEAR_DAYS
        children["tb_event"] = (
            children["tb_date"].notna()
            & children["tb_date"].ge(children["dob"])
            & children["tb_date"].lt(children["censor_date"])
            & children["raw_tb_age"].lt(MAX_AGE)
        ).astype(int)
        children["tb_age"] = np.where(children["tb_event"].eq(1), children["raw_tb_age"], np.nan)
    children["child_index"] = np.arange(len(children))
    scheduled, scheduled_audit = scheduled_for_scenario(sources, children, spec["lag_days"])
    child_index = children.set_index(ID)["child_index"]
    scheduled["child_index"] = scheduled[ID].map(child_index).astype(int)
    spline, spline_meta = make_spline(scheduled["age"].to_numpy(float))

    candidate = sources.candidate_episodes.loc[
        sources.candidate_episodes[ID].isin(set(children[ID]))
    ].copy()
    censor_map = children.set_index(ID)["censor_date"]
    candidate = candidate.loc[candidate["date"].lt(candidate[ID].map(censor_map))].copy()
    dedup, dedup_audit = deduplicate_episodes(candidate, spec["dedup_days"])
    intervals, episodes, episode_audit = build_episode_intervals(
        children, dedup, spec["lag_days"]
    )
    survival = build_survival_arrays(children)
    audit = {
        "scenario": scenario,
        "covariates": covariates,
        "children": int(len(children)),
        "children_excluded_for_covariate_missingness": int(len(sources.cohort) - len(children)),
        "tb_events": int(children["tb_event"].sum()),
        "administrative_censor": sources.source_audit["administrative_end"],
        "scheduled": scheduled_audit,
        "deduplication": dedup_audit,
        "episode_process": episode_audit,
        "age_spline": spline_meta,
    }
    return Prepared(
        scenario=scenario,
        covariates=covariates,
        children=children,
        scheduled=scheduled,
        episodes=episodes,
        episode_intervals=intervals,
        spline=spline,
        survival=survival,
        audit=audit,
    )


def build_respiratory_sources(baseline, tb_dates, wheeze, cohort_ids, *, administrative_end):
    """Construct the fixed cohort and observed scheduled/unscheduled processes."""
    base = baseline.copy()
    dates = tb_dates.copy()
    wide = wheeze.copy()
    for frame in (base, dates, wide):
        frame[ID] = clean_id(frame[ID])
        if frame[ID].duplicated().any():
            raise ValueError("Respiratory child-level source has duplicate keys.")
    if set(base[ID]) != set(dates[ID]):
        raise ValueError("Baseline and respiratory TB child sets differ.")
    ids = set(clean_id(pd.Series(list(cohort_ids))))
    if not ids or not ids.issubset(set(base[ID])) or not ids.issubset(set(wide[ID])):
        raise ValueError("The respiratory cohort is absent from a required source.")
    if not set(wide[ID]).issubset(set(base[ID])):
        raise ValueError("Wheeze source includes children absent from baseline.")
    admin = parse_date(pd.Series([administrative_end])).iloc[0]
    if pd.isna(admin):
        raise ValueError("Respiratory administrative endpoint is required.")
    fields = [ID, "childdob", "etq1", "sex", "Prenatal_smoking_composite", "birth_weight"]
    c = base.loc[base[ID].isin(ids), fields].copy()
    dates["tb_date"] = parse_date(dates["date_tb_diagnosed"])
    c = c.merge(dates[[ID, "tb_date"]], on=ID, how="left", validate="one_to_one")
    c["dob"] = parse_date(c.childdob)
    c["exit_date"] = parse_date(c.etq1)
    if c.dob.isna().any():
        raise ValueError("Respiratory cohort has missing dates of birth.")
    c["age_10_5_date"] = c.dob.map(lambda value: value + pd.DateOffset(years=10, months=6))
    c["admin_date"] = admin
    c["censor_date"] = c[["age_10_5_date", "admin_date", "exit_date"]].min(axis=1)
    c["censor_age"] = (c.censor_date-c.dob).dt.days / YEAR_DAYS
    c["raw_tb_age"] = (c.tb_date-c.dob).dt.days / YEAR_DAYS
    if (c.tb_date.notna() & c.tb_date.lt(c.dob)).any():
        raise ValueError("Respiratory TB date precedes birth.")
    if (~np.isfinite(c.censor_age) | c.censor_age.le(0)).any():
        raise ValueError("Respiratory follow-up must be finite and positive.")
    c["tb_event"] = (c.tb_date.notna() & c.tb_date.ge(c.dob)
                     & c.tb_date.lt(c.censor_date) & c.raw_tb_age.lt(MAX_AGE)).astype(int)
    c["tb_age"] = np.where(c.tb_event.eq(1), c.raw_tb_age, np.nan)
    sex = numeric(c.sex)
    if not sex.dropna().isin([1, 2]).all():
        raise ValueError("Baseline sex encoding is outside the declared categories.")
    c["male"] = np.where(sex.eq(2), 1., np.where(sex.eq(1), 0., np.nan))
    c["prenatal_smoking"] = numeric(c.Prenatal_smoking_composite)
    c["birthweight_g"] = numeric(c.birth_weight)
    mean = float(c.birthweight_g.mean())
    sd = float(c.birthweight_g.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Birthweight standardization has no finite positive scale.")
    c["birthweight_z"] = (c.birthweight_g-mean)/sd
    c = c.sort_values(ID).reset_index(drop=True)
    long, reshape_audit = reshape_wheeze(wide)
    info = [ID, "dob", "censor_date", "tb_date", "tb_event", "tb_age", "male", "prenatal_smoking", "birthweight_z"]
    long = long.merge(c[info], on=ID, how="inner", validate="many_to_one")
    long["age"] = (long.date-long.dob).dt.days/YEAR_DAYS
    valid = (long.date.notna() & long.date.ge(long.dob) & long.date.lt(long.censor_date)
             & long.age.ge(0) & long.age.lt(MAX_AGE))
    scheduled = long.loc[valid & long.visit_category.eq(1) & long.wheeze.notna()].copy()
    scheduled["wheeze"] = scheduled.wheeze.astype(int)
    candidates = long.loc[valid & (long.visit_category.eq(3)
                           | (long.visit_category.eq(2) & long.wheeze.eq(1)))].copy()
    candidates["episode_source"] = np.where(candidates.visit_category.eq(3),
        "dedicated wheeze visit", "sick/pneumonia visit with wheeze")
    audit = {"cohort_children": len(c), "administrative_end": str(admin.date()),
             "reshape": reshape_audit, "first_tb_events": int(c.tb_event.sum()),
             "scheduled_observations": len(scheduled), "scheduled_positive": int(scheduled.wheeze.sum()),
             "candidate_episode_contacts": len(candidates),
             "birthweight_standardization": {"mean_g": mean, "sd_g": sd},
             "missing_covariates": {key:int(c[key].isna().sum()) for key in ["male", "prenatal_smoking", "birthweight_z"]}}
    return Sources(c, scheduled, candidates, audit)


def _feno_records(pft, visits):
    columns = [ID, "child_dob", "date_of_test_ilf1", "feno_result", "height", "tb_first_dx", "male", "visit_label"]
    missing = set(columns)-set(pft.columns)
    if missing:
        raise ValueError("Prepared PFT lacks required FeNO fields.")
    d = pft[columns].copy()
    d[ID] = clean_id(d[ID])
    d["date"] = parse_date(d.date_of_test_ilf1)
    d["feno"] = numeric(d.feno_result)
    d["height"] = numeric(d.height)
    d = d.loc[d.visit_label.isin(visits) & d.feno.gt(0) & d.date.notna()].copy()
    duplicate = d.duplicated([ID, "date"], keep=False)
    conflicts = int(d.loc[duplicate].groupby([ID, "date"]).feno.nunique().gt(1).sum())
    before = len(d)
    d["priority"] = d.visit_label.eq("FeNO").astype(int)
    d = d.sort_values([ID, "date", "priority"], ascending=[True, True, False]).drop_duplicates([ID, "date"]).copy()
    d["log_feno"] = np.log(d.feno)
    return d, {"duplicate_rows_removed": before-len(d), "conflicting_duplicate_groups": conflicts}


def _feno_design(frame, terms, standardize):
    f = frame.dropna(subset=[ID, "log_feno"]+terms).copy()
    x = pd.DataFrame({"Intercept": np.ones(len(f))}, index=f.index)
    for term in terms:
        x[term] = f[term].astype(float)
        if term in standardize:
            sd = x[term].std(ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                raise ValueError("FeNO adjustment has no finite positive scale.")
            x[term] = (x[term]-x[term].mean())/sd
    if not len(f) or np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise ValueError("FeNO design is empty or rank deficient.")
    return f, x


def prepare_main_feno(pft, respiratory_cohort):
    """Prepare the five selected main FeNO samples and Gaussian-GEE matrices."""
    # Cohort selection precedes duplicate handling in the selected analysis.
    selected = pft.loc[pft[ID].isin(set(respiratory_cohort[ID]))]
    d, audit = _feno_records(selected, FENO_VISITS)
    d = d.drop(columns=["male"]).merge(respiratory_cohort[[ID, "dob", "censor_date", "tb_date", "tb_age", "male"]],
                                     on=ID, how="inner", validate="many_to_one")
    d["age"] = (d.date-d.dob).dt.days/YEAR_DAYS
    d["infancy_post"] = (d.tb_age.lt(1) & d.date.ge(d.tb_date)).astype(int)
    d["later_post"] = (d.tb_age.ge(1) & d.date.ge(d.tb_date)).astype(int)
    samples = {}
    for visit in FENO_VISITS:
        if visit == "12 months":
            keep = d.visit_label.eq(visit) & d.date.lt(d.censor_date) & (~d.tb_age.lt(1) | d.infancy_post.eq(1))
        else:
            keep = d.visit_label.eq(visit) & d.age.ge(1) & d.age.lt(MAX_AGE) & d.date.lt(d.censor_date)
        f = d.loc[keep].copy()
        terms = ["infancy_post"]+(["later_post"] if f.later_post.sum() else [])+["male", "age", "height"]
        f, x = _feno_design(f, terms, {"age", "height"})
        samples[visit] = {"frame": f, "design": x, "response": f.log_feno.to_numpy(float),
                          "target": "infancy_post", "audit": {"observations":len(f), "children":int(f[ID].nunique()),
                          "infancy_post":int(f.infancy_post.sum()), "later_post":int(f.later_post.sum()), "terms":list(x.columns)}}
    return {"samples": samples, "audit": {**audit, "visits":{k:v["audit"] for k,v in samples.items()}}}


def prepare_supplementary_feno(pft):
    """Keep the supplementary future/post-TB contrast and within-visit scaling."""
    visits = ["6-10 weeks"]+FENO_VISITS
    d, audit = _feno_records(pft, visits)
    d["age_years"] = (d.date-parse_date(d.child_dob)).dt.days/YEAR_DAYS
    tb = parse_date(d.tb_first_dx)
    d["tb_before_measurement"] = (tb.notna() & d.date.ge(tb)).astype(float)
    d["tb_after_measurement"] = (tb.notna() & d.date.lt(tb)).astype(float)
    samples = {}
    for visit in visits:
        f = d.loc[d.visit_label.eq(visit)].copy()
        # The reference specification scales before complete-case selection.
        for col, out in [("age_years", "age_z"), ("height", "height_z")]:
            sd = f[col].std(ddof=1)
            f[out] = (f[col]-f[col].mean())/sd if np.isfinite(sd) and sd > 0 else np.nan
        terms = [col for col in ["tb_before_measurement", "tb_after_measurement"] if f[col].sum()>0]
        terms += ["age_z", "height_z", "male"]
        f, x = _feno_design(f, terms, set())
        target = ("tb_before_measurement" if f.tb_before_measurement.sum()>0
                  else "tb_after_measurement" if f.tb_after_measurement.sum()>0 else None)
        status = "ready" if target is not None else "no_exposure_contrast"
        samples[visit] = {"frame":f, "design":x, "response":f.log_feno.to_numpy(float), "target":target,
                          "audit":{"observations":len(f), "children":int(f[ID].nunique()),
                          "tb_before":int(f.tb_before_measurement.sum()), "tb_after":int(f.tb_after_measurement.sum()),
                          "terms":list(x.columns), "status":status}}
    return {"samples":samples, "audit":{**audit, "visits":{k:v["audit"] for k,v in samples.items()}}}


def prepare_respiratory_inputs(pft, baseline, tb_dates, wheeze, cohort_ids, *, administrative_end):
    sources = build_respiratory_sources(baseline, tb_dates, wheeze, cohort_ids, administrative_end=administrative_end)
    wheeze_samples = {name:prepare_scenario(sources, name) for name in SCENARIOS}
    main = prepare_main_feno(pft, sources.cohort)
    supplement = prepare_supplementary_feno(pft)
    return {"wheeze":wheeze_samples, "main_feno":main, "supplementary_feno":supplement,
            "audit":{"sources":sources.source_audit, "wheeze":{k:v.audit for k,v in wheeze_samples.items()},
                     "main_feno":main["audit"], "supplementary_feno":supplement["audit"]}}
