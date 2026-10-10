"""Read authorized source extracts with explicit schemas and aggregate audits.

Readers keep every source row and column. Stata category codes are never silently
replaced by labels; labels are retained as metadata for explicit downstream use.
Errors contain neutral roles and field names, never source paths or row values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
import re
import struct
from typing import Any, Mapping

import numpy as np
import pandas as pd


class InputValidationError(ValueError):
    """An input failed a source-format or schema rule."""


@dataclass
class InputTable:
    """Private in-memory records and metadata; no automatic disk export."""

    data: pd.DataFrame
    metadata: dict[str, Any]


BASELINE_FIELDS = (
    "pid_child childdob etq1 sex maternal_hiv Prenatal_smoking_composite "
    "birth_weight lrti maternal_age_birth education Prenatal_alcohol_composite "
    "parity crowd household_income smoke_hh lowbirthweight TB tbcat tbtrt "
    "date_tb_diagnosed date_tb_investigated age_tb TB2 tbcat2 tbtrt2 "
    "date_tb_diagnosed2 timeTBneg timeTBpos survivaltb survival_tb_new "
    "survivaltb_final agetb_final"
).split()
PFT_FIELDS = (
    "pid_child visit child_dob sex height date_of_test date_of_test_ilf1 "
    "date_of_test_ilf2 fev1 fvc fev1_fvc fev1_post fvc_post fev1_fvc_post "
    "mmef mmef_post resistance compliance ree rei xee xei i_lci_2_5 feno_result "
    "reactance_6hz reactance_8hz reactance_10hz resistance_6hz resistance_8hz "
    "resistance_10hz fres_final r6_r20 ax_mean dr_mean dx_mean hiv_exposed "
    "hiv_infected site ethnicity preterm gestation_delivery birth_weight"
).split()
WHEEZE_FIELDS = ["pid_child"] + [
    f"{stem}{slot}" for slot in range(1, 37)
    for stem in ("redcap_event_name", "visitcat", "date", "ageyears", "wheeze_current")
]
SCHEMAS = {
    "baseline": BASELINE_FIELDS,
    "tb_events": "pid_child date_tb_diagnosed date_tb_investigated childdob tbcat tbtrt".split(),
    "pulmonary_function": PFT_FIELDS,
    "respiratory_tb_dates": ["pid_child", "date_tb_diagnosed"],
    "wheeze_assessments": WHEEZE_FIELDS,
    "wheeze_csv_check": WHEEZE_FIELDS,
}
REQUIRED_TABLES = frozenset(SCHEMAS) - {"wheeze_csv_check"}
FILE_ONLY_ROLES = {"gli_lookup", "pft_dictionary", "wheeze_dictionary"}
DATE_FIELDS = {
    "baseline": ["childdob", "etq1", "date_tb_diagnosed", "date_tb_investigated", "date_tb_diagnosed2"],
    "tb_events": ["childdob", "date_tb_diagnosed", "date_tb_investigated"],
    "pulmonary_function": ["child_dob", "date_of_test", "date_of_test_ilf1", "date_of_test_ilf2"],
    "respiratory_tb_dates": ["date_tb_diagnosed"],
    "wheeze_assessments": [f"date{i}" for i in range(1, 37)],
    "wheeze_csv_check": [f"date{i}" for i in range(1, 37)],
}
UNIQUE_CHILD_ROLES = {"baseline", "respiratory_tb_dates", "wheeze_assessments", "wheeze_csv_check"}
VISIT_CATEGORY_LABELS = {"Scheduled visit": 1, "Pneumonia episode": 2, "Wheeze episode": 3}
TB_EVENT_CATEGORIES = frozenset({"Confirmed tb", "Unconfirmed tb"})
MONTHS = {name: str(i).zfill(2) for i, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def normalize_identifiers(series: pd.Series, field: str = "pid_child") -> pd.Series:
    """Trim string keys; retain leading zeros and reject ambiguous/missing keys.

    Integral numeric keys are accepted losslessly, except floats outside their
    exact integer range. A mixed numeric/string source is rejected because its
    representation cannot establish whether a leading zero was lost upstream.
    """
    if series.isna().any():
        raise InputValidationError(f"{field}: missing identifiers are not allowed.")
    numeric = series.map(lambda value: isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)))
    strings = series.map(lambda value: isinstance(value, str))
    if numeric.all() and len(series):
        values = pd.to_numeric(series, errors="coerce")
        if (not np.isfinite(values.to_numpy(dtype=float)).all()
                or not values.eq(np.floor(values)).all()
                or (pd.api.types.is_float_dtype(values) and values.abs().gt(2**53).any())):
            raise InputValidationError(f"{field}: numeric identifiers must be exact finite integers.")
        result = values.map(lambda value: str(int(value))).astype("string")
    elif strings.all():
        original = series.astype("string")
        result = original.str.strip()
        if result.eq("").any() or result.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
            raise InputValidationError(f"{field}: blank or control-character identifiers are not allowed.")
        if pd.DataFrame({"raw": original, "key": result}).drop_duplicates().duplicated("key").any():
            raise InputValidationError(f"{field}: identifier normalization creates a collision.")
    else:
        raise InputValidationError(f"{field}: mixed or unsupported identifier encoding.")
    result.name = series.name
    result.index = series.index
    return result


def normalize_dates(series: pd.Series, field: str, *, allow_missing: bool = True) -> pd.Series:
    """Parse date-only ISO or English Stata-export dates without guessing.

    Declared Stata dates arrive as datetime values. Undeclared numeric dates,
    ambiguous slash dates, time zones and non-midnight timestamps are rejected.
    Empty strings are missing; nonempty invalid strings cannot become missing.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise InputValidationError(f"{field}: timezone dates are not supported.")
    if pd.api.types.is_datetime64_any_dtype(series):
        result = series.astype("datetime64[ns]")
    else:
        values = series.copy()
        values = values.map(lambda value: pd.NA if isinstance(value, str) and not value.strip() else value)
        present = values.dropna()
        if present.empty:
            result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]", name=series.name)
        elif present.map(lambda value: isinstance(value, (date, datetime, pd.Timestamp))).all():
            try:
                result = pd.to_datetime(values, errors="raise")
            except (ValueError, TypeError, OverflowError):
                raise InputValidationError(f"{field}: invalid date encoding.") from None
        elif present.map(lambda value: isinstance(value, str)).all():
            text = values.astype("string").str.strip()
            iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
            stata = text.str.fullmatch(r"\d{2}[A-Za-z]{3}\d{4}", na=False)
            if (~(iso | stata) & text.notna()).any():
                raise InputValidationError(f"{field}: unsupported or ambiguous date encoding.")
            if stata.any():
                selected = text.loc[stata]
                months = selected.str[2:5].str.lower().map(MONTHS)
                if months.isna().any():
                    raise InputValidationError(f"{field}: invalid calendar month.")
                text.loc[stata] = selected.str[5:9] + "-" + months + "-" + selected.str[:2]
            try:
                result = pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
            except (ValueError, TypeError, OverflowError):
                raise InputValidationError(f"{field}: invalid calendar date.") from None
        else:
            raise InputValidationError(f"{field}: undeclared numeric or mixed date encoding.")
    if isinstance(result.dtype, pd.DatetimeTZDtype):
        raise InputValidationError(f"{field}: timezone dates are not supported.")
    if not result.dropna().eq(result.dropna().dt.normalize()).all():
        raise InputValidationError(f"{field}: time-of-day is not supported.")
    if not allow_missing and result.isna().any():
        raise InputValidationError(f"{field}: missing dates are not allowed.")
    return result.astype("datetime64[ns]")


def _number(series: pd.Series, field: str) -> pd.Series:
    value = series.replace(r"^\s*$", pd.NA, regex=True)
    try:
        numeric = pd.to_numeric(value, errors="raise")
    except (ValueError, TypeError):
        raise InputValidationError(f"{field}: nonnumeric values are not allowed.") from None
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise InputValidationError(f"{field}: infinite values are not allowed.")
    return numeric


def _file_digest(path: Path, role: str, expected_sha256: str | None) -> tuple[str, int]:
    if expected_sha256 is not None and (not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None):
        raise InputValidationError(f"{role}: SHA256 must have 64 hexadecimal characters.")
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        size = path.stat().st_size
    except (OSError, ValueError):
        raise InputValidationError(f"{role}: source is not a readable file.") from None
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256.lower():
        raise InputValidationError(f"{role}: source SHA256 does not match its binding.")
    return digest.hexdigest(), size


def _read_csv(path: Path, role: str) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = csv.reader(handle, strict=True)
            header = next(records)
            if any(len(row) != len(header) for row in records):
                raise InputValidationError(f"{role}: CSV rows do not match the header width.")
        serialized_index = bool(header and header[0] == "" and role == "respiratory_tb_dates")
        if serialized_index:
            # This source export contains a pandas row-index column. Preserve it
            # with an explicit name and validate it; do not drop it silently.
            header[0] = "Unnamed: 0"
        if not header or len(set(header)) != len(header) or any(not col or col != col.strip() for col in header):
            raise InputValidationError(f"{role}: CSV header has empty, duplicate or padded columns.")
        # Preserve IDs (including leading zeros) and empty strings without the
        # broad default NA vocabulary. Other numeric fields are converted below.
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False,
                            names=header, header=0, na_values=[""], on_bad_lines="error")
        if role == "respiratory_tb_dates" and "date_tb_diagnosed" in frame:
            # This CSV export uses the literal R missing token only here in
            # the required date field. Do not apply an NA vocabulary to IDs.
            frame["date_tb_diagnosed"] = frame["date_tb_diagnosed"].mask(
                frame["date_tb_diagnosed"].eq("NA"), pd.NA)
            frame.attrs["date_missing_tokens"] = ["", "NA"]
        if serialized_index:
            index = _number(frame["Unnamed: 0"], "serialized_source_index")
            starts = [start for start in (0, 1)
                      if np.array_equal(index.to_numpy(), np.arange(start, start + len(frame)))]
            if not starts:
                raise InputValidationError(f"{role}: serialized row index must be a consecutive zero- or one-based sequence.")
            frame.attrs["serialized_index_column"] = "Unnamed: 0"
            frame.attrs["serialized_index_start"] = starts[0]
        return frame
    except InputValidationError:
        raise
    except (OSError, UnicodeError, ValueError, StopIteration, csv.Error):
        raise InputValidationError(f"{role}: unreadable UTF-8 CSV or invalid record structure.") from None


def _wheeze_checks(frame: pd.DataFrame, role: str) -> dict[str, int]:
    pieces = []
    for slot in range(1, 37):
        cat = f"visitcat{slot}"
        if pd.api.types.is_string_dtype(frame[cat]) or frame[cat].dtype == object:
            source = frame[cat].astype("string").str.strip()
            labels = source.isin(VISIT_CATEGORY_LABELS)
            source.loc[labels] = source.loc[labels].map(VISIT_CATEGORY_LABELS).astype("string")
            frame[cat] = source
        for stem in ("visitcat", "wheeze_current", "ageyears", "agemonths", "agedays"):
            column = f"{stem}{slot}"
            if column in frame:
                frame[column] = _number(frame[column], column)
        if not frame[cat].dropna().isin([1, 2, 3]).all():
            raise InputValidationError(f"{role}: unexpected visit-category code.")
        if not frame[f"wheeze_current{slot}"].dropna().isin([0, 1]).all():
            raise InputValidationError(f"{role}: current wheeze must be binary.")
        dedicated = frame[cat].eq(3).fillna(False)
        if (frame.loc[dedicated, f"date{slot}"].isna().any()
                or not frame.loc[dedicated, f"wheeze_current{slot}"].eq(1).fillna(False).all()):
            raise InputValidationError(f"{role}: dedicated wheeze visits must be dated and positive.")
        pieces.append(pd.DataFrame({"pid_child": frame.pid_child, "date": frame[f"date{slot}"]}))
    if "count_wheeze" in frame:
        frame["count_wheeze"] = _number(frame["count_wheeze"], "count_wheeze")
    dates = pd.concat(pieces, ignore_index=True).dropna(subset=["date"])
    duplicates = int(dates.duplicated(["pid_child", "date"]).sum())
    if duplicates:
        raise InputValidationError(f"{role}: {duplicates} duplicate child/date assessments.")
    return {"dated_assessments": len(dates), "duplicate_child_dates": duplicates}


def read_input(path: str | Path, input_id: str, expected_sha256: str | None = None) -> InputTable:
    """Read one neutral input role, validate it, and preserve row order/count.

    Only registered tabular roles are accepted. Computational GLI data and
    dictionaries use their dedicated readers/file checks in the orchestration.
    """
    if input_id not in SCHEMAS:
        raise InputValidationError("Unknown tabular input role.")
    path = Path(path)
    suffix = path.suffix.lower()
    allowed = {".csv"} if input_id in {"respiratory_tb_dates", "wheeze_csv_check"} else {".dta"}
    if suffix not in allowed:
        raise InputValidationError(f"{input_id}: unsupported source format.")
    digest, size = _file_digest(path, input_id, expected_sha256)
    metadata: dict[str, Any] = {"role": input_id, "sha256": digest, "bytes": size,
                                "format": suffix[1:], "value_labels": {}, "variable_labels": {},
                                "variable_label_sets": {}}
    if suffix == ".dta":
        try:
            with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
                metadata["variable_labels"] = reader.variable_labels()
                metadata["value_labels"] = {name: {int(key): value for key, value in labels.items()}
                                              for name, labels in reader.value_labels().items()}
                # pandas exposes variable-to-label assignments only through
                # these StataReader metadata attributes; covered by fixtures.
                metadata["variable_label_sets"] = dict(zip(reader._varlist, reader._lbllist))
                frame = reader.read()
        except (OSError, UnicodeError, ValueError, KeyError, EOFError, struct.error, StopIteration):
            raise InputValidationError(f"{input_id}: invalid Stata source.") from None
    else:
        frame = _read_csv(path, input_id)
        metadata["serialized_index_column"] = frame.attrs.get("serialized_index_column")
        metadata["serialized_index_start"] = frame.attrs.get("serialized_index_start")
        metadata["date_missing_tokens"] = frame.attrs.get("date_missing_tokens", [""])
    if frame.empty:
        raise InputValidationError(f"{input_id}: empty study sources are not allowed.")
    if frame.columns.duplicated().any():
        raise InputValidationError(f"{input_id}: duplicate source columns.")
    missing = sorted(set(SCHEMAS[input_id]) - set(frame.columns))
    if missing:
        raise InputValidationError(f"{input_id}: missing {len(missing)} required fields: {', '.join(missing)}.")
    source_rows, source_columns = frame.shape
    source_index = frame.index.copy()
    frame["pid_child"] = normalize_identifiers(frame["pid_child"])
    if input_id in UNIQUE_CHILD_ROLES and frame.pid_child.duplicated().any():
        raise InputValidationError(f"{input_id}: duplicate child identifiers.")
    for column in DATE_FIELDS[input_id]:
        required = column in {"childdob", "child_dob"} or (input_id == "tb_events" and column == "date_tb_diagnosed")
        frame[column] = normalize_dates(frame[column], column, allow_missing=not required)
    if input_id == "tb_events":
        categories=frame["tbcat"]
        if (categories.isna().any()
                or not categories.map(lambda value:isinstance(value,str)).all()
                or not categories.isin(TB_EVENT_CATEGORIES).all()):
            raise InputValidationError("tb_events: tbcat must use the declared nonmissing string vocabulary.")
        metadata["category_checks"]={"tbcat":{"allowed_values":sorted(TB_EVENT_CATEGORIES),
                                               "encoding":"exact_string","missing_allowed":False}}
    if input_id == "pulmonary_function":
        for column in set(PFT_FIELDS) - {"pid_child"} - set(DATE_FIELDS[input_id]):
            frame[column] = _number(frame[column], column)
    elif input_id == "baseline":
        for column in ("sex", "maternal_hiv", "Prenatal_smoking_composite", "birth_weight", "lrti",
                       "maternal_age_birth", "education", "Prenatal_alcohol_composite", "parity",
                       "crowd", "household_income", "smoke_hh", "lowbirthweight"):
            frame[column] = _number(frame[column], column)
    if input_id.startswith("wheeze_"):
        metadata["assessment_checks"] = _wheeze_checks(frame, input_id)
    # No imputation, dropping or deduplication is permitted in the reader.
    if frame.shape != (source_rows, source_columns) or not frame.index.equals(source_index):
        raise InputValidationError(f"{input_id}: import changed source rows or columns.")
    # Rehash after parsing to detect an input edited during the read.
    _file_digest(path, input_id, digest)
    metadata.update(rows=source_rows, columns=source_columns, unique_children=int(frame.pid_child.nunique()),
                    row_preservation=True, missing_counts={field: int(frame[field].isna().sum()) for field in SCHEMAS[input_id]},
                    dtypes={field: str(frame[field].dtype) for field in SCHEMAS[input_id]},
                    repeated_child_rows=int(frame.pid_child.duplicated().sum()))
    return InputTable(frame, metadata)


def compare_wheeze_copies(dta: InputTable | pd.DataFrame, csv_copy: InputTable | pd.DataFrame,
                          *, rtol: float = 1e-6, atol: float = 1e-7) -> dict[str, Any]:
    """Verify all common source fields after explicit encoding normalization.

    Float tolerance accommodates Stata float32 values written as decimal CSV.
    IDs, strings, binary/category codes and dates must match exactly. Row order
    may differ; child keys are required to be unique in each representation.
    """
    label_sets = dta.metadata.get("variable_label_sets", {}) if isinstance(dta, InputTable) else {}
    label_definitions = dta.metadata.get("value_labels", {}) if isinstance(dta, InputTable) else {}
    left = (dta.data if isinstance(dta, InputTable) else dta).copy()
    right = (csv_copy.data if isinstance(csv_copy, InputTable) else csv_copy).copy()
    if set(left.columns) != set(right.columns):
        raise InputValidationError("wheeze comparison: source column sets differ.")
    for frame in (left, right):
        frame["pid_child"] = normalize_identifiers(frame["pid_child"])
        if frame.pid_child.duplicated().any():
            raise InputValidationError("wheeze comparison: duplicate child identifiers.")
    if set(left.pid_child) != set(right.pid_child):
        raise InputValidationError("wheeze comparison: child-key sets differ.")
    left = left.set_index("pid_child").sort_index()
    right = right.set_index("pid_child").loc[left.index]
    numeric_fields = date_fields = text_fields = 0
    maximum_numeric_difference = 0.0
    failures = 0
    for column in left:
        a, b = left[column], right[column]
        if re.fullmatch(r"date\d+", column):
            a, b = normalize_dates(a, column), normalize_dates(b, column)
            same = a.eq(b) | (a.isna() & b.isna())
            date_fields += 1
        elif pd.api.types.is_numeric_dtype(a):
            declared_labels = label_definitions.get(label_sets.get(column), {})
            if declared_labels:
                if len(set(declared_labels.values())) != len(declared_labels):
                    raise InputValidationError("wheeze comparison: ambiguous declared category labels.")
                reverse = {label: code for code, label in declared_labels.items()}
            elif re.fullmatch(r"visitcat\d+", column):
                reverse = VISIT_CATEGORY_LABELS
            else:
                reverse = {}
            if reverse:
                b = b.astype("string")
                labeled = b.isin(reverse)
                b.loc[labeled] = b.loc[labeled].map(reverse).astype("string")
            b = _number(b, column)
            av = a.to_numpy(dtype=float, na_value=np.nan)
            bv = b.to_numpy(dtype=float, na_value=np.nan)
            exact = bool(declared_labels or re.fullmatch(r"(?:visitcat|wheeze_current)\d+|count_wheeze", column))
            same = np.isclose(av, bv, rtol=0 if exact else rtol, atol=0 if exact else atol, equal_nan=True)
            valid = np.isfinite(av) & np.isfinite(bv)
            if valid.any():
                maximum_numeric_difference = max(maximum_numeric_difference, float(np.max(np.abs(av[valid] - bv[valid]))))
            numeric_fields += 1
        else:
            a = a.astype("string").str.strip().replace("", pd.NA)
            b = b.astype("string").str.strip().replace("", pd.NA)
            same = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
            text_fields += 1
        failures += int((~np.asarray(same, dtype=bool)).sum())
    if failures:
        raise InputValidationError(f"wheeze comparison: {failures} unequal cells after encoding normalization.")
    return {"equivalent": True, "children": len(left), "columns": len(left.columns) + 1,
            "date_fields": date_fields, "numeric_fields": numeric_fields, "text_fields": text_fields,
            "unequal_cells": 0, "rtol": rtol, "atol": atol,
            "maximum_numeric_absolute_difference": maximum_numeric_difference}


def load_inputs(config_inputs: Mapping[str, Any], base_dir: str | Path | None = None) -> dict[str, InputTable]:
    """Read configured study tables; validate optional file-only bindings.

    Requires the five study-table roles; the GLI loader belongs to preparation.
    Optional dictionaries are checked for readability/hash without loading rows.
    This function does not create directories or write study data.
    """
    if not isinstance(config_inputs, Mapping):
        raise InputValidationError("Input configuration must be a mapping.")
    if set(config_inputs) - set(SCHEMAS) - FILE_ONLY_ROLES:
        raise InputValidationError("Input configuration contains unknown roles.")
    if REQUIRED_TABLES - set(config_inputs):
        raise InputValidationError("Input configuration omits a required study-table role.")
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    tables = {}
    for role, specification in config_inputs.items():
        if not isinstance(specification, Mapping) or set(specification) - {"path", "sha256"}:
            raise InputValidationError(f"{role}: invalid input specification.")
        value = specification.get("path")
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise InputValidationError(f"{role}: input path is required.")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else base / path
        if role in FILE_ONLY_ROLES:
            _file_digest(path, role, specification.get("sha256"))
            continue
        tables[role] = read_input(path, role, specification.get("sha256"))
    if "wheeze_csv_check" in tables:
        tables["wheeze_assessments"].metadata["companion_comparison"] = compare_wheeze_copies(
            tables["wheeze_assessments"], tables["wheeze_csv_check"])
    return tables
