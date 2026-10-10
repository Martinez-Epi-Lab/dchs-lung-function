"""Fabricated inputs exercise import and comparison failure modes."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dchs_lung_function.inputs import (
    InputTable, InputValidationError, SCHEMAS, compare_wheeze_copies, load_inputs,
    normalize_dates, normalize_identifiers, read_input,
)


class NormalizationTests(unittest.TestCase):
    def test_identifier_whitespace_and_leading_zeros(self):
        actual = normalize_identifiers(pd.Series([" 001 ", "A-002"]))
        self.assertEqual(actual.tolist(), ["001", "A-002"])

    def test_identifier_invalid_cases_and_private_error(self):
        for values in ([None], [" "], ["A", " A "], [1.2], [np.inf],
                       [float(2**54)], [1, "1"], [True], ["private\x00value"]):
            with self.subTest(values=values), self.assertRaises(InputValidationError) as caught:
                normalize_identifiers(pd.Series(values))
            self.assertNotIn("private", str(caught.exception))

    def test_integral_numeric_identifiers_preserved(self):
        self.assertEqual(normalize_identifiers(pd.Series([1, 2])).tolist(), ["1", "2"])

    def test_explicit_date_formats_and_missingness(self):
        result = normalize_dates(pd.Series(["2020-02-29", "01jan2021", "", None]), "test_date")
        self.assertEqual(result.iloc[0], pd.Timestamp("2020-02-29"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2021-01-01"))
        self.assertEqual(int(result.isna().sum()), 2)

    def test_invalid_dates_never_silently_become_missing(self):
        for values in (["2021-02-29"], ["01/02/2020"], [21915],
                       ["01xxx2021"], ["2020-01-01T01:00:00"],
                       [pd.Timestamp("2020-01-01 01:00")],
                       [pd.Timestamp("2020-01-01", tz="UTC")]):
            with self.subTest(values=values), self.assertRaises(InputValidationError):
                normalize_dates(pd.Series(values), "test_date")
        with self.assertRaises(InputValidationError):
            normalize_dates(pd.Series([None]), "birthdate", allow_missing=False)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_csv_missing_diagnosis_and_rows_preserved(self):
        path = self.folder / "dates.csv"
        path.write_text("pid_child,date_tb_diagnosed,extra\n001,2020-01-02,z\n002,,q\n", encoding="utf-8")
        before = path.read_bytes()
        result = read_input(path, "respiratory_tb_dates", hashlib.sha256(before).hexdigest())
        self.assertEqual(result.data.pid_child.tolist(), ["001", "002"])
        self.assertEqual(result.data.columns.tolist(), ["pid_child", "date_tb_diagnosed", "extra"])
        self.assertEqual(result.metadata["missing_counts"]["date_tb_diagnosed"], 1)
        self.assertTrue(result.metadata["row_preservation"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.folder.iterdir()), [path])

    def test_malformed_csv_rejected(self):
        cases = ["pid_child,pid_child,date_tb_diagnosed\nA,A,2020-01-01\n",
                 "pid_child,date_tb_diagnosed\nA,2020-01-01,unexpected\n",
                 "pid_child,date_tb_diagnosed\nA\n",
                 "pid_child, date_tb_diagnosed\nA,2020-01-01\n",
                 "pid_child\nA\n",
                 "pid_child,date_tb_diagnosed\nA,2020-01-01\nA,2020-01-02\n"]
        for i, content in enumerate(cases):
            path = self.folder / f"fixture_{i}.csv"
            path.write_text(content, encoding="utf-8")
            with self.subTest(i=i), self.assertRaises(InputValidationError):
                read_input(path, "respiratory_tb_dates")

    def test_invalid_encoding_hash_and_format(self):
        path = self.folder / "private-name.csv"
        path.write_bytes(b"pid_child,date_tb_diagnosed\n\xff,2020-01-01\n")
        with self.assertRaises(InputValidationError) as caught:
            read_input(path, "respiratory_tb_dates")
        self.assertNotIn("private-name", str(caught.exception))
        with self.assertRaisesRegex(InputValidationError, "SHA256"):
            read_input(path, "respiratory_tb_dates", "0" * 64)
        with self.assertRaisesRegex(InputValidationError, "format"):
            read_input(path, "baseline")

    def test_serialized_csv_index_preserved_and_validated(self):
        path = self.folder / "indexed.csv"
        path.write_text(",pid_child,date_tb_diagnosed\n0,001,2020-01-01\n1,002,\n", encoding="utf-8")
        table = read_input(path, "respiratory_tb_dates")
        self.assertEqual(table.metadata["serialized_index_column"], "Unnamed: 0")
        self.assertEqual(table.data["Unnamed: 0"].tolist(), ["0", "1"])
        path.write_text(",pid_child,date_tb_diagnosed\n5,001,2020-01-01\n", encoding="utf-8")
        with self.assertRaisesRegex(InputValidationError, "serialized row index"):
            read_input(path, "respiratory_tb_dates")

    def test_date_missing_token_does_not_change_identifier(self):
        path = self.folder / "dates.csv"
        path.write_text("pid_child,date_tb_diagnosed\nNA,NA\n", encoding="utf-8")
        table = read_input(path, "respiratory_tb_dates")
        self.assertEqual(table.data.pid_child.iloc[0], "NA")
        self.assertTrue(pd.isna(table.data.date_tb_diagnosed.iloc[0]))

    def test_truncated_stata_fails_cleanly(self):
        path = self.folder / "private-location.dta"
        path.write_bytes(b"\x76\x02")
        with self.assertRaises(InputValidationError) as caught:
            read_input(path, "tb_events")
        self.assertNotIn("private-location", str(caught.exception))

    def test_stata_repeated_episodes_labels_and_dates(self):
        source = pd.DataFrame({"pid_child": ["C01", "C01"],
            "date_tb_diagnosed": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "date_tb_investigated": pd.to_datetime(["2019-12-01", None]),
            "childdob": pd.to_datetime(["2010-01-01"] * 2),
            "tbcat": ["Confirmed tb", "Unconfirmed tb"], "tbtrt": ["yes", "yes"],
            "sex": [2, 2]})
        path = self.folder / "events.dta"
        source.to_stata(path, write_index=False, version=118,
                        convert_dates={"date_tb_diagnosed": "td", "date_tb_investigated": "td", "childdob": "td"},
                        value_labels={"sex": {1: "Female", 2: "Male"}})
        result = read_input(path, "tb_events")
        self.assertEqual(result.metadata["repeated_child_rows"], 1)
        self.assertEqual(result.data.sex.tolist(), [2, 2])
        label_set = result.metadata["variable_label_sets"]["sex"]
        self.assertEqual(result.metadata["value_labels"][label_set][2], "Male")
        self.assertEqual(result.data.date_tb_diagnosed.iloc[1], pd.Timestamp("2021-01-01"))
        self.assertEqual(result.metadata['category_checks']['tbcat']['encoding'],'exact_string')
        self.assertEqual(result.data.tbcat.tolist(),source.tbcat.tolist())

    def test_tb_category_rejects_unknown_blank_and_labelled_numeric_values(self):
        dates={name:'td' for name in ['childdob','date_tb_diagnosed','date_tb_investigated']}
        for category in ['Probable tb','confirmed tb','Confirmed tb ','',1,np.nan]:
            source=pd.DataFrame({'pid_child':['synthetic'], 'childdob':pd.to_datetime(['2010-01-01']),
                'date_tb_diagnosed':pd.to_datetime(['2011-01-01']),
                'date_tb_investigated':pd.to_datetime(['2011-01-01']), 'tbcat':[category], 'tbtrt':['yes']})
            path=self.folder/'categories.dta'
            numeric=not isinstance(category,str)
            source.to_stata(path,write_index=False,version=118,convert_dates=dates,
                            value_labels={'tbcat':{1:'Confirmed tb'}} if numeric else None)
            with self.subTest(category=category),self.assertRaisesRegex(InputValidationError,'string vocabulary') as caught:
                read_input(path,'tb_events')
            self.assertNotIn('synthetic',str(caught.exception))
            self.assertNotIn(str(path),str(caught.exception))

    def test_schema_and_missing_identifier_rejected(self):
        path = self.folder / "events.dta"
        pd.DataFrame({"pid_child": ["A"]}).to_stata(path, write_index=False, version=118)
        with self.assertRaisesRegex(InputValidationError, "required fields"):
            read_input(path, "tb_events")
        path = self.folder / "dates.csv"
        path.write_text("pid_child,date_tb_diagnosed\n,2020-01-01\n", encoding="utf-8")
        with self.assertRaisesRegex(InputValidationError, "missing identifiers"):
            read_input(path, "respiratory_tb_dates")

    def test_load_config_rejects_unknown_or_incomplete_roles(self):
        with self.assertRaisesRegex(InputValidationError, "unknown roles"):
            load_inputs({"unexpected": {"path": "missing"}}, self.folder)
        with self.assertRaisesRegex(InputValidationError, "required"):
            load_inputs({}, self.folder)


def wheeze_frame():
    columns = {"pid_child": ["S001", "S002"], "count_wheeze": [1, 0]}
    for slot in range(1, 37):
        columns[f"redcap_event_name{slot}"] = ["", ""]
        columns[f"visitcat{slot}"] = [np.nan] * 2
        columns[f"date{slot}"] = [pd.NaT] * 2
        columns[f"ageyears{slot}"] = [np.nan] * 2
        columns[f"wheeze_current{slot}"] = [np.nan] * 2
    frame = pd.DataFrame(columns)
    frame.loc[0, ["redcap_event_name1", "visitcat1", "date1", "ageyears1", "wheeze_current1"]] = [
        "scheduled", 1, pd.Timestamp("2020-01-01"), np.float32(2.34567), 1]
    return frame


class WheezeTests(unittest.TestCase):
    def test_declared_category_labels_compare_exactly(self):
        left, right = wheeze_frame(), wheeze_frame()
        left["phenotype"] = [1, 2]
        right["phenotype"] = ["First category", "Second category"]
        table = InputTable(left, {"variable_label_sets": {"phenotype": "classification"},
                                 "value_labels": {"classification": {1: "First category", 2: "Second category"}}})
        self.assertTrue(compare_wheeze_copies(table, right)["equivalent"])
        right.loc[0, "phenotype"] = "Second category"
        with self.assertRaisesRegex(InputValidationError, "unequal cells"):
            compare_wheeze_copies(table, right)

    def test_comparison_all_fields_with_row_reorder_and_float_rounding(self):
        left = wheeze_frame()
        right = left.iloc[::-1].copy()
        right["date1"] = right.date1.dt.strftime("%d%b%Y").str.lower()
        right["visitcat1"] = right.visitcat1.map({1: "Scheduled visit"})
        right["ageyears1"] = right.ageyears1.round(6)
        result = compare_wheeze_copies(left, right)
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["unequal_cells"], 0)
        self.assertEqual(result["columns"], len(left.columns))

    def test_date_and_event_name_differences_cannot_hide(self):
        for field, value in [("date1", pd.Timestamp("2020-01-02")),
                             ("redcap_event_name1", "different"), ("wheeze_current1", 0),
                             ("count_wheeze", 2)]:
            left, right = wheeze_frame(), wheeze_frame()
            right.loc[0, field] = value
            with self.subTest(field=field), self.assertRaisesRegex(InputValidationError, "unequal cells"):
                compare_wheeze_copies(left, right)

    def test_wheeze_read_checks_duplicate_dates_and_dedicated_visits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wheeze.dta"
            for invalid in ("duplicate", "dedicated", "nonbinary"):
                frame = wheeze_frame()
                if invalid == "duplicate":
                    frame["date2"] = frame["date1"]
                elif invalid == "dedicated":
                    frame.loc[0, "visitcat1"] = 3
                    frame.loc[0, "wheeze_current1"] = 0
                else:
                    frame.loc[0, "wheeze_current1"] = 2
                frame.to_stata(path, write_index=False, version=118,
                                convert_dates={f"date{i}": "td" for i in range(1, 37)})
                with self.subTest(invalid=invalid), self.assertRaises(InputValidationError):
                    read_input(path, "wheeze_assessments")


if __name__ == "__main__":
    unittest.main()
