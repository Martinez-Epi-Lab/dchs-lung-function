"""Synthetic release lifecycle checks; no study files, fits or actual PDF rendering."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dchs_lung_function import full_execution as full
from dchs_lung_function import render_execution as render
from dchs_lung_function import pipeline
from dchs_lung_function.execution import canonical_bytes, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from dchs_lung_function.reporting.sources import digest
from test_reporting import fake_font

COVERAGE = {'figures': 9, 'panels': 17, 'forest_estimates': 56, 'trajectory_points': 808,
            'schedule_cells': 44, 'flow_nodes': 20, 'flow_count_bindings': 28, 'caption_count_bindings': 12}
PREP = 'a' * 64
UNCERTAINTY = 'corrected_actual_risk_membership'
LIMITATIONS = ['Four upstream antenatal flow counts remain unavailable',
               'Efron-fit/Breslow-score combination retained', 'Inclusive entry/stop event-risk boundaries retained']


def publication(directory, identity, summary, files, overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        (directory / name).write_bytes(b'Invented aggregate fixture.\n')
    for name, value in (overrides or {}).items():
        if isinstance(value, bytes): (directory / name).write_bytes(value)
        else: write_json(directory / name, value)
    write_json(directory / 'summary.json', summary)
    receipt = {'schema_version': 1, 'status': 'accepted', 'identity': identity,
               'fingerprint': digest(identity), 'summary': summary, 'completed_at': 'synthetic-completion',
               'files': {name: file_hash(directory / name) for name in files}}
    write_json(directory / 'receipt.json', receipt)
    return receipt


class SyntheticReleaseCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dchs-release-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'source'
        (self.root / 'config').mkdir(parents=True)
        for name in ('input_contract.json', 'manuscript_reporting.json', 'publication_figures.json'):
            write_json(self.root / 'config' / name, {})
        (self.root / 'README.md').write_text('Synthetic source only.\n')
        self.source = {'README.md': file_hash(self.root / 'README.md')}
        self.work = self.base / 'work'
        self.output = self.base / 'outputs'
        self.doc = self.base / 'document-python'
        self.pdf = self.base / 'pdf-fonts-tool'
        for path in (self.doc, self.pdf):
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o700)
        self.fonts = {'arial': self.base / 'arial.ttf', 'arial_bold': self.base / 'arial-bold.ttf'}
        self.fonts['arial'].write_bytes(fake_font())
        self.fonts['arial_bold'].write_bytes(fake_font(style='Bold'))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def patch(self, module, name, **kwargs):
        return self.stack.enter_context(mock.patch.object(module, name, **kwargs))

    def source_guard(self, root):
        if Path(root) != self.root or self.current_source(root) != self.source:
            raise ValidationError('Synthetic source changed.')

    def current_source(self, root):
        return {'README.md': file_hash(Path(root) / 'README.md')}

    def runtimes(self, module):
        self.patch(module, 'assert_loaded', side_effect=self.source_guard)
        self.patch(module, 'source_hashes', side_effect=self.current_source)
        self.patch(module, 'document_runtime', return_value={'python': 'synthetic'})
        self.patch(module, 'figure_runtime', return_value={'renderer': 'synthetic'})


class FullExecutionTests(SyntheticReleaseCase):
    def setUp(self):
        super().setUp()
        self.runtimes(full)
        self.patch(full, 'runtime_identity', return_value={'preparation': 'synthetic'})
        self.patch(full, 'model_runtime_identity', return_value={'models': 'synthetic'})
        self.inputs = {'invented_input': self.base / 'invented-input.txt'}
        self.inputs['invented_input'].write_text('Invented input; never analyzed.\n')
        self.paths = [self.base / (name + '.json') for name in
                      ('paths', 'analysis', 'primary', 'mechanics', 'supporting', 'wheeze', 'sensitivity', 'reporting', 'figures')]
        self.configuration = {'directories': {'work': str(self.work), 'outputs': str(self.output)},
                              'inputs': {'invented_input': {'path': str(self.inputs['invented_input'])}},
                              'fonts': {key: str(path) for key, path in self.fonts.items()}}
        for path in self.paths: write_json(path, {})
        write_json(self.paths[0], self.configuration)
        self.patch(full, 'validate_config', return_value=None)
        self.patch(full, 'resolve_paths', side_effect=lambda *_: (self.inputs, {'work': self.work, 'outputs': self.output}))
        self.patch(full, 'reporting_config', return_value={'document_python': self.doc})
        self.patch(full, 'figures_config', return_value={'pdf_fonts_tool': self.pdf})
        self.raw = {'invented/' + str(i): 'c' * 64 for i in range(37)}
        self.patch(full, 'raw_inventory', return_value=self.raw)
        self.cumulative = self.patch(full, 'run_figures', side_effect=self.fake_cumulative)
        self.mutate_result = None

    def fake_cumulative(self, *args, **kwargs):
        ri = {'schema_version': 1, 'source': self.source, 'preparation_fingerprint': PREP}
        rfp = digest(ri)
        rs = {'through': 'M13', 'accepted': True, 'status': 'accepted', 'fingerprint': rfp, 'new_model_fits': 0}
        rd = self.output / 'manuscript_reporting' / rfp
        if not rd.exists(): publication(rd, ri, rs, full.REPORTING_FILES)
        fi = {'schema_version': 1, 'source': self.source, 'reporting_fingerprint': rfp,
              'reporting_receipt_sha256': file_hash(rd / 'receipt.json'), 'preparation_fingerprint': PREP,
              'uncertainty_selection': UNCERTAINTY}
        ffp = digest(fi)
        result = {'through': 'M14', 'accepted': True, 'status': 'accepted', 'fingerprint': ffp,
                  'reporting_fingerprint': rfp, 'preparation_fingerprint': PREP, 'coverage': deepcopy(COVERAGE),
                  'preceding_modules_verified': 13, 'new_model_fits': 0, 'corrected_pretb_models': 17,
                  'corrected_pretb_coefficients': 51, 'corrected_pretb_covariance_cells': 306,
                  'pretb_point_estimates_preserved': True, 'uncertainty_selection': UNCERTAINTY,
                  'G08_status': 'actual event-time score membership corrected for Figure S3; source-reproduction artifacts retained',
                  'inherited_limitations': LIMITATIONS}
        fd = self.output / 'publication_figures' / ffp
        if not fd.exists(): publication(fd, fi, result, full.FIGURE_FILES)
        result['resumed'] = bool(kwargs['resume'])
        if self.mutate_result: self.mutate_result(result)
        return result

    def run_full(self, **kwargs):
        return full.run_full(self.root, *self.paths, **kwargs)

    def assert_no_completion(self):
        self.assertFalse((self.work / 'full_current.json').exists())
        self.assertEqual(list(self.output.glob('full_reconstruction/*/receipt.json')), [])
        self.assertFalse((self.work / full.LOCK).exists())
        self.assertFalse((self.output / full.LOCK).exists())

    def test_clean_origin_success_and_verified_resume(self):
        first = self.run_full(jobs=2)
        self.assertTrue(first['clean_origin'])
        self.assertEqual(first['joint_fits_verified'], 37)
        self.assertFalse(first['github_delivery_verified'])
        self.assertEqual((self.work / full.ORIGIN).read_bytes(), (self.output / full.ORIGIN).read_bytes())
        receipt = self.output / 'full_reconstruction' / first['fingerprint'] / 'receipt.json'
        initial = receipt.read_bytes()
        resumed = self.run_full(resume=True)
        self.assertTrue(resumed['resumed'])
        self.assertEqual(first['fingerprint'], resumed['fingerprint'])
        self.assertEqual(receipt.read_bytes(), initial)
        self.assertEqual(self.cumulative.call_count, 2)
        self.assertFalse((self.work / full.LOCK).exists())

    def test_fresh_run_rejects_hidden_or_cached_content(self):
        self.work.mkdir()
        (self.work / '.unrelated').write_bytes(b'keep')
        with self.assertRaises(ValidationError): self.run_full()
        self.cumulative.assert_not_called()
        self.assertEqual((self.work / '.unrelated').read_bytes(), b'keep')
        self.assert_no_completion()

    def test_resume_without_paired_origin_is_rejected(self):
        self.work.mkdir(); self.output.mkdir()
        with self.assertRaises(ValidationError): self.run_full(resume=True)
        self.cumulative.assert_not_called()
        self.assert_no_completion()

    def test_changed_input_config_or_source_rejects_resume_before_cumulative(self):
        self.run_full()
        for path in (self.inputs['invented_input'], self.paths[1], self.root / 'README.md'):
            original = path.read_bytes(); path.write_bytes(original + b' ')
            count = self.cumulative.call_count
            with self.subTest(path=path.name), self.assertRaises(ValidationError): self.run_full(resume=True)
            self.assertEqual(self.cumulative.call_count, count)
            path.write_bytes(original)

    def test_changed_or_missing_origin_rejects_resume(self):
        self.run_full()
        path = self.output / full.ORIGIN
        original = path.read_bytes()
        changed = json.loads(original); changed['clean_origin'] = 1
        write_json(path, changed)
        with self.assertRaises(ValidationError): self.run_full(resume=True)
        path.unlink()
        with self.assertRaises(ValidationError): self.run_full(resume=True)
        self.assertEqual(self.cumulative.call_count, 1)

    def test_jobs_and_resume_must_have_exact_types(self):
        for options in ({'jobs': True}, {'jobs': 0}, {'jobs': 4}, {'jobs': 1.0}, {'resume': 1}):
            with self.subTest(options=options), self.assertRaises(ValidationError): self.run_full(**options)
        self.cumulative.assert_not_called()

    def test_generated_roots_cannot_overlap_each_other_or_source(self):
        original_work = self.work
        for path in (self.output, self.output / 'nested', self.root / 'generated', self.root.parent):
            self.work = path
            with self.subTest(path=path), self.assertRaises(ValidationError): self.run_full()
        self.work = original_work
        self.cumulative.assert_not_called()

    def test_generated_root_cannot_contain_protected_file(self):
        self.work = self.inputs['invented_input'].parent
        with self.assertRaises(ValidationError): self.run_full()
        self.cumulative.assert_not_called()

    def test_symlink_generated_or_config_paths_reject_before_cumulative(self):
        self.work.symlink_to(self.base / 'absent', target_is_directory=True)
        with self.assertRaises(ValidationError): self.run_full()
        self.work.unlink()
        alias = self.base / 'alias.json'; alias.symlink_to(self.paths[1])
        self.paths[1] = alias
        with self.assertRaises(ValidationError): self.run_full()
        self.cumulative.assert_not_called()

    def test_other_owner_lock_is_preserved_and_own_lock_is_cleaned(self):
        self.work.mkdir(); self.output.mkdir()
        owned = self.output / full.LOCK
        owned.write_bytes(b'other owner')
        with self.assertRaises(ValidationError):
            with full.full_locks((self.work, self.output)): self.fail('must not enter')
        self.assertEqual(owned.read_bytes(), b'other owner')
        self.assertFalse((self.work / full.LOCK).exists())

    def test_cumulative_failure_retains_origin_but_no_completion(self):
        self.cumulative.side_effect = ValidationError('Synthetic model failure.')
        with self.assertRaises(ValidationError): self.run_full()
        self.assertTrue((self.work / full.ORIGIN).is_file())
        self.assert_no_completion()
        self.cumulative.side_effect = self.fake_cumulative
        self.assertTrue(self.run_full(resume=True)['accepted'])

    def test_incomplete_or_spoofed_cumulative_summary_rejected(self):
        attacks = [('preceding_modules_verified', 13.0), ('new_model_fits', 1),
                   ('corrected_pretb_covariance_cells', 305), ('uncertainty_selection', 'uncorrected'),
                   ('coverage', {**COVERAGE, 'trajectory_points': 807}), ('accepted', 1)]
        for key, value in attacks:
            with self.subTest(key=key):
                self.mutate_result = lambda result, k=key, v=value: result.__setitem__(k, v)
                with self.assertRaises(ValidationError): self.run_full(resume=(self.work / full.ORIGIN).exists())
                self.assert_no_completion()

    def test_input_source_and_origin_changes_during_cumulative_block_publication(self):
        for name in ('input', 'source', 'origin'):
            def changed(*args, **kwargs):
                result = self.fake_cumulative(*args, **kwargs)
                path = {'input': self.inputs['invented_input'], 'source': self.root / 'README.md',
                        'origin': self.work / full.ORIGIN}[name]
                path.write_bytes(path.read_bytes() + b' ')
                return result
            self.cumulative.side_effect = changed
            with self.subTest(name=name), self.assertRaises(ValidationError):
                self.run_full(resume=(self.work / full.ORIGIN).exists())
            self.assert_no_completion()
            # Rebuild the invented root state between independently tested failures.
            if self.work.exists(): shutil.rmtree(self.work)
            if self.output.exists(): shutil.rmtree(self.output)
            self.inputs['invented_input'].write_text('Invented input; never analyzed.\n')
            (self.root / 'README.md').write_text('Synthetic source only.\n')

    def test_upstream_publication_tamper_blocks_full_completion(self):
        def changed(*args, **kwargs):
            result = self.fake_cumulative(*args, **kwargs)
            (self.output / 'publication_figures' / result['fingerprint'] / full.FIGURE_FILES[0]).write_bytes(b'tamper')
            return result
        self.cumulative.side_effect = changed
        with self.assertRaises(ValidationError): self.run_full()
        self.assert_no_completion()

    def test_failed_completion_write_can_resume_without_empty_final_directory(self):
        with mock.patch.object(full.os, 'link', side_effect=OSError('Synthetic disk failure')), self.assertRaises(OSError): self.run_full()
        self.assert_no_completion()
        self.assertTrue(self.run_full(resume=True)['accepted'])

    def test_failed_second_origin_write_allows_clean_retry(self):
        original = full.write_pointer
        def fail_second(path, value):
            if Path(path) == self.output / full.ORIGIN:
                raise OSError('Synthetic second-origin disk failure')
            return original(path, value)
        with mock.patch.object(full, 'write_pointer', side_effect=fail_second), self.assertRaises(OSError):
            self.run_full()
        self.assertFalse((self.work / full.ORIGIN).exists())
        self.assertFalse((self.output / full.ORIGIN).exists())
        self.assert_no_completion()
        self.cumulative.assert_not_called()
        self.assertTrue(self.run_full()['accepted'])

    def test_cumulative_receipt_linkage_must_match_returned_summary(self):
        def change(*args, **kwargs):
            result = self.fake_cumulative(*args, **kwargs)
            directory = self.output / 'publication_figures' / result['fingerprint']
            receipt = json.loads((directory / 'receipt.json').read_text())
            receipt['summary']['corrected_pretb_models'] = 16
            write_json(directory / 'summary.json', receipt['summary'])
            receipt['files']['summary.json'] = file_hash(directory / 'summary.json')
            write_json(directory / 'receipt.json', receipt)
            return result
        self.cumulative.side_effect = change
        with self.assertRaises(ValidationError): self.run_full()
        self.assert_no_completion()

    def test_rehashed_full_completion_summary_tamper_rejected_on_resume(self):
        result = self.run_full()
        path = self.output / 'full_reconstruction' / result['fingerprint'] / 'receipt.json'
        receipt = json.loads(path.read_text()); receipt['summary']['joint_fits_verified'] = 36
        write_json(path, receipt)
        with self.assertRaises(ValidationError): self.run_full(resume=True)


class FullRawInventoryTests(unittest.TestCase):
    def setUp(self):
        from dchs_lung_function.models import spirometry, mechanics, wheeze, sensitivity
        from dchs_lung_function.supporting_sampling import TARGETS
        self.temp = tempfile.TemporaryDirectory(prefix='dchs-raw-inventory-test-')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.targets = {'primary_fits': tuple(spirometry.TARGETS), 'mechanics_fits': tuple(mechanics.TARGETS),
                        'supporting_fits': tuple(TARGETS), 'wheeze_fits': tuple(wheeze.TARGETS),
                        'sensitivity_fits': tuple(sensitivity.TARGETS)}
        self.paths = []
        for family, targets in self.targets.items():
            for target in targets:
                identity = {'schema_version': 1, 'target': target, 'family': family, 'synthetic': True}
                directory = self.work / family / target / digest(identity)
                directory.mkdir(parents=True)
                # Opaque invented bytes: this layer checks receipts; numeric validation is upstream.
                for name in ('posterior.npz', 'sample_stats.npz', 'sampling.json'):
                    (directory / name).write_bytes(b'Wholly invented opaque fixture.\n')
                receipt = {'schema_version': 1, 'status': 'sampled', 'fingerprint': digest(identity),
                           'identity': identity, 'preparation_fingerprint': PREP, 'completed_at': 'synthetic',
                           'files': {name: file_hash(directory / name) for name in
                                     ('posterior.npz', 'sample_stats.npz', 'sampling.json')}}
                write_json(directory / 'receipt.json', receipt)
                self.paths.append(directory)

    def verify(self):
        return full.raw_inventory(self.work, preparation_fingerprint=PREP)

    def test_exact_declared_37_fit_receipts_are_bound(self):
        actual = self.verify()
        self.assertEqual(len(actual), 37)
        for path in self.paths:
            self.assertEqual(actual[str(path.relative_to(self.work))], file_hash(path / 'receipt.json'))

    def test_unknown_target_cannot_replace_known_target_at_same_count(self):
        path = self.paths[0]; old = path.parent
        renamed = old.with_name('invented_unknown_target')
        old.rename(renamed)
        path = renamed / path.name
        receipt = json.loads((path / 'receipt.json').read_text())
        receipt['identity']['target'] = renamed.name
        receipt['fingerprint'] = digest(receipt['identity'])
        new = path.with_name(receipt['fingerprint']); path.rename(new)
        write_json(new / 'receipt.json', receipt)
        with self.assertRaises(ValidationError): self.verify()

    def test_missing_and_duplicate_target_fits_rejected(self):
        first = self.paths[0]
        moved = first.with_name('b' * 64)
        shutil.copytree(first, moved)
        with self.assertRaises(ValidationError): self.verify()
        shutil.rmtree(moved); shutil.rmtree(first)
        with self.assertRaises(ValidationError): self.verify()

    def test_raw_receipt_schema_status_preparation_and_target_are_required(self):
        path = self.paths[0] / 'receipt.json'; original = json.loads(path.read_text())
        for key, value in (('schema_version', True), ('status', 'failed'), ('preparation_fingerprint', 'b' * 64),
                           ('undeclared', 'private text')):
            write_json(path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValidationError): self.verify()
        changed = deepcopy(original); changed['identity']['target'] = 'wrong'
        write_json(path, changed)
        with self.assertRaises(ValidationError): self.verify()

    def test_raw_hash_tamper_and_extra_payload_rejected(self):
        path = self.paths[0] / 'posterior.npz'; original = path.read_bytes()
        path.write_bytes(b'tamper')
        with self.assertRaises(ValidationError): self.verify()
        path.write_bytes(original)
        (self.paths[0] / 'extra.txt').write_text('unexpected')
        with self.assertRaises(ValidationError): self.verify()

    def test_raw_directory_and_leaf_symlink_rejected(self):
        path = self.paths[0] / 'posterior.npz'; original = path.read_bytes()
        elsewhere = self.work / 'elsewhere'; elsewhere.write_bytes(original)
        path.unlink(); path.symlink_to(elsewhere)
        with self.assertRaises(ValidationError): self.verify()
        path.unlink(); path.write_bytes(original)
        directory = self.paths[0]; moved = self.work / 'moved-fit'; directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValidationError): self.verify()

    def test_completion_publication_does_not_clobber_competing_directory(self):
        destination = self.work / 'completion'; destination.mkdir()
        (destination / 'keep').write_bytes(b'other owner')
        with self.assertRaises((ValidationError, FileExistsError)):
            full.publish_completion(destination, {'synthetic': True})
        self.assertEqual((destination / 'keep').read_bytes(), b'other owner')
        self.assertEqual(list(self.work.glob('.full-completion-*')), [])


class RenderExecutionTests(SyntheticReleaseCase):
    def setUp(self):
        super().setUp()
        self.runtimes(render)
        self.tables = [{'invented': 'aggregate table'}]
        self.checks = {'wording_review_required': False, 'coverage': {'tables': 10, 'numeric_cells': 304, 'sentences': 42}}
        self.data = {'coverage': deepcopy(COVERAGE), 'figures': [{'title': 'Invented figure', 'caption': {'text': 'Count: 3.'}}]}
        self.registry = SimpleNamespace(export=lambda: {'invented': 'aggregate registry'})
        self.patch(render, 'registry_from_export', return_value=self.registry)
        self.patch(render, 'build_tables', return_value=self.tables)
        self.patch(render, 'build_checks', return_value=self.checks)
        self.patch(render, '_comparison', return_value={'status': 'reference_not_supplied'})
        self.correction = self.patch(render, 'load_correction', return_value=SimpleNamespace())
        self.patch(render, 'build_data', return_value=self.data)
        self.patch(render, 'points_csv', return_value='invented,aggregate\n3,1\n')
        ri = {'schema_version': 1, 'source': self.source, 'preparation_fingerprint': PREP}
        self.rfp = digest(ri); self.report = self.base / 'accepted-tables' / self.rfp
        rs = {'through': 'M13', 'accepted': True, 'status': 'accepted', 'fingerprint': self.rfp, 'new_model_fits': 0}
        self.report_receipt = publication(self.report, ri, rs, render.TABLE_FILES,
            {'registry.json': {}, 'tables.json': {'schema_version': 1, 'tables': self.tables},
             'numerical_text.json': self.checks, 'reference_comparison.json': {'status': 'reference_not_supplied'}, 'metadata.json': {}})
        fi = {'schema_version': 1, 'source': self.source, 'preparation_fingerprint': PREP,
              'reporting_fingerprint': self.rfp, 'reporting_receipt_sha256': file_hash(self.report / 'receipt.json'),
              'uncertainty_selection': UNCERTAINTY}
        self.ffp = digest(fi); self.figure = self.base / 'accepted-figures' / self.ffp
        fs = {'through': 'M14', 'accepted': True, 'status': 'accepted', 'fingerprint': self.ffp, 'new_model_fits': 0,
              'uncertainty_selection': UNCERTAINTY, 'pretb_point_estimates_preserved': True, 'inherited_limitations': LIMITATIONS}
        self.figure_receipt = publication(self.figure, fi, fs, render.FIGURE_FILES,
            {'plot_data.json': self.data, 'plot_points.csv': b'invented,aggregate\n3,1\n', 'metadata.json': {}})
        self.config_path = self.base / 'render.json'
        self.config = {'schema_version': 1, 'reporting_publication': str(self.report),
                       'reporting_receipt_sha256': file_hash(self.report / 'receipt.json'),
                       'figures_publication': str(self.figure), 'figures_receipt_sha256': file_hash(self.figure / 'receipt.json'),
                       'work_directory': str(self.work), 'output_directory': str(self.output),
                       'document_python': str(self.doc), 'pdf_fonts_tool': str(self.pdf),
                       'fonts': {key: str(path) for key, path in self.fonts.items()}}
        write_json(self.config_path, self.config)
        self.table_renderer = self.patch(render, 'render_tables', side_effect=self.fake_tables)
        self.figure_renderer = self.patch(render, 'render_figures', side_effect=self.fake_figures)
        self.font_validator = self.patch(render, 'verify_pdf_fonts', return_value={'status': 'synthetic-font-check'})

    def fake_tables(self, destination, *args):
        for name in render.TABLE_FILES: (destination / name).write_bytes(('Invented table payload: ' + name).encode())

    def fake_figures(self, data, destination, fonts, **kwargs):
        for name in render.IMAGE_FILES: (destination / name).write_bytes(('Invented image bytes, not a PDF: ' + name).encode())
        return {'files': {name: file_hash(destination / name) for name in render.IMAGE_FILES}}

    def run_render(self, **kwargs):
        return render.run_render_only(self.root, self.config_path, **kwargs)

    def assert_no_completion(self):
        self.assertFalse((self.work / 'render_current.json').exists())
        self.assertEqual(list(self.output.glob('render_only/*/receipt.json')), [])
        self.assertFalse((self.work / 'render_only/.preparation.lock').exists())
        self.assertFalse((self.output / 'render_only/.preparation.lock').exists())

    def test_render_regenerates_and_compares_without_analysis_or_raw_inputs(self):
        with mock.patch.object(full, 'run_figures', side_effect=AssertionError('analysis forbidden')):
            result = self.run_render()
            self.assertEqual(result['new_model_fits'], 0)
            for key in ('participant_data_read', 'raw_posteriors_read', 'correction_recomputed', 'analytical_revalidation_performed'):
                self.assertIs(result[key], False)
            directory = self.output / 'render_only' / result['fingerprint']
            before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
            self.assertEqual(len(before), 52)
            self.assertEqual(set(before), set(render.RESULT_FILES) | {'receipt.json'})
            resumed = self.run_render(resume=True)
            self.assertTrue(resumed['resumed'])
            self.assertEqual(self.table_renderer.call_count, 2)
            self.assertEqual(self.figure_renderer.call_count, 2)
            self.assertEqual(before, {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()})

    def test_repeated_render_without_resume_is_rejected(self):
        self.run_render()
        with self.assertRaises(ValidationError): self.run_render()
        self.assertEqual(self.table_renderer.call_count, 1)

    def test_configuration_schema_hashes_types_and_resume_rejected(self):
        original = deepcopy(self.config)
        for key, value in (('schema_version', True), ('reporting_receipt_sha256', 'bad'),
                           ('work_directory', []), ('undeclared', 'private text')):
            changed = deepcopy(original); changed[key] = value
            write_json(self.config_path, changed)
            with self.subTest(key=key), self.assertRaises(ValidationError): self.run_render()
        write_json(self.config_path, original)
        with self.assertRaises(ValidationError): self.run_render(resume=1)
        self.table_renderer.assert_not_called()

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_overflow(self):
        for text in ('{"a":1,"a":2}', '{"value":NaN}', '{"value":1e400}'):
            self.config_path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValidationError): render.read_json(self.config_path)

    def test_modified_or_untrusted_source_receipt_rejected(self):
        for key in ('reporting_receipt_sha256', 'figures_receipt_sha256'):
            self.config[key] = '0' * 64; write_json(self.config_path, self.config)
            with self.subTest(key=key), self.assertRaises(ValidationError): self.run_render()
            self.config[key] = file_hash((self.report if key.startswith('reporting') else self.figure) / 'receipt.json')
        self.table_renderer.assert_not_called()

    def test_source_publication_extra_file_or_symlink_rejected(self):
        path = self.report / 'extra'
        path.write_text('unexpected')
        with self.assertRaises(ValidationError): self.run_render()
        path.unlink()
        payload = self.report / render.TABLE_FILES[0]
        data = payload.read_bytes(); payload.unlink(); payload.symlink_to(self.base / 'missing')
        with self.assertRaises(ValidationError): self.run_render()
        payload.unlink(); payload.write_bytes(data)
        self.table_renderer.assert_not_called()

    def test_provenance_linkage_cannot_be_repaired_with_outer_receipt_hash(self):
        receipt = deepcopy(self.figure_receipt)
        receipt['identity']['reporting_fingerprint'] = 'b' * 64
        receipt['fingerprint'] = digest(receipt['identity'])
        receipt['summary']['fingerprint'] = receipt['fingerprint']
        write_json(self.figure / 'summary.json', receipt['summary'])
        receipt['files']['summary.json'] = file_hash(self.figure / 'summary.json')
        write_json(self.figure / 'receipt.json', receipt)
        self.config['figures_receipt_sha256'] = file_hash(self.figure / 'receipt.json')
        write_json(self.config_path, self.config)
        with self.assertRaises(ValidationError): self.run_render()
        self.table_renderer.assert_not_called()

    def test_generated_and_protected_roots_cannot_overlap(self):
        original = deepcopy(self.config)
        for key, value in (('work_directory', str(self.output)), ('output_directory', str(self.report)),
                           ('output_directory', str(self.root / 'generated')), ('work_directory', str(self.base))):
            self.config = {**original, key: value}; write_json(self.config_path, self.config)
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError): self.run_render()
        self.table_renderer.assert_not_called()

    def test_symlink_root_or_font_rejected(self):
        self.work.symlink_to(self.base / 'absent', target_is_directory=True)
        with self.assertRaises(ValidationError): self.run_render()
        self.work.unlink()
        original = self.fonts['arial'].read_bytes(); self.fonts['arial'].unlink()
        self.fonts['arial'].symlink_to(self.fonts['arial_bold'])
        with self.assertRaises(ValidationError): self.run_render()
        self.table_renderer.assert_not_called()

    def test_changed_scientific_table_or_plot_payload_rejected(self):
        self.patch(render, 'build_tables', return_value=[{'changed': 'estimate'}])
        with self.assertRaises(ValidationError): self.run_render()
        self.table_renderer.assert_not_called()

    def test_changed_plot_selection_rejected(self):
        self.patch(render, 'build_data', return_value={**self.data, 'changed': 'selection'})
        with self.assertRaises(ValidationError): self.run_render()
        self.table_renderer.assert_not_called()

    def test_renderer_failure_has_no_completion_and_cleans_locks(self):
        self.figure_renderer.side_effect = RuntimeError('Synthetic renderer failure')
        with self.assertRaises(RuntimeError): self.run_render()
        self.assert_no_completion()
        self.assertEqual(len(list((self.work / 'render_only').glob('failed-*'))), 1)
        self.figure_renderer.side_effect = self.fake_figures
        self.assertTrue(self.run_render()['accepted'])

    def test_missing_or_extra_rendered_payload_rejected(self):
        def missing(destination, *args):
            self.fake_tables(destination, *args)
            (destination / render.TABLE_FILES[0]).unlink()
        self.table_renderer.side_effect = missing
        with self.assertRaises(ValidationError): self.run_render()
        self.assert_no_completion()

    def test_font_validation_failure_blocks_completion(self):
        self.font_validator.side_effect = ValidationError('Synthetic Arial rejection')
        with self.assertRaises(ValidationError): self.run_render()
        self.assert_no_completion()

    def test_mutated_source_or_aggregate_during_render_blocks_completion(self):
        for path in (self.root / 'README.md', self.report / 'registry.json'):
            original = path.read_bytes()
            def mutate(destination, *args):
                self.fake_tables(destination, *args)
                path.write_bytes(original + b' ')
            self.table_renderer.side_effect = mutate
            with self.subTest(path=path.name), self.assertRaises(ValidationError): self.run_render()
            self.assert_no_completion()
            path.write_bytes(original)

    def test_hash_repaired_cached_payload_is_compared_to_regeneration(self):
        result = self.run_render(); directory = self.output / 'render_only' / result['fingerprint']
        payload = directory / 'tables' / render.TABLE_FILES[0]
        payload.write_bytes(b'forged aggregate')
        receipt = json.loads((directory / 'receipt.json').read_text())
        receipt['files']['tables/' + render.TABLE_FILES[0]] = file_hash(payload)
        write_json(directory / 'receipt.json', receipt)
        with self.assertRaises(ValidationError): self.run_render(resume=True)
        self.assertEqual(self.table_renderer.call_count, 2)

    def test_cached_extra_empty_directory_rejected(self):
        result = self.run_render(); directory = self.output / 'render_only' / result['fingerprint']
        (directory / 'undeclared-empty').mkdir()
        with self.assertRaises(ValidationError): self.run_render(resume=True)

    def test_cached_receipt_boolean_spoof_or_extra_field_rejected(self):
        result = self.run_render(); path = self.output / 'render_only' / result['fingerprint'] / 'receipt.json'
        original = json.loads(path.read_text())
        for key, value in (('schema_version', True), ('undeclared', 'private annotation')):
            write_json(path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValidationError): self.run_render(resume=True)

    def test_render_other_owner_output_lock_is_preserved(self):
        work = self.work / 'render_only'; work.mkdir(parents=True)
        output = self.output / 'render_only'; output.mkdir(parents=True)
        lock = output / '.preparation.lock'; lock.write_bytes(b'other owner')
        with self.assertRaises(ValidationError): self.run_render()
        self.assertEqual(lock.read_bytes(), b'other owner')
        self.assertFalse((work / '.preparation.lock').exists())
        self.table_renderer.assert_not_called()

    def test_render_never_calls_analysis_fitting_or_prepared_input_loaders(self):
        from dchs_lung_function import execution, primary_execution, wheeze_execution, sensitivity_execution
        from dchs_lung_function.analysis import pretb_uncertainty
        with ExitStack() as guards:
            sentinels = [guards.enter_context(mock.patch.object(module, function,
                         side_effect=AssertionError('Analysis is forbidden in render-only mode')))
                         for module, function in ((execution, 'load_artifacts'),
                             (primary_execution, 'sample_contract'), (wheeze_execution, 'sample_contract'),
                             (sensitivity_execution, 'sample_contract'), (pretb_uncertainty, 'calculate'))]
            self.assertTrue(self.run_render()['accepted'])
            for sentinel in sentinels: sentinel.assert_not_called()

    def test_pointer_write_failure_rolls_back_new_publication(self):
        with mock.patch.object(render, 'write_pointer', side_effect=OSError('Synthetic pointer failure')), self.assertRaises(ValidationError):
            self.run_render()
        self.assert_no_completion()
        self.assertTrue(self.run_render()['accepted'])

    def test_destination_symlink_race_rejected_before_publication(self):
        def race(destination, *args):
            self.fake_tables(destination, *args)
            identity = self.current_identity
            (self.output / 'render_only' / digest(identity)).symlink_to(self.base / 'missing', target_is_directory=True)
        original = render.digest
        def capture(value):
            if isinstance(value, dict) and value.get('mode') == 'render_only': self.current_identity = value
            return original(value)
        self.patch(render, 'digest', side_effect=capture)
        self.table_renderer.side_effect = race
        with self.assertRaises(ValidationError): self.run_render()
        self.assertFalse((self.work / 'render_current.json').exists())


class ReleaseCliTests(unittest.TestCase):
    def invoke(self, arguments):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err): status = pipeline.main(arguments, root=ROOT)
        return status, out.getvalue(), err.getvalue()

    def test_render_cli_routes_only_aggregate_config_and_resume(self):
        with mock.patch.object(render, 'run_render_only', return_value={'accepted': True}) as runner:
            status, _, _ = self.invoke(['--render', '--render-config', 'synthetic.json', '--resume'])
        self.assertEqual(status, 0)
        runner.assert_called_once_with(ROOT, 'synthetic.json', resume=True)

    def test_render_rejects_every_analysis_option_before_execution(self):
        options = [('--config', 'x'), ('--analysis', 'x'), ('--sampling', 'x'), ('--mechanics-sampling', 'x'),
                   ('--supporting-sampling', 'x'), ('--wheeze-sampling', 'x'), ('--sensitivity-sampling', 'x'),
                   ('--reporting', 'x'), ('--figures', 'x'), ('--targets', 'ratio_pre'), ('--jobs', '2')]
        with mock.patch.object(render, 'run_render_only') as runner:
            for option in options:
                with self.subTest(option=option):
                    status, _, _ = self.invoke(['--render', '--render-config', 'x', *option])
                    self.assertEqual(status, 2)
        runner.assert_not_called()

    def test_full_cli_routes_all_controls_and_never_accepts_subset(self):
        args = ['--full', '--config', 'p', '--analysis', 'a', '--reporting', 'r', '--figures', 'f',
                '--sampling', 's1', '--mechanics-sampling', 's2', '--supporting-sampling', 's3',
                '--wheeze-sampling', 's4', '--sensitivity-sampling', 's5', '--jobs', '3', '--resume']
        with mock.patch.object(full, 'run_full', return_value={'accepted': True, 'joint_fits_verified': 37, 'coverage': COVERAGE}) as runner:
            self.assertEqual(self.invoke(args)[0], 0)
            runner.assert_called_once_with(ROOT, 'p', 'a', 's1', 's2', 's3', 's4', 's5', 'r', 'f', jobs=3, resume=True)
            self.assertEqual(self.invoke(args + ['--targets', 'ratio_pre'])[0], 2)
            self.assertEqual(runner.call_count, 1)

    def test_modes_fail_closed_on_false_acceptance(self):
        with mock.patch.object(render, 'run_render_only', return_value={'accepted': False}):
            self.assertEqual(self.invoke(['--render', '--render-config', 'x'])[0], 2)
        with mock.patch.object(full, 'run_full', return_value={'accepted': False, 'joint_fits_verified': 0, 'coverage': COVERAGE}):
            self.assertEqual(self.invoke(['--full', '--config', 'p', '--analysis', 'a', '--reporting', 'r', '--figures', 'f'])[0], 2)


if __name__ == '__main__': unittest.main()
