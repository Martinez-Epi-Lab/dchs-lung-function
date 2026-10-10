"""Independent source-release attacks using invented, source-only trees."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dchs_independent_exporter', ROOT / 'tools/export_source.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_archive(path, payloads, manifest=None, *, mutate_info=None, suffix=b'', prefix=b''):
    """Independent ZIP writer, also used to build internally consistent forgeries."""
    if manifest is None:
        manifest = {'schema_version': 1, 'format': 'dchs-source-release-v1',
                    'allowlist_sha256': sha(payloads['config/source_allowlist.json']),
                    'files': {name: {'sha256': sha(data), 'size': len(data)}
                              for name, data in sorted(payloads.items())}}
    manifest_bytes = manifest if isinstance(manifest, bytes) else canonical(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted({**payloads, 'SOURCE_MANIFEST.json': manifest_bytes}.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            if mutate_info:
                mutate_info(info)
            archive.writestr(info, data)
    path.write_bytes(prefix + stream.getvalue() + suffix)
    return manifest


class ExportSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dchs-source-test-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / 'source'
        self.source.mkdir()
        self.names = ['README.md', 'config/source_allowlist.json', 'tools/export_source.py']
        self.payloads = {'README.md': b'Wholly invented source fixture.\n',
                         'tools/export_source.py': b'"""Invented text, never executed."""\n',
                         'config/source_allowlist.json': canonical({'schema_version': 1, 'files': self.names})}
        for name, data in self.payloads.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.archive = self.base / 'source.zip'

    def exported(self):
        return exporter.export_source(self.source, self.archive)

    def assert_clean(self):
        self.assertEqual(list(self.base.glob('.source-*')), [])

    def test_deterministic_roundtrip_and_external_identity(self):
        first = self.exported()
        (self.source / 'README.md').chmod(0o600)
        os.utime(self.source / 'README.md', (1234567890, 1234567890))
        second = exporter.export_source(self.source, self.base / 'second.zip')
        self.assertEqual(self.archive.read_bytes(), (self.base / 'second.zip').read_bytes())
        self.assertEqual(first['source_manifest_sha256'], second['source_manifest_sha256'])
        unchecked = exporter.verify_archive(self.archive)
        self.assertFalse(unchecked['reviewed_manifest_matched'])
        checked = exporter.verify_archive(self.archive, expected_manifest_sha256=first['source_manifest_sha256'])
        self.assertTrue(checked['reviewed_manifest_matched'])
        destination = self.base / 'extracted'
        exporter.extract_archive(self.archive, destination, expected_manifest_sha256=first['source_manifest_sha256'])
        exporter.verify_tree(destination, expected_manifest_sha256=first['source_manifest_sha256'])
        exporter.export_source(destination, self.base / 'third.zip')
        self.assertEqual(self.archive.read_bytes(), (self.base / 'third.zip').read_bytes())
        self.assertEqual({p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()},
                         set(self.names) | {'SOURCE_MANIFEST.json'})
        self.assert_clean()

    def test_consistently_rehashed_forgery_requires_reviewed_anchor(self):
        original = self.exported()
        forged = dict(self.payloads, **{'README.md': b'Changed scientific source.\n'})
        write_archive(self.base / 'forged.zip', forged)
        self.assertFalse(exporter.verify_archive(self.base / 'forged.zip')['reviewed_manifest_matched'])
        with self.assertRaises(exporter.ExportError):
            exporter.verify_archive(self.base / 'forged.zip', expected_manifest_sha256=original['source_manifest_sha256'])

    def test_wrong_and_malformed_external_hash_rejected(self):
        self.exported()
        for value in ('0' * 64, 'A' * 64, 'x', True, 17):
            with self.subTest(value=value), self.assertRaises(exporter.ExportError):
                exporter.verify_archive(self.archive, expected_manifest_sha256=value)

    def test_source_inventory_rejects_unlisted_file_and_empty_directory(self):
        for name in ('secrets.txt', 'unused'):
            path = self.source / name
            path.mkdir() if name == 'unused' else path.write_text('Invented secret')
            with self.subTest(name=name), self.assertRaises(exporter.ExportError):
                self.exported()
            path.rmdir() if path.is_dir() else path.unlink()
        self.assertFalse(self.archive.exists())
        self.assert_clean()

    def test_development_git_and_bytecode_are_excluded_but_not_arbitrary_cache_payload(self):
        (self.source / '.git').mkdir()
        (self.source / '.git' / 'invented').write_bytes(b'not exported')
        cache = self.source / 'tools/__pycache__'
        cache.mkdir()
        (cache / 'invented.pyc').write_bytes(b'not exported')
        (self.source / '.DS_Store').write_bytes(b'not exported')
        self.exported()
        with zipfile.ZipFile(self.archive) as z:
            self.assertEqual(set(z.namelist()), set(self.names) | {'SOURCE_MANIFEST.json'})
        (cache / 'private.txt').write_text('not an allowed cache')
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(self.source, self.base / 'another.zip')

    def test_manifested_git_checkout_exports_exact_payload_but_verify_tree_stays_strict(self):
        original = self.exported()
        checkout = self.base / 'checkout'
        exporter.extract_archive(self.archive, checkout)
        (checkout / '.git').mkdir()
        (checkout / '.git/config').write_text('Invented development metadata')
        cache = checkout / 'tools/__pycache__'
        cache.mkdir()
        (cache / 'invented.pyc').write_bytes(b'not exported')
        (checkout / '.DS_Store').write_bytes(b'not exported')
        second = self.base / 'checkout.zip'
        result = exporter.export_source(checkout, second)
        self.assertEqual(result['source_manifest_sha256'], original['source_manifest_sha256'])
        self.assertEqual(second.read_bytes(), self.archive.read_bytes())
        with self.assertRaises(exporter.ExportError):
            exporter.verify_tree(checkout)
        with zipfile.ZipFile(second) as archive:
            self.assertEqual(set(archive.namelist()), set(self.names) | {'SOURCE_MANIFEST.json'})
        self.assert_clean()

    def test_manifested_checkout_rejects_extra_directory_cache_payload_and_changed_source(self):
        self.exported()
        checkout = self.base / 'checkout'
        exporter.extract_archive(self.archive, checkout)
        (checkout / '.git').mkdir()
        destination = self.base / 'rejected.zip'
        extra = checkout / 'unexpected'
        extra.mkdir()
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(checkout, destination)
        extra.rmdir()
        cache = checkout / 'tools/__pycache__'
        cache.mkdir()
        extra = cache / 'private.txt'
        extra.write_text('Invented non-bytecode content')
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(checkout, destination)
        extra.unlink()
        source = checkout / 'README.md'
        source.write_bytes(source.read_bytes() + b'Changed source\n')
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(checkout, destination)
        self.assertFalse(destination.exists())
        self.assert_clean()

    def test_manifest_change_during_manifested_checkout_snapshot_is_rejected(self):
        self.exported()
        checkout = self.base / 'checkout'
        exporter.extract_archive(self.archive, checkout)
        (checkout / '.git').mkdir()
        original_read = exporter._regular_bytes
        manifest = checkout / 'SOURCE_MANIFEST.json'
        def read(path):
            data = original_read(path)
            if Path(path) == checkout / 'README.md':
                manifest.write_bytes(manifest.read_bytes() + b'\n')
            return data
        destination = self.base / 'rejected.zip'
        with mock.patch.object(exporter, '_regular_bytes', side_effect=read):
            with self.assertRaisesRegex(exporter.ExportError, 'manifest changed'):
                exporter.export_source(checkout, destination)
        self.assertFalse(destination.exists())
        self.assert_clean()

    def test_unsafe_and_private_allowlist_paths_rejected(self):
        invalid = ('../README.md', '/README.md', 'config/../README.md', 'config/a\\b.json',
                   'data/fake.py', 'src/private/fake.py', 'config/paths.local.json', 'config/.env.json',
                   'docs/CON.md', 'docs/a..', 'src/participant.csv', 'config/a.json/extra.json',
                   'SOURCE_MANIFEST.json', '.github/workflows/unsafe.yaml')
        for name in invalid:
            names = sorted(self.names + [name])
            (self.source / 'config/source_allowlist.json').write_bytes(canonical({'schema_version': 1, 'files': names}))
            with self.subTest(name=name), self.assertRaises(exporter.ExportError):
                self.exported()
        self.assertFalse(self.archive.exists())

    def test_allowlist_schema_order_duplicates_and_required_members(self):
        invalid = [dict(schema_version=True, files=self.names), dict(schema_version=1, files=list(reversed(self.names))),
                   dict(schema_version=1, files=sorted(self.names + ['README.md'])),
                   dict(schema_version=1, files=sorted(self.names + ['readme.md'])),
                   dict(schema_version=1, files=self.names[:-1]), dict(schema_version=1, files=self.names, extra=1),
                   dict(schema_version=1, files=None)]
        for value in invalid:
            (self.source / 'config/source_allowlist.json').write_bytes(canonical(value))
            with self.subTest(value=value), self.assertRaises(exporter.ExportError):
                self.exported()

    def test_duplicate_json_keys_nonfinite_and_invalid_text_rejected(self):
        path = self.source / 'config/source_allowlist.json'
        for data in (b'{"schema_version":1,"schema_version":1,"files":[]}', b'{"schema_version":NaN,"files":[]}', b'\xff'):
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(exporter.ExportError):
                self.exported()
        path.write_bytes(self.payloads['config/source_allowlist.json'])
        for data in (b'\xff', b'source\x00binary'):
            (self.source / 'README.md').write_bytes(data)
            with self.subTest(data=data), self.assertRaises(exporter.ExportError):
                self.exported()

    def test_source_symlinks_hardlinks_and_special_files_rejected(self):
        path = self.source / 'README.md'
        outside = self.base / 'outside.md'
        outside.write_bytes(self.payloads['README.md'])
        for kind in ('symlink', 'dangling', 'hardlink', 'fifo'):
            path.unlink(missing_ok=True)
            if kind == 'symlink': path.symlink_to(outside)
            elif kind == 'dangling': path.symlink_to(self.base / 'absent')
            elif kind == 'hardlink': os.link(outside, path)
            else: os.mkfifo(path)
            with self.subTest(kind=kind), self.assertRaises(exporter.ExportError):
                self.exported()
        path.unlink()

    def test_source_ancestor_symlink_rejected(self):
        alias = self.base / 'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(alias, self.archive)

    def test_missing_source_and_foreign_allowlist_rejected(self):
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(self.source, self.archive, allowlist=self.base / 'other.json')
        (self.source / 'README.md').unlink()
        with self.assertRaises(exporter.ExportError):
            self.exported()

    def test_source_file_and_total_bounds_checked_before_publication(self):
        for limit in ('MAX_FILE_BYTES', 'MAX_TOTAL_BYTES'):
            with self.subTest(limit=limit), mock.patch.object(exporter, limit, 8), self.assertRaises(exporter.ExportError):
                self.exported()
        self.assertFalse(self.archive.exists())
        self.assert_clean()

    def test_export_never_overwrites_or_writes_inside_source(self):
        self.archive.write_bytes(b'keep')
        with self.assertRaises(exporter.ExportError): self.exported()
        self.assertEqual(self.archive.read_bytes(), b'keep')
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(self.source, self.source / 'archive.zip')

    def test_destination_symlink_including_dangling_is_rejected(self):
        for target in (self.base / 'missing', self.source / 'README.md'):
            self.archive.symlink_to(target)
            with self.subTest(target=target), self.assertRaises(exporter.ExportError): self.exported()
            self.archive.unlink()
        parent = self.base / 'alias'
        parent.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(exporter.ExportError):
            exporter.export_source(self.source, parent / 'archive.zip')

    def test_export_source_mutation_before_and_after_publication_rolls_back(self):
        original = exporter._snapshot
        for mutate_call in (2, 3):
            calls = 0
            def snapshot(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == mutate_call:
                    (self.source / 'README.md').write_text('source mutated')
                return original(*args, **kwargs)
            (self.source / 'README.md').write_bytes(self.payloads['README.md'])
            with self.subTest(mutate_call=mutate_call), mock.patch.object(exporter, '_snapshot', side_effect=snapshot), self.assertRaises(exporter.ExportError):
                self.exported()
            self.assertFalse(self.archive.exists())
            self.assert_clean()

    def test_export_destination_race_preserves_competing_file(self):
        original = os.link
        def collide(source, destination):
            Path(destination).write_bytes(b'other owner')
            return original(source, destination)
        with mock.patch.object(exporter.os, 'link', side_effect=collide), self.assertRaises(exporter.ExportError):
            self.exported()
        self.assertEqual(self.archive.read_bytes(), b'other owner')
        self.assert_clean()

    def test_archive_payload_hash_and_size_are_verified(self):
        manifest = write_archive(self.archive, self.payloads)
        for field, value in (('sha256', '0' * 64), ('size', 900), ('size', True)):
            changed = deepcopy(manifest)
            changed['files']['README.md'][field] = value
            write_archive(self.archive, self.payloads, changed)
            with self.subTest(field=field, value=value), self.assertRaises(exporter.ExportError):
                exporter.verify_archive(self.archive)

    def test_archive_manifest_exact_schema_and_canonical_bytes(self):
        manifest = write_archive(self.archive, self.payloads)
        cases = [dict(manifest, schema_version=True), dict(manifest, extra='undeclared'),
                 dict(manifest, allowlist_sha256='0' * 64), json.dumps(manifest, indent=2).encode(),
                 canonical(manifest).replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')]
        for value in cases:
            write_archive(self.archive, self.payloads, value)
            with self.subTest(value=str(value)[:30]), self.assertRaises(exporter.ExportError):
                exporter.verify_archive(self.archive)

    def test_archive_prefix_suffix_and_truncated_payload_rejected(self):
        self.exported()
        original = self.archive.read_bytes()
        for data in (b'private-prefix' + original, original + b'private-suffix', original[:-7], b'not zip'):
            self.archive.write_bytes(data)
            with self.subTest(size=len(data)), self.assertRaises(exporter.ExportError):
                exporter.verify_archive(self.archive)

    def test_archive_members_must_be_normalized_regular_uncompressed(self):
        changes = (lambda i: setattr(i, 'compress_type', zipfile.ZIP_DEFLATED),
                   lambda i: setattr(i, 'external_attr', (stat.S_IFLNK | 0o777) << 16),
                   lambda i: setattr(i, 'create_system', 0),
                   lambda i: setattr(i, 'comment', b'private comment'),
                   lambda i: setattr(i, 'date_time', (2020, 1, 1, 0, 0, 0)))
        for change in changes:
            write_archive(self.archive, self.payloads, mutate_info=change)
            with self.subTest(change=change), self.assertRaises(exporter.ExportError):
                exporter.verify_archive(self.archive)

    def test_archive_traversal_or_extra_member_rejected(self):
        for name in ('../escape.py', 'docs/unlisted.md', 'config/paths.local.json'):
            payloads = {**self.payloads, name: b'invented'}
            write_archive(self.archive, payloads)
            with self.subTest(name=name), self.assertRaises(exporter.ExportError):
                exporter.extract_archive(self.archive, self.base / 'extracted')
            self.assertFalse((self.base / 'extracted').exists())
            self.assertFalse((self.base / 'escape.py').exists())

    def test_extracted_tree_rejects_tampering_missing_extras_and_caches(self):
        result = self.exported()
        destination = self.base / 'extracted'
        exporter.extract_archive(self.archive, destination)
        path = destination / 'README.md'
        path.write_text('altered')
        with self.assertRaises(exporter.ExportError): exporter.verify_tree(destination)
        path.unlink()
        with self.assertRaises(exporter.ExportError): exporter.verify_tree(destination)
        path.write_bytes(self.payloads['README.md'])
        extra = destination / '.DS_Store'
        extra.write_bytes(b'cache')
        with self.assertRaises(exporter.ExportError): exporter.verify_tree(destination)
        extra.unlink()
        exporter.verify_tree(destination, expected_manifest_sha256=result['source_manifest_sha256'])

    def test_extract_no_clobber_and_ancestor_symlink(self):
        self.exported()
        destination = self.base / 'extracted'
        destination.mkdir()
        (destination / 'keep').write_bytes(b'other owner')
        with self.assertRaises(exporter.ExportError): exporter.extract_archive(self.archive, destination)
        self.assertEqual((destination / 'keep').read_bytes(), b'other owner')
        alias = self.base / 'alias'
        alias.symlink_to(destination, target_is_directory=True)
        with self.assertRaises(exporter.ExportError): exporter.extract_archive(self.archive, alias / 'subdir')
        self.assert_clean()

    def test_extract_destination_race_preserves_competing_tree(self):
        self.exported()
        destination = self.base / 'extracted'
        original = exporter._rename_directory_noreplace
        def collide(source, target):
            Path(target).mkdir()
            (Path(target) / 'keep').write_bytes(b'other owner')
            original(source, target)
        with mock.patch.object(exporter, '_rename_directory_noreplace', side_effect=collide), self.assertRaises(exporter.ExportError):
            exporter.extract_archive(self.archive, destination)
        self.assertEqual((destination / 'keep').read_bytes(), b'other owner')
        self.assert_clean()

    def test_extract_final_validation_failure_removes_only_own_tree(self):
        self.exported()
        destination = self.base / 'extracted'
        original = exporter.verify_tree
        def fail_final(path, **kwargs):
            if Path(path) == destination:
                raise exporter.ExportError('Synthetic final validation failure.')
            return original(path, **kwargs)
        with mock.patch.object(exporter, 'verify_tree', side_effect=fail_final), self.assertRaises(exporter.ExportError):
            exporter.extract_archive(self.archive, destination)
        self.assertFalse(destination.exists())
        self.assert_clean()

    def test_extract_archive_mutation_before_publication_rejected(self):
        self.exported()
        original = exporter._write_tree
        def mutate(*args):
            original(*args)
            with self.archive.open('ab') as stream: stream.write(b'changed')
        with mock.patch.object(exporter, '_write_tree', side_effect=mutate), self.assertRaises(exporter.ExportError):
            exporter.extract_archive(self.archive, self.base / 'extracted')
        self.assertFalse((self.base / 'extracted').exists())
        self.assert_clean()

    def test_cli_success_is_aggregate_and_errors_are_sanitized(self):
        out, error = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(error):
            self.assertEqual(exporter.main(['export', '--root', str(self.source), '--output', str(self.archive)]), 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result['source_files'], 3)
        self.assertNotIn('manifest', result)
        self.assertNotIn(str(self.base), out.getvalue())
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(exporter.main(['verify', str(self.base / 'invented_private_name.zip')]), 2)
        self.assertNotIn(str(self.base), error.getvalue())
        self.assertNotIn('invented_private_name', error.getvalue())


if __name__ == '__main__':
    unittest.main()
