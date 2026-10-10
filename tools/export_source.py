"""Create and verify a deterministic, explicitly allowlisted source ZIP.

The reviewed allowlist is a confidentiality boundary, not a file-discovery glob.
Archive verification checks integrity. A trusted external manifest SHA256 is
required to establish that the package is the particular reviewed release.
Only the standard library is used; no study inputs or analysis modules are read.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tempfile
import zipfile


ALLOWLIST = 'config/source_allowlist.json'
MANIFEST = 'SOURCE_MANIFEST.json'
FORMAT = 'dchs-source-release-v1'
MAX_FILES = 4096
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
STAMP = (1980, 1, 1, 0, 0, 0)
MODE = stat.S_IFREG | 0o644
_FORBIDDEN = {'data', 'private', 'work', 'outputs', 'draws', 'release', 'reviews',
              'results', 'posterior', '.venv', 'venv', '__pycache__', '.pytest_cache', '.git'}
_RESERVED = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)),
             *(f'lpt{i}' for i in range(1, 10))}


class ExportError(ValueError):
    """A source package failed its explicit release contract."""


def _require(condition, message):
    if not condition:
        raise ExportError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'JSON contains a duplicate key.')
        result[key] = value
    return result


def _json(data):
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ExportError('Nonfinite JSON value.')))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError('A package configuration or manifest is not valid UTF-8 JSON.') from error


def _no_symlinks(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             'Source and destination paths must not contain symbolic links.')
    return path


def _resolve(path):
    return _no_symlinks(Path(path).expanduser()).resolve()


def _name(name, *, payload=True):
    _require(isinstance(name, str) and 0 < len(name) <= 240,
             'A package path is invalid.')
    parts = name.split('/')
    _require(not PurePosixPath(name).is_absolute() and all(
        part not in {'', '.', '..'} and re.fullmatch(r'[A-Za-z0-9_.-]+', part)
        and not part.endswith('.') and part.split('.')[0].lower() not in _RESERVED
        for part in parts), 'A package path is unsafe or not portable.')
    if not payload:
        return name
    _require(name != MANIFEST and not any(p.lower() in _FORBIDDEN for p in parts)
             and not any('.local.' in p.lower() or p.lower().startswith('.env') for p in parts),
             'The allowlist contains a private or generated path.')
    allowed = (name in {'.gitignore', '.gitattributes', 'README.md', 'run.py',
                        'requirements-models.lock.txt', 'requirements-preparation.lock.txt'}
               or len(parts) == 3 and parts[:2] == ['.github', 'workflows'] and parts[-1].endswith('.yml')
               or parts[0] in {'src', 'tests', 'tools'} and len(parts) > 1 and parts[-1].endswith('.py')
               or parts[0] == 'config' and len(parts) == 2 and parts[-1].endswith('.json')
               or parts[0] == 'docs' and len(parts) == 2 and parts[-1].endswith('.md'))
    _require(allowed, 'Only reviewed code, documentation, examples, locks and synthetic checks may be exported.')
    return name


def _names(names):
    _require(isinstance(names, list) and 1 <= len(names) <= MAX_FILES
             and all(isinstance(n, str) for n in names), 'The source allowlist must contain an explicit file list.')
    _require(names == sorted(names) and len({n.casefold() for n in names}) == len(names),
             'Allowlisted paths must be sorted, unique and case-distinct.')
    for name in names:
        _name(name)
    return names


def _allowlist(data):
    value = _json(data)
    _require(isinstance(value, dict) and set(value) == {'schema_version', 'files'}
             and type(value['schema_version']) is int and value['schema_version'] == 1,
             'The source allowlist schema is invalid.')
    names = _names(value['files'])
    _require(ALLOWLIST in names and 'tools/export_source.py' in names,
             'The allowlist must include itself and the source exporter.')
    return names


def _regular_bytes(path):
    path = _no_symlinks(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                     'Release inputs must be regular files, without hard links.')
            _require(before.st_size <= MAX_FILE_BYTES, 'A release input exceeds the source-file size limit.')
            data = stream.read(MAX_FILE_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as error:
        raise ExportError('A required release file cannot be read safely.') from error
    _require(len(data) == before.st_size and len(data) <= MAX_FILE_BYTES
             and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
             == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns),
             'A release input changed while being read.')
    _no_symlinks(path)
    return data


def _text(data):
    try:
        data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ExportError('Allowlisted files must be UTF-8 source text.') from error
    _require(b'\0' not in data, 'Binary data cannot be included in a source release.')


def _inventory(root, names, *, development=False, include_manifest=False):
    expected = set(names) | ({MANIFEST} if include_manifest else set())
    parents = {str(p) for n in expected for p in PurePosixPath(n).parents if str(p) != '.'}
    found = set()
    _require(root.is_dir(), 'The source root must be a directory.')
    for here, directories, files in os.walk(root, followlinks=False):
        here = Path(here)
        for name in list(directories):
            p = here / name
            _no_symlinks(p)
            relative = p.relative_to(root).as_posix()
            if development and relative == '.git':
                directories.remove(name)
            elif development and '__pycache__' in p.relative_to(root).parts:
                continue
            else:
                _require(relative in parents, 'The source tree contains an unexpected directory.')
        for name in files:
            p = here / name
            _no_symlinks(p)
            _require(stat.S_ISREG(p.stat().st_mode), 'The source tree contains a special file.')
            relative = p.relative_to(root).as_posix()
            if development and (relative == '.git' or name == '.DS_Store'
                                or '__pycache__' in p.relative_to(root).parts and name.endswith('.pyc')):
                continue
            found.add(relative)
    _require(found == expected, 'The source tree contains missing or unexpected files.')


def _snapshot(root, allowlist=None):
    root = _resolve(root)
    chosen = root / ALLOWLIST if allowlist is None else _resolve(allowlist)
    _require(chosen == root / ALLOWLIST, 'Use the reviewed allowlist at config/source_allowlist.json.')
    data = _regular_bytes(chosen)
    names = _allowlist(data)
    existing_manifest = (root / MANIFEST).exists()
    if existing_manifest:
        verify_tree(root)
    _inventory(root, names, development=True, include_manifest=existing_manifest)
    payloads = {n: _regular_bytes(root / n) for n in names}
    _require(payloads[ALLOWLIST] == data, 'The allowlist changed during source selection.')
    _require(sum(map(len, payloads.values())) <= MAX_TOTAL_BYTES, 'The release exceeds the source-size limit.')
    for content in payloads.values():
        _text(content)
    _inventory(root, names, development=True, include_manifest=existing_manifest)
    return payloads


def _manifest(payloads):
    return {'schema_version': 1, 'format': FORMAT, 'allowlist_sha256': _sha(payloads[ALLOWLIST]),
            'files': {n: {'sha256': _sha(data), 'size': len(data)} for n, data in sorted(payloads.items())}}


def _validate_payloads(payloads, manifest_data, expected_manifest_sha256=None):
    manifest = _json(manifest_data)
    _require(isinstance(manifest, dict) and set(manifest) == {'schema_version', 'format', 'allowlist_sha256', 'files'}
             and type(manifest['schema_version']) is int and manifest['schema_version'] == 1
             and manifest['format'] == FORMAT and isinstance(manifest['files'], dict),
             'The source manifest schema is invalid.')
    _require(manifest_data == _json_bytes(manifest), 'The source manifest is not in canonical form.')
    if expected_manifest_sha256 is not None:
        _require(isinstance(expected_manifest_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256),
                 'The expected manifest SHA256 is invalid.')
        _require(_sha(manifest_data) == expected_manifest_sha256, 'The package does not match the reviewed manifest.')
    names = _names(list(manifest['files']))
    _require(set(payloads) == set(names) and ALLOWLIST in payloads,
             'The package payload inventory differs from its source manifest.')
    _require(_allowlist(payloads[ALLOWLIST]) == names and manifest['allowlist_sha256'] == _sha(payloads[ALLOWLIST]),
             'The archived allowlist and source manifest differ.')
    _require(sum(map(len, payloads.values())) <= MAX_TOTAL_BYTES, 'The package exceeds the source-size limit.')
    for name, data in payloads.items():
        entry = manifest['files'][name]
        _require(isinstance(entry, dict) and set(entry) == {'sha256', 'size'}
                 and type(entry['size']) is int and 0 <= entry['size'] <= MAX_FILE_BYTES
                 and isinstance(entry['sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['sha256'])
                 and entry['size'] == len(data) and entry['sha256'] == _sha(data),
                 'A source payload differs from its declared size or hash.')
        _text(data)
    return {'schema_version': 1, 'status': 'verified', 'source_files': len(names),
            'source_manifest_sha256': _sha(manifest_data),
            'reviewed_manifest_matched': expected_manifest_sha256 is not None,
            'manifest': manifest}


def _archive_payloads(archive):
    archive = _resolve(archive)
    _require(archive.is_file() and archive.stat().st_size <= MAX_TOTAL_BYTES + 4 * 1024 * 1024,
             'The archive is missing or exceeds the release-size limit.')
    try:
        with zipfile.ZipFile(archive, 'r') as z:
            infos = z.infolist()
            _require(not z.comment and 2 <= len(infos) <= MAX_FILES + 1, 'The archive inventory is invalid.')
            names = [i.filename for i in infos]
            _require(names == sorted(names) and len({n.casefold() for n in names}) == len(names)
                     and MANIFEST in names, 'Archive paths must be sorted, unique and include the source manifest.')
            for entry in infos:
                _name(entry.filename, payload=entry.filename != MANIFEST)
                _require(not entry.is_dir() and entry.create_system == 3 and entry.date_time == STAMP
                         and entry.external_attr == MODE << 16 and entry.internal_attr == 0
                         and entry.compress_type == zipfile.ZIP_STORED and entry.flag_bits == 0
                         and not entry.extra and not entry.comment and entry.file_size == entry.compress_size
                         and 0 <= entry.file_size <= MAX_FILE_BYTES,
                         'Archive members must be normalized uncompressed regular source files.')
            _require(sum(i.file_size for i in infos) <= MAX_TOTAL_BYTES + MAX_FILE_BYTES,
                     'The archive expands beyond the source-size limit.')
            payloads = {entry.filename: z.read(entry) for entry in infos}
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ExportError('The source archive cannot be read or has invalid ZIP integrity.') from error
    manifest_data = payloads.pop(MANIFEST)
    canonical = io.BytesIO()
    _write_zip(canonical, payloads, manifest_data)
    _require(_file_sha(archive) == _sha(canonical.getvalue()),
             'The archive contains noncanonical headers or bytes outside its exact source members.')
    return payloads, manifest_data


def verify_archive(archive, *, expected_manifest_sha256=None):
    """Verify exact contents; an external manifest hash anchors reviewed identity."""
    archive = _resolve(archive)
    before_hash = _file_sha(archive)
    payloads, manifest_data = _archive_payloads(archive)
    result = _validate_payloads(payloads, manifest_data, expected_manifest_sha256)
    after_hash = _file_sha(archive)
    _require(after_hash == before_hash, 'The archive changed during verification.')
    return {**result, 'archive_sha256': after_hash, 'archive_bytes': archive.stat().st_size}


def _file_sha(path):
    path = _no_symlinks(path)
    h = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_TOTAL_BYTES + MAX_FILE_BYTES,
                     'The archive must be a bounded regular file.')
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as error:
        raise ExportError('The archive cannot be read safely.') from error
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
             == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns),
             'The archive changed while being hashed.')
    _no_symlinks(path)
    return h.hexdigest()


def verify_tree(root, *, expected_manifest_sha256=None):
    """Verify an extracted package, rejecting all extras including caches/.git."""
    root = _resolve(root)
    _require((root / MANIFEST).exists() or (root / MANIFEST).is_symlink(),
             "SOURCE_MANIFEST.json is missing; verify-tree requires an extracted source release, not the development checkout.")
    manifest_data = _regular_bytes(root / MANIFEST)
    manifest = _json(manifest_data)
    _require(isinstance(manifest, dict) and isinstance(manifest.get('files'), dict),
             'The extracted source manifest is invalid.')
    names = _names(list(manifest['files']))
    _inventory(root, names, include_manifest=True)
    payloads = {n: _regular_bytes(root / n) for n in names}
    result = _validate_payloads(payloads, manifest_data, expected_manifest_sha256)
    _inventory(root, names, include_manifest=True)
    _require(all(_regular_bytes(root / n) == content for n, content in payloads.items()),
             'An extracted source file changed during verification.')
    _require(_regular_bytes(root / MANIFEST) == manifest_data, 'The extracted manifest changed during verification.')
    return result


def _destination(path, root=None):
    path = _resolve(path)
    _require(not path.exists(), 'The destination already exists; source exports never overwrite files or trees.')
    if root is not None:
        _require(path != root and root not in path.parents, 'The export destination must be outside the source tree.')
    _no_symlinks(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(path.parent)
    return path


def _write_tree(directory, payloads, manifest_data):
    for name, data in sorted({**payloads, MANIFEST: manifest_data}.items()):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        path.chmod(0o644)


def _write_zip(destination, payloads, manifest_data):
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as z:
        for name, data in sorted({**payloads, MANIFEST: manifest_data}.items()):
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.create_system = 3
            info.external_attr = MODE << 16
            info.internal_attr = 0
            info.compress_type = zipfile.ZIP_STORED
            z.writestr(info, data)


def _rename_directory_noreplace(source, destination):
    """Atomic directory publication that never replaces an existing target."""
    system = platform.system()
    if system == 'Windows':
        os.rename(source, destination)  # Windows rename refuses an existing target.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if system == 'Darwin' and hasattr(libc, 'renamex_np'):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif system == 'Linux' and hasattr(libc, 'renameat2'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise ExportError('This platform does not provide safe no-clobber directory publication.')
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise ExportError('The extraction destination appeared during verification.')
        raise ExportError('Atomic source-directory publication failed.')


def extract_archive(archive, destination, *, expected_manifest_sha256=None):
    """Validate, stage, verify and atomically publish an exact source tree."""
    archive = _resolve(archive)
    before = _file_sha(archive)
    payloads, manifest_data = _archive_payloads(archive)
    result = _validate_payloads(payloads, manifest_data, expected_manifest_sha256)
    destination = _destination(destination)
    _require(destination != archive and destination not in archive.parents,
             'The extraction destination must not contain the source archive.')
    staged = Path(tempfile.mkdtemp(prefix='.source-extract-', dir=destination.parent))
    staged_inode = staged.stat().st_ino
    published = False
    try:
        _write_tree(staged, payloads, manifest_data)
        verify_tree(staged, expected_manifest_sha256=result['source_manifest_sha256'])
        _require(_file_sha(archive) == before, 'The archive changed during extraction.')
        _no_symlinks(destination)
        _rename_directory_noreplace(staged, destination)
        published = True
        verify_tree(destination, expected_manifest_sha256=result['source_manifest_sha256'])
        _require(_file_sha(archive) == before, 'The archive changed during extraction publication.')
    except Exception:
        if published and destination.is_dir() and not destination.is_symlink() and destination.stat().st_ino == staged_inode:
            shutil.rmtree(destination)
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {**result, 'status': 'extracted', 'archive_sha256': before}


def export_source(root, archive, *, allowlist=None):
    """Export reviewed text paths; reject unlisted files and changed sources."""
    root = _resolve(root)
    payloads = _snapshot(root, allowlist)
    manifest_data = _json_bytes(_manifest(payloads))
    manifest_hash = _sha(manifest_data)
    archive = _destination(archive, root)
    descriptor, temporary = tempfile.mkstemp(prefix='.source-archive-', suffix='.zip', dir=archive.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    published_inode = None
    try:
        _write_zip(temporary, payloads, manifest_data)
        result = verify_archive(temporary, expected_manifest_sha256=manifest_hash)
        with tempfile.TemporaryDirectory(prefix='.source-check-', dir=archive.parent) as directory:
            _write_tree(Path(directory), payloads, manifest_data)
            verify_tree(directory, expected_manifest_sha256=manifest_hash)
        _require(_snapshot(root, allowlist) == payloads, 'Source files changed while the archive was being prepared.')
        _no_symlinks(archive)
        try:
            os.link(temporary, archive)  # Atomic creation; an existing destination is never overwritten.
        except OSError as error:
            raise ExportError('The archive destination appeared or cannot be published safely.') from error
        published_inode = archive.stat().st_ino
        _require(_snapshot(root, allowlist) == payloads, 'Source files changed during archive publication.')
        _require(_file_sha(archive) == result['archive_sha256'], 'The published archive changed.')
        return {**result, 'status': 'exported'}
    except Exception:
        if published_inode is not None and archive.exists() and not archive.is_symlink() and archive.stat().st_ino == published_inode:
            archive.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='create a reviewed-source-only ZIP')
    export.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--allowlist', type=Path)
    for name in ('verify', 'extract', 'verify-tree'):
        command = commands.add_parser(name)
        command.add_argument('source', type=Path)
        if name == 'extract':
            command.add_argument('destination', type=Path)
        command.add_argument('--manifest-sha256')
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'export':
            result = export_source(arguments.root, arguments.output, allowlist=arguments.allowlist)
        elif arguments.command == 'verify':
            result = verify_archive(arguments.source, expected_manifest_sha256=arguments.manifest_sha256)
        elif arguments.command == 'extract':
            result = extract_archive(arguments.source, arguments.destination, expected_manifest_sha256=arguments.manifest_sha256)
        else:
            result = verify_tree(arguments.source, expected_manifest_sha256=arguments.manifest_sha256)
        print(json.dumps({k: v for k, v in result.items() if k != 'manifest'}, sort_keys=True))
        return 0
    except (ExportError, OSError) as error:
        print('Source release validation failed: ' + (str(error) if isinstance(error, ExportError) else 'A filesystem operation failed.'), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
