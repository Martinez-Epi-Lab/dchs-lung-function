"""Crash-released local process locks and atomic no-clobber publication.

Directory descriptors are the lock inodes. They are never unlinked/replaced by
this module; every independently opened descriptor must acquire the same lock.
Supported execution filesystems must provide local POSIX flock/rename semantics.
"""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import os
from pathlib import Path
import platform
import stat

_HELD = set()


def _after_fork():
    # close, never LOCK_UN: the parent's shared open-file description owns it.
    for fd in tuple(_HELD):
        try: os.close(fd)
        except OSError: pass
    _HELD.clear()


os.register_at_fork(after_in_child=_after_fork)


def _safe_directory(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            system_alias = (platform.system()=='Darwin' and str(item) in {'/var','/tmp','/etc'}
                            and item.resolve()==Path('/private')/item.name)
            if not system_alias: raise ValueError('Process lock directories cannot be symlinks.')
    return path.resolve()


@contextmanager
def directory_locks(directories, *, legacy_name=None):
    """Nonblocking, ordered locks; process death releases all held descriptors."""
    opened = []
    owner_pid = os.getpid()
    try:
        for path in sorted({_safe_directory(p) for p in directories}):
            if legacy_name and ((path/legacy_name).exists() or (path/legacy_name).is_symlink()):
                raise ValueError('A legacy lock file requires explicit owner verification before migration.')
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = os.fstat(fd)
                current = path.stat(follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
                    raise ValueError('A process lock directory changed while acquiring ownership.')
            except BaseException:
                os.close(fd)
                raise
            _HELD.add(fd)
            opened.append((path, fd, info.st_dev, info.st_ino))
        yield
        for path, fd, device, inode in opened if os.getpid()==owner_pid else ():
            current = path.stat(follow_symlinks=False)
            if path.is_symlink() or (current.st_dev, current.st_ino) != (device, inode):
                raise ValueError('A process lock directory changed while owned.')
    finally:
        for _path, fd, _device, _inode in reversed(opened):
            if os.getpid()==owner_pid and fd in _HELD:  # never close a reused child descriptor
                _HELD.remove(fd)
                os.close(fd)


def rename_noreplace(source, destination):
    """Atomically expose a complete directory, refusing every existing target."""
    source, destination = Path(source), Path(destination)
    _safe_directory(source.parent); _safe_directory(destination.parent)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Publication requires a real staged directory.')
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == 'Darwin' and hasattr(libc, 'renamex_np'):
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif system == 'Linux' and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise RuntimeError('Safe publication requires Linux renameat2 or macOS renamex_np.')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def sync_tree(path):
    for item in Path(path).iterdir():
        if item.is_symlink() or not item.is_file():
            raise ValueError('Publication staging must contain only regular files.')
        fd = os.open(item, os.O_RDONLY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
    sync_directory(path)
