# Numerical runtime and interrupted-run recovery

Use the pinned interpreter with `python -B run.py ...`. The launcher sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `BLIS_NUM_THREADS` to `1`
before importing scientific libraries. `tests/run_tests.py` uses the same
bootstrap. Library callers must call
`dchs_lung_function.runtime_bootstrap.bootstrap()` before importing NumPy,
SciPy, PyTensor or PyMC. A late call is rejected, including when the environment
already happens to contain these values. Subsequent changes to these variables
are rejected. The requested controls propagate to newly launched worker processes.

The model runtime identity records the Python and package versions, OS/kernel
release and version, architecture, resolved C++ compiler command and binary hash,
compiler version, PyTensor linker/precision/BLAS/compiler settings, relevant
compiler environment variables, NumPy BLAS/LAPACK build identities and requested
thread controls. The macOS linker adapter remains recorded. Private compiler
cache paths are excluded from identity so moving the cache does not change the
scientific runtime. A small isolated configuration probe obtains the actual
PyTensor compiler settings before the analysis process initializes its private
compiler cache; it does not construct or fit a model.

Requested limits and observed limits are different fields. Runtime setup records
`threadpoolctl` observations for supported loaded libraries. That controller does
not observe Apple Accelerate's thread count. An empty controller list is recorded
as empty, with the capability limitation explicit; it is never reported as an
observed single-thread execution. The early Accelerate environment setting is a
request, not a measurement. Exact replication across different platforms or
compiler/BLAS versions is not promised. Changes to these runtime identities
require a new run identity and do not retroactively amend historical receipts.

## Locks and publication

Linux and macOS execution use nonblocking POSIX `flock` on opened generated
**directory descriptors**, acquired in sorted order. The directory inode is the
lock object. Lock files are not created or deleted. A process exit or SIGKILL
closes its descriptors and releases ownership. Independently opened descriptors,
including those in another thread, must acquire the same lock. Forked children
close inherited lock descriptors without unlocking the parent's open-file
description; close-on-exec also prevents a launched child from retaining them.
Full-run locks cover the configured roots, while each analytical module locks
its own generated subdirectories, avoiding nested acquisition of the same inode.

These guarantees require a local filesystem with working POSIX advisory-lock and
atomic-rename semantics. Shared/network filesystems are not certified. Other
programs must not remove or replace an actively locked directory. Symlinked lock
directories and changed directory inodes are rejected. Old `.preparation.lock`
or `.full_run.lock` files from earlier versions are rejected conservatively;
inspect the previous process and keep the accepted run intact rather than
blindly deleting a possible active owner's lock. Use new roots for the revised
source and runtime.

The full completion receipt is written into a complete staging directory, then
published with Linux `renameat2(RENAME_NOREPLACE)` or macOS
`renamex_np(RENAME_EXCL)`. Readers see either no final directory or the complete
receipt. Existing empty directories, symlinks and other destinations are never
overwritten. A process death before publication can leave an unreferenced staging
directory, which is not an accepted result and does not prevent a retry.

Preparation publishes two directories, which cannot be committed atomically by
one filesystem rename. Before publication, it flushes the complete staged files
and writes a transaction journal binding the fingerprint, stage names and receipt
hash. `--resume` under both directory locks validates the current source/input
identity, complete artifact and aggregate inventories, hashes and summaries.
It then finishes any missing no-clobber rename and removes the journal. Missing,
altered, symlinked or ambiguous stages are rejected without adopting or deleting
another result. Completed results and their receipt bytes are preserved. Normal
caught publication failures roll back owned directories; process death retains
journaled evidence for recovery. A separate pre-analysis journal supports recovery
of a partially written paired full-run origin. No analytical files may exist
while that origin transaction is being recovered.

File and directory `fsync` calls order publication writes. These process-crash
checks do not certify power-loss behavior of every storage device or filesystem.
Synthetic tests kill subprocesses at the commit boundaries, verify that active
owners remain excluded, and confirm verified resume without scientific refitting.
