"""Run a nonempty synthetic suite using a temporary writable compiler cache."""
from pathlib import Path
import argparse
import atexit
import shutil
import sys
import tempfile
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-only',action='store_true')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),str(here.parent/'src')]
    from dchs_lung_function.runtime_bootstrap import bootstrap
    bootstrap()
    if not list((here.parent/'src/dchs_lung_function').rglob('*.py')):
        raise SystemExit('No source files discovered.')
    cache=tempfile.mkdtemp(prefix='dchs-synthetic-tests-')
    # Registered before PyTensor so its own exit hooks finish before removal.
    atexit.register(shutil.rmtree,cache,ignore_errors=True)
    if args.preparation_only:
        names=['test_foundation','test_inputs','test_reference_scores','test_preparation',
               'test_cohorts','test_descriptive','test_respiratory_preparation','test_execution']
        suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    else:
        from dchs_lung_function.model_runtime import configure_runtime
        configure_runtime(cache)
        suite=unittest.defaultTestLoader.discover(str(here),pattern='test_*.py')
    if suite.countTestCases()==0:raise SystemExit('No synthetic tests discovered.')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
