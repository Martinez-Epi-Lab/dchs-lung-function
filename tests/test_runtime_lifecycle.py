"""Independent synthetic process-death, ownership, runtime and recovery tests."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function import lifecycle, execution as ex, full_execution as full
from dchs_lung_function.pipeline import ValidationError

ROOT=Path(__file__).resolve().parents[1]


def child(action, code=73):
    pid=os.fork()
    if pid==0:
        try: action()
        except BaseException: os._exit(99)
        os._exit(code)
    _,status=os.waitpid(pid,0)
    return os.waitstatus_to_exitcode(status)


class RuntimeIdentityTests(unittest.TestCase):
    def script(self,source,**kwargs):
        env=dict(os.environ, PYTHONPATH=str(ROOT/'src'))
        return subprocess.run([sys.executable,'-B','-c',source],env=env,text=True,
                              capture_output=True,timeout=60,**kwargs)

    def test_bootstrap_precedes_numpy_and_overrides_conflicting_requests(self):
        result=self.script("import os; os.environ['OPENBLAS_NUM_THREADS']='8'; from dchs_lung_function.runtime_bootstrap import bootstrap; c=bootstrap(); import numpy; assert all(v=='1' for v in c.values()); print('early')")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'early')

    def test_late_bootstrap_rejected_even_if_environment_happens_to_match(self):
        result=self.script("import numpy; from dchs_lung_function.runtime_bootstrap import bootstrap; bootstrap()")
        self.assertNotEqual(result.returncode,0)
        self.assertIn('cannot be applied retroactively',result.stderr)

    def test_environment_mutation_after_bootstrap_rejected(self):
        result=self.script("from dchs_lung_function.runtime_bootstrap import bootstrap; bootstrap(); import os; os.environ['OMP_NUM_THREADS']='2'; bootstrap()")
        self.assertNotEqual(result.returncode,0)
        self.assertIn('changed after',result.stderr)

    def test_runtime_identity_captures_toolchain_os_and_controls_and_is_cache_independent(self):
        result=self.script("""from dchs_lung_function.runtime_bootstrap import bootstrap
bootstrap()
from dchs_lung_function.model_runtime import model_runtime_identity,configure_runtime
import tempfile,json
before=model_runtime_identity()
with tempfile.TemporaryDirectory() as cache:
 setup=configure_runtime(cache)
 after=model_runtime_identity()
 assert before==after
 assert before['compiler']['version'] and len(before['compiler']['executable_sha256'])==64
 assert before['os_release'] and before['os_version'] and before['numpy_blas']
 assert all(v=='1' for v in before['requested_thread_environment'].values())
 assert 'blas_threads_per_chain' not in setup
 assert setup['thread_controls']['accelerate_thread_count_observed'] is False
 print('bound')
""")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'bound')

    def test_runtime_identity_changes_with_os_and_actual_compiler_configuration(self):
        from dchs_lung_function import model_runtime as runtime
        first=runtime.model_runtime_identity()
        with mock.patch.object(runtime.platform,'release',return_value='different-kernel'):
            self.assertNotEqual(first,runtime.model_runtime_identity())
        settings=runtime._compiler_settings()
        with mock.patch.object(runtime,'_compiler_settings',return_value={**settings,'gcc__cxxflags':'-DTEST_CHANGED'}):
            self.assertNotEqual(first,runtime.model_runtime_identity())
        with mock.patch.dict(os.environ,{'CXXFLAGS':'-DTEST_CHANGED'}):
            self.assertNotEqual(first,runtime.model_runtime_identity())

    def test_fresh_runtime_identity_survives_real_pymc_import_and_graph_compilation(self):
        result=self.script("""from dchs_lung_function.runtime_bootstrap import bootstrap
bootstrap()
from dchs_lung_function.model_runtime import model_runtime_identity,configure_runtime
import tempfile,sys,atexit,shutil
before=model_runtime_identity()
assert 'pytensor' not in sys.modules and 'pymc' not in sys.modules
cache=tempfile.mkdtemp(prefix='dchs-import-regression-')
atexit.register(shutil.rmtree,cache,ignore_errors=True)
configure_runtime(cache)
configured=model_runtime_identity()
import pymc as pm
import pytensor
assert before==configured==model_runtime_identity()
assert before['pytensor_configuration']['gcc__cxxflags']==pytensor.config.gcc__cxxflags
with pm.Model() as model:
 pm.Normal('fabricated_scalar',0,1)
logp=model.compile_logp()
assert logp({'fabricated_scalar':0.0})<0
assert before==model_runtime_identity()
print('effective-runtime-stable')
""")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'effective-runtime-stable')

    def test_empty_threadpool_report_never_claims_observed_single_thread(self):
        from dchs_lung_function.model_runtime import thread_observations
        with mock.patch('threadpoolctl.threadpool_info',return_value=[]):
            evidence=thread_observations()
        self.assertEqual(evidence['observed_pools'],[])
        self.assertFalse(evidence['accelerate_thread_count_observed'])
        self.assertIn('not observed',evidence['observation_scope'])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve()
        self.a=self.base/'a';self.b=self.base/'b';self.a.mkdir();self.b.mkdir()

    def test_active_owner_rejected_and_child_does_not_unlock_parent(self):
        with lifecycle.directory_locks((self.a,self.b)):
            with self.assertRaises(BlockingIOError):
                with lifecycle.directory_locks((self.b,self.a)): self.fail('entered')
            def attempt():
                try:
                    with lifecycle.directory_locks((self.a,)): os._exit(98)
                except BlockingIOError: os._exit(73)
            self.assertEqual(child(attempt),73)
            with self.assertRaises(BlockingIOError):
                with lifecycle.directory_locks((self.a,)): pass
        with lifecycle.directory_locks((self.a,self.b)): pass
        self.assertEqual(list(self.a.iterdir()),[])

    def test_sigkill_releases_owner_without_manual_cleanup(self):
        r,w=os.pipe();pid=os.fork()
        if pid==0:
            try:
                os.close(r)
                with lifecycle.directory_locks((self.a,self.b)):
                    os.write(w,b'1');os.close(w);signal.pause()
            finally: os._exit(99)
        os.close(w)
        try:
            import select
            self.assertTrue(select.select([r],[],[],10)[0])
            self.assertEqual(os.read(r,1),b'1')
            with self.assertRaises(BlockingIOError):
                with lifecycle.directory_locks((self.a,)): pass
        finally:
            os.kill(pid,signal.SIGKILL);os.waitpid(pid,0);os.close(r)
        with lifecycle.directory_locks((self.a,self.b)): pass
        self.assertEqual(list(self.a.iterdir()),[])

    def test_second_lock_failure_releases_first_but_preserves_existing_owner(self):
        with lifecycle.directory_locks((self.b,)):
            with self.assertRaises(BlockingIOError):
                with lifecycle.directory_locks((self.a,self.b)): pass
            with lifecycle.directory_locks((self.a,)): pass
            with self.assertRaises(BlockingIOError):
                with lifecycle.directory_locks((self.b,)): pass

    def test_standard_temporary_directory_alias_uses_the_same_lock_inode(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            with lifecycle.directory_locks((path,)):
                with self.assertRaises(BlockingIOError):
                    with lifecycle.directory_locks((path.resolve(),)):pass

    def test_lock_rejects_symlink_and_dangling_legacy_lock(self):
        alias=self.base/'alias';alias.symlink_to(self.a,target_is_directory=True)
        with self.assertRaises(ValueError):
            with lifecycle.directory_locks((alias,)): pass
        (self.a/'.preparation.lock').symlink_to(self.base/'absent')
        with self.assertRaises(ValidationError):
            with ex.preparation_locks(self.a,self.b): pass

    def test_no_clobber_publication_rejects_even_empty_or_dangling_destination(self):
        source=self.a/'stage';source.mkdir();(source/'receipt.json').write_text('complete')
        destination=self.a/'final';destination.mkdir()
        with self.assertRaises(FileExistsError): lifecycle.rename_noreplace(source,destination)
        self.assertEqual((source/'receipt.json').read_text(),'complete')
        destination.rmdir();destination.symlink_to(self.b/'absent')
        with self.assertRaises(FileExistsError): lifecycle.rename_noreplace(source,destination)
        self.assertTrue(destination.is_symlink())

    def test_full_receipt_hard_exit_before_and_after_atomic_rename(self):
        for moment in ('before','after'):
            final=self.a/moment
            def killed():
                original=full.rename_noreplace
                def rename(stage,destination):
                    if moment=='before': os._exit(73)
                    original(stage,destination);os._exit(73)
                with mock.patch.object(full,'rename_noreplace',side_effect=rename):
                    full.publish_completion(final,{'status':'accepted','synthetic':True})
            self.assertEqual(child(killed),73)
            if moment=='before':
                self.assertFalse(final.exists())
                full.publish_completion(final,{'status':'accepted','synthetic':True})
            self.assertEqual(json.loads((final/'receipt.json').read_text()),{'status':'accepted','synthetic':True})
            before=(final/'receipt.json').read_bytes()
            with self.assertRaises(FileExistsError): full.publish_completion(final,{'different':True})
            self.assertEqual((final/'receipt.json').read_bytes(),before)

    def test_origin_half_pair_recovery_requires_same_identity_and_no_analysis_files(self):
        identity={'synthetic':True}; fp='f'*64
        origin={'schema_version':1,'identity':identity,'fingerprint':fp,'clean_origin':True,'started_at':'synthetic-time'}
        ex.write_json(self.a/full.ORIGIN_JOURNAL,origin);ex.write_json(self.a/full.ORIGIN,origin)
        with self.assertRaises(ValidationError): full.finish_origin_transaction(self.a,self.b,{'different':True},fp)
        (self.b/'unexpected').write_text('keep')
        with self.assertRaises(ValidationError): full.finish_origin_transaction(self.a,self.b,identity,fp)
        (self.b/'unexpected').unlink()
        full.finish_origin_transaction(self.a,self.b,identity,fp)
        self.assertEqual((self.a/full.ORIGIN).read_bytes(),(self.b/full.ORIGIN).read_bytes())
        self.assertFalse((self.a/full.ORIGIN_JOURNAL).exists())


class PreparationCrashTests(unittest.TestCase):
    def fixture(self):
        import test_execution
        case=test_execution.ExecutionTests(methodName='runTest');case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def crash(self,case,when):
        def action():
            original=ex.rename_noreplace; count=0
            def rename(stage,destination):
                nonlocal count
                if when==0 and count==0: os._exit(73)
                original(stage,destination);count+=1
                if count==when: os._exit(73)
            with mock.patch.object(ex,'rename_noreplace',side_effect=rename):case.run_preparation()
        self.assertEqual(child(action),73)

    def test_resume_recovers_each_two_directory_commit_boundary_without_recomputation(self):
        for when in (0,1,2):
            with self.subTest(published_directories=when):
                case=self.fixture();self.crash(case,when)
                with mock.patch.object(ex,'compute_preparation',side_effect=AssertionError('must use verified staging')):
                    result=ex.run_preparation(case.root,case.config_path,case.analysis_path,'M01',resume=True)
                self.assertTrue(result['resumed'])
                work,out=case.run_dirs(result)
                before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in (work,out) for p in d.iterdir()}
                case.run_preparation(resume=True)
                self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in (work,out) for p in d.iterdir()})
                self.assertEqual(list(work.parent.glob('.transaction-*')),[])

    def test_recovery_rejects_changed_artifact_without_publishing_missing_side(self):
        case=self.fixture();self.crash(case,1)
        work=Path(case.configuration['directories']['work'])/'preparation_runs'
        stage=next(work.glob('.staging-*'));(stage/'artifacts.json').write_text('{}')
        with self.assertRaises(ValidationError):case.run_preparation(resume=True)
        self.assertEqual([p for p in work.iterdir() if not p.name.startswith('.')],[])
        self.assertTrue(next(work.glob('.transaction-*')).is_file())

    def test_recovery_rejects_traversal_and_does_not_delete_ambiguous_existing_directory(self):
        case=self.fixture();self.crash(case,0)
        work=Path(case.configuration['directories']['work'])/'preparation_runs'
        journal=next(work.glob('.transaction-*'));state=json.loads(journal.read_text());original=dict(state)
        state['stage_work']='../outside';ex.write_json(journal,state)
        with self.assertRaises(ValidationError):case.run_preparation(resume=True)
        ex.write_json(journal,original)
        destination=work/original['fingerprint'];destination.mkdir();(destination/'keep').write_text('other owner')
        with self.assertRaises(ValidationError):case.run_preparation(resume=True)
        self.assertEqual((destination/'keep').read_text(),'other owner')

    def test_source_change_does_not_adopt_previous_transaction(self):
        case=self.fixture();self.crash(case,1)
        work=Path(case.configuration['directories']['work'])/'preparation_runs'
        journal=next(work.glob('.transaction-*'));before=journal.read_bytes()
        (case.root/'src/example.py').write_text('VALUE = 9\n')
        result=case.run_preparation(resume=True)
        self.assertFalse(result['resumed'])
        self.assertEqual(journal.read_bytes(),before)

class FullCrashTests(unittest.TestCase):
    def fixture(self):
        import test_release
        case=test_release.FullExecutionTests(methodName='runTest');case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_full_resume_after_receipt_publication_death_preserves_accepted_bytes(self):
        case=self.fixture()
        def action():
            original=full.rename_noreplace
            def rename(stage,destination):
                original(stage,destination);os._exit(73)
            with mock.patch.object(full,'rename_noreplace',side_effect=rename):case.run_full()
        self.assertEqual(child(action),73)
        receipt=next(case.output.glob('full_reconstruction/*/receipt.json'))
        before=receipt.read_bytes()
        self.assertFalse((case.work/'full_current.json').exists())
        result=case.run_full(resume=True)
        self.assertTrue(result['accepted']);self.assertTrue(result['resumed'])
        self.assertEqual(before,receipt.read_bytes())

    def test_full_resume_after_first_origin_hard_exit_uses_journal(self):
        case=self.fixture()
        def action():
            original=full.write_pointer
            def pointer(path,value):
                original(path,value)
                if path==case.work/full.ORIGIN: os._exit(73)
            with mock.patch.object(full,'write_pointer',side_effect=pointer):case.run_full()
        self.assertEqual(child(action),73)
        self.assertTrue((case.work/full.ORIGIN_JOURNAL).is_file())
        self.assertFalse((case.output/full.ORIGIN).exists())
        result=case.run_full(resume=True)
        self.assertTrue(result['accepted'])
        self.assertEqual((case.work/full.ORIGIN).read_bytes(),(case.output/full.ORIGIN).read_bytes())
        self.assertFalse((case.work/full.ORIGIN_JOURNAL).exists())

    def test_lock_wrapper_preserves_body_errors_instead_of_mislabeling_ownership(self):
        with tempfile.TemporaryDirectory() as base:
            a=Path(base).resolve()/'a';b=a.parent/'b';a.mkdir();b.mkdir()
            for manager in (ex.preparation_locks(a,b),full.full_locks((a,b))):
                with self.assertRaisesRegex(ValueError,'synthetic scientific error'):
                    with manager:raise ValueError('synthetic scientific error')

class AdequacyBoundaryTests(unittest.TestCase):
    def test_real_aggregate_evaluator_passes_and_preserves_small_unevaluable_counts(self):
        from test_covariance_ppc import fabricated_case
        from dchs_lung_function.model_adequacy import evaluate_covariance,adequacy_status
        contract,posterior=fabricated_case()
        evidence=evaluate_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertTrue(evidence['accepted'])
        self.assertEqual(adequacy_status(True,evidence['accepted']),'diagnostics_passed')
        contract,posterior=fabricated_case(children=3)
        evidence=evaluate_covariance(posterior,contract,family='primary',target='ratio_pre')
        self.assertEqual(evidence['counts']['selected_children'],3)
        self.assertFalse(evidence['accepted'])
        self.assertEqual(evidence['status'],'not_evaluable')
        self.assertEqual(adequacy_status(True,evidence['accepted']),'model_adequacy_failed')
        self.assertEqual(adequacy_status(False,evidence['accepted']),'diagnostics_failed')

    def test_undeclared_or_spoofed_evidence_cannot_enter_saved_aggregate_artifacts(self):
        from copy import deepcopy
        from test_covariance_ppc import fabricated_case
        from dchs_lung_function import model_adequacy as gate
        contract,posterior=fabricated_case()
        original=gate.evaluate_covariance(posterior,contract,family='primary',target='ratio_pre')
        attacks=[lambda v:v.update(participant_records=['invented']),
                 lambda v:v['predictive']['icc'].update(private_field='invented'),
                 lambda v:v['observed'].update(icc='invented identifier'),
                 lambda v:v['predictive']['icc'].update(mean='invented identifier'),
                 lambda v:v.update(accepted=1),lambda v:v.update(target='wrong'),
                 lambda v:v.update(model_specification_version='wrong'),
                 lambda v:v['protocol']['criteria'].update(replicates=1)]
        for attack in attacks:
            value=deepcopy(original);attack(value)
            with mock.patch.object(gate,'evaluate_target_covariance',return_value=value):
                with self.assertRaises(ValidationError):gate.evaluate_covariance(posterior,contract,family='primary',target='ratio_pre')

    def test_nonfinite_repaired_cache_returns_controlled_failure(self):
        from dchs_lung_function.model_adequacy import verify_covariance
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'covariance_ppc.json').write_text('{"accepted":true,"changed":NaN}')
            with self.assertRaises(ValidationError):
                verify_covariance(path,{'covariance_ppc_passed':True,'diagnostics_passed':True,'status':'diagnostics_passed'},
                                  {'passed':True},{'accepted':True})
