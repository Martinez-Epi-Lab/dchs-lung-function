"""Reconstruct the complete manuscript workflow from an explicitly empty origin.

The paired origin records distinguish a clean reconstruction (including a restart
of that reconstruction) from validation of unrelated pre-existing model caches.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import tempfile

from .execution import (canonical_bytes, file_hash, resolve_paths, runtime_identity,
                        write_json, write_pointer)
from .pipeline import ValidationError, read_json, validate_config
from .reporting.sources import digest, no_symlinks, require, verify_files
from .reporting_execution import (assert_loaded, source_hashes, reporting_config,
                                  document_runtime, RESULT_FILES as REPORTING_FILES)
from .figures_execution import (run_figures, figures_config, figure_runtime,
                                RESULT_FILES as FIGURE_FILES)
from .model_runtime import model_runtime_identity
from .lifecycle import directory_locks, rename_noreplace, sync_directory, sync_tree

ORIGIN = 'full_run_origin.json'
LOCK = '.full_run.lock'
ORIGIN_JOURNAL = '.full-origin-transaction.json'
RAW_COUNTS = {'primary_fits':12, 'mechanics_fits':6, 'supporting_fits':7,
              'wheeze_fits':2, 'sensitivity_fits':10}
FIGURE_COVERAGE = {'caption_count_bindings':12,'figures':9,'flow_count_bindings':28,
                  'flow_nodes':20,'forest_estimates':56,'panels':17,
                  'schedule_cells':44,'trajectory_points':808}


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


@contextmanager
def full_locks(directories):
    with ExitStack() as stack:
        try:
            stack.enter_context(directory_locks(directories, legacy_name=LOCK))
        except (BlockingIOError, ValueError) as error:
            raise ValidationError('Another full run holds a work or output lock, or lock ownership is unsafe.') from error
        yield


def finish_origin_transaction(work, output, identity, fingerprint):
    """Recover only the pre-analysis paired-origin transaction, never infer it."""
    journal=work/ORIGIN_JOURNAL
    no_symlinks(journal)
    origin=read_json(journal,'Full origin transaction')
    require(isinstance(origin,dict) and set(origin)=={'schema_version','identity','fingerprint','clean_origin','started_at'}
            and type(origin['schema_version']) is int and origin['schema_version']==1
            and origin['clean_origin'] is True and origin['fingerprint']==fingerprint
            and canonical_bytes(origin['identity'])==canonical_bytes(identity)
            and isinstance(origin['started_at'],str) and bool(origin['started_at']),
            'Full origin recovery identity differs.')
    for directory in (work,output):
        allowed={ORIGIN,ORIGIN_JOURNAL} if directory==work else {ORIGIN}
        require({p.name for p in directory.iterdir()}<=allowed,
                'Full origin recovery found unexpected files before analysis.')
        path=directory/ORIGIN;no_symlinks(path)
        if path.exists():
            require(canonical_bytes(read_json(path,'Full origin'))==canonical_bytes(origin),
                    'Full origin recovery found a different origin record.')
        else:
            write_pointer(path,origin)
    journal.unlink();sync_directory(work)
    return origin


def raw_inventory(work, *, preparation_fingerprint=None):
    """Bind every completed fresh-origin raw fit after cumulative validation."""
    from .models import spirometry, mechanics, wheeze, sensitivity
    from .supporting_sampling import TARGETS as SUPPORTING_TARGETS
    target_sets={'primary_fits':set(spirometry.TARGETS),'mechanics_fits':set(mechanics.TARGETS),
                 'supporting_fits':set(SUPPORTING_TARGETS),'wheeze_fits':set(wheeze.TARGETS),
                 'sensitivity_fits':set(sensitivity.TARGETS)}
    result={}
    for family, count in RAW_COUNTS.items():
        directory=work/family;no_symlinks(directory)
        paths=sorted(directory.glob('*/*/posterior.npz'))
        require(len(paths)==count, 'full reconstruction has an unexpected raw-fit count.')
        targets=set()
        for posterior in paths:
            fit=posterior.parent;no_symlinks(fit)
            receipt=read_json(fit/'receipt.json', 'Full raw-fit receipt')
            require(isinstance(receipt,dict) and set(receipt)=={'schema_version','status','fingerprint','identity',
                    'preparation_fingerprint','completed_at','files'}
                    and type(receipt['schema_version']) is int and receipt['schema_version']==1
                    and receipt.get('status')=='sampled' and isinstance(receipt.get('identity'),dict)
                    and receipt.get('fingerprint')==fit.name==digest(receipt.get('identity'))
                    and receipt['identity'].get('target')==fit.parent.name
                    and (preparation_fingerprint is None or receipt['preparation_fingerprint']==preparation_fingerprint)
                    and fit.parent.name not in targets,
                    'full raw-fit identity or target inventory differs.')
            targets.add(fit.parent.name)
            verify_files(fit,receipt.get('files'),('posterior.npz','sample_stats.npz','sampling.json'))
            result[str(fit.relative_to(work))]=file_hash(fit/'receipt.json')
        require(targets==target_sets[family], 'full raw-fit targets differ from the exact declared family.')
    return result


def publish_completion(destination, receipt):
    """Publish one complete receipt without clobbering or leaving an empty run."""
    no_symlinks(destination)
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    stage=Path(tempfile.mkdtemp(prefix='.full-completion-',dir=destination.parent))
    try:
        write_json(stage/'receipt.json',receipt)
        sync_tree(stage)
        rename_noreplace(stage,destination)
        sync_directory(destination.parent)
    finally:
        if stage.exists(): shutil.rmtree(stage)


def run_full(root, config_file, analysis_file, primary_sampling_file,
             mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file,
             sensitivity_sampling_file, reporting_file, figures_file, *, jobs=1, resume=False):
    """Run M01–M14 with clean-origin proof and a final M15 execution receipt.

    A resume is allowed only for the same source, inputs, settings, runtime and
    destinations recorded when both roots were empty. Individual modules always
    revalidate their own artifacts before accepting any resumed completion.
    """
    root=Path(root).absolute();no_symlinks(root);root=root.resolve();assert_loaded(root)
    require(type(resume) is bool and type(jobs) is int and 1<=jobs<=3,
            'full jobs/resume arguments are invalid.')
    paths=[Path(p).absolute() for p in (config_file,analysis_file,primary_sampling_file,
           mechanics_sampling_file,supporting_sampling_file,wheeze_sampling_file,
           sensitivity_sampling_file,reporting_file,figures_file)]
    for p in paths:no_symlinks(p)
    paths=[p.resolve() for p in paths]
    configuration=read_json(paths[0], 'Full local configuration')
    validate_config(configuration,read_json(root/'config/input_contract.json','Input contract'),paths[0],root)
    input_paths,directories=resolve_paths(configuration,paths[0])
    for value in (*configuration['directories'].values(),
                  *(s['path'] for s in configuration['inputs'].values()),
                  *configuration['fonts'].values()):
        p=Path(value).expanduser();no_symlinks(p if p.is_absolute() else paths[0].parent/p)
    work,output=directories['work'],directories['outputs']
    require(not _overlap(work,output) and not any(_overlap(p,root) for p in (work,output)),
            'full generated roots must be separate from source and one another.')
    report=reporting_config(paths[-2]);figures=figures_config(paths[-1])
    fonts={}
    for key,value in configuration['fonts'].items():
        p=Path(value).expanduser();fonts[key]=(p if p.is_absolute() else paths[0].parent/p).resolve()
    protected=[*paths,*input_paths.values(),*fonts.values(),*report.values(),figures['pdf_fonts_tool']]
    require(not any(p==d or d in p.parents for p in protected for d in (work,output)),
            'full generated roots cannot contain inputs, settings or runtimes.')
    file_hashes={p:file_hash(p) for p in protected}
    runtime={'preparation':runtime_identity(),'models':model_runtime_identity(),'document':document_runtime(report['document_python']),
             'figure':figure_runtime(figures['pdf_fonts_tool'])}
    identity={'schema_version':1,'mode':'full','source':source_hashes(root),
              'configuration_hashes':[file_hashes[p] for p in paths],
              'inputs':{k:file_hashes[p] for k,p in input_paths.items()},
              'fonts':{k:file_hashes[p] for k,p in fonts.items()},
              'references':{k:file_hashes[p] for k,p in report.items() if k!='document_python'},
              'runtime':runtime,'work_directory':str(work),'output_directory':str(output)}
    fingerprint=digest(identity)
    def unchanged():
        assert_loaded(root)
        require(source_hashes(root)==identity['source'], 'full source identity changed during execution.')
        for p,h in file_hashes.items():
            no_symlinks(p);require(p.is_file() and file_hash(p)==h,'full protected input changed during execution.')
        require(runtime=={'preparation':runtime_identity(),'models':model_runtime_identity(),'document':document_runtime(report['document_python']),
                          'figure':figure_runtime(figures['pdf_fonts_tool'])},
                'full runtime changed during execution.')
    for directory in (work,output):
        no_symlinks(directory)
        if resume:
            require(directory.is_dir(), 'full resume requires the original generated roots.')
        else:
            require(not directory.exists() or (directory.is_dir() and not any(directory.iterdir())),
                    'a clean full run requires empty work and output roots; use new roots or a matching full --resume.')
    for directory in (work,output):directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    with full_locks((work,output)):
        if resume and ((work/ORIGIN_JOURNAL).exists() or (work/ORIGIN_JOURNAL).is_symlink()):
            finish_origin_transaction(work,output,identity,fingerprint)
        if resume:
            origins=[]
            for directory in (work,output):
                no_symlinks(directory/ORIGIN)
                origins.append(read_json(directory/ORIGIN,'Full clean-origin record'))
            origin=origins[0]
            require(canonical_bytes(origin)==canonical_bytes(origins[1]) and isinstance(origin,dict)
                    and set(origin)=={'schema_version','identity','fingerprint','clean_origin','started_at'}
                    and type(origin['schema_version']) is int and origin['schema_version']==1
                    and origin['clean_origin'] is True and origin['fingerprint']==fingerprint
                    and canonical_bytes(origin['identity'])==canonical_bytes(identity)
                    and isinstance(origin['started_at'],str) and bool(origin['started_at']),
                    'full resume differs from its paired clean-origin identity; start with new empty roots.')
        else:
            require(all(not any(d.iterdir()) for d in (work,output)),
                    'full generated roots changed before clean-origin creation.')
            origin={'schema_version':1,'identity':identity,'fingerprint':fingerprint,
                    'clean_origin':True,'started_at':datetime.now(timezone.utc).isoformat()}
            try:
                write_pointer(work/ORIGIN_JOURNAL,origin)
                finish_origin_transaction(work,output,identity,fingerprint)
            except BaseException:
                # No analysis has started: remove only this exact attempted origin.
                for directory in (work,output):
                    path=directory/ORIGIN
                    if path.is_file() and not path.is_symlink() and path.read_bytes()==canonical_bytes(origin)+b'\n':
                        path.unlink()
                journal=work/ORIGIN_JOURNAL
                if journal.is_file() and not journal.is_symlink() and journal.read_bytes()==canonical_bytes(origin)+b"\n":
                    journal.unlink()
                raise
        origin_hashes={d/ORIGIN:file_hash(d/ORIGIN) for d in (work,output)}
        destination=output/'full_reconstruction'/fingerprint
        no_symlinks(destination)
        previous=None
        if destination.exists():
            require(resume and destination.is_dir() and {p.name for p in destination.iterdir()}=={'receipt.json'},
                    'full completion has an unexpected inventory.')
            no_symlinks(destination/'receipt.json')
            previous=read_json(destination/'receipt.json','Full completion')
            require(isinstance(previous,dict) and set(previous)=={'schema_version','status','identity','fingerprint',
                    'origin_sha256','figures_receipt_sha256','reporting_receipt_sha256','raw_receipts','summary','completed_at'}
                    and type(previous['schema_version']) is int and previous['schema_version']==1
                    and previous['status']=='accepted' and previous['fingerprint']==fingerprint
                    and canonical_bytes(previous['identity'])==canonical_bytes(identity)
                    and previous['origin_sha256']==origin_hashes[work/ORIGIN],
                    'full completion identity changed.')
        result=run_figures(root,*paths,jobs=jobs,resume=resume)
        require(isinstance(result,dict) and result.get('through')=='M14' and result.get('accepted') is True
                and result.get('status')=='accepted'
                and canonical_bytes({k:result.get(k) for k in ('preceding_modules_verified','corrected_pretb_models',
                       'corrected_pretb_coefficients','corrected_pretb_covariance_cells','new_model_fits')})==
                    canonical_bytes({'preceding_modules_verified':13,'corrected_pretb_models':17,
                       'corrected_pretb_coefficients':51,'corrected_pretb_covariance_cells':306,'new_model_fits':0})
                and canonical_bytes(result.get('coverage'))==canonical_bytes(FIGURE_COVERAGE)
                and result.get('uncertainty_selection')=='corrected_actual_risk_membership'
                and result.get('pretb_point_estimates_preserved') is True and type(result.get('resumed')) is bool
                and all(isinstance(result.get(k),str) and re.fullmatch('[0-9a-f]{64}',result[k])
                        for k in ('fingerprint','reporting_fingerprint','preparation_fingerprint')),
                'full execution did not complete all manuscript analyses and figures.')
        figure_dir=output/'publication_figures'/result['fingerprint']
        report_dir=output/'manuscript_reporting'/result['reporting_fingerprint']
        for directory,expected in ((figure_dir,FIGURE_FILES),(report_dir,REPORTING_FILES)):
            receipt=read_json(directory/'receipt.json','Full publication')
            require(receipt.get('status')=='accepted' and receipt.get('fingerprint')==directory.name
                    and receipt['fingerprint']==digest(receipt.get('identity'))
                    and receipt['identity'].get('source')==identity['source'],
                    'full publication is not bound to the current source.')
            verify_files(directory,receipt.get('files'),expected)
            require(canonical_bytes(receipt.get('summary'))==canonical_bytes(read_json(directory/'summary.json','Full publication summary')),
                    'full publication summary differs from its receipt.')
            if directory==figure_dir:
                require(canonical_bytes(receipt['summary'])==canonical_bytes({k:v for k,v in result.items() if k!='resumed'})
                        and receipt['identity'].get('reporting_fingerprint')==result['reporting_fingerprint']
                        and receipt['identity'].get('reporting_receipt_sha256')==file_hash(report_dir/'receipt.json'),
                        'full figure completion differs from the returned M14 validation.')
        raw=raw_inventory(work,preparation_fingerprint=result['preparation_fingerprint'])
        summary={'through':'M15','mode':'full','accepted':True,'status':'accepted',
                 'fingerprint':fingerprint,'clean_origin':True,'analytical_revalidation_performed':True,
                 'joint_fits_verified':sum(RAW_COUNTS.values()),'raw_fit_families':RAW_COUNTS,
                 'figures_fingerprint':result['fingerprint'],'reporting_fingerprint':result['reporting_fingerprint'],
                 'preparation_fingerprint':result['preparation_fingerprint'],
                 'coverage':result['coverage'],'github_delivery_verified':False}
        receipt={'schema_version':1,'status':'accepted','identity':identity,'fingerprint':fingerprint,
                 'origin_sha256':origin_hashes[work/ORIGIN],
                 'figures_receipt_sha256':file_hash(figure_dir/'receipt.json'),
                 'reporting_receipt_sha256':file_hash(report_dir/'receipt.json'),
                 'raw_receipts':raw,'summary':summary,'completed_at':datetime.now(timezone.utc).isoformat()}
        unchanged()
        require(all(file_hash(p)==h for p,h in origin_hashes.items()), 'full origin changed during execution.')
        if previous is not None:
            require(canonical_bytes({k:v for k,v in previous.items() if k!='completed_at'})==
                    canonical_bytes({k:v for k,v in receipt.items() if k!='completed_at'}),
                    'full completion differs from revalidated artifacts.')
        else:
            publish_completion(destination,receipt)
        write_pointer(work/'full_current.json',summary)
    return {**summary,'resumed':resume}
