"""Fingerprint, fit, and validate primary models in private output directories.

Raw sampling identity is separate from validation identity: changed reporting
or diagnostic code rechecks verified draws without pretending to refit them.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, verify_artifacts, write_json,
                        resolve_paths,run_preparation,write_pointer)
from .pipeline import ValidationError,read_json
from .models.spirometry import TARGETS,build_model,validate_contract
from .model_runtime import configure_runtime,model_runtime_identity
from .model_adequacy import evaluate_covariance, adequacy_status, verify_covariance
from .covariance_ppc import protocol as covariance_protocol
from .sampling import resolved_sampling,sample_contract

CRITERIA = {'min_chains':4,'min_draws':1200,'max_rhat':1.01,
            'min_bulk_ess':400,'min_tail_ess':400,'min_bfmi':.3,
            'max_divergences':0,'max_treedepth_hits':0}
COMPUTE_SOURCES = ('models/spirometry.py','sampling.py','model_runtime.py','runtime_bootstrap.py')
VALIDATION_SOURCES = ('lifecycle.py','execution.py','pipeline.py','model_adequacy.py','covariance_ppc.py','primary_execution.py','diagnostics.py','analysis/primary_results.py')
RESULT_FILES = ('diagnostics.json','covariance_ppc.json','summary.json','result_metadata.json',
                'estimates.csv','shared_factor.csv','sample_counts.csv')
_LOADED_RUNNER_HASH = file_hash(Path(__file__))


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def source_hashes(root, names):
    return {name:file_hash(Path(root)/'src/dchs_lung_function'/name) for name in names}


def contract_descriptor(preparation_directory, target):
    if target not in TARGETS:
        raise ValidationError('Unknown primary model target.')
    directory = Path(preparation_directory).resolve()
    receipt = read_json(directory/'receipt.json','Preparation receipt')
    if (receipt.get('through')!='M03' or receipt.get('status')!='prepared'
            or file_hash(directory/'artifacts.json')!=receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Primary fitting requires a complete verified M03 preparation.')
    if digest(receipt.get('identity'))!=receipt.get('fingerprint'):
        raise ValidationError('Preparation receipt fingerprint is inconsistent.')
    manifest = verify_artifacts(directory)
    node = artifact_subtree(manifest,('M03','pft','samples','primary/'+target))
    files = set()
    def visit(value):
        if isinstance(value,dict):
            if 'file' in value:files.add(value['file'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(node)
    if not files.issubset(manifest['files']):
        raise ValidationError('A primary contract refers to an unverified preparation file.')
    identity = {'tree':node,'files':{name:manifest['files'][name] for name in sorted(files)}}
    return {'sha256':digest(identity),'preparation_fingerprint':receipt['fingerprint']}


def sampling_identity(root, descriptor, target, settings):
    identity = {'schema_version':1,'target':target,'contract_sha256':descriptor['sha256'],
                'sampling':resolved_sampling(target,settings),
                'source':source_hashes(root,COMPUTE_SOURCES),
                'environment_lock_sha256':file_hash(Path(root)/'requirements-models.lock.txt'),
                'runtime':model_runtime_identity()}
    return digest(identity),identity


def validation_identity(root, fit_fingerprint, criteria):
    identity = {'schema_version':1,'fit_fingerprint':fit_fingerprint,
                'source':source_hashes(root,VALIDATION_SOURCES),'criteria':criteria,
                'covariance_ppc_protocol':covariance_protocol(),
                'runtime':model_runtime_identity()}
    return digest(identity),identity


def expected_posterior_shapes(contract,target,chains,draws):
    model = build_model(contract,target)
    return {variable.name:(chains,draws,*[len(model.coords[dim]) for dim in
              model.named_vars_to_dims.get(variable.name,())])
            for variable in [*model.free_RVs,*model.deterministics]}


def validate_posterior_inventory(posterior, expected):
    if set(posterior)!=set(expected):
        raise ValidationError('Saved posterior does not contain the complete model variable inventory.')
    if any(np.asarray(posterior[name]).shape!=shape for name,shape in expected.items()):
        raise ValidationError('Saved posterior does not have the complete model coordinate shapes.')


def validate_sampling_metadata(metadata,identity,posterior,stats):
    controls=identity['sampling']
    if not isinstance(metadata,dict) or any(metadata.get(key)!=expected for key,expected in (
        ('target',identity['target']),('controls',controls),('runtime',identity['runtime']),
        ('realized_chains',controls['chains']),('realized_draws_per_chain',controls['draws']),
        ('realized_tuning_steps',controls['tune']))):
        raise ValidationError('Realized sampling metadata disagrees with the requested fit identity.')
    for field,arrays in [('posterior_variables',posterior),('sample_stat_variables',stats)]:
        names=metadata.get(field)
        if not isinstance(names,list) or len(names)!=len(set(names)) or set(names)!=set(arrays):
            raise ValidationError('Realized sampling variable metadata disagrees with saved arrays.')


def read_array_archive(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def verify_files(directory, entries, required):
    if not isinstance(entries,dict) or not set(required).issubset(entries):
        raise ValidationError('Model receipt lacks required artifact identities.')
    for name,expected in entries.items():
        path=Path(directory)/name
        if (Path(name).name!=name or path.is_symlink() or not path.is_file()
                or file_hash(path)!=expected):
            raise ValidationError('A saved model artifact is missing or changed.')


def safe_roots(root,preparation_directory,work,output):
    root,prep,work,output=map(lambda p:Path(p).resolve(),(root,preparation_directory,work,output))
    overlap=lambda a,b:a==b or a in b.parents or b in a.parents
    if overlap(work,output) or any(overlap(folder,protected)
        for folder in (work,output) for protected in (root,prep)):
        raise ValidationError('Primary work/output roots must be separate from source and preparation artifacts.')
    for folder in (work,output):folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    return root,prep,work,output


def sample_one(root,preparation_directory,target,settings,work,output,*,resume=False):
    """Publish a complete raw fit, then a separately checked aggregate result.

    Failed diagnostic results retain their raw private evidence, with an
    explicit failed status. They never create a validated completion marker.
    """
    root,prep,work,output=safe_roots(root,preparation_directory,work,output)
    descriptor=contract_descriptor(prep,target)
    fingerprint,identity=sampling_identity(root,descriptor,target,settings)
    work=work/target;output=output/target
    work.mkdir(exist_ok=True,mode=0o700);output.mkdir(exist_ok=True,mode=0o700)
    final=work/fingerprint
    contract=load_artifacts(prep,('M03','pft','samples','primary/'+target))
    validated=validate_contract(contract,target)
    configure_runtime(work/'compiler_cache')
    controls=identity['sampling']
    expected=expected_posterior_shapes(contract,target,controls['chains'],controls['draws'])
    def unchanged():
        current=contract_descriptor(prep,target)
        if sampling_identity(root,current,target,settings)!=(fingerprint,identity):
            raise ValidationError('Model, inputs or sampling environment changed while the fit was running.')
    with preparation_locks(work,output):
        reused=final.exists()
        if reused:
            if not resume:raise ValidationError('An identical raw fit exists; use --resume to verify and revalidate it.')
            raw=read_json(final/'receipt.json','Raw fit receipt')
            if raw.get('identity')!=identity or raw.get('fingerprint')!=fingerprint or raw.get('status')!='sampled':
                raise ValidationError('Raw fit receipt identity is inconsistent.')
            verify_files(final,raw.get('files'),('posterior.npz','sample_stats.npz','sampling.json'))
            posterior=read_array_archive(final/'posterior.npz')
            stats=read_array_archive(final/'sample_stats.npz')
            metadata=read_json(final/'sampling.json','Realized sampling metadata')
        else:
            stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=work))
            try:
                posterior,stats,metadata=sample_contract(contract,target,settings,work/'compiler_cache')
                validate_posterior_inventory(posterior,expected)
                validate_sampling_metadata(metadata,identity,posterior,stats)
                np.savez_compressed(stage/'posterior.npz',**posterior)
                np.savez_compressed(stage/'sample_stats.npz',**stats)
                write_json(stage/'sampling.json',metadata)
                raw={'schema_version':1,'status':'sampled','fingerprint':fingerprint,'identity':identity,
                     'preparation_fingerprint':descriptor['preparation_fingerprint'],
                     'completed_at':datetime.now(timezone.utc).isoformat(),
                     'files':{name:file_hash(stage/name) for name in ('posterior.npz','sample_stats.npz','sampling.json')}}
                write_json(stage/'receipt.json',raw)
                unchanged()
                os.replace(stage,final)
            finally:
                if stage.exists():shutil.rmtree(stage)
        validate_posterior_inventory(posterior,expected)
        validate_sampling_metadata(metadata,identity,posterior,stats)
        if file_hash(Path(__file__))!=_LOADED_RUNNER_HASH:
            raise ValidationError('The loaded model runner changed; restart with --resume to validate saved draws.')
        validation_fp,validation_id=validation_identity(root,fingerprint,CRITERIA)
        destination=output/validation_fp
        if destination.exists():
            receipt=read_json(destination/'receipt.json','Validation receipt')
            if receipt.get('identity')!=validation_id or receipt.get('fingerprint')!=validation_fp:
                raise ValidationError('Validation receipt identity is inconsistent.')
            verify_files(destination,receipt.get('files'),RESULT_FILES)
            if set(receipt['files'])!=set(RESULT_FILES):
                raise ValidationError('Primary validation inventory differs.')
            summary=read_json(destination/'summary.json','Primary summary')
            if receipt.get('summary')!=summary:
                raise ValidationError('Validation receipt and verified primary summary disagree.')
            diagnostic=read_json(destination/'diagnostics.json','Primary diagnostics')
            if (type(diagnostic.get('passed')) is not bool
                    or diagnostic['passed']!=summary.get('diagnostics_passed')
                    or diagnostic.get('status')!=('passed' if diagnostic['passed'] else 'failed')
                    or summary.get('status')!=adequacy_status(diagnostic['passed'],summary.get('covariance_ppc_passed'))):
                raise ValidationError('Verified diagnostics and primary completion summary disagree.')
            if receipt.get('raw_receipt_sha256')!=file_hash(final/'receipt.json'):
                raise ValidationError('Validation receipt is not bound to the verified raw fit.')
            from .diagnostics import evaluate_diagnostics
            recomputed=evaluate_diagnostics(posterior,{n:stats[n] for n in ('diverging','energy','reached_max_treedepth') if n in stats},CRITERIA)
            if canonical_bytes(recomputed)!=canonical_bytes(diagnostic):
                raise ValidationError('Saved primary diagnostics differ from verified posterior draws.')
            covariance=evaluate_covariance(posterior,contract,family='primary',target=target)
            verify_covariance(destination,summary,diagnostic,covariance)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during cached validation.')
            return {**summary,'raw_fit_reused':True,'validation_reused':True}
        from .diagnostics import evaluate_diagnostics
        from .analysis.primary_results import summarize_primary
        essential={name:stats[name] for name in ('diverging','energy','reached_max_treedepth') if name in stats}
        diagnostics=evaluate_diagnostics(posterior,essential,CRITERIA)
        covariance=evaluate_covariance(posterior,contract,family='primary',target=target)
        results=summarize_primary(posterior,contract,target)
        summary={'target':target,'fit_fingerprint':fingerprint,'validation_fingerprint':validation_fp,
                 'diagnostics_passed':bool(diagnostics['passed']),
                 'covariance_ppc_passed':covariance['accepted'],
                 'status':adequacy_status(bool(diagnostics['passed']),covariance['accepted']),
                 'counts':validated,'result_rows':len(results.estimates),
                 'scientific_reference_comparison':'requires separate independent validation'}
        stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=output))
        try:
            write_json(stage/'diagnostics.json',diagnostics)
            write_json(stage/'covariance_ppc.json',covariance)
            write_json(stage/'summary.json',summary)
            write_json(stage/'result_metadata.json',results.metadata)
            for name,frame in [('estimates',results.estimates),('shared_factor',results.shared_factor),('sample_counts',results.sample_counts)]:
                frame.to_csv(stage/(name+'.csv'),index=False,float_format='%.17g')
            receipt={'schema_version':1,'fingerprint':validation_fp,'identity':validation_id,'summary':summary,
                     'raw_receipt_sha256':file_hash(final/'receipt.json'),
                     'files':{p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}}
            write_json(stage/'receipt.json',receipt)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during validation.')
            os.replace(stage,destination)
        finally:
            if stage.exists():shutil.rmtree(stage)
        return {**summary,'raw_fit_reused':reused,'validation_reused':False}


def run_primary(root,config_file,analysis_file,sampling_file,*,targets=None,jobs=1,resume=False):
    """Run cumulative M01-M04 and return only aggregate model completion data."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    from .sampling import validate_sampling
    selected=list(TARGETS) if targets is None else list(targets)
    if not selected or len(set(selected))!=len(selected) or not set(selected).issubset(TARGETS):
        raise ValidationError('Primary targets must be a nonempty unique selection of supported models.')
    if type(jobs) is not int or not 1<=jobs<=3:
        raise ValidationError('Primary model concurrency must be between one and three jobs.')
    settings=validate_sampling(read_json(sampling_file,'Sampling configuration'))
    sampling_file=Path(sampling_file).resolve()
    sampling_hash=file_hash(sampling_file)
    configuration=read_json(config_file,'Local configuration')
    _,directories=resolve_paths(configuration,config_file)
    for directory in directories.values():
        if directory==sampling_file or directory in sampling_file.parents:
            raise ValidationError('Generated directories cannot contain the sampling configuration.')
    preparation=run_preparation(root,config_file,analysis_file,'M03',resume=True)
    prepdir=directories['work']/'preparation_runs'/preparation['fingerprint']
    work=directories['work']/'primary_fits'
    output=directories['outputs']/'primary_models'
    results=[]
    if jobs==1:
        for target in selected:
            results.append(sample_one(root,prepdir,target,settings,work,output,resume=resume))
    else:
        with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(sample_one,root,prepdir,target,settings,work,output,resume=resume):target
                     for target in selected}
            for completed in as_completed(pending):results.append(completed.result())
    if file_hash(sampling_file)!=sampling_hash:
        raise ValidationError('Sampling configuration changed during primary execution.')
    # Reverify the requested inputs/settings and preparation identity before
    # writing a cumulative completion marker, including on a fully cached run.
    verified=run_preparation(root,config_file,analysis_file,'M03',resume=True)
    if verified['fingerprint']!=preparation['fingerprint']:
        raise ValidationError('Preparation dependencies changed during primary execution.')
    if len(results)!=len(selected) or sorted(r.get('target','') for r in results)!=sorted(selected):
        raise ValidationError('Primary completion must contain exactly the selected targets.')
    for r in results:
        if r.get('status')!=adequacy_status(r.get('diagnostics_passed'),r.get('covariance_ppc_passed')):
            raise ValidationError('Primary completion gates disagree.')
    results.sort(key=lambda value:selected.index(value['target']))
    mcmc_passed=all(value['diagnostics_passed'] for value in results)
    covariance_passed=all(value['covariance_ppc_passed'] for value in results)
    passed=mcmc_passed and covariance_passed
    summary={'through':'M04','preparation_fingerprint':preparation['fingerprint'],
             'selected_models':len(results),'all_twelve_models':set(selected)==set(TARGETS),
             'diagnostics_passed':mcmc_passed,'covariance_ppc_passed':covariance_passed,
             'accepted':passed,'status':adequacy_status(mcmc_passed,covariance_passed),'models':results,
             'scientific_reference_comparison':'requires separate independent validation'}
    if passed:
        name='primary_current.json' if summary['all_twelve_models'] else 'primary_selection_current.json'
        write_pointer(directories['work']/name,summary)
    return summary
