"""Fingerprint, fit, and validate supporting MMEF and X6 joint models in private output directories.

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
import pandas as pd

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, verify_artifacts, write_json,
                        resolve_paths,run_preparation,write_pointer)
from .pipeline import ValidationError,read_json as _read_json
from .supporting_sampling import (TARGETS, build_model, validate_contract, contract_key, fit_spec)
from .model_runtime import configure_runtime,model_runtime_identity
from .supporting_sampling import resolved_sampling,sample_contract

CRITERIA = {'min_chains':4,'min_draws':1600,'max_rhat':1.01,
            'min_bulk_ess':400,'min_tail_ess':400,'min_bfmi':.3,
            'max_divergences':0,'max_treedepth_hits':0}
COMPUTE_SOURCES = ('models/mmef.py','models/x6.py','supporting_sampling.py','sampling.py','model_runtime.py')
VALIDATION_SOURCES = ('supporting_execution.py','diagnostics.py','analysis/supporting_results.py','analysis/primary_results.py')
_LOADED_RUNNER_HASH = file_hash(Path(__file__))
RESULT_FILES = ('diagnostics.json', 'summary.json', 'result_metadata.json',
                'estimates.csv', 'hazard.csv', 'sample_counts.csv')


def read_json(path, label):
    value = _read_json(path, label)
    if not isinstance(value, dict):
        raise ValidationError(label+' must be an object.')
    return value


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def source_hashes(root, names):
    return {name:file_hash(Path(root)/'src/dchs_lung_function'/name) for name in names}



# Bind the imported execution session before any caller can change source files.
_LOADED_SOURCE_HASHES = {name:file_hash(Path(__file__).parent/name)
                         for name in (*COMPUTE_SOURCES, *VALIDATION_SOURCES)}


def assert_loaded_sources(root):
    local = Path(__file__).resolve().parents[2]
    if (source_hashes(root, _LOADED_SOURCE_HASHES) != _LOADED_SOURCE_HASHES
            or source_hashes(local, _LOADED_SOURCE_HASHES) != _LOADED_SOURCE_HASHES):
        raise ValidationError('Supporting sources changed after import or differ from the supplied source tree; restart the process.')


def contract_descriptor(preparation_directory, target):
    if target not in TARGETS:
        raise ValidationError('Unknown supporting model target.')
    directory = Path(preparation_directory).resolve()
    receipt = read_json(directory/'receipt.json','Preparation receipt')
    if (receipt.get('through')!='M03' or receipt.get('status')!='prepared'
            or file_hash(directory/'artifacts.json')!=receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Supporting fitting requires a complete verified M03 preparation.')
    if digest(receipt.get('identity'))!=receipt.get('fingerprint'):
        raise ValidationError('Preparation receipt fingerprint is inconsistent.')
    manifest = verify_artifacts(directory)
    node = artifact_subtree(manifest,('M03','pft','samples',contract_key(target)))
    files = set()
    def visit(value):
        if isinstance(value,dict):
            if 'file' in value:files.add(value['file'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(node)
    if not files.issubset(manifest['files']):
        raise ValidationError('A supporting contract refers to an unverified preparation file.')
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
        raise ValidationError('Supporting work/output roots must be separate from source and preparation artifacts.')
    for folder in (work,output):folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    return root,prep,work,output


def verify_aggregates(directory, posterior, stats, contract, target, validated, summary):
    """Recompute every aggregate from complete, verified private posterior draws."""
    from .diagnostics import evaluate_diagnostics
    from .analysis.supporting_results import summarize_supporting
    essential = {n:stats[n] for n in ('diverging', 'energy', 'reached_max_treedepth') if n in stats}
    diagnostics = evaluate_diagnostics(posterior, essential, CRITERIA)
    results = summarize_supporting(posterior, contract, target)
    if (read_json(directory/'diagnostics.json', 'Supporting diagnostics') != diagnostics
            or read_json(directory/'result_metadata.json', 'Supporting result metadata') != results.metadata
            or summary.get('counts') != validated or summary.get('target') != target
            or summary.get('result_rows') != fit_spec(target)['result_rows'] or type(summary.get('diagnostics_passed')) is not bool
            or summary['diagnostics_passed'] != diagnostics['passed']
            or summary.get('status') != ('diagnostics_passed' if diagnostics['passed'] else 'diagnostics_failed')):
        raise ValidationError('Supporting aggregates disagree with verified posterior draws or diagnostic gates.')
    for name in ('estimates', 'hazard', 'sample_counts'):
        try:
            observed = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip')
            pd.testing.assert_frame_equal(observed, getattr(results, name), check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
        except (AssertionError, ValueError, TypeError) as exc:
            raise ValidationError('Saved supporting tables disagree with recomputed posterior summaries.') from None


def sample_one(root,preparation_directory,target,settings,work,output,*,resume=False):
    """Publish a complete raw fit, then a separately checked aggregate result.

    Failed diagnostic results retain their raw private evidence, with an
    explicit failed status. They never create a validated completion marker.
    """
    root,prep,work,output=safe_roots(root,preparation_directory,work,output)
    assert_loaded_sources(root)
    descriptor=contract_descriptor(prep,target)
    fingerprint,identity=sampling_identity(root,descriptor,target,settings)
    work=work/target;output=output/target
    work.mkdir(exist_ok=True,mode=0o700);output.mkdir(exist_ok=True,mode=0o700)
    final=work/fingerprint
    contract=load_artifacts(prep,('M03','pft','samples',contract_key(target)))
    validated=validate_contract(contract,target)
    configure_runtime(work/'compiler_cache')
    controls=identity['sampling']
    expected=expected_posterior_shapes(contract,target,controls['chains'],controls['draws'])
    def unchanged():
        assert_loaded_sources(root)
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
            if set(raw['files']) != {'posterior.npz', 'sample_stats.npz', 'sampling.json'}:
                raise ValidationError('Supporting raw fit has an unexpected file inventory.')
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
            if set(receipt['files']) != set(RESULT_FILES):
                raise ValidationError('Supporting validation has an unexpected file inventory.')
            summary=read_json(destination/'summary.json','Supporting summary')
            if receipt.get('summary')!=summary:
                raise ValidationError('Validation receipt and verified supporting summary disagree.')
            diagnostic=read_json(destination/'diagnostics.json','Supporting diagnostics')
            if (type(diagnostic.get('passed')) is not bool
                    or diagnostic['passed']!=summary.get('diagnostics_passed')
                    or diagnostic.get('status')!=('passed' if diagnostic['passed'] else 'failed')
                    or summary.get('status')!=('diagnostics_passed' if diagnostic['passed'] else 'diagnostics_failed')):
                raise ValidationError('Verified diagnostics and supporting completion summary disagree.')
            if receipt.get('raw_receipt_sha256')!=file_hash(final/'receipt.json'):
                raise ValidationError('Validation receipt is not bound to the verified raw fit.')
            if summary.get('fit_fingerprint') != fingerprint or summary.get('validation_fingerprint') != validation_fp:
                raise ValidationError('Supporting summary fingerprints disagree with their receipts.')
            verify_aggregates(destination, posterior, stats, contract, target, validated, summary)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during cached validation.')
            return {**summary,'raw_fit_reused':True,'validation_reused':True}
        from .diagnostics import evaluate_diagnostics
        from .analysis.supporting_results import summarize_supporting
        essential={name:stats[name] for name in ('diverging','energy','reached_max_treedepth') if name in stats}
        diagnostics=evaluate_diagnostics(posterior,essential,CRITERIA)
        results=summarize_supporting(posterior,contract,target)
        summary={'target':target,'fit_fingerprint':fingerprint,'validation_fingerprint':validation_fp,
                 'diagnostics_passed':bool(diagnostics['passed']),
                 'status':'diagnostics_passed' if diagnostics['passed'] else 'diagnostics_failed',
                 'counts':validated,'result_rows':len(results.estimates),
                 'scientific_reference_comparison':'requires separate independent validation'}
        stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=output))
        try:
            write_json(stage/'diagnostics.json',diagnostics)
            write_json(stage/'summary.json',summary)
            write_json(stage/'result_metadata.json',results.metadata)
            for name,frame in [('estimates',results.estimates),('hazard',results.hazard),('sample_counts',results.sample_counts)]:
                frame.to_csv(stage/(name+'.csv'),index=False,float_format='%.17g')
            receipt={'schema_version':1,'fingerprint':validation_fp,'identity':validation_id,'summary':summary,
                     'raw_receipt_sha256':file_hash(final/'receipt.json'),
                     'files':{p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}}
            write_json(stage/'receipt.json',receipt)
            verify_aggregates(stage, posterior, stats, contract, target, validated, summary)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during validation.')
            os.replace(stage,destination)
        finally:
            if stage.exists():shutil.rmtree(stage)
        return {**summary,'raw_fit_reused':reused,'validation_reused':False}


def run_supporting(root, config_file, analysis_file, primary_sampling_file,
                   mechanics_sampling_file, supporting_sampling_file, *, jobs=1, resume=False):
    """Run cumulative M01-M08, verifying every preceding selected analysis."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .supporting_sampling import validate_sampling
    from .mechanics_execution import run_mechanics
    from .supporting_repeated_execution import fit_selected
    if type(jobs) is not int or not 1 <= jobs <= 3:
        raise ValidationError('Supporting model concurrency must be between one and three jobs.')
    assert_loaded_sources(root)
    config_paths = [Path(p) for p in (primary_sampling_file, mechanics_sampling_file, supporting_sampling_file)]
    sampling_hashes = [file_hash(p) for p in config_paths]
    settings = validate_sampling(read_json(config_paths[2], 'Supporting sampling configuration'))
    configuration = read_json(config_file, 'Local configuration')
    _, directories = resolve_paths(configuration, config_file)
    for directory in directories.values():
        if directory == config_paths[2].resolve() or directory in config_paths[2].resolve().parents:
            raise ValidationError('Generated directories cannot contain the supporting sampling configuration.')
    prior = run_mechanics(root, config_file, analysis_file, config_paths[0], config_paths[1],
                          jobs=jobs, resume=resume)
    if not prior['accepted'] or prior['selected_models'] != 6 or prior['shared_factor_rows'] != 6:
        raise ValidationError('Cumulative M08 requires all selected preceding analyses to pass.')
    prep = directories['work']/'preparation_runs'/prior['preparation_fingerprint']
    work, output = directories['work']/'supporting_fits', directories['outputs']/'supporting_joint'
    selected = list(TARGETS)
    if jobs == 1:
        results = [sample_one(root, prep, target, settings, work, output, resume=resume) for target in selected]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = [pool.submit(sample_one, root, prep, target, settings, work, output, resume=resume)
                       for target in selected]
            results = [done.result() for done in as_completed(pending)]
    if (len(results) != len(selected) or any(not isinstance(result, dict) for result in results)
            or sorted(result.get('target', '') for result in results) != sorted(selected)):
        raise ValidationError('Supporting execution requires exactly the seven selected joint fits.')
    for result in results:
        accepted = result.get('diagnostics_passed')
        if (type(accepted) is not bool
                or result.get('status') != ('diagnostics_passed' if accepted else 'diagnostics_failed')
                or result.get('result_rows') != fit_spec(result['target'])['result_rows']):
            raise ValidationError('Supporting joint completion contradicts its diagnostic status or row inventory.')
    results.sort(key=lambda result:selected.index(result['target']))
    repeated = fit_selected(root, prep, directories['work']/'supporting_repeated',
                            directories['outputs']/'supporting_repeated', resume=resume)
    if (not isinstance(repeated, dict) or type(repeated.get('accepted')) is not bool
            or repeated.get('fitted_models') != 10
            or repeated.get('status') != ('accepted' if repeated['accepted'] else 'fit_failed')):
        raise ValidationError('Supporting repeated completion requires ten consistently validated component fits.')
    verified = run_preparation(root, config_file, analysis_file, 'M03', resume=True)
    if (verified['fingerprint'] != prior['preparation_fingerprint']
            or [file_hash(p) for p in config_paths] != sampling_hashes):
        raise ValidationError('Preparation or sampling dependencies changed during supporting execution.')
    passed = all(result['diagnostics_passed'] for result in results) and repeated['accepted']
    summary = {'through':'M08', 'preparation_fingerprint':prior['preparation_fingerprint'],
               'selected_models':len(results), 'supporting_models':results,
               'deterministic_fits':repeated['fitted_models'], 'supporting_repeated':repeated,
               'primary_models_verified':12, 'binary_models_verified':28,
               'trajectory_analyses_verified':4, 'mechanics_models_verified':6,
               'shared_factor_rows_verified':6, 'accepted':passed,
               'status':'accepted' if passed else 'fit_or_diagnostics_failed',
               'scientific_reference_comparison':'requires separate independent validation'}
    if passed:
        write_pointer(directories['work']/'supporting_current.json', summary)
    return summary
