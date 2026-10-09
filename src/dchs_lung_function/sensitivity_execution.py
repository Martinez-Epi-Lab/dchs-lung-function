"""Fingerprint, fit, and validate the ten designated sensitivity joint models in private output directories.

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
from .models.sensitivity import TARGETS, build_model, validate_contract, contract_key, fit_spec
from .model_runtime import configure_runtime,model_runtime_identity
from .sensitivity_sampling import resolved_sampling,sample_contract,expected_posterior_shapes

CRITERIA = {'min_chains':4,'min_draws':3200,'max_rhat':1.01,
            'min_bulk_ess':400,'min_tail_ess':400,'min_bfmi':.3,
            'max_divergences':0,'max_treedepth_hits':0}
COMPUTE_SOURCES = ('models/sensitivity.py','models/spirometry.py','sensitivity_sampling.py','sampling.py','model_runtime.py')
VALIDATION_SOURCES = ('sensitivity_execution.py','diagnostics.py','execution.py','pipeline.py',
                      'analysis/sensitivities.py','analysis/primary_results.py')
_LOADED_RUNNER_HASH = file_hash(Path(__file__))
TABLES = ('estimates','hazard','sample_counts')
RESULT_FILES = ('diagnostics.json','summary.json','result_metadata.json',
                *(name+'.csv' for name in TABLES))


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
        raise ValidationError('Sensitivity sources changed after import or differ from the supplied source tree; restart the process.')


def contract_descriptor(preparation_directory, target):
    if target not in TARGETS:
        raise ValidationError('Unknown sensitivity model target.')
    directory = Path(preparation_directory).resolve()
    receipt = read_json(directory/'receipt.json','Preparation receipt')
    if (receipt.get('through')!='M03' or receipt.get('status')!='prepared'
            or file_hash(directory/'artifacts.json')!=receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Sensitivity fitting requires a complete verified M03 preparation.')
    if digest(receipt.get('identity'))!=receipt.get('fingerprint'):
        raise ValidationError('Preparation receipt fingerprint is inconsistent.')
    manifest = verify_artifacts(directory)
    node = artifact_subtree(manifest,tuple(fit_spec(target)['artifact_path']))
    files = set()
    def visit(value):
        if isinstance(value,dict):
            if 'file' in value:files.add(value['file'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(node)
    if not files.issubset(manifest['files']):
        raise ValidationError('A sensitivity contract refers to an unverified preparation file.')
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
    directory = Path(directory)
    if (directory.is_symlink() or not directory.is_dir()
            or (directory/'receipt.json').is_symlink()
            or not (directory/'receipt.json').is_file()):
        raise ValidationError('Sensitivity publication directories and receipts must be regular nonsymlink artifacts.')
    if (not isinstance(entries,dict) or set(entries) != set(required)
            or {p.name for p in directory.iterdir()} != set(required) | {'receipt.json'}):
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
        raise ValidationError('Sensitivity work/output roots must be separate from source and preparation artifacts.')
    for folder in (work,output):folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    return root,prep,work,output


def validate_results(results, contract, target, posterior):
    """Compare the complete declared aggregate schema with primitive draws."""
    from .analysis.sensitivities import summarize_sensitivity, FIT_TABLE_COLUMNS
    expected = summarize_sensitivity(posterior,contract,target)
    spec, info = fit_spec(target), validate_contract(contract,target)
    try:
        from .analysis.sensitivities import SUMMARY_COLUMNS, derive_contrast_draws, derive_hazard_draws
        from .analysis.primary_results import _sample_counts
        if len(results.estimates)!=spec['result_rows'] or len(results.hazard)!=spec['hazard_rows'] or len(results.sample_counts)!=30:
            raise ValueError()
        shape=np.asarray(posterior['direct_ratio']).shape
        contrasts=derive_contrast_draws(posterior,contract,target)
        expected_meta={**info,'family':spec['family'],'observation_window':spec['observation_window'],
            'outcome':spec['outcome'],'unit':spec['unit'],'contrasts':list(contrasts),
            'hazards':list(spec['hazard_variables'].values()),'interval':'equal-tailed 95% posterior',
            'quantile_method':'linear','mcse_computed':True,'chains':shape[0],'draws_per_chain':shape[1],
            'posterior_variables':len(posterior),'hazard_covariates':list(spec['hazard_covariates']),
            'lrti_role':'nuisance regressions only; exploratory ever-recorded covariate',
            'reference_group':('pre/no-TB observations excluding post-TB measurements following first TB at or after age five'
                              if spec['family']=='early' else 'pre/no-prior-TB observations'),
            'acceptance':'complete posterior diagnostics and independent scientific comparison are separate'}
        if canonical_bytes(results.metadata)!=canonical_bytes(expected_meta):raise ValueError()
        kinds=([v[1] for v in spec['contrasts'].values()] if spec['family']=='early' else ['total']*5+['interaction']*3)
        if results.estimates.estimand_type.tolist()!=kinds or not results.estimates.residual_scale.eq(info['target_scale']).all():
            raise ValueError()
        for frame,draws in ((results.estimates,derive_contrast_draws(posterior,contract,target)),
                            (results.hazard,derive_hazard_draws(posterior,target))):
            if list(frame.contrast)!=list(draws) or not frame.target.eq(target).all() or not frame.model_id.eq(spec['model_id']).all():
                raise ValueError()
            if len(frame) and (frame[SUMMARY_COLUMNS].to_numpy().dtype.kind not in 'iuf' or not np.isfinite(frame[SUMMARY_COLUMNS].to_numpy()).all()):
                raise ValueError()
            for index,value in enumerate(draws.values()):
                row=frame.iloc[index]; flat=value.reshape(-1)
                observed=row[['mean','median','lower','upper','probability_below_zero','probability_above_zero','probability_equal_zero']].to_numpy(float)
                actual=[flat.mean(),np.median(flat),*np.quantile(flat,[.025,.975]),np.mean(flat<0),np.mean(flat>0),np.mean(flat==0)]
                if not np.allclose(observed,actual,rtol=1e-12,atol=1e-12) or row.mcse_mean<0:raise ValueError()
        if not results.hazard.unit.eq('hazard ratio').all():raise ValueError()
        for key in ('family','observation_window','outcome','unit'):
            if not results.estimates[key].eq(spec[key]).all():raise ValueError()
        counts=_sample_counts(contract); counts.insert(0,'target',target)
        pd.testing.assert_frame_equal(results.sample_counts,counts,check_exact=True)

        for name in TABLES:
            observed = getattr(results,name)
            if not isinstance(observed,pd.DataFrame) or list(observed.columns)!=FIT_TABLE_COLUMNS[name]:
                raise ValueError()
            pd.testing.assert_frame_equal(observed,getattr(expected,name),check_exact=True)
        if canonical_bytes(results.metadata) != canonical_bytes(expected.metadata):
            raise ValueError()
    except (AssertionError,KeyError,AttributeError,TypeError,ValueError):
        raise ValidationError('Sensitivity result schema or values differ from primitive posterior summaries.') from None


def make_summary(target,fingerprint,validation_fp,validated,diagnostics,results):
    return {'target':target,'fit_fingerprint':fingerprint,'validation_fingerprint':validation_fp,
            'diagnostics_passed':bool(diagnostics['passed']),
            'status':'diagnostics_passed' if diagnostics['passed'] else 'diagnostics_failed',
            'counts':validated,'result_rows':{name:len(getattr(results,name)) for name in TABLES},
            'scientific_reference_comparison':'requires separate independent validation'}


def verify_aggregates(directory, posterior, stats, contract, target, validated, summary):
    """Recompute every aggregate from complete, verified private posterior draws."""
    from .diagnostics import evaluate_diagnostics
    from .analysis.sensitivities import summarize_sensitivity
    essential = {n:stats[n] for n in ('diverging', 'energy', 'reached_max_treedepth') if n in stats}
    diagnostics = evaluate_diagnostics(posterior, essential, CRITERIA)
    results = summarize_sensitivity(posterior, contract, target)
    validate_results(results,contract,target,posterior)
    if (canonical_bytes(summary) != canonical_bytes(make_summary(target,summary.get('fit_fingerprint'),summary.get('validation_fingerprint'),validated,diagnostics,results))
            or read_json(directory/'diagnostics.json', 'Sensitivity diagnostics') != diagnostics
            or read_json(directory/'result_metadata.json', 'Sensitivity result metadata') != results.metadata
            or summary.get('counts') != validated or summary.get('target') != target
            or summary.get('result_rows') != {name:len(getattr(results,name)) for name in TABLES} or type(summary.get('diagnostics_passed')) is not bool
            or summary['diagnostics_passed'] != diagnostics['passed']
            or summary.get('status') != ('diagnostics_passed' if diagnostics['passed'] else 'diagnostics_failed')):
        raise ValidationError('Sensitivity aggregates disagree with verified posterior draws or diagnostic gates.')
    for name in TABLES:
        try:
            observed = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip')
            pd.testing.assert_frame_equal(observed, getattr(results, name), check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
        except (AssertionError, ValueError, TypeError) as exc:
            raise ValidationError('Saved sensitivity tables disagree with recomputed posterior summaries.') from None


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
    if work.is_symlink() or output.is_symlink() or (work/'compiler_cache').is_symlink():
        raise ValidationError('Sensitivity target directories and compiler cache cannot be symbolic links.')
    work.mkdir(exist_ok=True,mode=0o700);output.mkdir(exist_ok=True,mode=0o700)
    final=work/fingerprint
    contract=load_artifacts(prep,tuple(fit_spec(target)['artifact_path']))
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
        reused=final.exists() or final.is_symlink()
        if reused:
            if not resume:raise ValidationError('An identical raw fit exists; use --resume to verify and revalidate it.')
            raw=read_json(final/'receipt.json','Raw fit receipt')
            if raw.get('identity')!=identity or raw.get('fingerprint')!=fingerprint or raw.get('status')!='sampled':
                raise ValidationError('Raw fit receipt identity is inconsistent.')
            verify_files(final,raw.get('files'),('posterior.npz','sample_stats.npz','sampling.json'))
            if set(raw['files']) != {'posterior.npz', 'sample_stats.npz', 'sampling.json'}:
                raise ValidationError('Sensitivity raw fit has an unexpected file inventory.')
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
                if final.exists() or final.is_symlink():
                    raise ValidationError('Raw fit destination appeared during execution.')
                os.replace(stage,final)
            finally:
                if stage.exists():shutil.rmtree(stage)
        validate_posterior_inventory(posterior,expected)
        validate_sampling_metadata(metadata,identity,posterior,stats)
        if file_hash(Path(__file__))!=_LOADED_RUNNER_HASH:
            raise ValidationError('The loaded model runner changed; restart with --resume to validate saved draws.')
        validation_fp,validation_id=validation_identity(root,fingerprint,CRITERIA)
        destination=output/validation_fp
        if destination.exists() or destination.is_symlink():
            receipt=read_json(destination/'receipt.json','Validation receipt')
            if (set(receipt)!={'schema_version','fingerprint','identity','summary','raw_receipt_sha256','files'}
                    or type(receipt.get('schema_version')) is not int or receipt['schema_version']!=1
                    or receipt.get('identity')!=validation_id or receipt.get('fingerprint')!=validation_fp):
                raise ValidationError('Validation receipt identity is inconsistent.')
            verify_files(destination,receipt.get('files'),RESULT_FILES)
            if set(receipt['files']) != set(RESULT_FILES):
                raise ValidationError('Sensitivity validation has an unexpected file inventory.')
            summary=read_json(destination/'summary.json','Sensitivity summary')
            if receipt.get('summary')!=summary:
                raise ValidationError('Validation receipt and verified sensitivity summary disagree.')
            diagnostic=read_json(destination/'diagnostics.json','Sensitivity diagnostics')
            if (type(diagnostic.get('passed')) is not bool
                    or diagnostic['passed']!=summary.get('diagnostics_passed')
                    or diagnostic.get('status')!=('passed' if diagnostic['passed'] else 'failed')
                    or summary.get('status')!=('diagnostics_passed' if diagnostic['passed'] else 'diagnostics_failed')):
                raise ValidationError('Verified diagnostics and sensitivity completion summary disagree.')
            if receipt.get('raw_receipt_sha256')!=file_hash(final/'receipt.json'):
                raise ValidationError('Validation receipt is not bound to the verified raw fit.')
            if summary.get('fit_fingerprint') != fingerprint or summary.get('validation_fingerprint') != validation_fp:
                raise ValidationError('Sensitivity summary fingerprints disagree with their receipts.')
            verify_aggregates(destination, posterior, stats, contract, target, validated, summary)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during cached validation.')
            return {**summary,'raw_fit_reused':True,'validation_reused':True}
        from .diagnostics import evaluate_diagnostics
        from .analysis.sensitivities import summarize_sensitivity
        essential={name:stats[name] for name in ('diverging','energy','reached_max_treedepth') if name in stats}
        diagnostics=evaluate_diagnostics(posterior,essential,CRITERIA)
        results=summarize_sensitivity(posterior,contract,target)
        validate_results(results,contract,target,posterior)
        summary=make_summary(target,fingerprint,validation_fp,validated,diagnostics,results)
        stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=output))
        try:
            write_json(stage/'diagnostics.json',diagnostics)
            write_json(stage/'summary.json',summary)
            write_json(stage/'result_metadata.json',results.metadata)
            for name in TABLES:
                frame=getattr(results,name)
                frame.to_csv(stage/(name+'.csv'),index=False,float_format='%.17g')
            receipt={'schema_version':1,'fingerprint':validation_fp,'identity':validation_id,'summary':summary,
                     'raw_receipt_sha256':file_hash(final/'receipt.json'),
                     'files':{p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}}
            write_json(stage/'receipt.json',receipt)
            verify_aggregates(stage, posterior, stats, contract, target, validated, summary)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during validation.')
            if destination.exists() or destination.is_symlink():
                raise ValidationError('Validation destination appeared during execution.')
            os.replace(stage,destination)
        finally:
            if stage.exists():shutil.rmtree(stage)
        return {**summary,'raw_fit_reused':reused,'validation_reused':False}


def run_sensitivities(root, config_file, analysis_file, primary_sampling_file,
                      mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file,
                      sensitivity_sampling_file, *, jobs=1, resume=False):
    """Run cumulative M01-M12 and publish the designated sensitivity table inputs."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .sensitivity_sampling import validate_sampling
    from .sampling import validate_sampling as validate_primary_sampling
    from .wheeze_sampling import validate_sampling as validate_wheeze_sampling
    from .feno_execution import run_feno
    from .sensitivity_tables_execution import publish_tables
    if type(jobs) is not int or not 1<=jobs<=3:
        raise ValidationError('Sensitivity concurrency must be between one and three jobs.')
    assert_loaded_sources(root)
    paths=[Path(p) for p in (config_file,analysis_file,primary_sampling_file,
        mechanics_sampling_file,supporting_sampling_file,wheeze_sampling_file,sensitivity_sampling_file)]
    hashes=[file_hash(p) for p in paths]
    settings=validate_sampling(read_json(paths[-1],'Sensitivity sampling configuration'))
    primary_settings=validate_primary_sampling(read_json(paths[2],'Primary sampling configuration'))
    wheeze_settings=validate_wheeze_sampling(read_json(paths[5],'Wheeze sampling configuration'))
    configuration=read_json(config_file,'Local configuration')
    _,directories=resolve_paths(configuration,config_file)
    for directory in directories.values():
        if directory==paths[-1].resolve() or directory in paths[-1].resolve().parents:
            raise ValidationError('Generated directories cannot contain the sensitivity sampling configuration.')
    prior=run_feno(root,*paths[:-1],jobs=jobs,resume=resume)
    expected={'through':'M11','selected_models':11,'primary_models_verified':12,
        'binary_models_verified':28,'trajectory_analyses_verified':4,'mechanics_models_verified':6,
        'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
        'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,
        'wheeze_models_verified':2,'status':'accepted'}
    if (not isinstance(prior,dict) or prior.get('accepted') is not True
            or any(type(prior.get(k)) is not type(v) or prior[k]!=v for k,v in expected.items())):
        raise ValidationError('Cumulative M12 requires all selected preceding analyses to pass.')
    prep=directories['work']/'preparation_runs'/prior['preparation_fingerprint']
    work,output=directories['work']/'sensitivity_fits',directories['outputs']/'sensitivity_models'
    selected=list(TARGETS)
    if jobs==1:
        results=[sample_one(root,prep,t,settings,work,output,resume=resume) for t in selected]
    else:
        with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(sample_one,root,prep,t,settings,work,output,resume=resume) for t in selected]
            results=[f.result() for f in as_completed(futures)]
    if (len(results)!=len(selected) or any(not isinstance(r,dict) or not isinstance(r.get('target'),str) for r in results)
            or sorted(r['target'] for r in results)!=sorted(selected)):
        raise ValidationError('Sensitivity execution requires exactly the ten selected new joint fits.')
    fields={'target','fit_fingerprint','validation_fingerprint','diagnostics_passed','status','counts',
            'result_rows','scientific_reference_comparison','raw_fit_reused','validation_reused'}
    for result in results:
        spec=fit_spec(result['target']);passed=result.get('diagnostics_passed')
        row_counts={'estimates':spec['result_rows'],'hazard':spec['hazard_rows'],'sample_counts':30}
        if (set(result)!=fields or type(passed) is not bool
                or result['status']!=('diagnostics_passed' if passed else 'diagnostics_failed')
                or not isinstance(result.get('result_rows'),dict)
                or result['result_rows']!=row_counts
                or any(type(v) is not int for v in result['result_rows'].values())
                or any(type(result[k]) is not bool for k in ('raw_fit_reused','validation_reused'))):
            raise ValidationError('Sensitivity completion status or aggregate row inventory is inconsistent.')
    results.sort(key=lambda r:selected.index(r['target']))
    passed=all(r['diagnostics_passed'] for r in results)
    tables=publish_tables(root,prep,directories,results,settings,primary_settings,wheeze_settings,resume=resume) if passed else None
    if passed:
        table_fields={'fingerprint','accepted','status','rows','new_fit_count','reused_fit_count',
            'source_fits_verified','reused_primary_models','reused_wheeze_models','scientific_reference_comparison','resumed'}
        required_values={'accepted':True,'status':'accepted','new_fit_count':10,'reused_fit_count':4,
            'source_fits_verified':14,'reused_primary_models':2,'reused_wheeze_models':2,
            'scientific_reference_comparison':'requires separate independent validation'}
        if (not isinstance(tables,dict) or set(tables)!=table_fields
                or any(type(tables.get(k)) is not type(v) or tables[k]!=v for k,v in required_values.items())
                or type(tables.get('resumed')) is not bool
                or canonical_bytes(tables.get('rows'))!=canonical_bytes({'table_s1':10,'table_s3':24,'table_s4':12})
                or not isinstance(tables.get('fingerprint'),str) or len(tables['fingerprint'])!=64
                or any(c not in '0123456789abcdef' for c in tables['fingerprint'])):
            raise ValidationError('Sensitivity table completion is incomplete or unaccepted.')
    verified=run_preparation(root,config_file,analysis_file,'M03',resume=True)
    if verified['fingerprint']!=prior['preparation_fingerprint'] or [file_hash(p) for p in paths]!=hashes:
        raise ValidationError('Preparation or configuration changed during sensitivity execution.')
    assert_loaded_sources(root)
    summary={'through':'M12','preparation_fingerprint':prior['preparation_fingerprint'],
        'selected_models':10,'reused_models':4,'sensitivity_models':results,'sensitivity_tables':tables,
        **{k:v for k,v in expected.items() if k.endswith('_verified')},'feno_models_verified':11,
        'accepted':passed,'status':'accepted' if passed else 'diagnostics_failed',
        'inherited_limitations':['G08: pre-TB source covariance limitation remains unresolved',
                                'Four upstream antenatal flow counts remain unavailable'],
        'scientific_reference_comparison':'requires separate independent validation'}
    if passed:write_pointer(directories['work']/'sensitivity_current.json',summary)
    return summary
