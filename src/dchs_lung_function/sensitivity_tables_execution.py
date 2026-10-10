"""Verify fourteen existing fits and publish aggregate sensitivity table inputs.

This module never samples. Raw posteriors and participant-bearing preparation
objects stay in memory and private storage; published tables contain aggregates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .execution import (canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, write_json)
from .pipeline import ValidationError, read_json as _read_json
from .model_adequacy import evaluate_covariance, verify_covariance

SOURCES = ('lifecycle.py','model_adequacy.py','covariance_ppc.py','runtime_bootstrap.py','analysis/attenuation_precision.py','sensitivity_tables_execution.py', 'sensitivity_execution.py',
           'primary_execution.py', 'wheeze_execution.py',
           'analysis/sensitivities.py', 'analysis/primary_results.py',
           'analysis/wheeze_results.py', 'analysis/wheeze_standardization.py',
           'models/sensitivity.py', 'models/spirometry.py', 'models/wheeze.py',
           'sensitivity_sampling.py', 'sampling.py', 'wheeze_sampling.py',
           'model_runtime.py', 'diagnostics.py', 'execution.py', 'pipeline.py',
           'respiratory_preparation.py')
TABLES = ('table_s1', 'table_s3', 'table_s4')
ROWS = {'table_s1':10, 'table_s3':24, 'table_s4':12}
RESULT_FILES = (*[name+'.csv' for name in TABLES], 'metadata.json', 'summary.json')
RAW_FILES = ('posterior.npz', 'sample_stats.npz', 'sampling.json')
PRIMARY_FILES = ('covariance_ppc.json','diagnostics.json', 'summary.json', 'result_metadata.json',
                 'estimates.csv', 'shared_factor.csv', 'sample_counts.csv')
_LOADED_SOURCES = {name:file_hash(Path(__file__).parent/name) for name in SOURCES}


def _require(condition, message):
    if not condition:
        raise ValidationError('Sensitivity tables '+message)


def read_json(path, label):
    value = _read_json(path, label)
    _require(isinstance(value, dict), label+' must be an object.')
    return value


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _equal(actual, expected):
    """JSON identity with types preserved, including booleans versus integers."""
    try:
        return canonical_bytes(actual) == canonical_bytes(expected)
    except (ValueError, TypeError, OverflowError):
        return False


def source_hashes(root):
    return {name:file_hash(Path(root)/'src/dchs_lung_function'/name) for name in SOURCES}


def assert_loaded_sources(root):
    _require(source_hashes(root) == _LOADED_SOURCES
             and {n:file_hash(Path(__file__).parent/n) for n in SOURCES} == _LOADED_SOURCES,
             'sources changed after import or differ from the supplied tree; restart the process.')


def _no_symlinks(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             'artifact paths and their parent directories cannot be symbolic links.')


def verify_files(directory, entries, required):
    directory = Path(directory)
    _no_symlinks(directory)
    _require(directory.is_dir() and isinstance(entries, dict) and set(entries) == set(required)
             and {p.name for p in directory.iterdir()} == set(required)|{'receipt.json'},
             'artifact inventory is incomplete or contains extra files.')
    _require((directory/'receipt.json').is_file() and not (directory/'receipt.json').is_symlink(),
             'receipt must be a regular nonsymlink file.')
    for name, value in entries.items():
        path = directory/name
        _require(Path(name).name == name and path.is_file() and not path.is_symlink()
                 and isinstance(value, str) and file_hash(path) == value,
                 'a required artifact is missing or changed.')


def _array_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), 'array archive contains duplicated names.')
        return {name:archive[name] for name in archive.files}


def _csv_bytes(frame):
    return frame.to_csv(index=False, float_format='%.17g').encode('utf-8')


def _verify_csv(path, expected):
    _require(Path(path).read_bytes() == _csv_bytes(expected),
             'saved aggregate CSV differs from exact posterior recomputation.')


def _family_api(family):
    # Delay runner imports: the cumulative sensitivity runner calls this module.
    if family == 'sensitivity':
        from . import sensitivity_execution as api
    elif family == 'primary':
        from . import primary_execution as api
    elif family == 'wheeze':
        from . import wheeze_execution as api
    else:
        raise ValidationError('Unsupported sensitivity-table fit family.')
    return api


def _verified_fit(family, root, prep, directories, settings, target, supplied=None):
    """Return recomputed aggregate entry, public-safe binding and private hashes."""
    from .diagnostics import evaluate_diagnostics
    from .model_runtime import configure_runtime

    api = _family_api(family)
    if hasattr(api, 'assert_loaded_sources'):
        api.assert_loaded_sources(root)
    else:
        _require(api._LOADED_RUNNER_HASH == file_hash(Path(api.__file__)),
                 'the imported primary runner changed; restart the process.')
    descriptor = api.contract_descriptor(prep, target)
    fit_fp, fit_identity = api.sampling_identity(root, descriptor, target, settings)
    validation_fp, validation_identity = api.validation_identity(root, fit_fp, api.CRITERIA)
    work_name, output_name = {'sensitivity':('sensitivity_fits','sensitivity_models'),
                             'primary':('primary_fits','primary_models'),
                             'wheeze':('wheeze_fits','wheeze_models')}[family]
    raw = directories['work']/work_name/target/fit_fp
    aggregate = directories['outputs']/output_name/target/validation_fp
    for folder in (raw, aggregate):
        _no_symlinks(folder)
        _require((folder/'receipt.json').is_file() and not (folder/'receipt.json').is_symlink(),
                 'require complete existing source fits; this publisher cannot sample.')
    rr = read_json(raw/'receipt.json', 'Raw fit receipt')
    vr = read_json(aggregate/'receipt.json', 'Validation receipt')
    _require(set(rr) == {'schema_version','status','fingerprint','identity','preparation_fingerprint','completed_at','files'}
             and type(rr.get('schema_version')) is int and rr['schema_version'] == 1
             and rr.get('status') == 'sampled' and _equal(rr.get('identity'), fit_identity)
             and rr.get('fingerprint') == fit_fp == digest(fit_identity),
             'raw receipt disagrees with independently derived fit identity.')
    _require(set(vr) == {'schema_version','fingerprint','identity','summary','raw_receipt_sha256','files'}
             and type(vr.get('schema_version')) is int and vr['schema_version'] == 1
             and _equal(vr.get('identity'), validation_identity)
             and vr.get('fingerprint') == validation_fp == digest(validation_identity)
             and vr.get('raw_receipt_sha256') == file_hash(raw/'receipt.json'),
             'validation receipt disagrees with the derived identity or raw receipt.')
    aggregate_files = PRIMARY_FILES if family == 'primary' else api.RESULT_FILES
    verify_files(raw, rr.get('files'), RAW_FILES)
    verify_files(aggregate, vr.get('files'), aggregate_files)
    path = (('M03','pft','samples','primary/'+target) if family == 'primary'
            else tuple(api.fit_spec(target)['artifact_path']))
    contract = load_artifacts(prep, path)
    info = api.validate_contract(contract, target)
    cache = directories['work']/'sensitivity_table_validation'/'compiler_cache'
    _no_symlinks(cache)
    configure_runtime(cache)
    posterior = _array_archive(raw/'posterior.npz')
    stats = _array_archive(raw/'sample_stats.npz')
    controls = fit_identity['sampling']
    api.validate_posterior_inventory(posterior,
        api.expected_posterior_shapes(contract, target, controls['chains'], controls['draws']))
    sampling = read_json(raw/'sampling.json', 'Realized sampling metadata')
    api.validate_sampling_metadata(sampling, fit_identity, posterior, stats)
    for key in ('realized_chains','realized_draws_per_chain','realized_tuning_steps'):
        _require(type(sampling.get(key)) is int, 'realized sampling dimensions must be integers.')
    _require(_equal(sampling.get('controls'), controls) and _equal(sampling.get('runtime'), fit_identity['runtime']),
             'sampling controls or runtime metadata has inconsistent types or values.')
    shape = (controls['chains'], controls['draws'])
    _require(stats and all(np.asarray(a).shape == shape and np.asarray(a).dtype.kind in 'biuf' for a in stats.values()),
             'complete sampler-statistic arrays have invalid coordinates or types.')
    essential = {name:stats[name] for name in ('diverging','energy','reached_max_treedepth') if name in stats}
    diagnostic = evaluate_diagnostics(posterior, essential, api.CRITERIA)
    _require(diagnostic.get('passed') is True and diagnostic.get('status') == 'passed'
             and _equal(read_json(aggregate/'diagnostics.json', 'Source diagnostics'), diagnostic),
             'a source fit fails recomputed complete-posterior diagnostics.')

    if family!='wheeze':
        covariance=evaluate_covariance(posterior,contract,family=family,target=target)
        source_summary=read_json(aggregate/'summary.json','Source completion')
        verify_covariance(aggregate,source_summary,diagnostic,covariance)
        _require(covariance['accepted'] is True,'a source fit fails covariance adequacy.')

    if family == 'primary':
        from .analysis.primary_results import summarize_primary
        result = summarize_primary(posterior, contract, target)
        source_tables = {name:getattr(result,name) for name in ('estimates','shared_factor','sample_counts')}
        row_inventory = len(result.estimates)
        entry_estimates, entry_metadata = result.estimates, result.metadata
    elif family == 'sensitivity':
        from .analysis.sensitivities import summarize_sensitivity
        result = summarize_sensitivity(posterior, contract, target)
        source_tables = {name:getattr(result,name) for name in api.TABLES}
        row_inventory = {name:len(table) for name,table in source_tables.items()}
        entry_estimates, entry_metadata = result.estimates, result.metadata
    else:
        from .analysis.wheeze_results import (summarize_wheeze, summarize_effects,
                                              validate_posterior)
        values, _ = validate_posterior(posterior, info)
        ppc_seed = controls['random_seed'] + 77
        result = summarize_wheeze(posterior, contract, target, ppc_seed=ppc_seed)
        api.validate_results(result, target, info, posterior, ppc_seed)
        source_tables = {name:getattr(result,name) for name in api.TABLES}
        row_inventory = {name:len(table) for name,table in source_tables.items()}
        entry_estimates, entry_metadata = summarize_effects(values, target), info
        replicates = _array_archive(aggregate/'ppc_replicates.npz')
        _require(set(replicates) == set(result.ppc_replicates)
                 and all(np.array_equal(replicates[n], result.ppc_replicates[n], equal_nan=True) for n in replicates),
                 'wheeze predictive aggregates differ from their recorded posterior and seed.')
    _require(_equal(read_json(aggregate/'result_metadata.json', 'Source result metadata'), result.metadata),
             'source aggregate metadata differs from posterior recomputation.')
    for name, table in source_tables.items():
        _verify_csv(aggregate/(name+'.csv'), table)
    summary = {'target':target, 'fit_fingerprint':fit_fp, 'validation_fingerprint':validation_fp,
               'diagnostics_passed':True, 'status':'diagnostics_passed', 'counts':info,
               'result_rows':row_inventory,
               'scientific_reference_comparison':'requires separate independent validation'}
    if family!='wheeze': summary['covariance_ppc_passed']=True
    _require(_equal(read_json(aggregate/'summary.json', 'Source completion'), summary)
             and _equal(vr.get('summary'), summary),
             'source completion schema or values differ from verified posterior results.')
    if supplied is not None:
        _require(isinstance(supplied, dict)
                 and set(supplied) == set(summary)|{'raw_fit_reused','validation_reused'}
                 and all(type(supplied[k]) is bool for k in ('raw_fit_reused','validation_reused'))
                 and _equal({k:supplied[k] for k in summary}, summary),
                 'provided new-fit completion differs from its verified source artifacts.')
    bindings = {'family':family, 'target':target, 'contract_sha256':descriptor['sha256'],
                'fit_fingerprint':fit_fp, 'validation_fingerprint':validation_fp,
                'raw_receipt_sha256':file_hash(raw/'receipt.json'),
                'validation_receipt_sha256':file_hash(aggregate/'receipt.json')}
    input_files = {folder/name:value for folder,receipt in ((raw,rr),(aggregate,vr)) for name,value in receipt['files'].items()}
    input_files.update({folder/'receipt.json':file_hash(folder/'receipt.json') for folder in (raw,aggregate)})
    return {'entry':{'estimates':entry_estimates,'metadata':entry_metadata,'summary':summary},
            'binding':bindings, 'input_files':input_files}


def check_tables(result, expected):
    """Require only the selected aggregate schemas and exact computed content."""
    from .analysis.sensitivities import TABLE_COLUMNS, DEFAULT_TARGETS, WHEEZE_TARGETS
    from .analysis.attenuation_precision import MAX_MCSE_BOUND_PP
    expected_metadata = {
        'schema_version':1, 'new_model_targets':list(DEFAULT_TARGETS),
        'reused_primary_targets':['ratio_pre','ratio_post'], 'reused_wheeze_targets':list(WHEEZE_TARGETS),
        'rows':dict(ROWS),
        'attenuation':'100*(1-abs(adjusted posterior mean)/abs(early core posterior mean))',
        'attenuation_interval':'none; this is a transformation of posterior means',
        'attenuation_mcse':'first-order covariance-agnostic upper bound from marginal posterior-mean MCSEs',
        'attenuation_max_mcse_bound_pp':MAX_MCSE_BOUND_PP,
        'attenuation_precision_passed':bool(expected.table_s1.attenuation_precision_passed.all()),
        'zero_core':'undefined_zero_core; no epsilon substitution',
        'negative_attenuation':'retained unchanged; increased effect magnitude',
        'reused_fit_count':4, 'new_fit_count':10, 'rendered_manuscript':False}
    try:
        for name in TABLES:
            table = getattr(result,name)
            _require(isinstance(table,pd.DataFrame) and list(table.columns) == TABLE_COLUMNS[name]
                     and len(table) == ROWS[name], 'table schema or selected row inventory differs.')
            pd.testing.assert_frame_equal(table, getattr(expected,name), check_exact=True)
        _require(_equal(result.metadata, expected.metadata) and _equal(result.metadata, expected_metadata)
                 and _equal(result.metadata.get('rows'), ROWS)
                 and type(result.metadata.get('new_fit_count')) is int and result.metadata['new_fit_count'] == 10
                 and type(result.metadata.get('reused_fit_count')) is int and result.metadata['reused_fit_count'] == 4,
                 'aggregate table metadata differs from the selected scope.')
    except (AssertionError, AttributeError, TypeError, ValueError, KeyError):
        raise ValidationError('Sensitivity table schemas or exact computed contents are invalid.') from None


def verify_result(directory, fingerprint, identity, result, summary, metadata):
    directory = Path(directory)
    _no_symlinks(directory)
    _require(not (directory/'receipt.json').is_symlink(), 'table receipt cannot be a symbolic link.')
    receipt = read_json(directory/'receipt.json', 'Sensitivity-table receipt')
    _require(set(receipt) == {'schema_version','status','identity','fingerprint','summary','completed_at','files'}
             and type(receipt['schema_version']) is int and receipt['schema_version'] == 1
             and receipt['status'] == 'accepted' and receipt['fingerprint'] == fingerprint
             and _equal(receipt['identity'], identity) and digest(receipt['identity']) == fingerprint,
             'table receipt identity or schema differs.')
    verify_files(directory, receipt['files'], RESULT_FILES)
    _require(_equal(receipt['summary'], summary)
             and _equal(read_json(directory/'summary.json', 'Sensitivity-table summary'), summary)
             and _equal(read_json(directory/'metadata.json', 'Sensitivity-table metadata'), metadata),
             'table completion or aggregate metadata differs from verified source fits.')
    for name in TABLES:
        _verify_csv(directory/(name+'.csv'), getattr(result,name))


def publish_tables(root, prep, directories, new_results, sensitivity_settings,
                   primary_settings, wheeze_settings, *, resume=False):
    """Verify ten new and four reused fits; atomically publish S1/S3/S4 inputs."""
    from .analysis.sensitivities import build_tables, DEFAULT_TARGETS, WHEEZE_TARGETS
    from .model_runtime import model_runtime_identity
    from .sensitivity_sampling import validate_sampling as validate_sensitivity
    from .sampling import validate_sampling as validate_primary
    from .wheeze_sampling import validate_sampling as validate_wheeze

    _require(type(resume) is bool and isinstance(directories, dict)
             and set(directories) == {'work','outputs'}, 'publication arguments are malformed.')
    for path in (root, prep, *directories.values()):
        _no_symlinks(path)
    root, prep = Path(root).resolve(), Path(prep).resolve()
    directories = {name:Path(path).resolve() for name,path in directories.items()}
    work, output = directories['work']/'sensitivity_table_validation', directories['outputs']/'sensitivity_tables'
    overlap = lambda a,b:a==b or a in b.parents or b in a.parents
    _require(not overlap(directories['work'], directories['outputs'])
             and not any(overlap(folder, root) for folder in directories.values())
             and not any(overlap(folder, prep) for folder in (work,output)),
             'work/output roots must be separate from source and preparation artifacts.')
    for path in (work,output):
        _no_symlinks(path);path.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_loaded_sources(root)
    settings = {'sensitivity':validate_sensitivity(sensitivity_settings),
                'primary':validate_primary(primary_settings),'wheeze':validate_wheeze(wheeze_settings)}
    _require(isinstance(new_results, (list,tuple)) and len(new_results) == len(DEFAULT_TARGETS)
             and all(isinstance(v,dict) and isinstance(v.get('target'),str) for v in new_results)
             and {v['target'] for v in new_results} == set(DEFAULT_TARGETS),
             'require exactly ten distinct new-fit completions.')
    supplied = {v['target']:v for v in new_results}
    selection = {'sensitivity':list(DEFAULT_TARGETS),'primary':['ratio_pre','ratio_post'],'wheeze':list(WHEEZE_TARGETS)}
    bindings, input_files, entries = [], {}, {family:{} for family in selection}
    for family,targets in selection.items():
        for target in targets:
            verified = _verified_fit(family,root,prep,directories,settings[family],target,
                                     supplied[target] if family=='sensitivity' else None)
            entries[family][target] = verified['entry']
            bindings.append(verified['binding']);input_files.update(verified['input_files'])
    result = build_tables(entries['sensitivity'], entries['primary'], entries['wheeze'])
    # The pure selector is called again so mutable/partial return objects cannot
    # bypass the declared table inventory before publication or cached reuse.
    check_tables(result, build_tables(entries['sensitivity'],entries['primary'],entries['wheeze']))
    _require(result.metadata['attenuation_precision_passed'] is True,
             'Table S1 attenuation numerical precision failed its prespecified MCSE bound; '
             'retain the fits and follow the prespecified common draw-extension protocol.')
    identity = {'schema_version':1, 'selection':selection, 'fits':bindings,
                'source':source_hashes(root),
                'environment_lock_sha256':file_hash(root/'requirements-models.lock.txt'),
                'runtime':model_runtime_identity()}
    fingerprint = digest(identity)
    summary = {'fingerprint':fingerprint, 'accepted':True, 'status':'accepted',
               'rows':dict(ROWS), 'new_fit_count':10, 'reused_fit_count':4,
               'source_fits_verified':14, 'reused_primary_models':2, 'reused_wheeze_models':2,
               'scientific_reference_comparison':'requires separate independent validation'}
    metadata = {'table_metadata':result.metadata, 'fit_bindings':bindings}
    destination = output/fingerprint

    def unchanged():
        assert_loaded_sources(root)
        _require(file_hash(root/'requirements-models.lock.txt') == identity['environment_lock_sha256']
                 and _equal(model_runtime_identity(), identity['runtime']), 'model runtime or environment lock changed.')
        for path,value in input_files.items():
            _no_symlinks(path)
            _require(path.is_file() and file_hash(path) == value, 'a verified upstream artifact changed during publication.')
        for binding in bindings:
            api = _family_api(binding['family'])
            descriptor = api.contract_descriptor(prep,binding['target'])
            fp, _ = api.sampling_identity(root,descriptor,binding['target'],settings[binding['family']])
            vf, _ = api.validation_identity(root,fp,api.CRITERIA)
            _require(descriptor['sha256'] == binding['contract_sha256']
                     and fp == binding['fit_fingerprint'] and vf == binding['validation_fingerprint'],
                     'prepared inputs or upstream fit identities changed during publication.')
        for path in (work, output, destination):
            _no_symlinks(path)

    with preparation_locks(work,output):
        reused = destination.exists() or destination.is_symlink()
        _require(not reused or resume, 'identical table results exist; use --resume to verify them.')
        if reused:
            verify_result(destination,fingerprint,identity,result,summary,metadata)
            unchanged()
            return {**summary,'resumed':True}
        stage = Path(tempfile.mkdtemp(prefix='.staging-',dir=output))
        try:
            for name in TABLES:
                (stage/(name+'.csv')).write_bytes(_csv_bytes(getattr(result,name)))
            write_json(stage/'metadata.json',metadata);write_json(stage/'summary.json',summary)
            write_json(stage/'receipt.json',{'schema_version':1,'status':'accepted','identity':identity,
                'fingerprint':fingerprint,'summary':summary,'completed_at':datetime.now(timezone.utc).isoformat(),
                'files':{name:file_hash(stage/name) for name in RESULT_FILES}})
            verify_result(stage,fingerprint,identity,result,summary,metadata)
            unchanged()
            _require(not destination.exists() and not destination.is_symlink(),
                     'publication destination appeared during execution; refusing to overwrite it.')
            os.replace(stage,destination)
        finally:
            if stage.exists():shutil.rmtree(stage)
    return {**summary,'resumed':False}
