"""Resolve reporting inputs from verified contracts and artifact inventories.

The cumulative M13 runner first re-executes M01–M12 validation. This layer then
independently derives every source identity and checks exact artifact inventories
before reading selected aggregate CSVs. It exposes no participant records.
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from ..execution import canonical_bytes, file_hash
from ..pipeline import ValidationError, read_json
from .results import ResultRegistry


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValidationError('Manuscript reporting '+message)


def no_symlinks(path):
    p = Path(path).absolute()
    require(not any(x.is_symlink() for x in (p,*p.parents)), 'paths must not contain symbolic links.')


def verify_files(directory, files, expected):
    no_symlinks(directory)
    directory = Path(directory)
    require(isinstance(files,dict) and set(files)==set(expected)
            and directory.is_dir() and {p.name for p in directory.iterdir()}==set(expected)|{'receipt.json'},
            'artifact inventory differs from its declared exact schema.')
    for name in (*expected,'receipt.json'):
        p = directory/name
        require(Path(name).name==name and p.is_file() and not p.is_symlink(),
                'requires regular artifact files.')
        if name in files:
            require(file_hash(p)==files[name], 'an upstream artifact changed.')


def verify_receipt(directory, identity, files, *, status='accepted'):
    directory = Path(directory)
    no_symlinks(directory)
    require((directory/'receipt.json').is_file() and not (directory/'receipt.json').is_symlink(),
            'requires an existing complete upstream publication.')
    receipt = read_json(directory/'receipt.json','Aggregate source receipt')
    common = {'schema_version','identity','fingerprint','files'}
    schemas = ([common|{'status','preparation_fingerprint','completed_at'}] if status=='sampled' else
               [common|{'summary','raw_receipt_sha256'}, common|{'summary'}] if status is None else
               [common|{'summary','status','completed_at'},
                common|{'summary','status','completed_at','preparation_fingerprint'}])
    require(isinstance(receipt,dict) and type(receipt.get('schema_version')) is int
            and receipt['schema_version']==1 and canonical_bytes(receipt.get('identity'))==canonical_bytes(identity)
            and receipt.get('fingerprint')==digest(identity) and set(receipt) in schemas,
            'source receipt identity or schema differs from current contracts.')
    if status is not None:
        require(receipt.get('status')==status, 'source receipt is unaccepted.')
    else:
        require('status' not in receipt, 'validation receipt has an unexpected status field.')
    verify_files(directory,receipt.get('files'),files)
    if 'summary.json' in files:
        summary = read_json(directory/'summary.json','Aggregate source summary')
        require(isinstance(summary,dict) and canonical_bytes(summary)==canonical_bytes(receipt.get('summary')),
                'source completion differs from its receipt.')
        predicates = [summary[k] is True for k in ('accepted','diagnostics_passed') if k in summary]
        joint = 'raw_receipt_sha256' in receipt
        require(predicates and all(predicates)
                and summary.get('diagnostics_passed' if joint else 'accepted') is True
                and summary.get('status') == ('diagnostics_passed' if joint else 'accepted'),
                'source analysis did not pass or has contradictory completion fields.')
    return receipt


def collect_sources(root, prep, directories, settings):
    """Return the keyed registry, public-safe provenance, and private hash ledger."""
    from .. import (primary_execution as primary, mechanics_execution as mechanics,
        supporting_execution as supporting, wheeze_execution as wheeze,
        sensitivity_execution as sensitivity, binary_execution as binary,
        trajectory_execution as trajectory, supporting_repeated_execution as repeated,
        pretb_execution as pretb, feno_execution as feno, shared_factor_execution as shared,
        sensitivity_tables_execution as sensitivity_tables)
    from ..model_runtime import model_runtime_identity
    from .facts import derive_facts

    root, prep = Path(root),Path(prep)
    for p in (root,prep,*directories.values()):no_symlinks(p)
    registry, inputs, bindings = ResultRegistry(),{},[]

    def remember(directory, receipt):
        for name in (*receipt['files'],'receipt.json'):
            p = directory/name
            inputs[p] = file_hash(p)

    def add_files(family, directory, receipt, tables, extra=None):
        remember(directory,receipt)
        provenance = {'fingerprint':receipt['fingerprint'],
                      'receipt_sha256':file_hash(directory/'receipt.json'), **(extra or {})}
        bindings.append({'family':family,**provenance})
        for name in tables:
            source = family+'/'+name
            frame = pd.read_csv(directory/(name+'.csv'),float_precision='round_trip')
            registry.add(source,frame,{**provenance,'artifact_sha256':file_hash(directory/(name+'.csv'))})

    # M02 aggregates are covered by the freshly verified M03 preparation receipt.
    pr = read_json(prep/'receipt.json','Preparation receipt')
    require(pr.get('status')=='prepared' and pr.get('through')=='M03'
            and pr.get('fingerprint')==digest(pr.get('identity'))
            and file_hash(prep/'artifacts.json')==pr.get('artifact_manifest_sha256'),
            'requires a verified M03 preparation.')
    from ..execution import verify_artifacts
    manifest = verify_artifacts(prep)
    inputs.update({prep/n:h for n,h in manifest['files'].items()})
    inputs[prep/'artifacts.json'] = file_hash(prep/'artifacts.json')
    inputs[prep/'receipt.json'] = file_hash(prep/'receipt.json')
    aggregate_prep = directories['outputs']/'preparation_runs'/pr['fingerprint']
    expected_prep = ('cohort_ledger.csv','flow.csv','modality_schedule.csv','summary.json',
                     'table1.csv','table1_missingness.csv','table1_statistics.csv')
    no_symlinks(aggregate_prep)
    require(set(pr.get('aggregate_files',{}))==set(expected_prep)
            and {p.name for p in aggregate_prep.iterdir()}==set(expected_prep),
            'preparation aggregate inventory differs.')
    for name,value in pr['aggregate_files'].items():
        p = aggregate_prep/name
        require(not p.is_symlink() and p.is_file() and file_hash(p)==value,
                'preparation aggregate content changed.')
        inputs[p] = value
    for name in ('flow','table1_statistics','cohort_ledger','modality_schedule'):
        registry.add('preparation/'+name,pd.read_csv(aggregate_prep/(name+'.csv'),float_precision='round_trip'),
                     {'preparation_fingerprint':pr['fingerprint'],'artifact_sha256':inputs[aggregate_prep/(name+'.csv')]})
    # Public aggregate context is derived from the already verified preparation
    # identity, so render-only consumers never need the private configuration.
    event_end = pr.get('identity',{}).get('settings',{}).get('event_end')
    try:
        canonical_end = date.fromisoformat(event_end).isoformat()
    except (ValueError, TypeError) as exc:
        raise ValidationError('Manuscript reporting requires a valid configured event endpoint.') from exc
    require(event_end == canonical_end, 'configured event endpoint must be a canonical ISO date.')
    registry.add('preparation/reporting_context',pd.DataFrame([{'field':'event_end','value':event_end}]),
                 {'preparation_fingerprint':pr['fingerprint'],'receipt_sha256':inputs[prep/'receipt.json'],
                  'identity_setting':'event_end'})
    facts = derive_facts(prep)
    for name,frame in facts.items():
        registry.add('facts/'+name.removeprefix('facts/'),frame,
                     {'preparation_fingerprint':pr['fingerprint'],'derivation':'reporting/facts.py'})

    joint = {'primary':(primary,'primary_fits','primary_models',('estimates','shared_factor','sample_counts')),
             'mechanics':(mechanics,'mechanics_fits','mechanics_models',('estimates','hazard','sample_counts')),
             'supporting':(supporting,'supporting_fits','supporting_joint',('estimates','hazard','sample_counts')),
             'wheeze':(wheeze,'wheeze_fits','wheeze_models',wheeze.TABLES),
             'sensitivity_model':(sensitivity,'sensitivity_fits','sensitivity_models',sensitivity.TABLES)}
    joint_bindings = {}
    for family,(api,work_name,output_name,tables) in joint.items():
        joint_bindings[family] = {}
        for target in api.TARGETS:
            descriptor = api.contract_descriptor(prep,target)
            fp, identity = api.sampling_identity(root,descriptor,target,settings[family])
            vf, vi = api.validation_identity(root,fp,api.CRITERIA)
            raw = directories['work']/work_name/target/fp
            rr = verify_receipt(raw,identity,('posterior.npz','sample_stats.npz','sampling.json'),status='sampled')
            remember(raw,rr)
            aggregate = directories['outputs']/output_name/target/vf
            expected = api.RESULT_FILES
            vr = verify_receipt(aggregate,vi,expected,status=None)
            require(vr.get('raw_receipt_sha256')==file_hash(raw/'receipt.json'),
                    'validated aggregates belong to another raw fit.')
            diagnostics = read_json(aggregate/'diagnostics.json','Model diagnostics')
            require(diagnostics.get('passed') is True, 'source diagnostic gate failed.')
            if family!='wheeze':
                from ..covariance_ppc import protocol
                evidence=read_json(aggregate/'covariance_ppc.json','Covariance adequacy evidence')
                require(vr['summary'].get('covariance_ppc_passed') is True
                        and evidence.get('accepted') is True and evidence.get('joint_pass') is True
                        and evidence.get('status')=='passed' and evidence.get('target')==target
                        and canonical_bytes(evidence.get('protocol'))==canonical_bytes(protocol()),
                        'source covariance adequacy gate failed.')
            binding = {'target':target,'contract_sha256':descriptor['sha256'],
                       'fit_fingerprint':fp,'validation_fingerprint':vf,
                       'raw_receipt_sha256':file_hash(raw/'receipt.json'),
                       'validation_receipt_sha256':file_hash(aggregate/'receipt.json')}
            joint_bindings[family][target] = binding
            add_files(family+'/'+target,aggregate,vr,tables,{'fit_fingerprint':fp,
                'support':vr['summary']['counts']})

    deterministic = [
        ('binary',binary,'binary_models',binary.binary_contracts,binary.binary_identity,('estimates','pattern_counts')),
        ('trajectory',trajectory,'trajectory_models',trajectory.trajectory_contracts,trajectory.trajectory_identity,
         ('global_tests','age_contrasts','predictions')),
        ('supporting_repeated',repeated,'supporting_repeated',repeated.contract_descriptor,repeated.fit_identity,
         tuple(n for n in repeated.TABLES if n not in ('coefficients','covariance'))),
        ('pretb',pretb,'pretb_cox',pretb.contract_descriptor,pretb.fit_identity,('estimates',)),
        ('feno',feno,'feno_models',feno.contract_descriptor,feno.fit_identity,('estimates','counts'))]
    for family,api,folder,describe,identify,tables in deterministic:
        fp, identity = identify(root,describe(prep))
        aggregate = directories['outputs']/folder/fp
        receipt = verify_receipt(aggregate,identity,api.RESULT_FILES,
                                 status='accepted' if family in {'binary','trajectory'} else None)
        add_files(family,aggregate,receipt,tables)

    from ..analysis.shared_factor import DEFAULT_TARGETS as shared_targets
    identity = {'schema_version':1,'selection':list(shared_targets),
                'primary_fits':[joint_bindings['primary'][t] for t in shared_targets],
                'source':primary.source_hashes(root,shared.SOURCES),
                'environment_lock_sha256':file_hash(root/'requirements-models.lock.txt'),
                'runtime':model_runtime_identity()}
    aggregate = directories['outputs']/'shared_factor'/digest(identity)
    receipt = verify_receipt(aggregate,identity,shared.RESULT_FILES)
    add_files('shared_factor',aggregate,receipt,('estimates',))

    from ..analysis.sensitivities import DEFAULT_TARGETS, WHEEZE_TARGETS
    selection = {'sensitivity':list(DEFAULT_TARGETS),'primary':['ratio_pre','ratio_post'],
                 'wheeze':list(WHEEZE_TARGETS)}
    selected_bindings = []
    for family,targets in selection.items():
        for target in targets:
            selected_bindings.append({'family':family,**joint_bindings['sensitivity_model' if family=='sensitivity' else family][target]})
    identity = {'schema_version':1,'selection':selection,'fits':selected_bindings,
                'source':sensitivity_tables.source_hashes(root),
                'environment_lock_sha256':file_hash(root/'requirements-models.lock.txt'),
                'runtime':model_runtime_identity()}
    aggregate = directories['outputs']/'sensitivity_tables'/digest(identity)
    receipt = verify_receipt(aggregate,identity,sensitivity_tables.RESULT_FILES)
    add_files('sensitivity',aggregate,receipt,sensitivity_tables.TABLES)
    return registry,bindings,inputs
