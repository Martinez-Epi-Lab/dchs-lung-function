"""Verify selected primary fits and publish six derived shared-factor rows."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile

import pandas as pd

from . import primary_execution as primary
from .analysis.shared_factor import DEFAULT_TARGETS, summarize_shared_factor
from .execution import (file_hash, load_artifacts, preparation_locks, write_json)
from .pipeline import ValidationError, read_json as _read_json
from .model_runtime import model_runtime_identity, configure_runtime

SOURCES = ('shared_factor_execution.py', 'analysis/shared_factor.py', 'primary_execution.py',
           'analysis/primary_results.py', 'diagnostics.py', 'models/spirometry.py',
           'sampling.py', 'model_runtime.py')
RESULT_FILES = ('estimates.csv', 'metadata.json', 'summary.json')
_LOADED_HASH = file_hash(Path(__file__))


def read_json(path, label):
    result = _read_json(path, label)
    if not isinstance(result, dict):
        raise ValidationError(label+' must be an object.')
    return result


def _verified_primary(root, prep, work_root, output_root, settings, target):
    """Read and verify an existing complete M04 fit without a sampling path."""
    descriptor = primary.contract_descriptor(prep, target)
    fit_fp, fit_identity = primary.sampling_identity(root, descriptor, target, settings)
    validation_fp, validation_identity = primary.validation_identity(root, fit_fp, primary.CRITERIA)
    raw = Path(work_root)/'primary_fits'/target/fit_fp
    validated = Path(output_root)/'primary_models'/target/validation_fp
    if not (raw/'receipt.json').is_file() or not (validated/'receipt.json').is_file():
        raise ValidationError('Shared-factor summaries require an existing complete accepted primary fit.')
    raw_receipt = read_json(raw/'receipt.json', 'Primary raw receipt')
    validation_receipt = read_json(validated/'receipt.json', 'Primary validation receipt')
    if (raw_receipt.get('status') != 'sampled' or raw_receipt.get('identity') != fit_identity
            or raw_receipt.get('fingerprint') != fit_fp
            or validation_receipt.get('identity') != validation_identity
            or validation_receipt.get('fingerprint') != validation_fp
            or validation_receipt.get('raw_receipt_sha256') != file_hash(raw/'receipt.json')):
        raise ValidationError('Shared-factor primary fit identities are inconsistent.')
    primary.verify_files(raw, raw_receipt.get('files'), ('posterior.npz', 'sample_stats.npz', 'sampling.json'))
    primary.verify_files(validated, validation_receipt.get('files'),
        ('diagnostics.json', 'summary.json', 'estimates.csv', 'shared_factor.csv', 'sample_counts.csv', 'result_metadata.json'))
    contract = load_artifacts(prep, ('M03', 'pft', 'samples', 'primary/'+target))
    configure_runtime(Path(work_root)/'shared_factor'/'compiler_cache')
    posterior = primary.read_array_archive(raw/'posterior.npz')
    stats = primary.read_array_archive(raw/'sample_stats.npz')
    controls = fit_identity['sampling']
    primary.validate_posterior_inventory(posterior,
        primary.expected_posterior_shapes(contract, target, controls['chains'], controls['draws']))
    primary.validate_sampling_metadata(read_json(raw/'sampling.json', 'Primary sampling metadata'),
                                       fit_identity, posterior, stats)
    from .diagnostics import evaluate_diagnostics
    diagnostics = evaluate_diagnostics(posterior,
        {n:stats[n] for n in ('diverging', 'energy', 'reached_max_treedepth') if n in stats}, primary.CRITERIA)
    summary = read_json(validated/'summary.json', 'Primary summary')
    if (not diagnostics['passed'] or read_json(validated/'diagnostics.json', 'Primary diagnostics') != diagnostics
            or validation_receipt.get('summary') != summary or summary.get('diagnostics_passed') is not True
            or summary.get('status') != 'diagnostics_passed' or summary.get('target') != target
            or summary.get('fit_fingerprint') != fit_fp or summary.get('validation_fingerprint') != validation_fp):
        raise ValidationError('Shared-factor source primary fit does not pass complete posterior diagnostics.')
    input_files = {}
    for directory, receipt in ((raw, raw_receipt), (validated, validation_receipt)):
        input_files.update({directory/name:digest for name,digest in receipt['files'].items()})
        input_files[directory/'receipt.json'] = file_hash(directory/'receipt.json')
    binding = {'target':target, 'fit_fingerprint':fit_fp, 'validation_fingerprint':validation_fp,
               'raw_receipt_sha256':file_hash(raw/'receipt.json'),
               'validation_receipt_sha256':file_hash(validated/'receipt.json'),
               'contract_sha256':descriptor['sha256']}
    return {'posterior':posterior, 'contract':contract, 'binding':binding, 'input_files':input_files}


def summarize_selected_shared_factors(root, preparation, work_root, output_root, settings, *, resume=False):
    """Derive Figure S5 inputs only from complete diagnostically accepted M04 fits."""
    root, prep, work, output = primary.safe_roots(root, preparation,
        Path(work_root)/'shared_factor', Path(output_root)/'shared_factor')
    rows, metadata, bindings, input_files = [], [], [], {}
    for target in DEFAULT_TARGETS:
        verified = _verified_primary(root, prep, work_root, output_root, settings, target)
        result = summarize_shared_factor(verified['posterior'], verified['contract'], target)
        if (len(result.estimates) != 1 or list(result.estimates.target) != [target]
                or result.metadata.get('target') != target or result.metadata.get('new_fit_required') is not False):
            raise ValidationError('Shared-factor producer returned an invalid selected row inventory.')
        rows.append(result.estimates); metadata.append(result.metadata)
        bindings.append(verified['binding']); input_files.update(verified['input_files'])
    table = pd.concat(rows, ignore_index=True)
    if list(table.target) != list(DEFAULT_TARGETS) or len(table) != 6:
        raise ValidationError('Shared-factor output must contain the six selected primary targets once each.')
    source = primary.source_hashes(root, SOURCES)
    identity = {'schema_version':1, 'selection':list(DEFAULT_TARGETS), 'primary_fits':bindings,
                'source':source, 'environment_lock_sha256':file_hash(root/'requirements-models.lock.txt'),
                'runtime':model_runtime_identity()}
    fingerprint = primary.digest(identity)
    summary = {'through':'M07_shared_factor', 'fingerprint':fingerprint, 'result_rows':6,
               'source_primary_models':6, 'new_fits':0, 'accepted':True, 'status':'accepted',
               'scientific_reference_comparison':'requires separate independent validation'}
    meta = {'models':metadata, 'primary_fit_bindings':bindings}
    destination = output/fingerprint

    def unchanged():
        if (file_hash(Path(__file__)) != _LOADED_HASH or primary.source_hashes(root, SOURCES) != source
                or file_hash(root/'requirements-models.lock.txt') != identity['environment_lock_sha256']
                or model_runtime_identity() != identity['runtime']
                or any(not p.is_file() or p.is_symlink() or file_hash(p) != digest for p,digest in input_files.items())
                or any(primary.contract_descriptor(prep, b['target'])['sha256'] != b['contract_sha256'] for b in bindings)):
            raise ValidationError('Shared-factor source, inputs or runtime changed during execution.')

    def verify(directory):
        receipt = read_json(directory/'receipt.json', 'Shared-factor receipt')
        if (receipt.get('identity') != identity or receipt.get('fingerprint') != fingerprint
                or receipt.get('status') != 'accepted' or set(receipt.get('files', {})) != set(RESULT_FILES)):
            raise ValidationError('Shared-factor completion identity or inventory is invalid.')
        primary.verify_files(directory, receipt['files'], RESULT_FILES)
        if (read_json(directory/'summary.json', 'Shared-factor summary') != summary
                or receipt.get('summary') != summary
                or read_json(directory/'metadata.json', 'Shared-factor metadata') != meta):
            raise ValidationError('Shared-factor completion contradicts its verified source fits.')
        try:
            observed = pd.read_csv(directory/'estimates.csv', float_precision='round_trip')
            pd.testing.assert_frame_equal(observed, table, check_dtype=False,
                                          check_exact=False, atol=1e-12, rtol=1e-12)
        except (AssertionError, ValueError, TypeError):
            raise ValidationError('Shared-factor estimates disagree with verified primary posterior draws.') from None

    with preparation_locks(work, output):
        if destination.exists():
            if not resume:
                raise ValidationError('An identical shared-factor result exists; use --resume to verify it.')
            verify(destination); unchanged()
            return {**summary, 'resumed':True}
        staged = Path(tempfile.mkdtemp(prefix='.staging-', dir=output))
        try:
            table.to_csv(staged/'estimates.csv', index=False, float_format='%.17g')
            write_json(staged/'metadata.json', meta); write_json(staged/'summary.json', summary)
            write_json(staged/'receipt.json', {'schema_version':1, 'status':'accepted',
                'identity':identity, 'fingerprint':fingerprint, 'summary':summary,
                'completed_at':datetime.now(timezone.utc).isoformat(),
                'files':{n:file_hash(staged/n) for n in RESULT_FILES}})
            verify(staged); unchanged(); os.replace(staged, destination)
        finally:
            if staged.exists():
                shutil.rmtree(staged)
    return {**summary, 'resumed':False}
