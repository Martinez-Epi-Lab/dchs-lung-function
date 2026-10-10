"""Execute and verify the selected binary spirometry models outside source."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, resolve_paths, runtime_identity,
                        run_preparation, verify_artifacts, write_json, write_pointer)
from .pipeline import ValidationError, read_json


SOURCES = ('src/dchs_lung_function/models/binary_spirometry.py',
           'src/dchs_lung_function/binary_execution.py',
           'src/dchs_lung_function/preparation.py',
           'src/dchs_lung_function/execution.py',
           'src/dchs_lung_function/pipeline.py', 'run.py')
RESULT_FILES = ('estimates.csv', 'coefficients.csv', 'covariance.csv',
                'pattern_counts.csv', 'diagnostics.json', 'summary.json')
_LOADED_HASH = file_hash(Path(__file__))


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def binary_contracts(preparation_directory):
    """Verify M03 and fingerprint just the selected binary contract content."""
    from .models.binary_spirometry import DEFAULT_FIT_SELECTION
    directory = Path(preparation_directory).resolve()
    receipt = read_json(directory/'receipt.json', 'Preparation receipt')
    if (receipt.get('through') != 'M03' or receipt.get('status') != 'prepared'
            or _digest(receipt.get('identity')) != receipt.get('fingerprint')
            or file_hash(directory/'artifacts.json') != receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Binary fitting requires a complete verified M03 preparation.')
    manifest = verify_artifacts(directory)
    descriptors = {}
    for stage, outcome in dict.fromkeys((s, o) for s, o, _ in DEFAULT_FIT_SELECTION):
        key = f'binary/{stage}/{outcome}'
        node = artifact_subtree(manifest, ('M03', 'pft', 'samples', key))
        files = set()
        def visit(value):
            if isinstance(value, dict):
                if 'file' in value:
                    files.add(value['file'])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(node)
        if not files.issubset(manifest['files']):
            raise ValidationError('Binary contract refers to an unverified preparation artifact.')
        descriptors[key] = _digest({'tree':node, 'files':{f:manifest['files'][f] for f in sorted(files)}})
    return {'contracts':descriptors, 'preparation_fingerprint':receipt['fingerprint']}


def binary_identity(root, descriptor):
    from .models.binary_spirometry import DEFAULT_FIT_SELECTION
    root = Path(root)
    identity = {'schema_version':1, 'contracts':descriptor['contracts'],
                'selection':[list(v) for v in DEFAULT_FIT_SELECTION],
                'source':{name:file_hash(root/name) for name in SOURCES},
                'environment_lock_sha256':file_hash(root/'requirements-preparation.lock.txt'),
                'runtime':runtime_identity()}
    return _digest(identity), identity


def _roots(root, preparation, work, output):
    root, preparation, work, output = [Path(p).resolve() for p in (root, preparation, work, output)]
    def overlap(a, b):
        return a == b or a in b.parents or b in a.parents
    if overlap(work, output) or any(overlap(p, protected)
                                  for p in (work, output) for protected in (root, preparation)):
        raise ValidationError('Binary work/output roots must be separate from source and preparation.')
    for p in (work, output):
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root, preparation, work, output


def _verify_result(directory, fingerprint, identity):
    receipt = read_json(directory/'receipt.json', 'Binary completion receipt')
    if (not isinstance(receipt, dict)
            or receipt.get('schema_version') != 1 or receipt.get('fingerprint') != fingerprint
            or receipt.get('identity') != identity):
        raise ValidationError('Binary completion identity is inconsistent.')
    files = receipt.get('files')
    if not isinstance(files, dict) or set(files) != set(RESULT_FILES):
        raise ValidationError('Binary completion receipt has an incomplete output inventory.')
    for name, expected in files.items():
        path = directory/name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValidationError('A saved binary result is missing or changed.')
    summary = read_json(directory/'summary.json', 'Binary summary')
    diagnostics = read_json(directory/'diagnostics.json', 'Binary diagnostics')
    if not isinstance(summary, dict) or not isinstance(diagnostics, dict):
        raise ValidationError('Binary summary and diagnostics must be objects.')
    from .models.binary_spirometry import DEFAULT_FIT_SELECTION, fit_spec
    expected_models = [f'{s}/{o}/{d}' for s, o, d in DEFAULT_FIT_SELECTION]
    models = diagnostics.get('models', [])
    if (not isinstance(models, list)
            or any(not isinstance(m, dict) or not isinstance(m.get('metadata'), dict) for m in models)
            or [m.get('model_id') for m in models] != expected_models
            or any(type(m.get('metadata', {}).get('accepted')) is not bool for m in models)):
        raise ValidationError('Binary diagnostics lack the complete selected model inventory.')
    try:
        estimates = pd.read_csv(directory/'estimates.csv')
        coefficients = pd.read_csv(directory/'coefficients.csv')
        covariance = pd.read_csv(directory/'covariance.csv')
        patterns = pd.read_csv(directory/'pattern_counts.csv')
        expected_rows, expected_coefficients, expected_covariance = [], [], []
        for model, (stage, outcome, design) in zip(models, DEFAULT_FIT_SELECTION):
            meta = model['metadata']
            spec = fit_spec(stage, outcome, design)
            model_id = model['model_id']
            if any(model.get(k) != v or meta.get('design_key' if k == 'design' else k) != v
                   for k, v in [('stage', stage), ('outcome', outcome), ('design', design)]):
                raise ValidationError('Binary diagnostic labels disagree with the selected specification.')
            failure_reasons = meta.get('fit_failure_reasons')
            if not isinstance(failure_reasons, list) or meta['accepted'] != (not failure_reasons):
                raise ValidationError('Binary acceptance contradicts its recorded fitting failures.')
            if (meta['accepted'] and meta.get('expected_nonestimable_reason') is None
                    and (meta.get('converged') is not True or meta.get('fit_status') != 'converged')):
                raise ValidationError('An accepted binary fit lacks convergence or an explicit structural limitation.')
            subset = estimates.loc[estimates.model_id.eq(model_id)]
            if (list(subset.term) != spec['reported_terms']
                    or not subset.estimable.map(lambda v: isinstance(v, (bool, np.bool_))).all()):
                raise ValidationError('Binary results lack the complete selected contrast inventory.')
            n_estimable = int(subset.estimable.sum())
            status = ('fit_failed' if not meta['accepted'] else 'success' if n_estimable == len(subset)
                      else 'non_estimable' if n_estimable == 0 else 'partial')
            if (meta.get('status') != status or meta.get('estimable_contrasts') != n_estimable
                    or meta.get('non_estimable_contrasts') != len(subset)-n_estimable):
                raise ValidationError('Binary fit status contradicts its reported contrasts.')
            for field in ('n_observations', 'n_children', 'events'):
                if not subset[field].eq(meta[field]).all():
                    raise ValidationError('Binary result counts disagree with model metadata.')
            for _, row in subset.iterrows():
                if (row['stage'], row['outcome'], row['design']) != (stage, outcome, design):
                    raise ValidationError('Binary result labels disagree with model metadata.')
                if row.estimable:
                    values = row[['beta', 'se', 'rr', 'lower', 'upper', 'p_value']].to_numpy(float)
                    if (row.status != 'estimable' or not np.isfinite(values).all()
                            or not pd.isna(row.reason_code) or row.se <= 0
                            or not 0 < row.lower <= row.rr <= row.upper or not 0 <= row.p_value <= 1):
                        raise ValidationError('An estimable binary result has invalid inference fields.')
                elif (row.status != 'non_estimable' or pd.isna(row.reason_code)
                      or not row[['beta', 'se', 'rr', 'lower', 'upper', 'p_value']].isna().all()):
                    raise ValidationError('A non-estimable binary result lacks explicit null inference fields.')
            terms = meta.get('terms')
            if not isinstance(terms, list) or not terms or len(set(terms)) != len(terms):
                raise ValidationError('Binary coefficient terms are missing or duplicated.')
            coefficient_values = coefficients.loc[coefficients.model_id.eq(model_id), 'coefficient'].to_numpy(float)
            covariance_values = covariance.loc[covariance.model_id.eq(model_id), 'covariance'].to_numpy(float)
            finite_coefficients = bool(np.isfinite(coefficient_values).all())
            finite_covariance = bool(np.isfinite(covariance_values).all())
            if (type(meta.get('coefficients_finite')) is not bool
                    or type(meta.get('covariance_finite')) is not bool
                    or meta['coefficients_finite'] != finite_coefficients
                    or meta['covariance_finite'] != finite_covariance
                    or (meta['accepted'] and meta.get('expected_nonestimable_reason') is None
                        and not (finite_coefficients and finite_covariance))):
                raise ValidationError('Binary coefficient or covariance finiteness contradicts its fit metadata.')
            expected_rows.extend((model_id, term) for term in spec['reported_terms'])
            expected_coefficients.extend((model_id, term) for term in terms)
            expected_covariance.extend((model_id, a, b) for a in terms for b in terms)
        if (list(zip(estimates.model_id, estimates.term)) != expected_rows
                or list(zip(coefficients.model_id, coefficients.term)) != expected_coefficients
                or list(zip(covariance.model_id, covariance.row_term, covariance.column_term)) != expected_covariance):
            raise ValidationError('Binary aggregate files have missing, extra or duplicate model terms.')
        expected_patterns = {(stage, state, pattern) for stage in ('pre', 'post')
            for state in ('no_recorded_tb', 'future_tb', 'post_infancy', 'post_age1_before5', 'post_at_after5')
            for pattern in ('Normal', 'Obstructive', 'Possible restriction', 'Mixed')}
        observed_patterns = list(zip(patterns.stage, patterns.exposure_state, patterns.pattern))
        if len(observed_patterns) != len(expected_patterns) or set(observed_patterns) != expected_patterns:
            raise ValidationError('The binary pattern ledger must retain every category and exposure state.')
        for (_, _), group in patterns.groupby(['stage', 'exposure_state']):
            values = group[['observations', 'children', 'event_observations', 'event_children']].to_numpy(float)
            if (not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any()
                    or group.observations.nunique() != 1 or group.children.nunique() != 1
                    or group.event_observations.sum() != group.observations.iloc[0]
                    or (group.event_children > group.children).any()
                    or (group.event_children > group.event_observations).any()
                    or (group.children > group.observations).any()):
                raise ValidationError('Binary pattern counts are internally inconsistent.')
            denominator = int(group.observations.iloc[0])
            prevalence = group.prevalence.to_numpy(float)
            if ((denominator == 0 and not np.isnan(prevalence).all())
                    or (denominator > 0 and (not np.isfinite(prevalence).all()
                        or not np.allclose(prevalence, group.event_observations/denominator,
                                           rtol=1e-12, atol=1e-12)))):
                raise ValidationError('Binary pattern prevalence does not match its exact counts.')
        if (summary.get('estimate_rows') != len(estimates)
                or summary.get('estimable_rows') != int(estimates.estimable.sum())
                or summary.get('non_estimable_rows') != int((~estimates.estimable).sum())):
            raise ValidationError('Binary completion counts disagree with the contrast table.')
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError('Binary aggregate result structure is invalid.') from None
    accepted = all(m['metadata']['accepted'] for m in models)
    if (receipt.get('summary') != summary or summary.get('fingerprint') != fingerprint
            or summary.get('models') != len(expected_models)
            or type(summary.get('accepted')) is not bool or summary['accepted'] != accepted
            or summary.get('status') != ('accepted' if accepted else 'fit_failed')
            or receipt.get('status') != summary['status']):
        raise ValidationError('Binary receipt, diagnostics and completion summary disagree.')
    return summary


def fit_prepared_binary(root, preparation_directory, work, output, *, resume=False):
    """Publish aggregate fits transactionally; expected boundary rows stay explicit."""
    from .models.binary_spirometry import DEFAULT_FIT_SELECTION, fit_binary, pattern_counts
    root, prep, work, output = _roots(root, preparation_directory, work, output)
    descriptor = binary_contracts(prep)
    fingerprint, identity = binary_identity(root, descriptor)
    destination = output/fingerprint

    def unchanged():
        if (file_hash(Path(__file__)) != _LOADED_HASH
                or binary_identity(root, binary_contracts(prep)) != (fingerprint, identity)):
            raise ValidationError('Binary model, preparation or runtime changed during execution.')

    with preparation_locks(work, output):
        if destination.exists():
            if not resume:
                raise ValidationError('An identical binary run exists; use --resume to verify it.')
            summary = _verify_result(destination, fingerprint, identity)
            unchanged()
            return {**summary, 'resumed':True}
        estimates, coefficients, covariances, patterns, models = [], [], [], [], []
        contracts = {}
        for stage, outcome, design in DEFAULT_FIT_SELECTION:
            key = f'binary/{stage}/{outcome}'
            if key not in contracts:
                contracts[key] = load_artifacts(prep, ('M03', 'pft', 'samples', key))
            contract = contracts[key]
            if contract['audit'].get('stage') != stage or contract['audit'].get('outcome') != outcome:
                raise ValidationError('Binary contract does not match the requested outcome and stage.')
            result = fit_binary(contract, design)
            labels = {'model_id':f'{stage}/{outcome}/{design}', 'stage':stage,
                      'outcome':outcome, 'design':design}
            def labelled(frame):
                frame = frame.copy()
                for name, value in reversed(list(labels.items())):
                    if name in frame:
                        if not frame[name].eq(value).all():
                            raise ValidationError('Binary result labels do not match the selected fit.')
                        frame = frame.drop(columns=name)
                    frame.insert(0, name, value)
                return frame
            estimates.append(labelled(result.estimates))
            coefficients.append(labelled(pd.DataFrame({'term':result.coefficients.index,
                                                      'coefficient':result.coefficients.to_numpy()})))
            covariances.append(labelled(pd.DataFrame([
                {'row_term':row, 'column_term':col, 'covariance':result.covariance.loc[row, col]}
                for row in result.covariance.index for col in result.covariance.columns])))
            models.append({**labels, 'metadata':result.metadata})
        for stage in ('pre', 'post'):
            patterns.append(pattern_counts(contracts[f'binary/{stage}/obstructive_pattern']))
        combined = pd.concat(estimates, ignore_index=True)
        accepted = all(m['metadata']['accepted'] for m in models)
        summary = {'through':'M05', 'fingerprint':fingerprint, 'models':len(models),
                   'accepted':accepted, 'status':'accepted' if accepted else 'fit_failed',
                   'estimate_rows':len(combined),
                   'estimable_rows':int(combined.estimable.sum()),
                   'non_estimable_rows':int((~combined.estimable).sum()),
                   'scientific_reference_comparison':'requires separate independent validation'}
        staged = Path(tempfile.mkdtemp(prefix='.staging-', dir=output))
        try:
            for name, frame in [('estimates', combined),
                                ('coefficients', pd.concat(coefficients, ignore_index=True)),
                                ('covariance', pd.concat(covariances, ignore_index=True)),
                                ('pattern_counts', pd.concat(patterns, ignore_index=True))]:
                frame.to_csv(staged/(name+'.csv'), index=False, float_format='%.17g')
            write_json(staged/'diagnostics.json', {'models':models})
            write_json(staged/'summary.json', summary)
            receipt = {'schema_version':1, 'fingerprint':fingerprint, 'identity':identity,
                       'status':summary['status'], 'summary':summary,
                       'preparation_fingerprint':descriptor['preparation_fingerprint'],
                       'completed_at':datetime.now(timezone.utc).isoformat(),
                       'files':{name:file_hash(staged/name) for name in RESULT_FILES}}
            write_json(staged/'receipt.json', receipt)
            _verify_result(staged, fingerprint, identity)
            unchanged()
            os.replace(staged, destination)
        finally:
            if staged.exists():
                shutil.rmtree(staged)
        return {**summary, 'resumed':False}


def run_binary(root, config_file, analysis_file, sampling_file, *, jobs=1, resume=False):
    """Run cumulative M01-M05, requiring every primary model to pass first."""
    from .primary_execution import run_primary
    sampling_hash = file_hash(Path(sampling_file))
    primary = run_primary(root, config_file, analysis_file, sampling_file, jobs=jobs, resume=resume)
    if not primary['all_twelve_models'] or not primary['diagnostics_passed'] or primary.get('covariance_ppc_passed') is not True:
        raise ValidationError('Cumulative M05 requires diagnostic acceptance of all twelve primary fits.')
    configuration = read_json(config_file, 'Local configuration')
    _, directories = resolve_paths(configuration, config_file)
    prep = directories['work']/'preparation_runs'/primary['preparation_fingerprint']
    summary = fit_prepared_binary(root, prep, directories['work']/'binary_models',
                                  directories['outputs']/'binary_models', resume=resume)
    verified = run_preparation(root, config_file, analysis_file, 'M03', resume=True)
    if (verified['fingerprint'] != primary['preparation_fingerprint']
            or file_hash(Path(sampling_file)) != sampling_hash):
        raise ValidationError('Preparation dependencies changed during binary execution.')
    summary = {**summary, 'preparation_fingerprint':primary['preparation_fingerprint'],
               'primary_models_verified':12}
    if summary['accepted']:
        write_pointer(directories['work']/'binary_current.json', summary)
    return summary
