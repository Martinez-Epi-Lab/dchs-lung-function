"""Fit, publish and verify the selected pre-TB Cox models privately."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, resolve_paths, run_preparation,
                        runtime_identity, verify_artifacts, write_json, write_pointer)
from .pipeline import ValidationError, read_json
from .models import pretb_cox as model

SOURCES = ('models/pretb_cox.py', 'pretb_execution.py', 'preparation.py',
           'execution.py', 'pipeline.py')
TABLES = ('estimates', 'coefficients', 'covariance')
RESULT_FILES = tuple(name+'.csv' for name in TABLES) + ('diagnostics.json', 'summary.json')
_LOADED_SOURCES = {name:file_hash(Path(__file__).parent/name) for name in SOURCES}


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValidationError('Pre-TB Cox '+message)


def assert_loaded_sources(root):
    for base in (Path(root)/'src/dchs_lung_function', Path(__file__).resolve().parent):
        _require({n:file_hash(base/n) for n in SOURCES} == _LOADED_SOURCES,
                 'sources changed after import or differ from the supplied source tree; restart the process.')


def safe_roots(root, preparation, work, output):
    root, prep, work, output = [Path(p).resolve() for p in (root, preparation, work, output)]
    def overlap(a, b):
        return a == b or a in b.parents or b in a.parents
    _require(not overlap(work, output) and not any(overlap(p, q)
             for p in (work, output) for q in (root, prep)),
             'work/output roots must be separate from source and preparation.')
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    return root, prep, work, output


def contract_descriptor(preparation):
    directory = Path(preparation).resolve()
    receipt = read_json(directory/'receipt.json', 'Preparation receipt')
    _require(isinstance(receipt, dict) and receipt.get('through') == 'M03'
             and receipt.get('status') == 'prepared'
             and digest(receipt.get('identity')) == receipt.get('fingerprint')
             and file_hash(directory/'artifacts.json') == receipt.get('artifact_manifest_sha256'),
             'execution requires complete verified M03 preparation.')
    manifest = verify_artifacts(directory)
    descriptors = {}
    for target in model.DEFAULT_TARGETS:
        key = model.contract_key(target)
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
        _require(files.issubset(manifest['files']), 'contract refers to an unverified preparation artifact.')
        descriptors[key] = digest({'tree':node, 'files':{n:manifest['files'][n] for n in sorted(files)}})
    return {'contracts':descriptors, 'preparation_fingerprint':receipt['fingerprint']}


def fit_identity(root, descriptor):
    root = Path(root)
    identity = {'schema_version':1, 'contracts':descriptor['contracts'],
                'selection':list(model.DEFAULT_TARGETS),
                'source':{n:file_hash(root/'src/dchs_lung_function'/n) for n in SOURCES},
                'environment_lock_sha256':file_hash(root/'requirements-models.lock.txt'),
                'runtime':runtime_identity()}
    return digest(identity), identity


def _close(actual, expected):
    return np.allclose(np.asarray(actual, float), np.asarray(expected, float),
                       rtol=1e-8, atol=1e-10, equal_nan=False)


def _integer(value, minimum=0):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and value >= minimum and value == int(value)


def _risk_evidence(frame, source_scores):
    """Reconstruct both memberships without PHReg's reverse-time bookkeeping."""
    start, stop, event = frame.start.to_numpy(float), frame.stop.to_numpy(float), frame.event.to_numpy(bool)
    times = np.unique(stop[event])
    risk = (start[None, :] <= times[:, None]) & (times[:, None] <= stop[None, :])
    failures = event[None, :] & (times[:, None] == stop[None, :])
    def bit_hash(value):
        return hashlib.sha256(np.packbits(value, axis=None, bitorder='little').tobytes()).hexdigest()
    codes = pd.Categorical(frame.pid_child).codes
    rows = []
    for age, members, events in zip(times, risk, failures):
        entry = members & (start == age)
        children = len(np.unique(codes[members]))
        rows.append({'event_age_years':float(age), 'events':int(events.sum()),
            'risk_rows':int(members.sum()), 'risk_children':children,
            'entry_at_event_rows':int(entry.sum()),
            'entry_at_event_children':len(np.unique(codes[entry])),
            'additional_same_child_risk_rows':int(members.sum())-children})
    result = {'membership_shape':list(risk.shape), 'membership_sha256':bit_hash(risk),
              'event_membership_sha256':bit_hash(failures), 'event_times':rows,
              'intervals_outside_event_risk_sets':int((~risk.any(axis=0)).sum())}
    if source_scores:
        _require(len(times) > 0, 'source-score membership needs an observed event.')
        retained = (start <= times[-1]) & (stop >= times[0])
        extra = retained & ~risk.any(axis=0)
        support = risk | (extra[None, :] & (times[:, None] <= stop[None, :]))
        result.update(likelihood_membership_contributions=int(risk.sum()),
            source_score_membership_contributions=int(support.sum()),
            source_score_membership_sha256=bit_hash(support),
            source_score_membership_difference_cells=int(np.count_nonzero(risk != support)),
            source_score_support_intervals=int(support.any(axis=0).sum()),
            source_undefined_score_intervals=int((~support.any(axis=0)).sum()))
    return result


def check_result(result, contracts=None):
    """Reject incomplete, contradictory or non-aggregate producer output."""
    try:
        _check_result(result, contracts)
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, ArithmeticError):
        raise ValidationError('Pre-TB Cox result or prepared contract has malformed values.') from None


def _check_result(result, contracts=None):
    _require(isinstance(result, dict) and set(result) == {'tables', 'metadata'}, 'result schema differs.')
    tables, metadata = result['tables'], result['metadata']
    schemas = dict(zip(TABLES, [model.ESTIMATE_COLUMNS, model.COEFFICIENT_COLUMNS, model.COVARIANCE_COLUMNS]))
    _require(isinstance(tables, dict) and set(tables) == set(TABLES), 'table inventory differs.')
    for name, schema in schemas.items():
        _require(isinstance(tables[name], pd.DataFrame) and tuple(tables[name].columns) == tuple(schema),
                 'table columns differ from the declared aggregate schema.')
    _require(isinstance(metadata, dict) and set(metadata) == {'schema_version', 'accepted', 'status', 'selection', 'models', 'multiplicity'}
             and type(metadata['schema_version']) is int and metadata['schema_version'] == 1
             and type(metadata['accepted']) is bool
             and metadata['status'] == ('accepted' if metadata['accepted'] else 'fit_failed')
             and metadata['selection'] == list(model.DEFAULT_TARGETS), 'family metadata differs.')
    _require(isinstance(metadata['models'], list) and len(metadata['models']) == 17
             and all(isinstance(m, dict) for m in metadata['models'])
             and [m.get('target') for m in metadata['models']] == list(model.DEFAULT_TARGETS),
             'model inventory differs.')
    estimates, coefficients, covariance = [tables[n] for n in TABLES]
    terms = list(model.TERMS)
    _require(list(estimates.target) == list(model.DEFAULT_TARGETS), 'estimate inventory differs.')
    _require(list(zip(coefficients.target, coefficients.term)) == [(t, term) for t in model.DEFAULT_TARGETS for term in terms],
             'coefficient inventory differs.')
    _require(list(zip(covariance.target, covariance.covariance_type, covariance.row_term, covariance.column_term))
             == [(t, kind, a, b) for t in model.DEFAULT_TARGETS for kind in ('robust', 'model') for a in terms for b in terms],
             'covariance inventory differs.')
    _require(all(m['accepted'] is True for m in metadata['models']) == metadata['accepted'],
             'family acceptance contradicts individual models.')
    meta_keys = {'target', 'contract_key', 'terms', 'counts', 'specification', 'accepted', 'status',
                 'reason', 'warnings', 'optimizer', 'diagnostics', 'covariance_correction', 'covariance_method', 'risk_sets'}
    count_keys = {'n_children', 'n_intervals', 'n_subsequent_tb', 'person_years'}
    diagnostic_keys = {'log_partial_likelihood', 'score', 'max_absolute_score', 'information_condition_number',
        'model_covariance_min_eigenvalue', 'robust_covariance_min_eigenvalue', 'robust_covariance_max_asymmetry',
        'score_residual_group_sum', 'undefined_score_rows_zeroed'}
    optimizer_keys = {'method', 'iterations', 'converged', 'warnflag', 'maxiter', 'tol', 'ridge_factor'}
    risk_keys = {'membership_shape', 'membership_sha256', 'event_membership_sha256', 'event_times', 'intervals_outside_event_risk_sets'}
    score_keys = {'likelihood_membership_contributions', 'source_score_membership_contributions',
        'source_score_membership_sha256', 'source_score_membership_difference_cells',
        'source_score_support_intervals', 'source_undefined_score_intervals'}
    event_keys = {'event_age_years', 'events', 'risk_rows', 'risk_children', 'entry_at_event_rows',
                  'entry_at_event_children', 'additional_same_child_risk_rows'}
    for meta in metadata['models']:
        target = meta['target']; spec = model.fit_spec(target)
        row = estimates.loc[estimates.target.eq(target)].iloc[0]
        coef = coefficients.loc[coefficients.target.eq(target)]
        cov = covariance.loc[covariance.target.eq(target)]
        _require(set(meta) == meta_keys and meta['contract_key'] == model.contract_key(target)
                 and meta['terms'] == terms and meta['specification'] == spec
                 and meta['covariance_method'] == spec['covariance_method'], 'model metadata differs.')
        status = meta['status']; estimated = status == 'estimated'
        _require(status in {'estimated', 'non_estimable', 'fit_failed'} and type(meta['accepted']) is bool
                 and meta['accepted'] == (status != 'fit_failed') and isinstance(meta['reason'], str)
                 and ((estimated and meta['reason'] == '') or (not estimated and bool(meta['reason']))),
                 'model acceptance and reason contradict its status.')
        _require(isinstance(meta['warnings'], list) and all(isinstance(v, str) for v in meta['warnings'])
                 and row.warning == '; '.join(meta['warnings']), 'warnings differ from model metadata.')
        counts = meta['counts']
        _require(isinstance(counts, dict) and set(counts) == count_keys
                 and all(_integer(counts[k]) for k in count_keys-{'person_years'})
                 and counts['n_subsequent_tb'] <= counts['n_children'] <= counts['n_intervals']
                 and np.isfinite(counts['person_years']) and 0 <= counts['person_years'] <= 2*counts['n_intervals']+1e-9,
                 'sample counts are invalid.')
        n, g, events = counts['n_intervals'], counts['n_children'], counts['n_subsequent_tb']
        _require(all(_close(row[k], v) for k, v in counts.items()), 'table counts differ from metadata.')
        if status == 'non_estimable':
            _require(events < 3 and meta['reason'] == 'source_minimum_event_rule'
                     and meta['covariance_correction'] is None and meta['optimizer'] == {} and meta['diagnostics'] == {},
                     'source-policy non-estimable state is invalid.')
        else:
            _require(events >= 3, 'fit status contradicts the minimum-event source policy.')
        for table in (estimates.loc[estimates.target.eq(target)], coef, cov):
            _require(table.status.eq(status).all()
                     and all(isinstance(v, (bool, np.bool_)) and v == meta['accepted'] for v in table.accepted),
                     'table acceptance contradicts the model state.')
        _require(row.reason == meta['reason'] and coef.reason.eq(meta['reason']).all(), 'table reasons differ.')
        _require(all(row[k] == spec[k] for k in ('measure', 'measure_label', 'source_column', 'scale'))
                 and row.carry_forward_window_years == 2., 'outcome labels, scales or window differ.')
        risk = meta['risk_sets']
        has_source = meta['covariance_correction'] is not None
        _require(isinstance(risk, dict) and set(risk) == risk_keys | (score_keys if has_source else set())
                 and isinstance(risk['event_times'], list)
                 and all(isinstance(v, dict) and set(v) == event_keys for v in risk['event_times']),
                 'risk-support metadata has an undeclared field or missing evidence.')
        _require(risk['membership_shape'] == [len(risk['event_times']), n]
                 and sum(v['events'] for v in risk['event_times']) == events,
                 'event-risk support contradicts sample counts.')
        if contracts is not None:
            contract = contracts[model.contract_key(target)]
            checked = model.validate_contract(contract, target)
            _require(checked['counts'] == counts, 'counts differ from the verified prepared contract.')
            _require(risk == _risk_evidence(contract['frame'], has_source),
                     'likelihood or legacy score memberships differ from the prepared intervals.')
        _require(isinstance(meta['optimizer'], dict) and set(meta['optimizer']).issubset(optimizer_keys)
                 and isinstance(meta['diagnostics'], dict) and set(meta['diagnostics']).issubset(diagnostic_keys),
                 'fit diagnostics contain undeclared fields.')
        matrices = {kind:cov.loc[cov.covariance_type.eq(kind), 'value'].to_numpy(float).reshape(3, 3)
                    for kind in ('robust', 'model')}
        if estimated:
            _require(g > 1 and n > 3 and has_source and _close(meta['covariance_correction'], g/(g-1)*(n-1)/(n-3)),
                     'finite-sample covariance correction differs.')
            opt, diag = meta['optimizer'], meta['diagnostics']
            _require(set(opt) == optimizer_keys and opt['method'] == 'newton'
                     and opt['converged'] is True and opt['warnflag'] == 0
                     and _integer(opt['iterations'], 1) and opt['iterations'] <= 100
                     and opt['maxiter'] == 100 and opt['tol'] == 1e-8 and opt['ridge_factor'] == 1e-10,
                     'estimated fit lacks selected optimizer convergence evidence.')
            _require(set(diag) == diagnostic_keys and np.isfinite(diag['log_partial_likelihood'])
                     and np.asarray(diag['score']).shape == (3,) and np.isfinite(diag['score']).all()
                     and _close(diag['max_absolute_score'], np.max(np.abs(diag['score'])))
                     and diag['max_absolute_score'] <= 1e-5
                     and diag['undefined_score_rows_zeroed'] == risk['source_undefined_score_intervals']
                     and np.asarray(diag['score_residual_group_sum']).shape == (3,)
                     and np.isfinite(diag['score_residual_group_sum']).all(), 'estimated fit diagnostics are invalid.')
            for kind, matrix in matrices.items():
                eig = np.linalg.eigvalsh(matrix)
                _require(np.isfinite(matrix).all() and _close(matrix, matrix.T) and (np.diag(matrix) > 0).all()
                         and (eig.min() > 0 if kind == 'model' else eig.min() >= -1e-10*max(1., np.linalg.norm(matrix, 2))),
                         'estimated covariance is invalid.')
                _require(_close(diag[kind+'_covariance_min_eigenvalue'], eig.min()), 'covariance eigenvalue evidence differs.')
            _require(_close(diag['robust_covariance_max_asymmetry'], np.max(abs(matrices['robust']-matrices['robust'].T)))
                     and _close(diag['information_condition_number'], np.linalg.cond(matrices['model'])),
                     'covariance diagnostics differ from saved matrices.')
            beta = coef.coefficient.to_numpy(float); se = np.sqrt(np.diag(matrices['robust']))
            p = 2*norm.sf(abs(beta/se))
            values = coef[['coefficient','robust_se','model_se','hazard_ratio','ci_low','ci_high','p_value']].to_numpy(float)
            expected = np.column_stack([beta,se,np.sqrt(np.diag(matrices['model'])),np.exp(beta),
                                        np.exp(beta-1.96*se),np.exp(beta+1.96*se),p])
            _require(np.isfinite(values).all() and _close(values, expected), 'coefficient inference contradicts covariance.')
            _require(_close(row[['coefficient','robust_se','hr_per_adverse_sd','ci_low','ci_high','p_value']],
                            [beta[0],se[0],np.exp(beta[0]),np.exp(beta[0]-1.96*se[0]),np.exp(beta[0]+1.96*se[0]),p[0]]),
                     'target inference contradicts its fitted coefficient and covariance.')
        else:
            _require(coef[['coefficient','robust_se','model_se','hazard_ratio','ci_low','ci_high','p_value']].isna().all().all()
                     and all(np.isnan(v).all() for v in matrices.values())
                     and row[['coefficient','robust_se','hr_per_adverse_sd','ci_low','ci_high','p_value','fdr_q_value']].isna().all(),
                     'unestimated fit retains unsupported numerical inference.')
    p = estimates.p_value.to_numpy(float); finite = np.flatnonzero(np.isfinite(p))
    q = np.full(len(p), np.nan)
    if len(finite):
        order = finite[np.argsort(p[finite])]
        q[order] = np.minimum(1., np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    _require(np.allclose(estimates.fdr_q_value.to_numpy(float), q, rtol=1e-10, atol=1e-12, equal_nan=True)
             and metadata['multiplicity'] == {'method':'Benjamini-Hochberg', 'family':'selected_17', 'finite_tests':len(finite)},
             'multiplicity correction differs from the complete selected finite-test family.')


def _summary(result, fingerprint):
    metadata = result['metadata']
    return {'fingerprint':fingerprint, 'accepted':metadata['accepted'],
            'status':metadata['status'], 'selected_models':len(model.DEFAULT_TARGETS),
            'estimable_models':sum(m['status'] == 'estimated' for m in metadata['models']),
            'non_estimable_models':sum(m['status'] == 'non_estimable' for m in metadata['models']),
            'rows':{n:len(result['tables'][n]) for n in TABLES},
            'scientific_reference_comparison':'requires separate independent validation'}


def verify_result(directory, fingerprint, identity, result):
    """Check the entire saved publication against newly recomputed fits."""
    check_result(result)
    _require(directory.is_dir() and not directory.is_symlink()
             and all((directory/n).is_file() and not (directory/n).is_symlink()
                     for n in (*RESULT_FILES, 'receipt.json')), 'publication must contain regular nonsymlink files.')
    receipt = read_json(directory/'receipt.json', 'Pre-TB Cox receipt')
    _require(isinstance(receipt, dict) and receipt.get('schema_version') == 1
             and receipt.get('fingerprint') == fingerprint and receipt.get('identity') == identity,
             'receipt identity is inconsistent.')
    files = receipt.get('files')
    _require(isinstance(files, dict) and set(files) == set(RESULT_FILES)
             and {p.name for p in directory.iterdir()} == set(RESULT_FILES) | {'receipt.json'},
             'publication inventory is incomplete or contains unexpected artifacts.')
    _require(all(file_hash(directory/n) == h for n, h in files.items()), 'publication file hashes differ.')
    summary = read_json(directory/'summary.json', 'Pre-TB Cox summary')
    _require(summary == _summary(result, fingerprint) and receipt.get('summary') == summary
             and read_json(directory/'diagnostics.json', 'Pre-TB Cox diagnostics') == result['metadata'],
             'publication metadata differs from recomputed fits.')
    for name in TABLES:
        expected = result['tables'][name]
        try:
            observed = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip',
                keep_default_na=False, na_values={c:[''] for c in expected
                    if pd.api.types.is_numeric_dtype(expected[c])})
            pd.testing.assert_frame_equal(observed, expected, check_dtype=False,
                check_exact=False, rtol=1e-10, atol=1e-12)
        except (AssertionError, ValueError, TypeError):
            raise ValidationError('Pre-TB Cox saved table differs from recomputed fit evidence.') from None
    return summary


def fit_selected(root, preparation, work, output, *, resume=False):
    root, prep, work, output = safe_roots(root, preparation, work, output)
    assert_loaded_sources(root)
    descriptor = contract_descriptor(prep)
    fingerprint, identity = fit_identity(root, descriptor)
    destination = output/fingerprint
    with preparation_locks(work, output):
        reused = destination.exists()
        _require(not reused or resume, 'identical results exist; use --resume to verify them.')
        contracts = {model.contract_key(t):load_artifacts(prep, ('M03', 'pft', 'samples', model.contract_key(t)))
                     for t in model.DEFAULT_TARGETS}
        result = model.fit_selected(contracts)
        check_result(result, contracts)
        def unchanged():
            assert_loaded_sources(root)
            _require(fit_identity(root, contract_descriptor(prep)) == (fingerprint, identity),
                     'dependencies changed during execution.')
        if reused:
            summary = verify_result(destination, fingerprint, identity, result)
            unchanged()
            return {**summary, 'resumed':True}
        stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=output))
        try:
            for name in TABLES:
                result['tables'][name].to_csv(stage/(name+'.csv'), index=False, float_format='%.17g')
            summary = _summary(result, fingerprint)
            write_json(stage/'diagnostics.json', result['metadata'])
            write_json(stage/'summary.json', summary)
            write_json(stage/'receipt.json', {'schema_version':1, 'fingerprint':fingerprint,
                'identity':identity, 'summary':summary,
                'files':{n:file_hash(stage/n) for n in RESULT_FILES}})
            verify_result(stage, fingerprint, identity, result)
            unchanged()
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return {**summary, 'resumed':False}


def run_pretb(root, config_file, analysis_file, primary_sampling_file,
              mechanics_sampling_file, supporting_sampling_file, *, jobs=1, resume=False):
    """Run cumulative M01-M09, including verification of preceding analyses."""
    from .supporting_execution import run_supporting
    _require(type(jobs) is int and 1 <= jobs <= 3, 'concurrency must be between one and three jobs.')
    assert_loaded_sources(root)
    paths = [Path(p) for p in (config_file, analysis_file, primary_sampling_file,
                              mechanics_sampling_file, supporting_sampling_file)]
    hashes = [file_hash(p) for p in paths]
    prior = run_supporting(root, *paths, jobs=jobs, resume=resume)
    expected = {'through':'M08', 'selected_models':7, 'deterministic_fits':10,
                'primary_models_verified':12, 'binary_models_verified':28,
                'trajectory_analyses_verified':4, 'mechanics_models_verified':6,
                'shared_factor_rows_verified':6, 'status':'accepted'}
    _require(isinstance(prior, dict) and prior.get('accepted') is True
             and all(prior.get(k) == v for k, v in expected.items()),
             'cumulative execution requires all preceding selected analyses to pass.')
    configuration = read_json(config_file, 'Local configuration')
    _, directories = resolve_paths(configuration, config_file)
    prep = directories['work']/'preparation_runs'/prior['preparation_fingerprint']
    result = fit_selected(root, prep, directories['work']/'pretb_cox_fits',
                          directories['outputs']/'pretb_cox', resume=resume)
    _require(isinstance(result, dict) and type(result.get('accepted')) is bool
             and result.get('selected_models') == 17
             and result.get('rows') == {'estimates':17, 'coefficients':51, 'covariance':306}
             and result.get('status') == ('accepted' if result['accepted'] else 'fit_failed'),
             'completion requires all 17 selected fits and their complete output inventories.')
    verified = run_preparation(root, config_file, analysis_file, 'M03', resume=True)
    _require(verified['fingerprint'] == prior['preparation_fingerprint']
             and [file_hash(p) for p in paths] == hashes,
             'preparation or configuration changed during cumulative execution.')
    assert_loaded_sources(root)
    summary = {'through':'M09', 'preparation_fingerprint':prior['preparation_fingerprint'],
               'selected_models':17, 'pretb_cox':result,
               'primary_models_verified':12, 'binary_models_verified':28,
               'trajectory_analyses_verified':4, 'mechanics_models_verified':6,
               'shared_factor_rows_verified':6, 'supporting_joint_models_verified':7,
               'supporting_deterministic_fits_verified':10,
               'accepted':result['accepted'], 'status':result['status'],
               'scientific_reference_comparison':'requires separate independent validation'}
    if result['accepted']:
        write_pointer(directories['work']/'pretb_current.json', summary)
    return summary
