"""Fit and verify selected main and supplementary FeNO regressions privately."""
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
from .models import feno as model

SOURCES = ('models/feno.py', 'feno_execution.py', 'respiratory_preparation.py',
           'execution.py', 'pipeline.py')
TABLES = ('estimates', 'coefficients', 'covariance', 'counts')
RESULT_FILES = tuple(name+'.csv' for name in TABLES) + ('diagnostics.json', 'summary.json')
_LOADED_SOURCES = {name:file_hash(Path(__file__).parent/name) for name in SOURCES}


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValidationError('FeNO '+message)


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
    for folder in (work, output):
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        node = artifact_subtree(manifest, model.artifact_path(target))
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
        descriptors[target] = digest({'tree':node, 'files':{n:manifest['files'][n] for n in sorted(files)}})
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
                       rtol=1e-10, atol=1e-10, equal_nan=False)


def _integer(value, minimum=0):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value) and value >= minimum and value == int(value))


def check_result(result, contracts=None):
    """Require complete aggregate output, bound to verified contracts if supplied."""
    try:
        _check_result(result, contracts)
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, ArithmeticError):
        raise ValidationError('FeNO result or prepared contract has malformed values.') from None


def _check_result(result, contracts):
    _require(isinstance(result, dict) and set(result) == {'tables','metadata'}, 'result schema differs.')
    tables, metadata = result['tables'], result['metadata']
    _require(isinstance(tables, dict) and set(tables) == set(TABLES), 'table inventory differs.')
    for name in TABLES:
        table = tables[name]
        _require(isinstance(table, pd.DataFrame) and list(table.columns) == list(model.TABLE_COLUMNS[name])
                 and not table.empty and table.target.isin(model.DEFAULT_TARGETS).all(), 'aggregate table schema differs.')
    _require(isinstance(metadata, dict) and set(metadata) == {'schema_version','accepted','status','selection','models'}
             and type(metadata['schema_version']) is int and metadata['schema_version'] == 1
             and type(metadata['accepted']) is bool and metadata['selection'] == list(model.DEFAULT_TARGETS)
             and isinstance(metadata['models'], list) and len(metadata['models']) == 11,
             'metadata schema or selection differs.')
    _require([m['target'] for m in metadata['models']] == list(model.DEFAULT_TARGETS)
             and list(tables['estimates'].target) == list(model.DEFAULT_TARGETS), 'selected models differ or are reordered.')
    _require(metadata['accepted'] == all(m['accepted'] is True for m in metadata['models'])
             and metadata['status'] == ('accepted' if metadata['accepted'] else 'fit_failed'), 'overall acceptance contradicts model status.')
    if contracts is not None:
        _require(isinstance(contracts, dict) and set(contracts) == set(model.DEFAULT_TARGETS), 'contract selection differs.')
    reasons = {'','source_no_exposure_contrast','singular_design','insufficient_residual_degrees_of_freedom',
               'optimizer_did_not_converge','nonfinite_fit','score_not_converged','invalid_covariance','numerical_fit_failure'}
    warning_classes = {'RuntimeWarning','UserWarning','ConvergenceWarning','IterationLimitWarning',
                       'ValueWarning','PerfectSeparationWarning'}
    meta_keys = {'target','family','visit','model_version','artifact_path','terms','plotted_term','counts',
                 'status','accepted','reason','converged','warnings','fit_controls','covariance_correction',
                 'diagnostics','coefficient_transform'}
    count_keys = {'observations','children','exposed_observations','exposed_children','reference_observations',
                  'reference_children','other_observations','other_children'}
    diagnostic_keys = {'rank','condition_number','score_norm','scale','df_resid','n_clusters','iterations',
                       'robust_min_eigenvalue','naive_min_eigenvalue','robust_max_asymmetry'}
    value_cols = ['coefficient','robust_se','naive_se','exponentiated_coefficient','ci_low','ci_high','p_value']
    target_cols = ['coefficient','robust_se','geometric_mean_ratio','ci_low','ci_high','percent_difference','p_value']
    for index, target in enumerate(model.DEFAULT_TARGETS):
        spec, meta, row = model.fit_spec(target), metadata['models'][index], tables['estimates'].iloc[index]
        _require(isinstance(meta, dict) and set(meta) == meta_keys, 'model metadata contains undeclared fields.')
        for field in ('target','family','visit','model_version','artifact_path','fit_controls','coefficient_transform'):
            _require(canonical_bytes(meta[field]) == canonical_bytes(spec[field]), 'model specification differs.')
        terms, plotted, counts = meta['terms'], meta['plotted_term'], meta['counts']
        _require(isinstance(terms, list) and len(terms) == len(set(terms)) and terms and terms[0] == 'Intercept'
                 and isinstance(counts, dict) and set(counts) == count_keys
                 and all(_integer(v) for v in counts.values()), 'model terms or counts are malformed.')
        main = spec['family'] == 'main'
        exposure_terms = ['infancy_post','later_post'] if main else ['tb_before_measurement','tb_after_measurement']
        adjustments = ['male','age','height'] if main else ['age_z','height_z','male']
        _require(terms == ['Intercept']+[t for t in exposure_terms if t in terms]+adjustments
                 and (not main or ('infancy_post' in terms and plotted == 'infancy_post'))
                 and (plotted in exposure_terms or (not main and plotted == ''))
                 and (not plotted or plotted in terms), 'selected terms or plotted contrast differ.')
        n, g, p = counts['observations'], counts['children'], len(terms)
        _require(0 < g <= n and n == sum(counts[k+'_observations'] for k in ('exposed','reference','other'))
                 and all(0 <= counts[k+'_children'] <= min(g, counts[k+'_observations']) for k in ('exposed','reference','other')),
                 'support counts are contradictory.')
        _require(meta['status'] in {'estimated','non_estimable','fit_failed'} and type(meta['accepted']) is bool
                 and type(meta['converged']) is bool and meta['reason'] in reasons
                 and isinstance(meta['warnings'], list) and all(w in warning_classes for w in meta['warnings'])
                 and type(meta['covariance_correction']) is float and meta['covariance_correction'] == 1., 'status, warnings or covariance rule differs.')
        estimated = meta['status'] == 'estimated'
        _require(meta['accepted'] == (meta['status'] != 'fit_failed')
                 and (not estimated or (meta['reason'] == '' and meta['converged'] and plotted))
                 and (estimated or meta['reason'] != '')
                 and (meta['status'] != 'non_estimable' or (not main and plotted == ''
                      and meta['reason'] == 'source_no_exposure_contrast' and not meta['converged']
                      and counts['exposed_observations'] == counts['other_observations'] == 0))
                 and (meta['status'] != 'fit_failed' or meta['reason'] not in {'','source_no_exposure_contrast'}),
                 'model acceptance contradicts its status or reason.')
        coef = tables['coefficients'].loc[tables['coefficients'].target.eq(target)]
        cov = tables['covariance'].loc[tables['covariance'].target.eq(target)]
        count_table = tables['counts'].loc[tables['counts'].target.eq(target)]
        _require(list(coef.term) == terms
                 and list(zip(cov.covariance_type,cov.row_term,cov.column_term)) ==
                    [(kind,a,b) for kind in ('robust','naive') for a in terms for b in terms]
                 and list(count_table.group) == list(model.COUNT_GROUPS[spec['family']]),
                 'coefficient, covariance or count inventory differs.')
        for frame in (coef, cov):
            _require(frame.status.eq(meta['status']).all() and frame.accepted.map(lambda v:isinstance(v,(bool,np.bool_))).all()
                     and frame.accepted.eq(meta['accepted']).all(), 'table status differs from metadata.')
        _require(coef.reason.eq(meta['reason']).all(), 'coefficient reason differs.')
        for frame in (coef, count_table):
            _require(frame.family.eq(spec['family']).all() and frame.visit.eq(spec['visit']).all(), 'table selector differs.')
        for field in ('target','family','visit','display_visit','nominal_age_years','scenario'):
            _require(row[field] == spec[field], 'estimate selector differs.')
        for field in (*count_keys,'status','accepted','reason','converged','plotted_term'):
            _require(row[field] == (counts[field] if field in counts else meta[field]), 'estimate support or status differs.')
        _require(isinstance(row.accepted,(bool,np.bool_)) and isinstance(row.converged,(bool,np.bool_))
                 and row.contrast == model.contrast_label(target, plotted), 'estimate contrast differs.')
        support = dict(zip(count_table.group, zip(count_table.observations,count_table.children)))
        _require(all(_integer(v) for pair in support.values() for v in pair)
                 and sum(pair[0] for pair in support.values()) == n, 'group counts differ from sample size.')
        labels = model.COUNT_GROUPS[spec['family']]
        exposed_index = 0 if main or plotted == 'tb_before_measurement' else 1
        for key, group_index in [('exposed',exposed_index),('other',1-exposed_index),('reference',2)]:
            _require(support[labels[group_index]] == (counts[key+'_observations'],counts[key+'_children']),
                     'group counts disagree with target support.')
        _require(np.isfinite(row[['age_min','age_max','age_median']].to_numpy(float)).all()
                 and row.age_min <= row.age_median <= row.age_max, 'age summaries are invalid.')
        diag = meta['diagnostics']
        _require(isinstance(diag, dict) and set(diag) == diagnostic_keys
                 and all(v is None or (not isinstance(v,(bool,np.bool_))
                         and isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(v))
                         for v in diag.values()), 'diagnostics contain undeclared or nonscalar fields.')
        _require(_integer(diag['rank']) and diag['rank'] <= p
                 and diag['df_resid'] == n-p and diag['n_clusters'] == g,
                 'design diagnostics contradict sample support.')
        matrices = {kind:cov.loc[cov.covariance_type.eq(kind),'value'].to_numpy(float).reshape(p,p)
                    for kind in ('robust','naive')}
        if contracts is not None:
            contract = contracts[target]
            model.validate_contract(contract, target)
            frame, design = contract['frame'], contract['design']
            _require(terms == list(design.columns) and plotted == (contract['target'] or '')
                     and n == len(frame) and g == frame.pid_child.nunique(), 'output differs from verified prepared contract.')
            first, second = [frame[t].eq(1) for t in exposure_terms]
            for label, mask in zip(labels,(first,second,~(first|second))):
                _require(support[label] == (int(mask.sum()),int(frame.loc[mask,'pid_child'].nunique())),
                         'output group counts differ from prepared records.')
            age = frame['age' if main else 'age_years']
            _require(_close(row[['age_min','age_max','age_median']], [age.min(),age.max(),age.median()]),
                     'age summaries differ from prepared records.')
        if estimated:
            _require(n > p and _integer(diag['rank']) and diag['rank'] == p
                     and _integer(diag['df_resid']) and diag['df_resid'] == n-p
                     and _integer(diag['n_clusters'],1) and diag['n_clusters'] == g
                     and _integer(diag['iterations'],1) and diag['iterations'] <= spec['fit_controls']['maxiter']
                     and np.isfinite([diag['condition_number'],diag['score_norm'],diag['scale']]).all()
                     and diag['condition_number'] >= 1 and 0 <= diag['score_norm'] < spec['fit_controls']['ctol']
                     and diag['scale'] > 0, 'estimated fit lacks valid convergence evidence.')
            for kind, matrix in matrices.items():
                eig = np.linalg.eigvalsh(matrix)
                _require(np.isfinite(matrix).all() and _close(matrix,matrix.T) and (np.diag(matrix)>0).all()
                         and (eig.min()>0 if kind=='naive' else eig.min()>=-1e-12)
                         and _close(diag[kind+'_min_eigenvalue'],eig.min()), 'covariance matrix or diagnostics differ.')
            _require(_close(diag['robust_max_asymmetry'], np.max(abs(matrices['robust']-matrices['robust'].T))),
                     'covariance asymmetry diagnostic differs.')
            beta = coef.coefficient.to_numpy(float)
            se = np.sqrt(np.diag(matrices['robust']))
            expected = np.column_stack([beta,se,np.sqrt(np.diag(matrices['naive'])),np.exp(beta),
                                        np.exp(beta-1.96*se),np.exp(beta+1.96*se),2*norm.sf(abs(beta/se))])
            _require(np.isfinite(coef[value_cols].to_numpy(float)).all() and _close(coef[value_cols],expected),
                     'coefficient inference contradicts covariance.')
            position = terms.index(plotted)
            b,s = beta[position],se[position]
            _require(_close(row[target_cols],[b,s,np.exp(b),np.exp(b-1.96*s),np.exp(b+1.96*s),
                                             100*np.expm1(b),2*norm.sf(abs(b/s))]), 'target inference differs.')
            if contracts is not None:
                x,y = design.to_numpy(float),np.asarray(contract['response'],float)
                q,r = np.linalg.qr(x,mode='reduced')
                analytic_beta = np.linalg.solve(r,q.T@y)
                residual = y-x@analytic_beta
                inv_r = np.linalg.solve(r,np.eye(p)); bread = inv_r@inv_r.T
                scores = x*residual[:,None]
                codes = pd.factorize(frame.pid_child,sort=True)[0]
                group_scores = np.zeros((g,p));np.add.at(group_scores,codes,scores)
                robust = bread@(group_scores.T@group_scores)@bread
                scale = float(residual@residual/(n-p))
                _require(_close(beta,analytic_beta) and _close(matrices['robust'],robust)
                         and _close(matrices['naive'],scale*bread) and _close(diag['scale'],scale)
                         and _close(diag['condition_number'],np.linalg.cond(x))
                         and np.linalg.norm(x.T@residual) < spec['fit_controls']['ctol'],
                         'GEE result differs from independent least-squares and child-score equations.')
        else:
            _require(coef[value_cols].isna().all().all() and row[target_cols].isna().all()
                     and all(np.isnan(v).all() for v in matrices.values()), 'unestimated fit retains unsupported inference.')



def _summary(result, fingerprint):
    metadata = result['metadata']
    return {'fingerprint':fingerprint, 'accepted':metadata['accepted'],
            'status':metadata['status'], 'selected_models':len(model.DEFAULT_TARGETS),
            'estimable_models':sum(m['status'] == 'estimated' for m in metadata['models']),
            'non_estimable_models':sum(m['status'] == 'non_estimable' for m in metadata['models']),
            'rows':{n:len(result['tables'][n]) for n in TABLES},
            'scientific_reference_comparison':'requires separate independent validation'}


def verify_result(directory, fingerprint, identity, result):
    """Check all saved outputs against freshly recomputed selected regressions."""
    check_result(result)
    _require(directory.is_dir() and not directory.is_symlink()
             and all((directory/n).is_file() and not (directory/n).is_symlink()
                     for n in (*RESULT_FILES, 'receipt.json')), 'publication must contain regular nonsymlink files.')
    receipt = read_json(directory/'receipt.json', 'FeNO receipt')
    _require(isinstance(receipt, dict) and set(receipt) == {'schema_version','fingerprint','identity','summary','files'}
             and type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1
             and receipt.get('fingerprint') == fingerprint
             and receipt.get('identity') == identity, 'receipt identity is inconsistent.')
    files = receipt.get('files')
    _require(isinstance(files, dict) and set(files) == set(RESULT_FILES)
             and {p.name for p in directory.iterdir()} == set(RESULT_FILES) | {'receipt.json'},
             'publication inventory is incomplete or contains unexpected artifacts.')
    _require(all(file_hash(directory/n) == h for n, h in files.items()), 'publication file hashes differ.')
    summary = read_json(directory/'summary.json', 'FeNO summary')
    _require(summary == _summary(result, fingerprint) and receipt.get('summary') == summary
             and read_json(directory/'diagnostics.json', 'FeNO diagnostics') == result['metadata'],
             'publication metadata differs from recomputed fits.')
    for name in TABLES:
        expected = result['tables'][name]
        try:
            observed = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip',
                keep_default_na=False, na_values={c:[''] for c in expected
                    if pd.api.types.is_numeric_dtype(expected[c])})
            pd.testing.assert_frame_equal(observed, expected, check_dtype=False,
                check_exact=False, rtol=1e-10, atol=1e-10)
        except (AssertionError, ValueError, TypeError):
            raise ValidationError('FeNO saved table differs from recomputed fit evidence.') from None
    return summary


def fit_selected(root, preparation, work, output, *, resume=False):
    root, prep, work, output = safe_roots(root, preparation, work, output)
    assert_loaded_sources(root)
    descriptor = contract_descriptor(prep)
    fingerprint, identity = fit_identity(root, descriptor)
    destination = output/fingerprint
    with preparation_locks(work, output):
        reused = destination.exists() or destination.is_symlink()
        _require(not reused or resume, 'identical results exist; use --resume to verify them.')
        contracts = {t:load_artifacts(prep, model.artifact_path(t)) for t in model.DEFAULT_TARGETS}
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
            _require(not destination.exists() and not destination.is_symlink(),
                     'publication destination appeared during execution; refusing to replace it.')
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return {**summary, 'resumed':False}


def run_feno(root, config_file, analysis_file, primary_sampling_file,
             mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file,
             *, jobs=1, resume=False):
    """Run cumulative M01-M11, verifying all preceding selected analyses."""
    from .wheeze_execution import run_wheeze
    _require(type(jobs) is int and 1 <= jobs <= 3, 'concurrency must be between one and three jobs.')
    assert_loaded_sources(root)
    paths = [Path(p) for p in (config_file, analysis_file, primary_sampling_file,
                              mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file)]
    hashes = [file_hash(p) for p in paths]
    prior = run_wheeze(root, *paths, jobs=jobs, resume=resume)
    expected = {'through':'M10', 'selected_models':2, 'primary_models_verified':12,
                'binary_models_verified':28, 'trajectory_analyses_verified':4,
                'mechanics_models_verified':6, 'shared_factor_rows_verified':6,
                'supporting_joint_models_verified':7, 'supporting_deterministic_fits_verified':10,
                'pretb_models_verified':17, 'status':'accepted'}
    _require(isinstance(prior, dict) and prior.get('accepted') is True
             and all(prior.get(k) == v for k, v in expected.items()),
             'cumulative execution requires all preceding selected analyses to pass.')
    configuration = read_json(config_file, 'Local configuration')
    _, directories = resolve_paths(configuration, config_file)
    prep = directories['work']/'preparation_runs'/prior['preparation_fingerprint']
    result = fit_selected(root, prep, directories['work']/'feno_fits',
                          directories['outputs']/'feno_models', resume=resume)
    widths = [len(load_artifacts(prep, model.artifact_path(t))['design'].columns) for t in model.DEFAULT_TARGETS]
    expected_rows = {'estimates':11, 'coefficients':sum(widths),
                     'covariance':sum(2*p*p for p in widths), 'counts':33}
    _require(isinstance(result, dict) and type(result.get('accepted')) is bool
             and result.get('selected_models') == 11
             and result.get('rows') == expected_rows
             and _integer(result.get('estimable_models')) and _integer(result.get('non_estimable_models'))
             and result['estimable_models']+result['non_estimable_models'] <= 11
             and (not result['accepted'] or result['estimable_models']+result['non_estimable_models'] == 11)
             and isinstance(result.get('rows'), dict) and set(result['rows']) == set(TABLES)
             and result['rows']['estimates'] == 11 and result['rows']['counts'] == 33
             and all(_integer(n,1) for n in result['rows'].values())
             and result.get('status') == ('accepted' if result['accepted'] else 'fit_failed'),
             'completion requires all eleven selected fits and their complete output inventories.')
    verified = run_preparation(root, config_file, analysis_file, 'M03', resume=True)
    _require(verified['fingerprint'] == prior['preparation_fingerprint']
             and [file_hash(p) for p in paths] == hashes,
             'preparation or configuration changed during cumulative execution.')
    assert_loaded_sources(root)
    summary = {'through':'M11', 'preparation_fingerprint':prior['preparation_fingerprint'],
               'selected_models':11, 'feno_models':result,
               **{k:v for k,v in expected.items() if k.endswith('_verified')},
               'wheeze_models_verified':2, 'accepted':result['accepted'], 'status':result['status'],
               'inherited_limitations':['G08: pre-TB source covariance limitation remains unresolved',
                                       'Four upstream antenatal flow counts remain unavailable'],
               'scientific_reference_comparison':'requires separate independent validation'}
    if result['accepted']:
        write_pointer(directories['work']/'feno_current.json', summary)
    return summary
