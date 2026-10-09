"""Publish and verify deterministic MMEF change and X6 supporting results."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2, norm
from .models import supporting_repeated as model

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, verify_artifacts, write_json, runtime_identity)
from .pipeline import ValidationError
from .supporting_execution import digest, read_json, safe_roots, verify_files

SOURCES = ('models/supporting_repeated.py', 'models/spirometry_trajectories.py',
           'preparation.py', 'supporting_repeated_execution.py', 'supporting_execution.py',
           'execution.py', 'pipeline.py')
_LOADED_HASH = file_hash(Path(__file__))
_LOADED_SOURCES = {name:file_hash(Path(__file__).parent/name) for name in SOURCES}
TABLES = ('mmef_change', 'mmef_descriptive', 'x6_repeated', 'x6_global_tests',
          'x6_predictions', 'x6_age_contrasts', 'coefficients', 'covariance')
RESULT_FILES = tuple(n+'.csv' for n in TABLES) + ('diagnostics.json', 'summary.json')


def contract_descriptor(preparation):
    from .models.supporting_repeated import CONTRACT_KEYS
    directory = Path(preparation).resolve()
    receipt = read_json(directory/'receipt.json', 'Preparation receipt')
    if (receipt.get('through') != 'M03' or receipt.get('status') != 'prepared'
            or digest(receipt.get('identity')) != receipt.get('fingerprint')
            or file_hash(directory/'artifacts.json') != receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Supporting repeated models require complete verified M03 preparation.')
    manifest = verify_artifacts(directory)
    descriptors = {}
    for key in CONTRACT_KEYS:
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
            raise ValidationError('Supporting contract references an unverified input artifact.')
        descriptors[key] = digest({'tree':node, 'files':{f:manifest['files'][f] for f in sorted(files)}})
    return {'contracts':descriptors, 'preparation_fingerprint':receipt['fingerprint']}


def fit_identity(root, descriptor):
    from .models.supporting_repeated import CONTRACT_KEYS
    root = Path(root)
    identity = {'schema_version':1, 'contracts':descriptor['contracts'],
                'selection':list(CONTRACT_KEYS),
                'source':{name:file_hash(root/'src/dchs_lung_function'/name) for name in SOURCES},
                'environment_lock_sha256':file_hash(root/'requirements-models.lock.txt'),
                'runtime':runtime_identity()}
    return digest(identity), identity


SCHEMAS = {'x6_predictions': ['model_id',
                    'fit',
                    'outcome',
                    'outcome_label',
                    'unit',
                    'tb_age_cutoff_years',
                    'age_years',
                    'group',
                    'predicted_mean',
                    'robust_se',
                    'ci_low',
                    'ci_high'],
 'coefficients': ['model_id',
                  'family',
                  'target',
                  'threshold',
                  'fit',
                  'term',
                  'coefficient',
                  'robust_se',
                  'model_se'],
 'mmef_descriptive': ['target',
                      'outcome',
                      'unit',
                      'group',
                      'n_pairs',
                      'n_children',
                      'mean_pre',
                      'mean_post',
                      'mean_change',
                      'sd_change',
                      'median_change'],
 'mmef_change': ['model_id',
                 'fit',
                 'design',
                 'target',
                 'outcome',
                 'unit',
                 'exposure',
                 'sensitivity',
                 'estimate',
                 'se',
                 'ci_low',
                 'ci_high',
                 'p_value',
                 'standardized_reference_change',
                 'standardized_reference_ci_low',
                 'standardized_reference_ci_high',
                 'standardized_exposed_change',
                 'standardized_exposed_ci_low',
                 'standardized_exposed_ci_high',
                 'bootstrap_replicates',
                 'bootstrap_ci_low',
                 'bootstrap_ci_high',
                 'leave_one_child_out_min',
                 'leave_one_child_out_max',
                 'n_pairs',
                 'n_children',
                 'n_exposed_pairs',
                 'n_exposed_children',
                 'converged',
                 'dropped_terms',
                 'adjustment'],
 'x6_global_tests': ['model_id',
                     'fit',
                     'outcome',
                     'outcome_label',
                     'unit',
                     'tb_age_cutoff_years',
                     'n_observations',
                     'n_children',
                     'n_exposed_observations',
                     'n_exposed_children',
                     'n_reference_observations',
                     'n_reference_children',
                     'overall_trajectory_wald_chi2',
                     'overall_trajectory_df',
                     'overall_trajectory_p',
                     'interaction_wald_chi2',
                     'interaction_df',
                     'interaction_p',
                     'parallel_trajectory_difference',
                     'parallel_trajectory_robust_se',
                     'parallel_trajectory_ci_low',
                     'parallel_trajectory_ci_high',
                     'parallel_trajectory_p',
                     'icc',
                     'condition_number',
                     'random_intercept_variance',
                     'residual_variance'],
 'covariance': ['model_id',
                'family',
                'target',
                'threshold',
                'fit',
                'covariance_type',
                'row_term',
                'column_term',
                'covariance'],
 'x6_repeated': ['model_id',
                 'fit',
                 'outcome',
                 'outcome_label',
                 'unit',
                 'tb_age_cutoff_years',
                 'contrast',
                 'estimate',
                 'robust_se',
                 'ci_low',
                 'ci_high',
                 'p_value',
                 'icc',
                 'condition_number',
                 'n_observations',
                 'n_children',
                 'n_exposed_observations',
                 'n_exposed_children',
                 'n_reference_observations',
                 'n_reference_children',
                 'reference_measurement_mean',
                 'reference_measurement_sd',
                 'interpretation'],
 'x6_age_contrasts': ['model_id',
                      'fit',
                      'outcome',
                      'outcome_label',
                      'unit',
                      'tb_age_cutoff_years',
                      'age_years',
                      'difference_tb_minus_reference',
                      'robust_se',
                      'ci_low',
                      'ci_high']}

FIXED_ROWS = {'mmef_change':2, 'mmef_descriptive':6, 'x6_repeated':4,
              'x6_global_tests':2, 'x6_predictions':404, 'x6_age_contrasts':16}
SELECTED_FITS = [("mmef/change/before5", "full"), ("mmef/change/infancy", "full")]
SELECTED_FITS += [(f"x6_repeated/{outcome}/{cutoff}", "full")
                 for outcome in ('x6_sa_z', 'reactance_6hz') for cutoff in (1, 5)]
SELECTED_FITS += [(f"trajectory/x6/{cutoff}", fit) for cutoff in (1, 5) for fit in ('full', 'reduced')]


def assert_loaded_sources(root):
    for name, expected in _LOADED_SOURCES.items():
        if (file_hash(Path(__file__).parent/name) != expected
                or file_hash(Path(root)/'src/dchs_lung_function'/name) != expected):
            raise ValidationError('Supporting repeated sources changed after import or differ from the supplied source tree; restart the process.')


def _require(condition, message):
    if not condition:
        raise ValidationError('Supporting repeated '+message)


def _acceptance(metadata):
    _require(isinstance(metadata, dict) and type(metadata.get('accepted')) is bool
             and isinstance(metadata.get('failure_reasons'), list)
             and all(isinstance(x, str) and x for x in metadata['failure_reasons']), 'fit status is malformed.')
    accepted = metadata['accepted']
    _require(accepted == (not metadata['failure_reasons'])
             and metadata.get('status') == ('accepted' if accepted else 'fit_failed'),
             'acceptance contradicts its fit evidence.')
    return accepted


def _integer(value, minimum=0):
    return not isinstance(value, (bool, np.bool_)) and np.isfinite(value) and value == int(value) and value >= minimum


def _close(actual, expected):
    return np.allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=1e-9, atol=1e-10)


def _normal(row, beta, covariance, term, names, fields):
    index = names.index(term)
    estimate, se = beta[index], np.sqrt(covariance[index,index])
    _require(np.isfinite(se) and se > 0, 'contrast has invalid uncertainty.')
    expected = [estimate, se, estimate-1.96*se, estimate+1.96*se, 2*norm.sf(abs(estimate/se))]
    _require(_close(row[list(fields)], expected), 'inference disagrees with coefficient and robust covariance.')


def _check_result(result, contracts=None):
    try:
        _check_result_schema(result, contracts)
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, np.linalg.LinAlgError):
        raise ValidationError('Supporting repeated result has malformed scientific evidence.') from None


def _check_result_schema(result, contracts):
    _require(isinstance(result, dict) and set(result) == {'tables', 'metadata'}
             and isinstance(result['tables'], dict) and set(result['tables']) == set(TABLES),
             'fit returned an invalid result inventory.')
    tables, metadata = result['tables'], result['metadata']
    for name, frame in tables.items():
        _require(isinstance(frame, pd.DataFrame) and list(frame) == SCHEMAS[name]
                 and frame.index.equals(pd.RangeIndex(len(frame))), 'table schema or row index is invalid.')
        if name in FIXED_ROWS:
            _require(len(frame) == FIXED_ROWS[name], 'table has an incomplete row inventory.')
    accepted = _acceptance(metadata)
    _require(metadata.get('fitted_models') == 10 and type(metadata['fitted_models']) is int
             and metadata.get('analyses') == 8 and metadata.get('contract_keys') == list(model.CONTRACT_KEYS)
             and metadata.get('row_counts') == {n:len(f) for n,f in tables.items()}
             and metadata.get('participant_records_returned') is False
             and metadata.get('interval_multiplier') == 1.96
             and metadata.get('p_value_adjustment') is None, 'fit inventory or inference declaration is inconsistent.')
    models = metadata.get('models')
    _require(isinstance(models, list) and all(isinstance(m, dict) for m in models)
             and [(m.get('model_id'), m.get('fit')) for m in models] == SELECTED_FITS,
             'diagnostics lack the exact ten selected component fits.')
    failures = [m['model_id']+'/'+m['fit']+': '+r for m in models for r in m['failure_reasons']]
    _require(metadata['failure_reasons'] == failures, 'overall failure evidence disagrees with component fits.')
    coefficients, covariances = tables['coefficients'], tables['covariance']
    _require(not coefficients.duplicated(['model_id','fit','term']).any()
             and not covariances.duplicated(['model_id','fit','covariance_type','row_term','column_term']).any(),
             'coefficient or covariance identities are duplicated.')
    _require(set(zip(coefficients.model_id, coefficients.fit)) == set(SELECTED_FITS)
             and set(zip(covariances.model_id, covariances.fit)) == set(SELECTED_FITS),
             'coefficient or covariance fit inventory is incomplete.')
    fitted = {}
    for meta in models:
        pair = meta['model_id'], meta['fit']
        good = _acceptance(meta)
        terms = meta['terms']; p = len(terms)
        _require(isinstance(terms, list) and p > 0 and len(set(terms)) == p
                 and all(isinstance(t,str) and t for t in terms) and meta['n_parameters'] == p,
                 'model terms are invalid.')
        if pair[0].startswith('mmef/'):
            expected_identity = ('MMEF-CHANGE-GEE','mmef',5 if pair[0].endswith('before5') else 1)
        elif pair[0].startswith('x6_repeated/'):
            expected_identity = ('X6-REPEATED',pair[0].split('/')[1],int(pair[0].split('/')[2]))
        else:
            expected_identity = ('X6-TRAJECTORY','x6_sa_z',int(pair[0].split('/')[2]))
        _require(tuple(meta[k] for k in ('family','target','threshold')) == expected_identity,
                 'component identity labels differ from the selected model.')
        n, g = meta['n_observations'], meta['n_children']
        _require(_integer(n,1) and _integer(g,1) and g <= n, 'sample counts are invalid.')
        if pair[0].startswith('trajectory/'):
            expected_terms = list(model.FULL_TERMS if pair[1] == 'full' else model.REDUCED_TERMS)
            _require(terms == expected_terms, 'trajectory terms differ from the selected design.')
        if contracts is not None:
            key = 'mmef/change' if pair[0].startswith('mmef/') else pair[0]
            contract = contracts[key]
            if pair[0].startswith('mmef/'):
                expected_terms = contract['designs'][pair[0].rsplit('/',1)[1]]['terms']
            else:
                expected_terms = contract['terms' if pair[1] == 'full' else 'reduced_terms']
            _require(terms == list(expected_terms) and n == len(contract['frame'])
                     and g == contract['frame'].pid_child.nunique(), 'fit differs from its prepared design or sample.')
        coeff = coefficients.loc[coefficients.model_id.eq(pair[0]) & coefficients.fit.eq(pair[1])]
        cov = covariances.loc[covariances.model_id.eq(pair[0]) & covariances.fit.eq(pair[1])]
        _require(list(coeff.term) == terms and len(cov) == 2*p*p, 'fit coefficient/covariance term inventory is incomplete.')
        for field in ('family','target','threshold'):
            _require(coeff[field].eq(meta[field]).all() and cov[field].eq(meta[field]).all(), 'fit identity labels disagree.')
        beta = coeff.coefficient.to_numpy(float); matrices = {}
        for kind in ('robust','model'):
            cells = cov.loc[cov.covariance_type.eq(kind)]
            _require(set(zip(cells.row_term,cells.column_term)) == {(a,b) for a in terms for b in terms},
                     'covariance matrix cells are incomplete.')
            matrix = cells.pivot(index='row_term',columns='column_term',values='covariance').loc[terms,terms].to_numpy(float)
            matrices[kind] = matrix
            if good:
                _require(np.isfinite(matrix).all() and _close(matrix,matrix.T)
                         and np.linalg.eigvalsh((matrix+matrix.T)/2).min() >= -1e-8*max(1.,np.abs(matrix).max()),
                         'accepted covariance is nonfinite, asymmetric or indefinite.')
                _require(_close(coeff['robust_se' if kind == 'robust' else 'model_se'], np.sqrt(np.maximum(np.diag(matrix),0.))),
                         'reported standard errors differ from covariance diagonals.')
            else:
                _require(np.isnan(matrix).all(), 'failed fit retains unsupported covariance inference.')
        if good:
            _require(np.isfinite(coeff[['coefficient','robust_se','model_se']].to_numpy(float)).all() and n>p and g>2,
                     'accepted fit has nonfinite coefficients or insufficient sample.')
            correction = 1. if pair[0].startswith('mmef/') else g/(g-1)*(n-1)/(n-p)
            _require(_close(meta['finite_sample_correction'],correction), 'covariance correction differs from its sample.')
            if not pair[0].startswith('mmef/'):
                _require(meta['sigma2']>0 and meta['tau2']>=0 and _close(meta['tau2'],meta['lambda']*meta['sigma2'])
                         and _close(meta['icc'],meta['tau2']/(meta['tau2']+meta['sigma2'])), 'variance components disagree.')
        else:
            _require(np.isnan(coeff[['coefficient','robust_se','model_se']].to_numpy(float)).all(),
                     'failed fit retains unsupported coefficient inference.')
        fitted[pair] = (meta,beta,matrices['robust'],terms)
    _require(accepted == all(m['accepted'] for m in models), 'overall acceptance differs from component fits.')
    _check_tables(tables, metadata, fitted, contracts)
    canonical_bytes(metadata)


def _check_tables(tables, metadata, fitted, contracts):
    for name, ids in [('mmef_change',[x[0] for x in SELECTED_FITS[:2]]),
                      ('x6_repeated',[x[0] for x in SELECTED_FITS[2:6]]),
                      ('x6_global_tests',['trajectory/x6/1','trajectory/x6/5'])]:
        _require(list(tables[name].model_id) == ids and tables[name].fit.eq('full').all(), 'table model inventory is inconsistent.')
    for name in ('mmef_change','x6_repeated'):
        for _, row in tables[name].iterrows():
            meta,beta,cov,terms = fitted[(row.model_id,'full')]
            mmef = name == 'mmef_change'
            nfield, efield = ('n_pairs','n_exposed_pairs') if mmef else ('n_observations','n_exposed_observations')
            _require(row[nfield] == meta['n_observations'] and row.n_children == meta['n_children']
                     and _integer(row[efield]) and _integer(row.n_exposed_children)
                     and row.n_exposed_children <= row[efield] <= row[nfield]
                     and row.n_exposed_children <= row.n_children, 'table sample counts disagree.')
            fields = ['estimate','se' if mmef else 'robust_se','ci_low','ci_high','p_value']
            if contracts is not None:
                frame=contracts['mmef/change' if mmef else row.model_id]['frame']
                exposed=(frame['post_before5' if row.design=='before5' else 'post_infancy'].eq(1)
                         if mmef else frame.post_any.eq(1)&frame.tb_age_first_years.lt(meta['threshold']))
                _require(row[efield]==int(exposed.sum()) and row.n_exposed_children==frame.loc[exposed,'pid_child'].nunique(),
                         'exposed counts differ from the prepared sample.')
                if not mmef:
                    reference=frame.tb_first_dx.isna()
                    _require(row.n_reference_observations==int(reference.sum())
                             and row.n_reference_children==frame.loc[reference,'pid_child'].nunique(),
                             'reference counts differ from the prepared sample.')
            if not mmef:
                z = meta['target'] == 'x6_sa_z'
                _require(row.outcome == ('x6_z' if z else 'x6_native')
                         and row.unit == ('reference SD' if z else 'hPa*s/L')
                         and row.tb_age_cutoff_years == meta['threshold'], 'X6 outcome scale or cutoff differs.')
                for role in ('reference','exposed'):
                    _require(_integer(row['n_'+role+'_observations']) and _integer(row['n_'+role+'_children'])
                             and row['n_'+role+'_children'] <= row['n_'+role+'_observations'] <= row.n_observations
                             and row['n_'+role+'_children'] <= row.n_children, 'X6 subgroup counts are invalid.')
            else:
                _require(row.target == 'mmef' and row.unit == 'L/s'
                         and row.design == row.model_id.rsplit('/',1)[1], 'MMEF outcome or design labels differ.')
            if meta['accepted']:
                term = ('post_before5' if row.design == 'before5' else 'post_infancy') if mmef else 'target'
                _normal(row,beta,cov,term,terms,fields)
                if mmef:
                    _require(row.converged == True and _close(row.standardized_exposed_change-row.standardized_reference_change,row.estimate), 'MMEF standardized effects disagree.')
                    if contracts is not None:
                        design = contracts['mmef/change']['designs'][row.design]
                        exposure_terms = ('post_before5','post_at_after5','future_tb') if row.design=='before5' else ('post_infancy','post_age1_before5','post_at_after5','future_tb')
                        reference = np.asarray(design['matrix'],float).mean(axis=0)
                        reference[[terms.index(t) for t in exposure_terms]] = 0.
                        exposed = reference.copy(); exposed[terms.index(term)] = 1.
                        for label,vector in [('reference',reference),('exposed',exposed)]:
                            estimate=vector@beta; uncertainty=np.sqrt(vector@cov@vector)
                            _require(_close(row[['standardized_'+label+'_change','standardized_'+label+'_ci_low','standardized_'+label+'_ci_high']],
                                            [estimate,estimate-1.96*uncertainty,estimate+1.96*uncertainty]),
                                     'MMEF standardized levels differ from the prepared design and fitted covariance.')
                    audit = metadata['mmef_resampling'][row.design]
                    _require(audit['seed'] == 20260903 and audit['requested_replicates'] == 2000
                             and audit['accepted_replicates'] == row.bootstrap_replicates
                             and _integer(row.bootstrap_replicates,1) and row.bootstrap_replicates <= 2000
                             and audit['rank_deficient_replicates'] == 2000-row.bootstrap_replicates
                             and sum(audit['stratum_child_counts'].values()) == row.n_children
                             and audit['leave_one_child_out_estimable'] + audit['leave_one_child_out_rank_deficient'] == row.n_children
                             and row.bootstrap_ci_low <= row.bootstrap_ci_high
                             and row.leave_one_child_out_min <= row.leave_one_child_out_max, 'MMEF resampling evidence is inconsistent.')
                else:
                    _require(_close([row.icc,row.condition_number],[meta['icc'],meta['condition_number']]), 'X6 fit metrics disagree.')
            else:
                if mmef:
                    fields = SCHEMAS['mmef_change'][8:24]
                    _require(row.converged == False, 'failed MMEF fit claims convergence.')
                else:
                    fields += ['icc','condition_number']
                _require(row[fields].isna().all(), 'failed fit retains table inference.')
    precision = metadata['mmef_delta_precision']
    _require(precision['formula'] == 'float64(mmef_post) - float64(mmef)'
             and precision['model_delta_dtype'] == 'float64' and precision['sample_and_designs_unchanged'] is True
             and _integer(precision['changed_numeric_values'])
             and precision['changed_numeric_values'] <= tables['mmef_change'].iloc[0].n_pairs
             and np.isfinite(precision['maximum_absolute_rounding_difference'])
             and precision['maximum_absolute_rounding_difference'] >= 0, 'paired-change precision audit is invalid.')
    desc = tables['mmef_descriptive']
    _require(desc.target.eq('mmef').all() and desc.outcome.eq('MMEF (FEF25–75)').all()
             and desc.unit.eq('L/s').all(), 'descriptive outcome labels differ.')
    _require(list(desc.group) == list(model.MMEF_GROUPS), 'MMEF descriptive categories are incomplete.')
    if contracts is not None:
        frame=contracts['mmef/change']['frame']
        masks=[frame.tb_first_dx.isna(),frame.future_tb.eq(1),frame.post_before5.eq(1),
               frame.post_infancy.eq(1),frame.post_age1_before5.eq(1),frame.post_at_after5.eq(1)]
        delta=frame.mmef_post.astype(float)-frame.mmef.astype(float)
        for (_,row),mask in zip(desc.iterrows(),masks):
            _require(row.n_pairs==int(mask.sum()) and row.n_children==frame.loc[mask,'pid_child'].nunique(),
                     'descriptive counts differ from the prepared sample.')
            expected=[frame.loc[mask,'mmef'].astype(float).mean(),frame.loc[mask,'mmef_post'].astype(float).mean(),
                      delta.loc[mask].mean(),delta.loc[mask].std(ddof=1),delta.loc[mask].median()]
            _require(np.allclose(row[['mean_pre','mean_post','mean_change','sd_change','median_change']].to_numpy(float),expected,
                                rtol=1e-9,atol=1e-10,equal_nan=True), 'descriptive statistics differ from the prepared sample.')
    for _, row in desc.iterrows():
        _require(_integer(row.n_pairs) and _integer(row.n_children) and row.n_children <= row.n_pairs,
                 'descriptive counts are invalid.')
        fields=['mean_pre','mean_post','mean_change','median_change']
        if row.n_pairs:
            _require(np.isfinite(row[fields].to_numpy(float)).all() and _close(row.mean_post-row.mean_pre,row.mean_change), 'descriptive means disagree.')
        else:
            _require(row[fields+['sd_change']].isna().all(), 'empty descriptive group has unsupported values.')
        _require((row.n_pairs<2 and pd.isna(row.sd_change)) or (row.n_pairs>=2 and np.isfinite(row.sd_change) and row.sd_change>=0), 'descriptive SD is invalid.')
    for cutoff in (1,5):
        key=f'trajectory/x6/{cutoff}'
        full,beta,cov,terms=fitted[(key,'full')]
        reduced,rbeta,rcov,rterms=fitted[(key,'reduced')]
        row=tables['x6_global_tests'].loc[tables['x6_global_tests'].model_id.eq(key)].iloc[0]
        predictions=tables['x6_predictions'].loc[tables['x6_predictions'].model_id.eq(key)]
        contrasts=tables['x6_age_contrasts'].loc[tables['x6_age_contrasts'].model_id.eq(key)]
        _require(list(zip(predictions.age_years,predictions.group)) == [(age,group) for age in np.linspace(3.,10.,101) for group in ('No recorded TB',f'First TB <{cutoff} years')]
                 and list(contrasts.age_years) == list(np.arange(3.,11.)), 'X6 trajectory prediction grid is incomplete.')
        for frame in (predictions,contrasts,tables['x6_global_tests'].loc[tables['x6_global_tests'].model_id.eq(key)]):
            _require(frame.fit.eq('full').all() and frame.outcome.eq('x6_sa_z').all()
                     and frame.unit.eq('reference SD').all() and frame.tb_age_cutoff_years.eq(cutoff).all(),
                     'trajectory outcome scale or cutoff differs.')
        _require(row.n_observations == full['n_observations'] == reduced['n_observations']
                 and row.n_children == full['n_children'] == reduced['n_children'], 'trajectory sample counts disagree.')
        if contracts is not None:
            frame=contracts[key]['frame']
            for role,mask in [('exposed',frame.post_any.eq(1)&frame.tb_age_first_years.lt(cutoff)),('reference',frame.tb_first_dx.isna())]:
                _require(row['n_'+role+'_observations']==int(mask.sum())
                         and row['n_'+role+'_children']==frame.loc[mask,'pid_child'].nunique(),
                         'trajectory subgroup counts differ from the prepared sample.')
        if full['accepted'] and reduced['accepted']:
            for prefix,indexes in [('interaction',[9,10,11]),('overall_trajectory',[1,9,10,11])]:
                matrix=cov[np.ix_(indexes,indexes)]; statistic=beta[indexes]@np.linalg.pinv(matrix)@beta[indexes]
                _require(np.linalg.matrix_rank(matrix)==len(indexes) and _close(row[[prefix+'_wald_chi2',prefix+'_df',prefix+'_p']],[statistic,len(indexes),chi2.sf(statistic,len(indexes))]), 'trajectory Wald evidence disagrees.')
            _normal(row,rbeta,rcov,'target',rterms,['parallel_trajectory_difference','parallel_trajectory_robust_se','parallel_trajectory_ci_low','parallel_trajectory_ci_high','parallel_trajectory_p'])
            _verify_predictions(metadata['x6_prediction_standardization'][str(cutoff)], predictions, contrasts, beta, cov)
            for frame,estimate in [(predictions,'predicted_mean'),(contrasts,'difference_tb_minus_reference')]:
                _require(np.isfinite(frame[[estimate,'robust_se','ci_low','ci_high']].to_numpy(float)).all()
                         and (frame.robust_se>=0).all() and _close(frame.ci_low,frame[estimate]-1.96*frame.robust_se)
                         and _close(frame.ci_high,frame[estimate]+1.96*frame.robust_se), 'trajectory interval evidence is inconsistent.')
        else:
            _require(row[SCHEMAS['x6_global_tests'][12:]].isna().all(), 'failed trajectory retains global inference.')
            for frame,estimate in [(predictions,'predicted_mean'),(contrasts,'difference_tb_minus_reference')]:
                _require(frame[[estimate,'robust_se','ci_low','ci_high']].isna().all().all(), 'failed trajectory retains inference.')


def _verify_predictions(state, predictions, contrasts, beta, covariance):
    height, spline = state['height'], state['spline']
    constants = np.asarray([height[k] for k in ('age_center','mean','sd','male_mean')],float)
    height_beta = np.asarray(height['coefficients'],float)
    knots, constraints = np.asarray(spline['knots'],float), np.asarray(spline['centering_constraints'],float)
    _require(np.isfinite(constants).all() and height['age_center']==7.5 and height['sd']>0
             and 0<=height['male_mean']<=1 and height_beta.shape==(5,) and np.isfinite(height_beta).all()
             and knots.shape==(4,) and np.isfinite(knots).all() and (np.diff(knots)>0).all()
             and knots[0]==3 and knots[-1]==10 and constraints.shape==(1,4)
             and np.isfinite(constraints).all() and np.linalg.norm(constraints)>0
             and spline['formula']==model.X6_SPLINE_FORMULA, 'trajectory standardization state is invalid.')
    for frame, field, difference in [(predictions,'predicted_mean',False),(contrasts,'difference_tb_minus_reference',True)]:
        age = frame.age_years.to_numpy(float)
        basis = np.asarray(patsy.cr(age,knots=knots[1:-1],lower_bound=3.,upper_bound=10.,constraints=constraints))
        centered = age-7.5
        expected_height = np.column_stack([np.ones(len(age)),centered,centered**2,centered**3,np.repeat(height['male_mean'],len(age))])@height_beta
        height_z = (expected_height-height['mean'])/height['sd']
        target = np.ones(len(age)) if difference else frame['group'].ne('No recorded TB').to_numpy(float)
        design = np.column_stack([np.ones(len(age)),target,np.zeros((len(age),2)),height_z,np.repeat(height['male_mean'],len(age)),basis,target[:,None]*basis])
        if difference:
            design[:,[0,2,3,4,5,6,7,8]]=0.
        variances = np.einsum('ij,jk,ik->i',design,covariance,design)
        _require((variances>=-1e-10).all() and _close(frame[field],design@beta)
                 and _close(frame.robust_se,np.sqrt(np.maximum(variances,0.))),
                 'trajectory predictions disagree with the fitted coefficient/covariance state.')


def _summary(result, fingerprint):
    return {'fingerprint':fingerprint, 'accepted':result['metadata']['accepted'],
            'status':result['metadata']['status'], 'fitted_models':10,
            'rows':{name:len(frame) for name, frame in result['tables'].items()},
            'scientific_reference_comparison':'requires separate independent validation'}


def verify_result(directory, fingerprint, identity, result):
    """Recompute all selected fits before accepting an existing aggregate cache."""
    _check_result(result)
    receipt = read_json(directory/'receipt.json', 'Supporting repeated receipt')
    if receipt.get('fingerprint') != fingerprint or receipt.get('identity') != identity:
        raise ValidationError('Supporting repeated receipt identity is inconsistent.')
    verify_files(directory, receipt.get('files'), RESULT_FILES)
    if set(receipt['files']) != set(RESULT_FILES):
        raise ValidationError('Supporting repeated output inventory is inconsistent.')
    summary = read_json(directory/'summary.json', 'Supporting repeated summary')
    if (summary != _summary(result, fingerprint) or receipt.get('summary') != summary
            or read_json(directory/'diagnostics.json', 'Supporting repeated diagnostics') != result['metadata']):
        raise ValidationError('Supporting repeated aggregates disagree with recomputed fits.')
    for name, frame in result['tables'].items():
        try:
            observed = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip',
                                   keep_default_na=False, na_values={column:[''] for column in frame
                                       if pd.api.types.is_numeric_dtype(frame[column])})
            pd.testing.assert_frame_equal(observed, frame, check_dtype=False, check_exact=False,
                                          rtol=1e-10, atol=1e-12)
        except (AssertionError, ValueError, TypeError):
            raise ValidationError('Supporting repeated table disagrees with recomputed fit evidence.') from None
    return summary


def fit_selected(root, preparation, work, output, *, resume=False):
    root, prep, work, output = safe_roots(root, preparation, work, output)
    descriptor = contract_descriptor(prep)
    fingerprint, identity = fit_identity(root, descriptor)
    destination = output/fingerprint
    assert_loaded_sources(root)
    with preparation_locks(work, output):
        reused = destination.exists()
        if reused and not resume:
            raise ValidationError('Identical supporting repeated results exist; use --resume to verify them.')
        contracts = {key:load_artifacts(prep, ('M03', 'pft', 'samples', key)) for key in model.CONTRACT_KEYS}
        result = model.fit_selected(contracts)
        _check_result(result, contracts)
        def unchanged():
            assert_loaded_sources(root)
            if (file_hash(Path(__file__)) != _LOADED_HASH
                    or fit_identity(root, contract_descriptor(prep)) != (fingerprint, identity)):
                raise ValidationError('Supporting repeated dependencies changed during execution.')
        if reused:
            summary = verify_result(destination, fingerprint, identity, result)
            unchanged()
            return {**summary, 'resumed':True}
        stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=output))
        try:
            for name, frame in result['tables'].items():
                frame.to_csv(stage/(name+'.csv'), index=False, float_format='%.17g')
            summary = _summary(result, fingerprint)
            write_json(stage/'diagnostics.json', result['metadata'])
            write_json(stage/'summary.json', summary)
            write_json(stage/'receipt.json', {'schema_version':1, 'fingerprint':fingerprint,
                'identity':identity, 'summary':summary,
                'files':{name:file_hash(stage/name) for name in RESULT_FILES}})
            verify_result(stage, fingerprint, identity, result)
            unchanged()
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return {**summary, 'resumed':False}
