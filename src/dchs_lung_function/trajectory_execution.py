"""Fingerprint, fit and verify selected spirometry trajectories privately."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import patsy

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, resolve_paths, runtime_identity,
                        run_preparation, verify_artifacts, write_json, write_pointer)
from .pipeline import ValidationError, read_json
from .models import spirometry_trajectories as _trajectory_model


SOURCES = ('src/dchs_lung_function/models/spirometry_trajectories.py',
           'src/dchs_lung_function/trajectory_execution.py',
           'src/dchs_lung_function/preparation.py', 'src/dchs_lung_function/execution.py',
           'src/dchs_lung_function/pipeline.py', 'run.py')
RESULT_FILES = ('global_tests.csv', 'predictions.csv', 'age_contrasts.csv',
                'coefficients.csv', 'covariance.csv', 'diagnostics.json', 'summary.json')
_LOADED_HASH = file_hash(Path(__file__))
_LOADED_MODEL_HASH = file_hash(Path(_trajectory_model.__file__))


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def trajectory_contracts(preparation_directory):
    """Verify the complete M03 manifest, then identify its four selected contracts."""
    from .models.spirometry_trajectories import DEFAULT_FIT_SELECTION
    directory = Path(preparation_directory).resolve()
    receipt = read_json(directory/'receipt.json', 'Preparation receipt')
    if (not isinstance(receipt, dict) or receipt.get('through') != 'M03'
            or receipt.get('status') != 'prepared'
            or _digest(receipt.get('identity')) != receipt.get('fingerprint')
            or file_hash(directory/'artifacts.json') != receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Trajectory fitting requires complete verified M03 preparation.')
    manifest = verify_artifacts(directory)
    descriptors = {}
    for outcome, threshold in DEFAULT_FIT_SELECTION:
        key = f'trajectory/{outcome}/{threshold}'
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
            raise ValidationError('Trajectory contract refers to an unverified preparation file.')
        descriptors[key] = _digest({'tree':node, 'files':{f:manifest['files'][f] for f in sorted(files)}})
    return {'contracts':descriptors, 'preparation_fingerprint':receipt['fingerprint']}


def trajectory_identity(root, descriptor):
    from .models.spirometry_trajectories import DEFAULT_FIT_SELECTION
    root = Path(root)
    identity = {'schema_version':1, 'contracts':descriptor['contracts'],
                'selection':[list(v) for v in DEFAULT_FIT_SELECTION],
                'source':{n:file_hash(root/n) for n in SOURCES},
                'environment_lock_sha256':file_hash(root/'requirements-preparation.lock.txt'),
                'runtime':runtime_identity()}
    return _digest(identity), identity


def _roots(root, preparation, work, output):
    root, preparation, work, output = [Path(p).resolve() for p in (root, preparation, work, output)]
    def overlap(a, b):
        return a == b or a in b.parents or b in a.parents
    if overlap(work, output) or any(overlap(p, protected)
                                  for p in (work, output) for protected in (root, preparation)):
        raise ValidationError('Trajectory work/output roots must be separate from source and preparation.')
    for p in (work, output):
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root, preparation, work, output


def _accepted_metadata(meta):
    if (not isinstance(meta, dict) or type(meta.get('accepted')) is not bool
            or not isinstance(meta.get('failure_reasons'), list)
            or meta['accepted'] != (not meta['failure_reasons'])
            or meta.get('status') != ('accepted' if meta['accepted'] else 'fit_failed')):
        raise ValidationError('Trajectory acceptance contradicts its fitting status or reasons.')
    return meta['accepted']


def _verify_predictions(model, predictions, contrasts, full):
    """Reconstruct the saved mean/contrast uncertainty from aggregate fit state."""
    state = model['metadata']['standardization']
    height, spline = state['height'], state['spline']
    constants = np.asarray([height[k] for k in ('age_center', 'mean', 'sd', 'male_mean')], float)
    height_beta = np.asarray(height['coefficients'], float)
    knots = np.asarray(spline['knots'], float)
    constraints = np.asarray(spline['centering_constraints'], float)
    if (constants.shape != (4,) or not np.isfinite(constants).all()
            or height['age_center'] != 7.5 or height['sd'] <= 0 or not 0 <= height['male_mean'] <= 1
            or height_beta.shape != (5,) or not np.isfinite(height_beta).all()
            or knots.shape != (4,) or not np.isfinite(knots).all() or not (np.diff(knots) > 0).all()
            or knots[0] != 5 or knots[-1] != 10 or constraints.shape != (1, 4)
            or not np.isfinite(constraints).all() or not np.linalg.norm(constraints) > 0
            or spline['formula'] != _trajectory_model.SPLINE_FORMULA):
        raise ValidationError('Trajectory standardization state is invalid.')
    for table, estimate, difference in [(predictions, 'predicted_mean', False),
                                        (contrasts, 'difference_tb_minus_reference', True)]:
        rows = table.loc[table.model_id.eq(model['model_id'])]
        ages = rows.age_years.to_numpy(float)
        basis = np.asarray(patsy.cr(ages, knots=knots[1:-1], lower_bound=5., upper_bound=10.,
                                    constraints=constraints))
        centered = ages-7.5
        expected_height = np.column_stack([np.ones(len(ages)), centered, centered**2,
            centered**3, np.repeat(height['male_mean'], len(ages))]) @ height_beta
        height_z = (expected_height-height['mean'])/height['sd']
        target = np.ones(len(ages)) if difference else rows['group'].ne('No recorded TB').to_numpy(float)
        design = np.column_stack([np.ones(len(ages)), target, np.zeros((len(ages), 2)),
                                  height_z, np.repeat(height['male_mean'], len(ages)),
                                  basis, target[:, None]*basis])
        if difference:
            design[:, [0, 2, 3, 4, 5, 6, 7, 8]] = 0.
        means = design @ full['beta']
        variances = np.einsum('ij,jk,ik->i', design, full['covariance'], design)
        if ((variances < -1e-10).any()
                or not np.allclose(rows[estimate], means, atol=1e-10, rtol=1e-10)
                or not np.allclose(rows.robust_se, np.sqrt(np.maximum(variances, 0.)), atol=1e-10, rtol=1e-10)):
            raise ValidationError('Trajectory predictions or contrasts disagree with fitted aggregate state.')


def _verify_result(directory, fingerprint, identity):
    """Check complete file inventories and agreement between results and statuses."""
    receipt = read_json(directory/'receipt.json', 'Trajectory receipt')
    if (not isinstance(receipt, dict) or receipt.get('schema_version') != 1
            or receipt.get('fingerprint') != fingerprint or receipt.get('identity') != identity):
        raise ValidationError('Trajectory completion identity is inconsistent.')
    files = receipt.get('files')
    if not isinstance(files, dict) or set(files) != set(RESULT_FILES):
        raise ValidationError('Trajectory completion has an incomplete output inventory.')
    for name, expected in files.items():
        path = directory/name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValidationError('A saved trajectory result is missing or changed.')
    summary = read_json(directory/'summary.json', 'Trajectory summary')
    diagnostics = read_json(directory/'diagnostics.json', 'Trajectory diagnostics')
    if not isinstance(summary, dict) or not isinstance(diagnostics, dict):
        raise ValidationError('Trajectory summary and diagnostics must be objects.')
    from .models.spirometry_trajectories import DEFAULT_FIT_SELECTION
    selected = [f'{outcome}/{threshold}' for outcome, threshold in DEFAULT_FIT_SELECTION]
    models = diagnostics.get('models')
    if (not isinstance(models, list) or any(not isinstance(m, dict) for m in models)
            or [m.get('model_id') for m in models] != selected):
        raise ValidationError('Trajectory diagnostics lack the selected analysis inventory.')
    try:
        tables = {n:pd.read_csv(directory/(n+'.csv'), float_precision='round_trip') for n in
                  ('global_tests', 'predictions', 'age_contrasts', 'coefficients', 'covariance')}
        tests, predictions, contrasts, coefficients, covariance = [tables[n] for n in tables]
        expected_predictions, expected_contrasts, expected_coefficients, expected_covariance = [], [], [], []
        accepted_models = []
        for model, (outcome, threshold) in zip(models, DEFAULT_FIT_SELECTION):
            model_id = model['model_id']
            spec = _trajectory_model.fit_spec(outcome, threshold)
            if (model.get('target'), model.get('threshold')) != (outcome, threshold):
                raise ValidationError('Trajectory labels disagree with the selected analysis.')
            accepted = _accepted_metadata(model.get('metadata'))
            for key in ('target', 'threshold', 'outcome', 'outcome_label', 'unit'):
                if model['metadata'].get(key) != spec[key]:
                    raise ValidationError('Trajectory metadata differs from the selected outcome specification.')
            parts = model.get('fits')
            if not isinstance(parts, dict) or set(parts) != {'full', 'reduced'}:
                raise ValidationError('Trajectory diagnostics lack full and reduced fits.')
            part_accepted = [_accepted_metadata(parts[n]) for n in ('full', 'reduced')]
            if accepted and not all(part_accepted):
                raise ValidationError('An accepted trajectory analysis contains a failed fit.')
            accepted_models.append(accepted)
            for table in tables.values():
                rows = table.loc[table.model_id.eq(model_id)]
                if not rows.target.eq(outcome).all() or not rows.threshold.eq(threshold).all():
                    raise ValidationError('Trajectory result labels disagree with their model.')
            test = tests.loc[tests.model_id.eq(model_id)]
            if len(test) != 1:
                raise ValidationError('Trajectory global tests must have one row per analysis.')
            for table in (tests, predictions, contrasts):
                rows = table.loc[table.model_id.eq(model_id)]
                for field, value in [('outcome', spec['outcome']), ('outcome_label', spec['outcome_label']),
                                     ('unit', spec['unit']), ('tb_age_cutoff_years', threshold)]:
                    if not rows[field].eq(value).all():
                        raise ValidationError('Trajectory table outcome labels or units are inconsistent.')
            count_fields = ['n_observations', 'n_children', *[f'n_{role}_{unit}'
                for role in ('exposed', 'reference', 'future', 'later') for unit in ('observations', 'children')]]
            row = test.iloc[0]
            if any(model['metadata']['counts'].get(k) != row[k] for k in count_fields):
                raise ValidationError('Trajectory metadata and global sample counts disagree.')
            all_counts = row[count_fields].to_numpy(float)
            if (not np.isfinite(all_counts).all() or (all_counts < 0).any()
                    or (all_counts != np.floor(all_counts)).any()
                    or not 0 < row.n_children <= row.n_observations
                    or sum(row[f'n_{role}_observations'] for role in ('exposed', 'reference', 'future', 'later')) != row.n_observations
                    or any(row[f'n_{role}_children'] > min(row.n_children, row[f'n_{role}_observations'])
                           for role in ('exposed', 'reference', 'future', 'later'))):
                raise ValidationError('Trajectory sample counts do not form valid observation groups.')
            saved_parts = {}
            for part, terms in [('full', ['Intercept', 'target', 'later', 'future_tb', 'height_z', 'sex:Male',
                                  *[f'age_spline_{i}' for i in range(1, 4)],
                                  *[f'target:age_spline_{i}' for i in range(1, 4)]]),
                                ('reduced', ['Intercept', 'target', 'later', 'future_tb', 'height_z', 'sex:Male',
                                  *[f'age_spline_{i}' for i in range(1, 4)]])]:
                expected_coefficients.extend((model_id, part, term) for term in terms)
                b = coefficients.loc[coefficients.model_id.eq(model_id) & coefficients.fit.eq(part)]
                if list(b.term) != terms:
                    raise ValidationError('Trajectory coefficient terms are missing or reordered.')
                matrices = {}
                for cov_type in ('robust', 'model'):
                    expected_covariance.extend((model_id, part, cov_type, a, b) for a in terms for b in terms)
                    c = covariance.loc[covariance.model_id.eq(model_id) & covariance.fit.eq(part)
                                       & covariance.covariance_type.eq(cov_type)]
                    if list(zip(c.row_term, c.column_term)) != [(a, b) for a in terms for b in terms]:
                        raise ValidationError('Trajectory covariance has missing or duplicate coordinates.')
                    matrix = c['covariance'].to_numpy(float).reshape(len(terms), len(terms))
                    matrices[cov_type] = matrix
                    if parts[part]['accepted']:
                        if (not np.isfinite(matrix).all()
                                or not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-8)
                                or (np.diag(matrix) <= 0).any()
                                or np.linalg.eigvalsh((matrix+matrix.T)/2).min() < -1e-8*max(1., np.max(abs(matrix)))):
                            raise ValidationError('An accepted trajectory covariance is invalid.')
                if parts[part]['accepted'] and not np.isfinite(b.coefficient.to_numpy(float)).all():
                    raise ValidationError('An accepted trajectory coefficient is nonfinite.')
                meta = parts[part]
                if meta.get('terms') != terms or meta.get('n_parameters') != len(terms):
                    raise ValidationError('Trajectory fit metadata disagrees with its coefficient inventory.')
                if meta['accepted']:
                    values = np.asarray([meta[k] for k in ('n_observations', 'n_children', 'lambda',
                        'sigma2', 'tau2', 'icc', 'objective', 'condition_number', 'finite_sample_correction')], float)
                    n, g, lam, sigma2, tau2, icc, _, condition, correction = values
                    if (not np.isfinite(values).all() or n != int(n) or g != int(g)
                            or n <= len(terms) or not 2 < g <= n or lam < 0 or sigma2 <= 0 or tau2 < 0
                            or condition < 1 or not 0 <= icc < 1
                            or not np.isclose(tau2, lam*sigma2, rtol=1e-10, atol=1e-12)
                            or not np.isclose(icc, tau2/(tau2+sigma2), rtol=1e-10, atol=1e-12)
                            or not np.isclose(correction, g/(g-1)*(n-1)/(n-len(terms)), rtol=1e-12, atol=1e-12)):
                        raise ValidationError('Trajectory fit variance, counts or covariance correction is inconsistent.')
                saved_parts[part] = {'beta':b.coefficient.to_numpy(float), 'covariance':matrices['robust']}
            for age in np.linspace(5., 10., 101):
                for group in ('No recorded TB', f'First TB <{threshold} years'):
                    expected_predictions.append((model_id, float(age), group))
            expected_contrasts.extend((model_id, float(age)) for age in range(5, 11))
            if accepted:
                from scipy.stats import chi2, norm
                row = test.iloc[0]
                numeric = ['n_observations', 'n_children', 'n_exposed_observations', 'n_exposed_children',
                           'n_reference_observations', 'n_reference_children', 'interaction_wald_chi2',
                           'interaction_df', 'interaction_p', 'overall_trajectory_wald_chi2',
                           'overall_trajectory_df', 'overall_trajectory_p', 'parallel_trajectory_difference',
                           'parallel_trajectory_robust_se', 'parallel_trajectory_ci_low',
                           'parallel_trajectory_ci_high', 'parallel_trajectory_p', 'icc',
                           'condition_number', 'random_intercept_variance', 'residual_variance']
                if not np.isfinite(row[numeric].to_numpy(float)).all():
                    raise ValidationError('An accepted trajectory global summary is nonfinite.')
                if (row.interaction_df != 3 or row.overall_trajectory_df != 4
                        or row.residual_variance <= 0 or row.random_intercept_variance < 0
                        or not 0 <= row.icc < 1 or row.parallel_trajectory_robust_se <= 0
                        or any(not 0 <= row[k] <= 1 for k in
                               ('interaction_p', 'overall_trajectory_p', 'parallel_trajectory_p'))):
                    raise ValidationError('An accepted trajectory has invalid inferential diagnostics.')
                count_fields = ['n_observations', 'n_children', 'n_exposed_observations',
                                'n_exposed_children', 'n_reference_observations', 'n_reference_children']
                counts = row[count_fields].to_numpy(float)
                if ((counts < 0).any() or (counts != np.floor(counts)).any()
                        or not 2 < row.n_children <= row.n_observations
                        or not 0 <= row.n_exposed_children <= min(row.n_children, row.n_exposed_observations)
                        or not 0 <= row.n_reference_children <= min(row.n_children, row.n_reference_observations)
                        or row.n_exposed_observations+row.n_reference_observations > row.n_observations
                        or row.n_exposed_children+row.n_reference_children > row.n_children
                        or any(parts[part]['n_observations'] != row.n_observations
                               or parts[part]['n_children'] != row.n_children for part in ('full', 'reduced'))):
                    raise ValidationError('Trajectory global sample counts disagree with fitted samples.')
                for field, source in [('icc', 'icc'), ('condition_number', 'condition_number'),
                                      ('residual_variance', 'sigma2'), ('random_intercept_variance', 'tau2')]:
                    if not np.isclose(row[field], parts['full'][source], rtol=1e-10, atol=1e-12):
                        raise ValidationError('Trajectory global diagnostics disagree with the full fit.')
                beta = saved_parts['full']['beta']; cov = saved_parts['full']['covariance']
                for prefix, indexes in [('interaction', [9, 10, 11]), ('overall_trajectory', [1, 9, 10, 11])]:
                    selected_cov = cov[np.ix_(indexes, indexes)]; selected_beta = beta[indexes]
                    degrees = int(np.linalg.matrix_rank(selected_cov))
                    statistic = float(selected_beta @ np.linalg.pinv(selected_cov) @ selected_beta)
                    if (row[prefix+'_wald_chi2'] < 0 or row[prefix+'_df'] != degrees
                            or not np.isclose(row[prefix+'_wald_chi2'], statistic, atol=1e-9, rtol=1e-9)
                            or not np.isclose(row[prefix+'_p'], chi2.sf(statistic, degrees), atol=1e-10, rtol=1e-9)):
                        raise ValidationError('Trajectory Wald results disagree with fitted coefficients and covariance.')
                reduced = saved_parts['reduced']; difference = reduced['beta'][1]
                se = float(np.sqrt(reduced['covariance'][1, 1]))
                for field, value in [('parallel_trajectory_difference', difference),
                    ('parallel_trajectory_robust_se', se), ('parallel_trajectory_ci_low', difference-1.96*se),
                    ('parallel_trajectory_ci_high', difference+1.96*se),
                    ('parallel_trajectory_p', 2*norm.sf(abs(difference/se)))]:
                    if not np.isclose(row[field], value, rtol=1e-9, atol=1e-10):
                        raise ValidationError('Parallel-trajectory inference disagrees with the reduced fit.')
                for table, estimate, se in [(predictions, 'predicted_mean', 'robust_se'),
                                             (contrasts, 'difference_tb_minus_reference', 'robust_se')]:
                    rows = table.loc[table.model_id.eq(model_id)]
                    values = rows[[estimate, se, 'ci_low', 'ci_high']].to_numpy(float)
                    if (not np.isfinite(values).all() or (rows[se] < 0).any()
                            or not np.allclose(rows.ci_low, rows[estimate]-1.96*rows[se], atol=1e-10, rtol=1e-10)
                            or not np.allclose(rows.ci_high, rows[estimate]+1.96*rows[se], atol=1e-10, rtol=1e-10)):
                        raise ValidationError('Trajectory prediction or contrast intervals are inconsistent.')
                _verify_predictions(model, predictions, contrasts, saved_parts['full'])
            else:
                global_inference = ['interaction_wald_chi2', 'interaction_df', 'interaction_p',
                    'overall_trajectory_wald_chi2', 'overall_trajectory_df', 'overall_trajectory_p',
                    'parallel_trajectory_difference', 'parallel_trajectory_robust_se',
                    'parallel_trajectory_ci_low', 'parallel_trajectory_ci_high', 'parallel_trajectory_p',
                    'icc', 'condition_number', 'random_intercept_variance', 'residual_variance']
                if not test[global_inference].isna().all().all():
                    raise ValidationError('A failed trajectory analysis contains finite inference.')
                for table, estimate in [(predictions, 'predicted_mean'), (contrasts, 'difference_tb_minus_reference')]:
                    rows = table.loc[table.model_id.eq(model_id)]
                    if not rows[[estimate, 'robust_se', 'ci_low', 'ci_high']].isna().all().all():
                        raise ValidationError('A failed trajectory analysis contains prediction inference.')
        if (list(tests.model_id) != selected
                or list(zip(predictions.model_id, predictions.age_years, predictions['group'])) != expected_predictions
                or list(zip(contrasts.model_id, contrasts.age_years)) != expected_contrasts
                or list(zip(coefficients.model_id, coefficients.fit, coefficients.term)) != expected_coefficients
                or list(zip(covariance.model_id, covariance.fit, covariance.covariance_type,
                            covariance.row_term, covariance.column_term)) != expected_covariance):
            raise ValidationError('Trajectory aggregate tables lack the complete selected row inventory.')
        accepted = all(accepted_models)
        expected = {'analyses':len(selected), 'fitted_models':2*len(selected),
                    'prediction_rows':len(expected_predictions), 'age_contrast_rows':len(expected_contrasts),
                    'accepted':accepted, 'status':'accepted' if accepted else 'fit_failed',
                    'fingerprint':fingerprint, 'through':'M06'}
        if (any(summary.get(k) != value for k, value in expected.items())
                or type(summary.get('accepted')) is not bool or receipt.get('summary') != summary
                or receipt.get('status') != summary['status']):
            raise ValidationError('Trajectory receipt, diagnostics and summary disagree.')
    except (ValueError, TypeError, KeyError, AttributeError, patsy.PatsyError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError('Trajectory aggregate structure is invalid.') from None
    return summary


def fit_prepared_trajectories(root, preparation_directory, work, output, *, resume=False):
    """Fit four full/reduced pairs and retain aggregate results atomically."""
    from .models.spirometry_trajectories import DEFAULT_FIT_SELECTION, fit_trajectory
    root, prep, work, output = _roots(root, preparation_directory, work, output)
    model_source = 'src/dchs_lung_function/models/spirometry_trajectories.py'
    if (file_hash(Path(_trajectory_model.__file__)) != _LOADED_MODEL_HASH
            or file_hash(root/model_source) != _LOADED_MODEL_HASH
            or file_hash(root/'src/dchs_lung_function/trajectory_execution.py') != _LOADED_HASH):
        raise ValidationError('Loaded trajectory source differs from the requested source; restart with matching code.')
    descriptor = trajectory_contracts(prep)
    fingerprint, identity = trajectory_identity(root, descriptor)
    destination = output/fingerprint
    def unchanged():
        if (file_hash(Path(__file__)) != _LOADED_HASH
                or file_hash(Path(_trajectory_model.__file__)) != _LOADED_MODEL_HASH
                or trajectory_identity(root, trajectory_contracts(prep)) != (fingerprint, identity)):
            raise ValidationError('Trajectory model, preparation or runtime changed during execution.')
    with preparation_locks(work, output):
        if destination.exists():
            if not resume:
                raise ValidationError('An identical trajectory run exists; use --resume to verify it.')
            summary = _verify_result(destination, fingerprint, identity)
            unchanged()
            return {**summary, 'resumed':True}
        tables = {n:[] for n in ('global_tests', 'predictions', 'age_contrasts', 'coefficients', 'covariance')}
        models = []
        for outcome, threshold in DEFAULT_FIT_SELECTION:
            key = f'trajectory/{outcome}/{threshold}'
            contract = load_artifacts(prep, ('M03', 'pft', 'samples', key))
            if (contract['audit'].get('outcome'), contract['audit'].get('threshold')) != (outcome, threshold):
                raise ValidationError('Trajectory contract does not match the selected outcome and cutoff.')
            result = fit_trajectory(contract)
            labels = {'model_id':f'{outcome}/{threshold}', 'target':outcome, 'threshold':threshold}
            def labelled(frame):
                frame = frame.copy()
                for name, value in reversed(list(labels.items())):
                    if name in frame:
                        if not frame[name].eq(value).all():
                            raise ValidationError('Trajectory result labels do not match the requested fit.')
                        frame = frame.drop(columns=name)
                    frame.insert(0, name, value)
                return frame
            for name in ('global_tests', 'predictions', 'age_contrasts'):
                tables[name].append(labelled(getattr(result, name)))
            for part in ('full', 'reduced'):
                fitted = getattr(result, part)
                tables['coefficients'].append(labelled(pd.DataFrame({'fit':part,
                    'term':fitted.coefficients.index, 'coefficient':fitted.coefficients.to_numpy()})))
                for cov_type, matrix in [('robust', fitted.covariance), ('model', fitted.model_covariance)]:
                    tables['covariance'].append(labelled(pd.DataFrame([
                        {'fit':part, 'covariance_type':cov_type, 'row_term':a, 'column_term':b,
                         'covariance':matrix.loc[a, b]} for a in matrix.index for b in matrix.columns])))
            models.append({**labels, 'metadata':result.metadata,
                           'fits':{n:getattr(result, n).metadata for n in ('full', 'reduced')}})
        accepted = all(_accepted_metadata(m['metadata']) for m in models)
        combined = {n:pd.concat(frames, ignore_index=True) for n, frames in tables.items()}
        summary = {'through':'M06', 'fingerprint':fingerprint, 'analyses':len(models),
                   'fitted_models':2*len(models), 'prediction_rows':len(combined['predictions']),
                   'age_contrast_rows':len(combined['age_contrasts']), 'accepted':accepted,
                   'status':'accepted' if accepted else 'fit_failed',
                   'scientific_reference_comparison':'requires separate independent validation'}
        staged = Path(tempfile.mkdtemp(prefix='.staging-', dir=output))
        try:
            for name, frame in combined.items():
                frame.to_csv(staged/(name+'.csv'), index=False, float_format='%.17g')
            write_json(staged/'diagnostics.json', {'models':models})
            write_json(staged/'summary.json', summary)
            receipt = {'schema_version':1, 'fingerprint':fingerprint, 'identity':identity,
                       'status':summary['status'], 'summary':summary,
                       'preparation_fingerprint':descriptor['preparation_fingerprint'],
                       'completed_at':datetime.now(timezone.utc).isoformat(),
                       'files':{n:file_hash(staged/n) for n in RESULT_FILES}}
            write_json(staged/'receipt.json', receipt)
            _verify_result(staged, fingerprint, identity)
            unchanged()
            os.replace(staged, destination)
        finally:
            if staged.exists():
                shutil.rmtree(staged)
        return {**summary, 'resumed':False}


def run_trajectories(root, config_file, analysis_file, sampling_file, *, jobs=1, resume=False):
    """Run cumulative M01-M06; prior modules must pass before trajectories publish."""
    from .binary_execution import run_binary
    sampling_hash = file_hash(Path(sampling_file))
    binary = run_binary(root, config_file, analysis_file, sampling_file, jobs=jobs, resume=resume)
    if not binary['accepted'] or binary['models'] != 28 or binary.get('primary_models_verified') != 12:
        raise ValidationError('Cumulative M06 requires all primary and selected binary models to pass.')
    configuration = read_json(config_file, 'Local configuration')
    _, directories = resolve_paths(configuration, config_file)
    prep = directories['work']/'preparation_runs'/binary['preparation_fingerprint']
    summary = fit_prepared_trajectories(root, prep, directories['work']/'trajectory_models',
                                       directories['outputs']/'trajectory_models', resume=resume)
    verified = run_preparation(root, config_file, analysis_file, 'M03', resume=True)
    if (verified['fingerprint'] != binary['preparation_fingerprint']
            or file_hash(Path(sampling_file)) != sampling_hash):
        raise ValidationError('Preparation or sampling dependencies changed during trajectory execution.')
    summary = {**summary, 'preparation_fingerprint':binary['preparation_fingerprint'],
               'primary_models_verified':12, 'binary_models_verified':28}
    if summary['accepted']:
        write_pointer(directories['work']/'trajectory_current.json', summary)
    return summary
