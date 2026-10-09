"""Six individual mechanics/first-TB joint models, specification version 1.0.0.

Each Gaussian outcome has its own shared child effect and first-event hazard.
Prepared outcomes are adverse-oriented; native-unit reporting must reverse
that orientation after residual rescaling. Graph construction neither samples
nor reads or writes data. Child coordinates are positional, not identifiers.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

MODEL_ID = 'MECHANICS-JOINT'
SPEC_VERSION = '1.0.0'
HAZARD_BREAKS = (0., 1., 3., 5., 8., 10., 20.)
HAZARD_LABELS = ('0-<1', '1-<3', '3-<5', '5-<8', '8-<10', '10-<20')
BASELINE_HAZARDS = (.045, .018, .006, .006, .006, .006)
POST_TERMS = ('post_before5', 'post_infancy', 'post_at_after5')
TARGETS = {
    'R': ('resistance', 1., 'Model-derived resistance', 'cmH2O*s/L'),
    'C': ('compliance', -1., 'Model-derived compliance', 'L/cmH2O'),
    'ReE': ('ree', 1., 'End-expiratory resistance', 'hPa*s/L'),
    'ReI': ('rei', 1., 'End-inspiratory resistance', 'hPa*s/L'),
    'XeE': ('xee', -1., 'End-expiratory reactance', 'hPa*s/L'),
    'XeI': ('xei', -1., 'End-inspiratory reactance', 'hPa*s/L'),
}
DEFAULT_TARGETS = tuple(TARGETS)
CONTRASTS = {
    'TB <1': 'shift_infancy',
    'TB 1-<5': 'shift_1to5',
    'TB >=5': 'shift_late',
    'TB <5 weighted': 'shift_lt5_child_weighted',
    'Any TB weighted': 'shift_any_child_weighted',
    'Infancy minus 1-<5': 'shift_infancy_additional',
    'Later minus infancy': 'shift_late_minus_infancy',
    'Later minus 1-<5': 'shift_late_minus_middle',
}
POSTERIOR_VARIABLES = (
    'child_z', 'tau_child', 'shared_outcome', 'shift_1to5',
    'shift_infancy_additional', 'shift_late', 'sigma_outcome',
    'log_baseline_hazard', 'assoc_outcome', 'hazard_male',
    'shift_infancy', 'shift_lt5_child_weighted', 'shift_any_child_weighted',
    'shift_late_minus_infancy', 'shift_late_minus_middle',
    'hazard_hr_per_between_child_sd',
)


def fit_spec(target: str) -> dict[str, Any]:
    """JSON-safe selected target, prior, sampling, and reporting specification."""
    if target not in TARGETS:
        raise ValueError('Unsupported individual mechanics target.')
    column, orientation, outcome, unit = TARGETS[target]
    return {
        'model_id': MODEL_ID, 'specification_version': SPEC_VERSION, 'target': target,
        'column': column, 'orientation': orientation, 'outcome': outcome, 'unit': unit,
        'reporting_multiplier': 1000. if target == 'C' else 1.,
        'reporting_unit': 'mL/cmH2O' if target == 'C' else unit,
        'measurement_date': 'date_of_test_ilf2', 'nuisance_formula': '1+C(visit_label)+height_z+male',
        'nuisance_training': 'pre/no-TB observations for this outcome',
        'residual_scale': 'training-residual SD with ddof equal to nuisance design columns',
        'post_diagnosis_terms': list(POST_TERMS),
        'normal_priors_mean_sd': {
            'child_z': [0., 1.], 'shift_1to5': [0., .75],
            'shift_infancy_additional': [0., .75], 'shift_late': [0., .75],
            'assoc_outcome': [0., .75], 'hazard_male': [0., .75],
        },
        'half_normal_prior_sd': {'tau_child': 1., 'sigma_outcome': 1.},
        'log_baseline_hazard_prior': {'mean': np.log(BASELINE_HAZARDS).tolist(), 'sd': 1.25},
        'hazard_breaks_years': list(HAZARD_BREAKS), 'hazard_interval_labels': list(HAZARD_LABELS),
        'latent_parameterization': 'shared_outcome = child_z*tau_child; noncentered',
        'gaussian_mean': 'shared_outcome + shift_1to5*before5 + shift_infancy_additional*infancy + shift_late*late',
        'hazard_log_mean': 'log_baseline_hazard + assoc_outcome*shared_outcome + hazard_male*male',
        'hazard_hr_transform': 'exp(assoc_outcome*tau_child)',
        'native_transform': 'standardized contrast * target residual scale / orientation',
        'reported_transform': 'native contrast * reporting_multiplier',
        'contrasts': dict(CONTRASTS), 'table_contrasts': list(CONTRASTS)[:5],
        'contrast_weights': 'distinct children with post-TB observations of the selected outcome',
        'posterior_variables': list(POSTERIOR_VARIABLES),
        'participant_posterior_variables': ['child_z', 'shared_outcome'],
        'runtime': {'pymc': '5.25.1', 'pytensor': '2.31.7', 'arviz': '0.22.0'},
        'sampling': {'draws': 1200, 'tune': 1600, 'chains': 4, 'cores': 2,
                     'target_accept': .95, 'random_seed': 2026092612+DEFAULT_TARGETS.index(target),
                     'init': 'jitter+adapt_diag', 'initvals': None,
                     'progressbar': False, 'return_inferencedata': True},
        'diagnostic_criteria': {'max_rhat': 1.01, 'min_bulk_ess': 400, 'divergences': 0,
                                'scope': 'all posterior variables including child effects'},
        'additional_diagnostics': ['tail_ess', 'bfmi', 'tree_depth', 'mcse', 'trace_inspection'],
    }


def _array(values, shape, label, *, binary=False, integer=False):
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label} must be numeric.') from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label} has an invalid shape or nonfinite values.')
    if binary and not np.isin(array, [0., 1.]).all():
        raise ValueError(f'{label} must be binary.')
    if integer and not np.equal(array, np.floor(array)).all():
        raise ValueError(f'{label} must contain integers.')
    return array


def validate_contract(contract: Mapping, target: str) -> dict[str, Any]:
    """Check the M03 single-process contract and return aggregate metadata only."""
    spec = fit_spec(target)
    try:
        audit = contract['audit']
        if audit['family'] != 'mechanics' or audit['target'] != spec['column'] or audit['covariates']:
            raise ValueError('The prepared contract is not the selected individual mechanics model.')
        if set(contract['frames']) != {'target'} or set(contract['designs']) != {'target'}:
            raise ValueError('Individual mechanics models require exactly one observed process.')
        children, frame, event = contract['children'], contract['frames']['target'], contract['event']
        n, m = len(children), len(frame)
        if n < 1 or m < 1:
            raise ValueError('The mechanics model requires children and observed measurements.')
        indices = _array(children['child_index'], (n,), 'Child indices', integer=True)
        if not np.array_equal(indices, np.arange(n)):
            raise ValueError('Child coordinates do not match the prepared child order.')
        male = _array(children['male'], (n,), 'Child sex', binary=True)
        idx = _array(frame['child_index'], (m,), 'Measurement child indices', integer=True).astype(int)
        if (idx < 0).any() or (idx >= n).any():
            raise ValueError('A measurement child index is outside the prepared child coordinates.')
        if children['pid_child'].isna().any() or children['pid_child'].duplicated().any() or frame['pid_child'].isna().any():
            raise ValueError('The mechanics child linkage is missing or duplicated.')
        if not np.array_equal(children['pid_child'].to_numpy()[idx], frame['pid_child'].to_numpy()):
            raise ValueError('Measurement child indexing does not match its linkage.')
        if list(contract['hazard_terms']) != ['male'] or list(contract['post_diagnosis_terms']) != list(POST_TERMS):
            raise ValueError('Mechanics hazard or post-diagnosis terms differ from the selected specification.')
        if not np.array_equal(_array(contract['hazard_design'], (n, 1), 'Hazard design')[:, 0], male):
            raise ValueError('The hazard design does not match child sex.')
        z = _array(frame[list(POST_TERMS)], (m, 3), 'Post-diagnosis indicators', binary=True)
        if (z[:, 1] > z[:, 0]).any() or (z[:, 0]+z[:, 2] > 1).any():
            raise ValueError('Infancy must be nested within before-five TB, with later TB separate.')
        if not np.array_equal(z, _array(contract['post_diagnosis_designs']['target'], (m, 3), 'Post-diagnosis design', binary=True)):
            raise ValueError('Post-diagnosis indicators and design disagree.')
        post = _array(frame['post_any'], (m,), 'Post-TB indicator', binary=True)
        if not np.array_equal(post, z[:, 0]+z[:, 2]):
            raise ValueError('Post-TB indicators do not reconcile.')
        y = _array(frame['y_std'], (m,), 'Standardized outcome')
        values = _array(frame['outcome_value'], (m,), 'Oriented outcome')
        raw = _array(frame[spec['column']], (m,), 'Native mechanics outcome')
        if not np.array_equal(values, raw*spec['orientation']):
            raise ValueError('Prepared mechanics values do not have the selected adverse orientation.')
        outcome_audit = audit['outcomes']['target']
        scale = float(outcome_audit['scale'])
        design = contract['designs']['target']
        terms = list(outcome_audit['nuisance_terms'])
        x = _array(design, (m, len(terms)), 'Nuisance design')
        if list(design.columns) != terms:
            raise ValueError('Nuisance design terms do not match the preparation audit.')
        beta = _array(outcome_audit['nuisance_coefficients'], (len(terms),), 'Nuisance coefficients')
        if not np.isfinite(scale) or scale <= 0 or not np.allclose(y, (values-x@beta)/scale, rtol=1e-12, atol=1e-12):
            raise ValueError('Standardized mechanics outcomes do not reconcile with the nuisance fit and scale.')
        process_counts = {'n_observations': m, 'n_children': int(np.unique(idx).size)}
        if any(outcome_audit[key] != value for key, value in process_counts.items()) or outcome_audit['n_training'] != int((post == 0).sum()):
            raise ValueError('Mechanics process sample counts do not match the preparation audit.')
        breaks = np.asarray(HAZARD_BREAKS)
        if not np.array_equal(event['breaks'], breaks):
            raise ValueError('The first-TB hazard intervals differ from the selected specification.')
        stop = _array(event['stop'], (n,), 'Event follow-up')
        events = _array(event['event'], (n,), 'First-TB event', binary=True)
        exposure = _array(event['exposure'], (n, 6), 'Hazard exposure')
        interval = _array(event['event_interval'], (n, 6), 'Hazard event interval', binary=True)
        expected_exposure = np.maximum(0., np.minimum(stop[:, None], breaks[1:])-breaks[:-1])
        expected_interval = events[:, None]*((stop[:, None] >= breaks[:-1]) & (stop[:, None] < breaks[1:]))
        if (stop <= 0).any() or (stop >= breaks[-1]).any() or not np.allclose(exposure, expected_exposure, rtol=0., atol=1e-12):
            raise ValueError('Event follow-up and hazard exposure do not reconcile.')
        if not np.array_equal(interval, expected_interval) or not np.array_equal(interval.sum(axis=1), events):
            raise ValueError('First-TB event intervals do not reconcile.')
        groups = [set(idx[z[:, 1] == 1]), set(idx[(z[:, 0] == 1) & (z[:, 1] == 0)]), set(idx[z[:, 2] == 1])]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('A mechanics child has inconsistent diagnosis-age groups.')
        weights = _array(contract['weights'], (3,), 'Exposed-child weights', integer=True)
        if not np.array_equal(weights, [len(group) for group in groups]) or weights[:2].sum() <= 0 or weights[2] <= 0:
            raise ValueError('Mechanics contrast weights require the selected early- and later-TB child support.')
        if audit['joint_children'] != n or audit['tb_events'] != int(events.sum()) or not np.array_equal(audit['weights'], weights):
            raise ValueError('Joint sample counts or contrast weights disagree with the preparation audit.')
        return {'model_id': MODEL_ID, 'specification_version': SPEC_VERSION, 'target': target,
                'joint_children': n, 'tb_events': int(events.sum()),
                'process_counts': {'target': process_counts}, 'weights': weights.astype(int).tolist(),
                'target_scale': scale, 'orientation': spec['orientation']}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('The mechanics contract is missing required M03 fields.') from exc


def build_model(contract: Mapping, target: str):
    """Build the exact selected PyMC graph without fitting or modifying inputs."""
    validated = validate_contract(contract, target)
    import pymc as pm
    import pytensor.tensor as pt

    frame, children, event = contract['frames']['target'], contract['children'], contract['event']
    weights = np.asarray(validated['weights'], dtype=int)
    with pm.Model(coords={'child': np.arange(len(children)), 'hazard_interval': np.arange(6)}) as model:
        child_z = pm.Normal('child_z', 0, 1, dims='child')
        tau = pm.HalfNormal('tau_child', 1)
        latent = pm.Deterministic('shared_outcome', child_z*tau, dims='child')
        middle = pm.Normal('shift_1to5', 0, .75)
        infant_extra = pm.Normal('shift_infancy_additional', 0, .75)
        late = pm.Normal('shift_late', 0, .75)
        sigma = pm.HalfNormal('sigma_outcome', 1)
        mu = (latent[frame.child_index.to_numpy()]+middle*frame.post_before5.to_numpy()
              +infant_extra*frame.post_infancy.to_numpy()+late*frame.post_at_after5.to_numpy())
        pm.Normal('obs_outcome', mu=mu, sigma=sigma, observed=frame.y_std.to_numpy())
        logbase = pm.Normal('log_baseline_hazard', mu=np.log(list(BASELINE_HAZARDS)), sigma=1.25, dims='hazard_interval')
        assoc = pm.Normal('assoc_outcome', 0, .75)
        male = pm.Normal('hazard_male', 0, .75)
        loghaz = logbase[None, :]+(assoc*latent+male*children.male.to_numpy())[:, None]
        pm.Potential('first_tb_loglik', pt.sum(event['event_interval']*loghaz-event['exposure']*pt.exp(loghaz)))
        infant = pm.Deterministic('shift_infancy', middle+infant_extra)
        pm.Deterministic('shift_lt5_child_weighted', (weights[0]*infant+weights[1]*middle)/weights[:2].sum())
        pm.Deterministic('shift_any_child_weighted', (weights[0]*infant+weights[1]*middle+weights[2]*late)/weights.sum())
        pm.Deterministic('shift_late_minus_infancy', late-infant)
        pm.Deterministic('shift_late_minus_middle', late-middle)
        pm.Deterministic('hazard_hr_per_between_child_sd', pt.exp(assoc*tau))
    return model
