"""Primary spirometry joint model, specification version 2.0.0.

The six observed processes use the residuals and scales prepared by M03.
The unit-variance mechanics factor is shared by the four mechanics indicators,
the spirometry target, and the first-TB hazard. An independent target child
intercept represents persistent target variation not shared with mechanics.
The separate LCI process has no
loading into either the target or hazard. No data are read or written and no
sampling occurs here. Child coordinates are positional indices, not identifiers.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

SPEC_VERSION = '2.0.0'
MODEL_ID = 'SPIRO-JOINT-PRIMARY'
MECHANICS = ('resistance', 'compliance', 'xee', 'ree')
PROCESSES = (*MECHANICS, 'lci', 'target')
HAZARD_BREAKS = (0., 1., 3., 5., 8., 10., 20.)
HAZARD_LABELS = ('0-<1', '1-<3', '3-<5', '5-<8', '8-<10', '10-<20')
BASELINE_HAZARDS = (.045, .018, .006, .006, .006, .006)
POST_TERMS = ('post_before5', 'post_infancy', 'post_at_after5')
CONTRASTS = {
    'TB <1': 'total_ratio_infancy',
    'TB 1-<5': 'total_ratio_1to5',
    'TB >=5': 'total_ratio_late',
    'TB <5 weighted': 'total_ratio_lt5_child_weighted',
    'Any TB weighted': 'total_ratio_any_child_weighted',
    'Infancy minus 1-<5': 'total_ratio_infancy_difference',
    'Later minus infancy': 'total_ratio_late_minus_infancy',
    'Later minus 1-<5': 'total_ratio_late_minus_1to5',
}

# Target order, seeds, measurement units, and transformations are fixed parts of
# the selected fit specification. The fitted variable names remain common to
# all targets to allow direct comparison of posterior definitions.
TARGETS = {
    'ratio_pre': ('fev1_fvc', 100., 'Pre-BD FEV1/FVC', 'percentage points', 2026092308),
    'ratio_pre_z': ('fev1_fvc_gli_z', 1., 'Pre-BD FEV1/FVC', 'GLI z-score', 2026092309),
    'ratio_post': ('fev1_fvc_post', 100., 'Post-BD FEV1/FVC', 'percentage points', 2026092310),
    'ratio_post_z': ('fev1_fvc_post_gli_z', 1., 'Post-BD FEV1/FVC', 'GLI z-score', 2026092311),
    'fev1_pre_z': ('fev1_gli_z', 1., 'Pre-BD FEV1', 'GLI z-score', 2026092400),
    'fvc_pre_z': ('fvc_gli_z', 1., 'Pre-BD FVC', 'GLI z-score', 2026092401),
    'fev1_post_z': ('fev1_post_gli_z', 1., 'Post-BD FEV1', 'GLI z-score', 2026092402),
    'fvc_post_z': ('fvc_post_gli_z', 1., 'Post-BD FVC', 'GLI z-score', 2026092403),
    'fev1_pre_native': ('fev1', 1., 'Pre-BD FEV1', 'L', 2026092404),
    'fvc_pre_native': ('fvc', 1., 'Pre-BD FVC', 'L', 2026092405),
    'fev1_post_native': ('fev1_post', 1., 'Post-BD FEV1', 'L', 2026092406),
    'fvc_post_native': ('fvc_post', 1., 'Post-BD FVC', 'L', 2026092407),
}


def fit_spec(target: str) -> dict[str, Any]:
    """Return independent, JSON-safe model and selected sampling metadata."""
    if target not in TARGETS:
        raise ValueError('Unsupported primary spirometry target.')
    column, multiplier, label, unit, seed = TARGETS[target]
    normal_priors = {
        'mechanics': [0., 1.], 'lci_raw': [0., 1.], 'target_raw': [0., 1.],
        'loading_ratio_mechanics': [0., 1.],
        **{name: [0., .75] for name in (
            'shift_mechanics', 'shift_mechanics_infancy', 'shift_mechanics_late',
            'shift_lci', 'shift_lci_infancy', 'shift_lci_late',
            'direct_ratio', 'direct_ratio_infancy', 'direct_ratio_late',
            'assoc_mechanics', 'hazard_male')},
    }
    return deepcopy({
        'model_id': MODEL_ID, 'specification_version': SPEC_VERSION,
        'target': target, 'column': column, 'measurement_multiplier': multiplier,
        'outcome': label, 'unit': unit,
        'runtime': {'pymc': '5.25.1', 'pytensor': '2.31.7', 'arviz': '0.22.0'},
        'normal_priors_mean_sd': normal_priors,
        'half_normal_prior_sd': {name: 1. for name in (
            'tau_lci', 'tau_target', 'loading_mechanics', 'sigma_mechanics', 'sigma_lci', 'sigma_ratio')},
        'log_baseline_hazard_prior': {'mean': np.log(BASELINE_HAZARDS).tolist(), 'sd': 1.25},
        'hazard_breaks_years': list(HAZARD_BREAKS),
        'mechanics_indicators': list(MECHANICS),
        'process_orientations': {'resistance': -1., 'compliance': 1., 'xee': 1.,
                                 'ree': -1., 'lci': -1., 'target': multiplier},
        'nuisance_formula': '1+C(visit_label)+height_z+male',
        'nuisance_training': 'pre/no-TB observations of each process',
        'residual_scale': 'sample SD of training residuals, ddof=nuisance design columns',
        'latent_parameterization': {'mechanics': 'centered Normal(0,1)',
                                    'lci_process': 'tau_lci*lci_raw; lci_raw Normal(0,1)',
                                    'target_process': 'tau_target*target_raw; target_raw Normal(0,1), independent of mechanics and LCI'},
        'target_child_effect_hazard_coupling': False,
        'participant_posterior_variables': ['mechanics', 'lci_raw', 'lci_process', 'target_raw', 'target_process'],
        'lci_coupling_to_target_or_hazard': False,
        'post_diagnosis_terms': list(POST_TERMS),
        'contrasts': dict(CONTRASTS),
        'contrast_scale': 'multiply each standardized total by target residual scale',
        'contrast_weights': 'distinct children with target observations after TB, by first-TB age',
        'sampling': {
            'draws': 1200, 'tune': 1600, 'chains': 4, 'cores': 2,
            'target_accept': .95, 'random_seed': seed, 'init': 'adapt_diag',
            'initvals': [{'loading_ratio_mechanics': v, 'loading_mechanics': [v]*4,
                          'sigma_ratio': .9} for v in (.25, .35, .45, .55)],
            'progressbar': False,
        },
        'sampling_authority': 'Historical default controls and initialization provenance; resolved_sampling with the active sampling configuration determines executed controls.',
        'diagnostic_criteria': {'divergences': 0, 'max_rhat': 1.01, 'min_bulk_ess': 400,
                                'scope': 'all posterior variables including child latent effects'},
        'additional_diagnostics': ['tail_ess', 'bfmi', 'mcse', 'trace_inspection'],
    })


def _array(value, shape, name, *, binary=False, integer=False):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Invalid shape or nonfinite values in {name}.')
    if binary and not np.isin(result, [0., 1.]).all():
        raise ValueError(f'{name} must be binary.')
    if integer and not np.equal(result, np.floor(result)).all():
        raise ValueError(f'{name} must contain integers.')
    return result


def validate_contract(contract: Mapping, target: str) -> dict[str, Any]:
    """Fail before graph construction on an inconsistent M03 primary contract.

    Validation returns aggregate metadata only. It never emits identifiers or
    rows, and does not impose study-specific sample counts on fabricated tests.
    """
    spec = fit_spec(target)
    try:
        audit = contract['audit']
        if audit['family'] != 'primary' or audit['target'] != spec['column'] or audit['covariates']:
            raise ValueError('The contract is not the selected primary target specification.')
        children, frames, event = contract['children'], contract['frames'], contract['event']
        n = len(children)
        if n < 1 or set(frames) != set(PROCESSES):
            raise ValueError('The primary model requires children and all six processes.')
        indices = _array(children['child_index'], (n,), 'child indices', integer=True)
        if not np.array_equal(indices, np.arange(n)):
            raise ValueError('Child indices must follow the M03 child order.')
        male = _array(children['male'], (n,), 'child sex', binary=True)
        if list(contract['hazard_terms']) != ['male'] or list(contract['post_diagnosis_terms']) != list(POST_TERMS):
            raise ValueError('Primary hazard or post-diagnosis terms differ from the fixed specification.')
        h = _array(contract['hazard_design'], (n, 1), 'hazard design')
        if not np.array_equal(h[:, 0], male):
            raise ValueError('Hazard design does not match the child sex covariate.')
        if not np.array_equal(np.asarray(event['breaks']), HAZARD_BREAKS):
            raise ValueError('Primary hazard intervals differ from the fixed specification.')
        exposure = _array(event['exposure'], (n, 6), 'hazard exposure')
        events = _array(event['event'], (n,), 'first-TB event', binary=True)
        event_interval = _array(event['event_interval'], (n, 6), 'hazard event interval', binary=True)
        stop = _array(event['stop'], (n,), 'hazard follow-up')
        breaks = np.asarray(HAZARD_BREAKS)
        expected_exposure = np.maximum(0., np.minimum(stop[:, None], breaks[1:])-breaks[:-1])
        expected_event = events[:, None]*((stop[:, None] >= breaks[:-1]) & (stop[:, None] < breaks[1:]))
        if (stop <= 0).any() or (stop > breaks[-1]).any() or not np.allclose(exposure, expected_exposure, rtol=0., atol=1e-12):
            raise ValueError('Hazard exposure does not reconcile with follow-up.')
        if not np.array_equal(event_interval, expected_event) or not np.array_equal(event_interval.sum(1), events):
            raise ValueError('First-TB event intervals do not reconcile.')
        counts = {}
        group_children = None
        for name in PROCESSES:
            frame = frames[name]
            m = len(frame)
            if m < 1:
                raise ValueError('Each observed process must contain at least one measurement.')
            child_index = _array(frame['child_index'], (m,), 'observation child indices', integer=True).astype(int)
            if (child_index < 0).any() or (child_index >= n).any():
                raise ValueError('Observation child index lies outside the child coordinate.')
            _array(frame['y_std'], (m,), 'standardized observations')
            z = _array(frame[list(POST_TERMS)], (m, 3), 'post-diagnosis indicators', binary=True)
            if (z[:, 1] > z[:, 0]).any() or (z[:, 0]+z[:, 2] > 1).any():
                raise ValueError('Infancy must be nested within before-five TB; later TB must be separate.')
            if not np.array_equal(_array(contract['post_diagnosis_designs'][name], (m, 3), 'post-diagnosis design', binary=True), z):
                raise ValueError('Post-diagnosis design differs from its prepared process frame.')
            for alias, position in [('post_tb_lt5', 0), ('post_tb_infancy', 1), ('post_tb_late', 2)]:
                if alias in frame and not np.array_equal(frame[alias].to_numpy(), z[:, position]):
                    raise ValueError('Post-diagnosis source aliases disagree.')
            if 'pid_child' in children and 'pid_child' in frame:
                if children['pid_child'].isna().any() or children['pid_child'].duplicated().any():
                    raise ValueError('The child linkage is missing or duplicated.')
                if not np.array_equal(children['pid_child'].to_numpy()[child_index], frame['pid_child'].to_numpy()):
                    raise ValueError('Observation child indexing does not match its linkage.')
            process_audit = audit['outcomes'][name]
            scale = float(process_audit['scale'])
            count = {'n_observations': m, 'n_children': int(np.unique(child_index).size)}
            if not np.isfinite(scale) or scale <= 0 or any(process_audit[k] != v for k, v in count.items()):
                raise ValueError('Process audit counts or residual scale are inconsistent.')
            counts[name] = count
            if name == 'target':
                group_children = [set(child_index[z[:, 1] == 1]),
                                  set(child_index[(z[:, 0] == 1) & (z[:, 1] == 0)]),
                                  set(child_index[z[:, 2] == 1])]
        weights = _array(contract['weights'], (3,), 'contrast weights', integer=True)
        if (weights < 0).any() or weights[:2].sum() <= 0:
            raise ValueError('Weighted early-TB contrasts require exposed target children.')
        if any(group_children[i] & group_children[j] for i in range(3) for j in range(i)):
            raise ValueError('A target child has inconsistent first-TB age groups.')
        if not np.array_equal(weights, [len(group) for group in group_children]):
            raise ValueError('Contrast weights differ from distinct exposed target children.')
        if audit['joint_children'] != n or audit['tb_events'] != int(events.sum()) or not np.array_equal(audit['weights'], weights):
            raise ValueError('Joint audit counts differ from the prepared contract.')
        return {'model_id': MODEL_ID, 'specification_version': SPEC_VERSION, 'target': target,
                'joint_children': n, 'tb_events': int(events.sum()), 'process_counts': counts,
                'weights': weights.astype(int).tolist(), 'target_scale': float(audit['outcomes']['target']['scale'])}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('The primary model contract is missing required M03 fields.') from exc


def build_model(contract: Mapping, target: str):
    """Build the selected PyMC graph without fitting or changing the contract."""
    validated = validate_contract(contract, target)
    import pymc as pm
    import pytensor.tensor as pt

    frames, children, event = contract['frames'], contract['children'], contract['event']
    coords = {'child': np.arange(validated['joint_children']),
              'mechanics_indicator': list(MECHANICS), 'hazard_interval': list(HAZARD_LABELS)}
    with pm.Model(coords=coords) as model:
        mechanics = pm.Normal('mechanics', 0., 1., dims='child')
        lci_raw = pm.Normal('lci_raw', 0., 1., dims='child')
        tau_lci = pm.HalfNormal('tau_lci', 1.)
        lci_process = pm.Deterministic('lci_process', tau_lci*lci_raw, dims='child')
        target_raw = pm.Normal('target_raw', 0., 1., dims='child')
        tau_target = pm.HalfNormal('tau_target', 1.)
        target_process = pm.Deterministic('target_process', tau_target*target_raw, dims='child')
        loading_mechanics = pm.HalfNormal('loading_mechanics', 1., dims='mechanics_indicator')
        sigma_mechanics = pm.HalfNormal('sigma_mechanics', 1., dims='mechanics_indicator')
        sigma_lci = pm.HalfNormal('sigma_lci', 1.)
        shift_mechanics = pm.Normal('shift_mechanics', 0., .75)
        shift_mechanics_infancy = pm.Normal('shift_mechanics_infancy', 0., .75)
        shift_lci = pm.Normal('shift_lci', 0., .75)
        shift_lci_infancy = pm.Normal('shift_lci_infancy', 0., .75)
        shift_mechanics_late = pm.Normal('shift_mechanics_late', 0., .75)
        shift_lci_late = pm.Normal('shift_lci_late', 0., .75)
        direct_ratio_late = pm.Normal('direct_ratio_late', 0., .75)

        for j, key in enumerate(MECHANICS):
            frame = frames[key]
            child_idx = frame['child_index'].to_numpy(dtype=int)
            z0, zinf, zlate = (frame[name].to_numpy(dtype=float) for name in POST_TERMS)
            current = mechanics[child_idx]+shift_mechanics*z0+shift_mechanics_infancy*zinf+shift_mechanics_late*zlate
            pm.Normal(f'obs_{key}', mu=loading_mechanics[j]*current,
                      sigma=sigma_mechanics[j], observed=frame['y_std'].to_numpy(dtype=float))

        lci = frames['lci']
        lci_idx = lci['child_index'].to_numpy(dtype=int)
        lci_z0, lci_zinf, lci_zlate = (lci[name].to_numpy(dtype=float) for name in POST_TERMS)
        current_lci = lci_process[lci_idx]+shift_lci*lci_z0+shift_lci_infancy*lci_zinf+shift_lci_late*lci_zlate
        pm.Normal('obs_lci', mu=current_lci, sigma=sigma_lci, observed=lci['y_std'].to_numpy(dtype=float))

        ratio = frames['target']
        ratio_idx = ratio['child_index'].to_numpy(dtype=int)
        ratio_z0, ratio_zinf, ratio_zlate = (ratio[name].to_numpy(dtype=float) for name in POST_TERMS)
        loading_ratio_mechanics = pm.Normal('loading_ratio_mechanics', 0., 1.)
        direct_ratio = pm.Normal('direct_ratio', 0., .75)
        direct_ratio_infancy = pm.Normal('direct_ratio_infancy', 0., .75)
        sigma_ratio = pm.HalfNormal('sigma_ratio', 1.)
        ratio_current_mechanics = mechanics[ratio_idx]+shift_mechanics*ratio_z0+shift_mechanics_infancy*ratio_zinf+shift_mechanics_late*ratio_zlate
        ratio_mu = loading_ratio_mechanics*ratio_current_mechanics+target_process[ratio_idx]+direct_ratio*ratio_z0+direct_ratio_infancy*ratio_zinf+direct_ratio_late*ratio_zlate
        pm.Normal('obs_fev1_fvc', mu=ratio_mu, sigma=sigma_ratio, observed=ratio['y_std'].to_numpy(dtype=float))

        log_baseline_hazard = pm.Normal('log_baseline_hazard', mu=np.log(np.asarray(BASELINE_HAZARDS)),
                                        sigma=1.25, dims='hazard_interval')
        assoc_mechanics = pm.Normal('assoc_mechanics', 0., .75)
        hazard_male = pm.Normal('hazard_male', 0., .75)
        eta = assoc_mechanics*mechanics+hazard_male*children['male'].to_numpy(dtype=float)
        exposure = pt.as_tensor_variable(np.asarray(event['exposure']))
        event_interval = pt.as_tensor_variable(np.asarray(event['event_interval']))
        log_h = log_baseline_hazard[None, :]+eta[:, None]
        pm.Potential('first_tb_loglik', pt.sum(event_interval*log_h-exposure*pt.exp(log_h)))

        total_1to5 = pm.Deterministic('total_ratio_1to5', direct_ratio+loading_ratio_mechanics*shift_mechanics)
        total_infancy = pm.Deterministic('total_ratio_infancy', direct_ratio+direct_ratio_infancy+
                                         loading_ratio_mechanics*(shift_mechanics+shift_mechanics_infancy))
        n_inf, n_noninf, n_late = validated['weights']
        weight_inf = n_inf/(n_inf+n_noninf)
        pm.Deterministic('total_ratio_lt5_child_weighted', weight_inf*total_infancy+(1.-weight_inf)*total_1to5)
        pm.Deterministic('total_ratio_infancy_difference', total_infancy-total_1to5)
        total_late = pm.Deterministic('total_ratio_late', direct_ratio_late+loading_ratio_mechanics*shift_mechanics_late)
        pm.Deterministic('total_ratio_any_child_weighted',
                         (n_inf*total_infancy+n_noninf*total_1to5+n_late*total_late)/(n_inf+n_noninf+n_late))
        pm.Deterministic('total_ratio_late_minus_infancy', total_late-total_infancy)
        pm.Deterministic('total_ratio_late_minus_1to5', total_late-total_1to5)
    return model
