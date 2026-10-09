"""Selected early-TB covariate and observation-window joint models.

Early models explicitly retain the first-TB-before-five graph. The two all-age
window variants delegate to the unchanged primary graph. Eligibility and the
configured window endpoint are bound to the selected M03 artifact by execution;
the builders never read inputs, sample, or load historical source code.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping
import numpy as np
from . import spirometry

SPEC_VERSION = '1.0.0'
MECHANICS = {'resistance': -1., 'compliance': 1., 'xee': 1., 'ree': -1.}
PROCESSES = (*MECHANICS, 'lci', 'target')
EARLY_CONTRASTS = {
    'First TB <5 years, child-weighted': ('total_ratio_lt5_child_weighted', 'total'),
    'First TB <1 year': ('total_ratio_infancy', 'total'),
    'First TB 1-<5 years': ('total_ratio_1to5', 'total'),
    'Infancy minus age 1-<5 years': ('total_ratio_infancy_difference', 'interaction'),
    'Direct component: first TB 1-<5 years': ('direct_ratio', 'component'),
    'Additional direct infancy component': ('direct_ratio_infancy', 'component'),
}
WINDOW_CONTRASTS = dict(spirometry.CONTRASTS)
BASE_POSTERIOR_VARIABLES = (
    'mechanics', 'lci_raw', 'tau_lci', 'lci_process', 'loading_mechanics',
    'sigma_mechanics', 'sigma_lci', 'shift_mechanics', 'shift_mechanics_infancy',
    'shift_lci', 'shift_lci_infancy', 'loading_ratio_mechanics', 'direct_ratio',
    'direct_ratio_infancy', 'sigma_ratio', 'log_baseline_hazard',
    'assoc_mechanics', 'hazard_male', 'total_ratio_1to5', 'total_ratio_infancy',
    'total_ratio_lt5_child_weighted', 'total_ratio_infancy_difference',
)
WINDOW_POSTERIOR_VARIABLES = (*BASE_POSTERIOR_VARIABLES,
    'shift_mechanics_late', 'shift_lci_late', 'direct_ratio_late',
    'total_ratio_late', 'total_ratio_any_child_weighted',
    'total_ratio_late_minus_infancy', 'total_ratio_late_minus_1to5')
TARGETS = {
    'early_core': {'contract_key': 'sensitivity/core', 'covariates': []},
    'early_maternal_hiv': {'contract_key': 'sensitivity/maternal_hiv', 'covariates': ['maternal_hiv']},
    'early_maternal_smoking': {'contract_key': 'sensitivity/maternal_smoking', 'covariates': ['maternal_smoking']},
    'early_birthweight': {'contract_key': 'sensitivity/birthweight', 'covariates': ['birthweight_z']},
    'early_all_baseline': {'contract_key': 'sensitivity/all_baseline', 'covariates': ['maternal_hiv', 'maternal_smoking', 'birthweight_z']},
    'early_ever_lrti': {'contract_key': 'sensitivity/ever_lrti', 'covariates': ['ever_lrti']},
    'early_all_four': {'contract_key': 'sensitivity/all_four', 'covariates': ['maternal_hiv', 'maternal_smoking', 'birthweight_z', 'ever_lrti']},
    'early_core_window': {'contract_key': 'sensitivity/core_window', 'covariates': []},
    'window_ratio_pre': {'contract_key': 'window/pre', 'covariates': [], 'source_target': 'ratio_pre'},
    'window_ratio_post': {'contract_key': 'window/post', 'covariates': [], 'source_target': 'ratio_post'},
}
DEFAULT_TARGETS = tuple(TARGETS)
COVARIATE_LABELS = {'maternal_hiv': 'Maternal HIV', 'maternal_smoking': 'Maternal smoking during pregnancy', 'birthweight_z': 'Birthweight (per SD)'}


def fit_spec(target):
    """Return independent JSON-safe graph, contrast, and source-control metadata."""
    if target not in TARGETS:
        raise ValueError('Unsupported spirometry sensitivity target.')
    chosen = deepcopy(TARGETS[target])
    window = target.startswith('window_')
    restricted = window or target == 'early_core_window'
    covariates = chosen['covariates']
    hazard_covariates = [name for name in covariates if name != 'ever_lrti']
    hazard_variables = {'assoc_mechanics': 'Higher pre-TB mechanics factor', 'hazard_male': 'Male sex'}
    hazard_variable_columns = {}
    for covariate in hazard_covariates:
        variable = 'hazard_prenatal_smoking' if target == 'early_maternal_smoking' else 'hazard_' + covariate
        hazard_variable_columns[variable] = covariate
        hazard_variables[variable] = COVARIATE_LABELS[covariate]
    if window:
        spec = spirometry.fit_spec(chosen['source_target'])
        spec['sampling']['random_seed'] = 2026092300 if target == 'window_ratio_pre' else 2026092302
        spec.update(family='window', posterior_variables=list(WINDOW_POSTERIOR_VARIABLES),
                    result_rows=8, hazard_rows=0, hazard_variables={}, hazard_variable_columns={},
                    sampling_provenance='Selected saved audit records seed, chains, draws, tune and target acceptance; source runner supplies dispersed adapt_diag initialization.')
    else:
        base = target in {'early_core', 'early_core_window'}
        priors = {name: [0., .75] for name in (
            'shift_mechanics', 'shift_mechanics_infancy', 'shift_lci', 'shift_lci_infancy',
            'direct_ratio', 'direct_ratio_infancy', 'assoc_mechanics', 'hazard_male',
            *hazard_variable_columns)}
        priors.update(mechanics=[0., 1.], lci_raw=[0., 1.], loading_ratio_mechanics=[0., 1.])
        spec = {
            'family': 'early', 'column': 'fev1_fvc', 'measurement_multiplier': 100.,
            'outcome': 'Pre-BD FEV1/FVC', 'unit': 'percentage points',
            'runtime': {'pymc': '5.25.1', 'pytensor': '2.31.7', 'arviz': '0.22.0'},
            'normal_priors_mean_sd': priors,
            'half_normal_prior_sd': {name: 1. for name in ('tau_lci', 'loading_mechanics', 'sigma_mechanics', 'sigma_lci', 'sigma_ratio')},
            'log_baseline_hazard_prior': {'mean': np.log([.045, .018, .006]).tolist(), 'sd': 1.25},
            'hazard_breaks_years': [0., 1., 3., 5.],
            'mechanics_indicators': list(MECHANICS),
            'process_orientations': {**MECHANICS, 'lci': -1., 'target': 100.},
            'nuisance_formula': '1+C(visit_label)+height_z+male' + ''.join('+' + c for c in covariates),
            'nuisance_training': 'pre/no-TB observations of each process',
            'residual_scale': 'sample SD of training residuals, ddof=nuisance design columns',
            'latent_parameterization': {'mechanics': 'centered Normal(0,1)', 'lci_process': 'tau_lci*lci_raw; lci_raw Normal(0,1)'},
            'lci_coupling_to_target_or_hazard': False,
            'post_diagnosis_terms': ['post_before5', 'post_infancy'],
            'contrasts': {name: list(value) for name, value in EARLY_CONTRASTS.items()},
            'contrast_scale': 'multiply standardized effects by target residual scale in percentage points',
            'contrast_weights': 'distinct target children observed after first TB before one or at ages one to before five',
            'hazard_variables': hazard_variables, 'hazard_variable_columns': hazard_variable_columns,
            'posterior_variables': [*BASE_POSTERIOR_VARIABLES, *hazard_variable_columns],
            'result_rows': 6, 'hazard_rows': len(hazard_variables),
            'sampling': {'draws': 1200 if base else 1600, 'tune': 1600 if base else 2200,
                         'chains': 4, 'cores': 4, 'target_accept': .92, 'random_seed': 20260813,
                         'init': 'jitter+adapt_diag', 'initvals': None},
            'sampling_provenance': 'Source defaults only. Selected saved arrays establish four chains and retained draws (1600 for full models; 1200 for restricted core), but do not establish historical seed, tune or initialization.',
        }
    spec.update(chosen)
    spec.update(target=target, specification_version=SPEC_VERSION,
        model_id='SPIRO-WINDOW-SENSITIVITY' if restricted else 'SPIRO-COVARIATE-SENSITIVITY',
        artifact_path=['M03', 'pft', 'samples', chosen['contract_key']],
        observation_window='restricted' if restricted else 'full',
        hazard_covariates=hazard_covariates, orientation=100.,
        participant_posterior_variables=['mechanics', 'lci_raw', 'lci_process'],
        fresh_sampling='Configured separately; recorded offset from source seed/default, dispersed positive adapt_diag starts, and full-posterior diagnostics.')
    return spec


def contract_key(target):
    return fit_spec(target)['contract_key']


def _array(value, shape, label, *, binary=False, integer=False):
    if np.iscomplexobj(value):
        raise ValueError(label + ' must be real-valued.')
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(label + ' must be numeric.') from None
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(label + ' has invalid shape or nonfinite values.')
    if binary and not np.isin(result, [0., 1.]).all():
        raise ValueError(label + ' must be binary.')
    if integer and not np.equal(result, np.floor(result)).all():
        raise ValueError(label + ' must contain integers.')
    return result


def validate_contract(contract, target):
    """Validate scientific M03 arrays; return aggregates without child records."""
    spec = fit_spec(target)
    if spec['family'] == 'window':
        result = spirometry.validate_contract(contract, spec['source_target'])
        # Graph reuse is deliberate; the execution layer binds the window artifact.
        result.update(model_id=spec['model_id'], target=target,
            specification_version=SPEC_VERSION, orientation=100., covariates=[],
            hazard_terms=['male'], process_scales={key: float(contract['audit']['outcomes'][key]['scale']) for key in PROCESSES})
        return result
    try:
        import patsy
        audit, children, frames = contract['audit'], contract['children'], contract['frames']
        covariates = spec['covariates']
        if audit['family'] != 'early' or audit['target'] != 'fev1_fvc' or list(audit['covariates']) != covariates:
            raise ValueError('Family, target or covariates differ from the selected early model.')
        if any(set(value) != set(PROCESSES) for value in (frames, contract['designs'], audit['outcomes'], contract['post_diagnosis_designs'])):
            raise ValueError('The selected early model requires all six processes.')
        n = len(children)
        if n < 1 or children.pid_child.isna().any() or children.pid_child.duplicated().any() or children.pid_child.astype(str).str.strip().eq('').any():
            raise ValueError('Joint child linkage is empty, missing or duplicated.')
        if not np.array_equal(_array(children.child_index, (n,), 'Child indices', integer=True), np.arange(n)):
            raise ValueError('Child coordinates differ from their prepared order.')
        _array(children.male, (n,), 'Child sex', binary=True)
        for covariate in covariates:
            _array(children[covariate], (n,), 'Child covariate', binary=covariate != 'birthweight_z')
        hazard_terms = ['male', *spec['hazard_covariates']]
        if list(contract['hazard_terms']) != hazard_terms or list(audit['hazard_terms']) != hazard_terms:
            raise ValueError('Hazard covariates differ from the selected baseline set.')
        if not np.array_equal(_array(contract['hazard_design'], (n, len(hazard_terms)), 'Hazard design'), children[hazard_terms].to_numpy(float)):
            raise ValueError('Hazard design differs from child covariates.')
        post_terms = ['post_before5', 'post_infancy']
        if list(contract['post_diagnosis_terms']) != post_terms or list(audit['post_diagnosis_terms']) != post_terms:
            raise ValueError('Early post-diagnosis terms differ from the selected model.')
        counts, scales = {}, {}
        for key in PROCESSES:
            frame = frames[key]; m = len(frame); a = audit['outcomes'][key]
            if m < 1 or frame.pid_child.isna().any():
                raise ValueError('Observed process has empty or missing linkage.')
            idx = _array(frame.child_index, (m,), 'Measurement child indices', integer=True).astype(int)
            if (idx < 0).any() or (idx >= n).any() or not np.array_equal(children.pid_child.to_numpy()[idx], frame.pid_child.to_numpy()):
                raise ValueError('Measurement linkage differs from child coordinates.')
            for name in ['tb_first_dx', 'tb_age_first_years']:
                if not np.array_equal(frame[name].to_numpy(), children[name].to_numpy()[idx], equal_nan=True):
                    raise ValueError('Measurement diagnosis fields differ from linked child diagnosis fields.')
            for covariate in ['male', *covariates]:
                if not np.array_equal(_array(frame[covariate], (m,), 'Measurement covariate'), children[covariate].to_numpy()[idx]):
                    raise ValueError('Measurement covariate differs from linked child covariate.')
            post = _array(frame.post_any, (m,), 'Post-TB indicator', binary=True)
            actual = (frame.tb_first_dx.notna() & frame.test_date.ge(frame.tb_first_dx)).to_numpy(int)
            if frame.test_date.isna().any() or frame.test_date.lt(frame.child_dob).any() or not np.array_equal(post, actual):
                raise ValueError('Post-TB status disagrees with exact measurement and diagnosis dates.')
            age = np.asarray(frame.tb_age_first_years, dtype=float)
            if (post.astype(bool) & (~np.isfinite(age) | (age >= 5.) | (age < 0.))).any():
                raise ValueError('Post-TB measurement violates the strict before-five definition.')
            z = _array(frame[post_terms], (m, 2), 'Post-diagnosis design', binary=True)
            if not np.array_equal(z, np.column_stack([post, post * (age < 1.)])) or not np.array_equal(z, _array(contract['post_diagnosis_designs'][key], (m, 2), 'Prepared post design', binary=True)):
                raise ValueError('Early/infancy post-diagnosis indicators disagree.')
            for alias, values in [('post_target_tb', post), ('post_tb_lt5', z[:, 0]), ('post_tb_infancy', z[:, 1])]:
                if not np.array_equal(_array(frame[alias], (m,), 'Exposure alias', binary=True), values):
                    raise ValueError('Prepared exposure aliases disagree.')
            column = 'fev1_fvc' if key == 'target' else ('i_lci_2_5' if key == 'lci' else key)
            values = _array(frame.outcome_value, (m,), 'Oriented outcome')
            # Preserve source precision: native multiplication precedes conversion.
            if not np.array_equal(values, np.asarray(frame[column] * spec['process_orientations'][key], dtype=float)):
                raise ValueError('Outcome orientation differs from its selected native scale.')
            height_z = (frame.height - frame.height.mean()) / frame.height.std(ddof=1)
            if not np.array_equal(_array(frame.height_z, (m,), 'Height z-score'), height_z.to_numpy(float)):
                raise ValueError('Height must be standardized on its complete process sample.')
            design = contract['designs'][key]; terms = list(a['nuisance_terms'])
            expected = patsy.dmatrix(spec['nuisance_formula'], frame, return_type='dataframe', NA_action='raise')
            if list(design.columns) != terms or list(expected.columns) != terms:
                raise ValueError('Nuisance formula or design terms differ from the selected covariates.')
            x = _array(design, (m, len(terms)), 'Nuisance matrix')
            if not np.array_equal(x, expected.to_numpy(float)):
                raise ValueError('Nuisance design values differ from their prepared covariates.')
            train = post == 0
            if train.sum() <= x.shape[1]:
                raise ValueError('Insufficient pre/no-TB observations for nuisance training.')
            beta = _array(a['nuisance_coefficients'], (len(terms),), 'Nuisance coefficients')
            fitted = np.linalg.lstsq(x[train], values[train], rcond=None)[0]
            if not np.allclose(x @ beta, x @ fitted, rtol=1e-11, atol=1e-11):
                raise ValueError('Nuisance predictions differ from the pre/no-TB fit.')
            residual = values - x @ beta
            scale = float(a['scale'])
            if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale, np.std(residual[train], ddof=x.shape[1]), rtol=1e-12, atol=1e-12):
                raise ValueError('Residual scale does not reconcile with nuisance training.')
            if not np.allclose(_array(frame.y_std, (m,), 'Standardized response'), residual / scale, rtol=1e-12, atol=1e-12):
                raise ValueError('Standardized response differs from its residual fit.')
            counts[key] = {'n_observations': m, 'n_children': int(np.unique(idx).size)}
            if any(a[k] != v for k, v in counts[key].items()) or a['n_training'] != int(train.sum()):
                raise ValueError('Process counts differ from the preparation audit.')
            scales[key] = scale
        event = contract['event']; breaks = np.array([0., 1., 3., 5.])
        if not np.array_equal(event['breaks'], breaks):
            raise ValueError('Early first-TB hazard requires the three selected intervals.')
        events = _array(event['event'], (n,), 'First-TB events', binary=True)
        stop = _array(event['stop'], (n,), 'Hazard stop ages')
        age = children.tb_age_first_years.to_numpy(float)
        if not np.array_equal(events, (age < 5.).astype(int)) or not np.array_equal(stop, np.where(age < 5., age, 5.)) or (stop <= 0).any():
            raise ValueError('Hazard censoring differs from first TB strictly before five.')
        exposure = _array(event['exposure'], (n, 3), 'Hazard exposure')
        interval = _array(event['event_interval'], (n, 3), 'Event intervals', binary=True)
        if not np.allclose(exposure, np.maximum(0., np.minimum(stop[:, None], breaks[1:]) - breaks[:-1]), rtol=0, atol=1e-12):
            raise ValueError('Hazard exposure does not reconcile with follow-up.')
        if not np.array_equal(interval, events[:, None] * ((stop[:, None] >= breaks[:-1]) & (stop[:, None] < breaks[1:]))):
            raise ValueError('First-TB interval membership does not reconcile.')
        post = frames['target'].loc[frames['target'].post_any.eq(1)].drop_duplicates('pid_child')
        weights = np.array([post.tb_age_first_years.lt(1).sum(), post.tb_age_first_years.between(1, 5, inclusive='left').sum(), 0], dtype=int)
        if weights.sum() < 1 or not np.array_equal(weights, _array(contract['weights'], (3,), 'Contrast weights', integer=True)) or not np.array_equal(weights, audit['weights']):
            raise ValueError('Contrast weights differ from distinct exposed target children.')
        if audit['joint_children'] != n or audit['tb_events'] != int(events.sum()):
            raise ValueError('Joint counts differ from preparation audit.')
        return {'model_id': spec['model_id'], 'specification_version': SPEC_VERSION, 'target': target,
            'joint_children': n, 'tb_events': int(events.sum()), 'process_counts': counts,
            'process_scales': scales, 'target_scale': scales['target'], 'weights': weights.tolist(),
            'orientation': 100., 'covariates': list(covariates), 'hazard_terms': hazard_terms}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Sensitivity contract lacks required M03 fields.') from exc


def build_model(prepared: Mapping, target: str):
    """Build the explicit selected model; retain original posterior variable names."""
    spec = fit_spec(target)
    validated = validate_contract(prepared, target)
    if spec["family"] == "window":
        return spirometry.build_model(prepared, spec["source_target"])
    import pymc as pm
    import pytensor.tensor as pt
    frames = prepared["frames"]
    children = prepared["children"]
    event = prepared["event"]
    n_children = len(children)

    mechanics_keys = ["resistance", "compliance", "xee", "ree"]
    coords = {
        "child": np.arange(n_children),
        "mechanics_indicator": mechanics_keys,
        "hazard_interval": ["0-<1", "1-<3", "3-<5"],
    }
    with pm.Model(coords=coords) as model:
        # Fixed unit-variance priors identify the two shared latent processes.
        mechanics = pm.Normal("mechanics", 0.0, 1.0, dims="child")
        lci_raw = pm.Normal("lci_raw", 0.0, 1.0, dims="child")
        tau_lci = pm.HalfNormal("tau_lci", 1.0)
        lci_process = pm.Deterministic("lci_process", tau_lci * lci_raw, dims="child")

        loading_mechanics = pm.HalfNormal("loading_mechanics", 1.0, dims="mechanics_indicator")
        sigma_mechanics = pm.HalfNormal("sigma_mechanics", 1.0, dims="mechanics_indicator")
        sigma_lci = pm.HalfNormal("sigma_lci", 1.0)

        # Post-TB shifts in the underlying processes; the infancy terms are additions.
        shift_mechanics = pm.Normal("shift_mechanics", 0.0, 0.75)
        shift_mechanics_infancy = pm.Normal("shift_mechanics_infancy", 0.0, 0.75)
        shift_lci = pm.Normal("shift_lci", 0.0, 0.75)
        shift_lci_infancy = pm.Normal("shift_lci_infancy", 0.0, 0.75)

        for j, key in enumerate(mechanics_keys):
            frame = frames[key]
            child_idx = frame["child_index"].to_numpy(dtype=int)
            z0 = frame["post_tb_lt5"].to_numpy(dtype=float)
            zinf = frame["post_tb_infancy"].to_numpy(dtype=float)
            current = mechanics[child_idx] + shift_mechanics * z0 + shift_mechanics_infancy * zinf
            pm.Normal(
                f"obs_{key}",
                mu=loading_mechanics[j] * current,
                sigma=sigma_mechanics[j],
                observed=frame["y_std"].to_numpy(dtype=float),
            )

        lci = frames["lci"]
        lci_idx = lci["child_index"].to_numpy(dtype=int)
        lci_z0 = lci["post_tb_lt5"].to_numpy(dtype=float)
        lci_zinf = lci["post_tb_infancy"].to_numpy(dtype=float)
        current_lci = lci_process[lci_idx] + shift_lci * lci_z0 + shift_lci_infancy * lci_zinf
        pm.Normal(
            "obs_lci",
            # Loading fixed at one: this process is identified by LCI itself.
            mu=current_lci,
            sigma=sigma_lci,
            observed=lci["y_std"].to_numpy(dtype=float),
        )

        ratio = frames["target"]
        ratio_idx = ratio["child_index"].to_numpy(dtype=int)
        ratio_z0 = ratio["post_tb_lt5"].to_numpy(dtype=float)
        ratio_zinf = ratio["post_tb_infancy"].to_numpy(dtype=float)
        loading_ratio_mechanics = pm.Normal("loading_ratio_mechanics", 0.0, 1.0)
        direct_ratio = pm.Normal("direct_ratio", 0.0, 0.75)
        direct_ratio_infancy = pm.Normal("direct_ratio_infancy", 0.0, 0.75)
        sigma_ratio = pm.HalfNormal("sigma_ratio", 1.0)
        ratio_current_mechanics = (
            mechanics[ratio_idx] + shift_mechanics * ratio_z0 + shift_mechanics_infancy * ratio_zinf
        )
        ratio_mu = (
            loading_ratio_mechanics * ratio_current_mechanics
            + direct_ratio * ratio_z0
            + direct_ratio_infancy * ratio_zinf
        )
        pm.Normal(
            "obs_fev1_fvc",
            mu=ratio_mu,
            sigma=sigma_ratio,
            observed=ratio["y_std"].to_numpy(dtype=float),
        )

        # Piecewise-exponential first-TB model sharing the mechanics factor.
        log_baseline_hazard = pm.Normal(
            "log_baseline_hazard",
            mu=np.log(np.array([0.045, 0.018, 0.006])),
            sigma=1.25,
            dims="hazard_interval",
        )
        assoc_mechanics = pm.Normal("assoc_mechanics", 0.0, 0.75)
        hazard_male = pm.Normal("hazard_male", 0.0, 0.75)
        eta = assoc_mechanics * mechanics + hazard_male * children["male"].to_numpy(dtype=float)
        for variable, column in spec["hazard_variable_columns"].items():
            coefficient = pm.Normal(variable, 0.0, 0.75)
            eta = eta + coefficient * children[column].to_numpy(dtype=float)
        exposure = pt.as_tensor_variable(event["exposure"])
        event_interval = pt.as_tensor_variable(event["event_interval"])
        log_h = log_baseline_hazard[None, :] + eta[:, None]
        event_logp = pt.sum(event_interval * log_h - exposure * pt.exp(log_h))
        pm.Potential("first_tb_loglik", event_logp)

        # Derived total associations include the empirically supported mechanics path.
        total_ratio_1to5 = pm.Deterministic(
            "total_ratio_1to5",
            direct_ratio
            + loading_ratio_mechanics * shift_mechanics,
        )
        total_ratio_infancy = pm.Deterministic(
            "total_ratio_infancy",
            direct_ratio
            + direct_ratio_infancy
            + loading_ratio_mechanics * (shift_mechanics + shift_mechanics_infancy),
        )
        n_inf = validated["weights"][0]
        n_noninf = validated["weights"][1]
        weight_inf = n_inf / (n_inf + n_noninf)
        pm.Deterministic(
            "total_ratio_lt5_child_weighted",
            weight_inf * total_ratio_infancy + (1.0 - weight_inf) * total_ratio_1to5,
        )
        pm.Deterministic(
            "total_ratio_infancy_difference",
            total_ratio_infancy - total_ratio_1to5,
        )
    return model
