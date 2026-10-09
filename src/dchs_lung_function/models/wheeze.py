"""Selected three-process age-specific wheeze joint model, version 1.0.0.

The observed processes are scheduled binary wheeze, recurrent episode counts,
and first TB. A standard-normal child propensity is shared by all three.
Construction reads validated in-memory preparation contracts; it never fits,
loads study files, or changes their measurement and person-time definitions.
"""
from __future__ import annotations
from collections.abc import Mapping

import numpy as np
import pandas as pd

MODEL_ID = 'WHEEZE-JOINT'
SPEC_VERSION = '1.0.0'
BASE_BREAKS = (0., .25, .5, 1., 2., 3., 5., 8., 10.5)
BASE_LABELS = ('0-<0.25', '0.25-<0.5', '0.5-<1', '1-<2', '2-<3', '3-<5', '5-<8', '8-10.5')
EFFECT_BREAKS = (0., 1., 3., 5., 10.5)
EFFECT_LABELS = ('0-<1 year', '1-<3 years', '3-<5 years', '5-10.5 years')
TARGET_BANDS = (1, 2, 3)
SPLINE_FORMULA = "0 + cr(age, df=5, lower_bound=0, upper_bound=10.5, constraints='center')"
TARGETS = {
    'primary_sex': {'covariates': ['male'], 'source_seed': 2026091133},
    'secondary_sex_smoking_birthweight': {
        'covariates': ['male', 'prenatal_smoking', 'birthweight_z'], 'source_seed': 2026092133},
}
DEFAULT_TARGETS = tuple(TARGETS)
POSTERIOR_VARIABLES = (
    'latent_propensity', 'sigma_scheduled', 'sigma_episode', 'scheduled_intercept',
    'scheduled_age', 'scheduled_covariate', 'scheduled_infancy_post_log_or',
    'scheduled_other_post_log_or', 'episode_log_baseline_rate', 'episode_covariate',
    'episode_infancy_post_log_rr', 'episode_other_post_log_rr', 'tb_log_baseline_hazard',
    'tb_covariate', 'tb_latent_association', 'scheduled_infancy_post_or',
    'scheduled_other_post_or', 'episode_infancy_post_rr', 'episode_other_post_rr',
    'tb_propensity_hr', 'scheduled_or_per_propensity_sd', 'episode_rr_per_propensity_sd',
)


def contract_key(target):
    if target not in TARGETS:
        raise ValueError('Unsupported selected wheeze scenario.')
    return target


def fit_spec(target):
    contract_key(target)
    return {'model_id': MODEL_ID, 'specification_version': SPEC_VERSION, 'target': target,
            'scenario': target, 'contract_key': target,
            'artifact_path': ['M03', 'respiratory', 'wheeze', target],
            'covariates': list(TARGETS[target]['covariates']), 'result_rows': 6,
            'baseline_age_breaks': list(BASE_BREAKS), 'baseline_age_labels': list(BASE_LABELS),
            'effect_age_breaks': list(EFFECT_BREAKS), 'effect_age_labels': list(EFFECT_LABELS),
            'selected_bands': list(TARGET_BANDS), 'spline_formula': SPLINE_FORMULA,
            'spline_standardization': 'column mean and sample SD over scheduled observations',
            'episode_rule': 'consecutive qualifying contact gaps <=30 days; first contact indexes episode',
            'post_tb_lag_days': 0, 'infancy_threshold_years': 1.,
            'scheduled_likelihood': 'Bernoulli(logit): intercept+spline+covariates+sigma_scheduled*latent+age-band infancy/other post-TB effects',
            'episode_likelihood': 'Poisson(person_years*exp(baseline_age_rate+covariates+sigma_episode*latent+age-band infancy/other post-TB effects))',
            'tb_likelihood': 'sum(event_interval*log_hazard-exposure*exp(log_hazard)); baseline+covariates+association*latent',
            'priors': {
                'latent_propensity': ['Normal', 0., 1.],
                'sigma_scheduled': ['HalfNormal', 1.], 'sigma_episode': ['HalfNormal', 1.],
                'scheduled_intercept': ['Normal', -3., 1.5],
                'scheduled_age': ['Normal', 0., 1.],
                'scheduled_covariate': ['Normal', 0., .75],
                'scheduled_infancy_post_log_or': ['Normal', 0., 1.],
                'scheduled_other_post_log_or': ['Normal', 0., 1.],
                'episode_log_baseline_rate': ['Normal', float(np.log(.05)), 1.5],
                'episode_covariate': ['Normal', 0., .75],
                'episode_infancy_post_log_rr': ['Normal', 0., 1.],
                'episode_other_post_log_rr': ['Normal', 0., 1.],
                'tb_log_baseline_hazard': ['Normal', float(np.log(.012)), 1.5],
                'tb_covariate': ['Normal', 0., .75],
                'tb_latent_association': ['Normal', 0., .75]},
            'posterior_variables': list(POSTERIOR_VARIABLES),
            'participant_posterior_variables': ['latent_propensity'],
            'sampling': {'random_seed': TARGETS[target]['source_seed'], 'init': 'jitter+adapt_diag',
                         'initvals': None},
            'historical_sampling': {'draws': 800, 'tune': 1000, 'chains': 4, 'cores': 4,
                                    'target_accept': .93, 'random_seed': TARGETS[target]['source_seed'],
                                    'init': 'jitter+adapt_diag', 'provenance': 'selected saved run arguments'},
            'diagnostic_criteria': {'min_chains': 4, 'max_rhat': 1.01, 'min_bulk_ess': 400,
                'min_tail_ess': 400, 'min_bfmi': .3, 'divergences': 0, 'max_tree_depth_hits': 0,
                'scope': 'all 22 posterior variables including every child propensity'}}


def _require(condition, message):
    if not condition:
        raise ValueError('Wheeze contract ' + message)


def _numeric(values, shape, label, *, integer=False, binary=False):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError('Wheeze contract '+label+' must be numeric.') from None
    _require(arr.shape == shape and np.isfinite(arr).all(), label+' has invalid shape or nonfinite values.')
    if integer:
        _require(np.equal(arr, np.floor(arr)).all(), label+' must be integer-valued.')
    if binary:
        _require(np.isin(arr, [0., 1.]).all(), label+' must be binary.')
    return arr


def _indices(frame, children, label):
    idx = _numeric(frame['child_index'], (len(frame),), label+' child indices', integer=True).astype(int)
    _require(((idx >= 0) & (idx < len(children))).all(), label+' child index is out of bounds.')
    _require(not frame.pid_child.isna().any() and
             np.array_equal(frame.pid_child.to_numpy(), children.pid_child.to_numpy()[idx]),
             label+' child linkage differs from its positional index.')
    return idx


def _band(age, breaks):
    return np.searchsorted(np.asarray(breaks)[1:], age, side='right').clip(0, len(breaks)-2)


def process_arrays(contract, target):
    """Return source-equivalent design/exposure arrays for graph and predictions.

    This helper does not validate a second time. Public consumers must first call
    validate_contract; returned arrays are private model inputs, not report rows.
    """
    cov = fit_spec(target)['covariates']
    children, sched, epi = contract['children'], contract['scheduled'], contract['episode_intervals']
    x = children[cov].to_numpy(float)
    tb_age = children.tb_age.to_numpy(float)
    s_age, e_age = tb_age[sched.child_index.to_numpy(int)], tb_age[epi.child_index.to_numpy(int)]
    s_post, e_post = sched.post_tb.to_numpy(float), epi.post_tb.to_numpy(float)
    return {'X_child': x, 'X_sched': sched[cov].to_numpy(float),
            'X_epi': x[epi.child_index.to_numpy(int)],
            'sched_inf': s_post*(s_age < 1.), 'sched_other': s_post*(s_age >= 1.),
            'epi_inf': e_post*(e_age < 1.), 'epi_other': e_post*(e_age >= 1.)}


def validate_contract(contract, target):
    """Reconcile all process arrays with the frozen preparation and audit."""
    spec = fit_spec(target)
    _require(isinstance(contract, Mapping), 'must be a mapping.')
    try:
        c, s, e, episodes = (contract[k] for k in ('children', 'scheduled', 'episode_intervals', 'episodes'))
        audit, survival = contract['audit'], contract['survival']
        _require(all(isinstance(f, pd.DataFrame) for f in (c, s, e, episodes)), 'frames are missing.')
        _require(contract['scenario'] == target and audit['scenario'] == target,
                 'scenario differs from its selected target.')
        _require(list(contract['covariates']) == spec['covariates'] == audit['covariates'],
                 'covariates differ from the selected scenario.')
        n, ns, ne, k = len(c), len(s), len(e), len(spec['covariates'])
        _require(n > 0 and ns >= 6 and ne > 0, 'requires children and both observed processes.')
        idx = _numeric(c.child_index, (n,), 'Child indices', integer=True)
        _require(np.array_equal(idx, np.arange(n)) and not c.pid_child.isna().any()
                 and not c.pid_child.duplicated().any(), 'child coordinates are missing or duplicated.')
        xc = _numeric(c[spec['covariates']], (n, k), 'Child covariates')
        _numeric(c.male, (n,), 'Child sex', binary=True)
        if 'prenatal_smoking' in spec['covariates']:
            _numeric(c.prenatal_smoking, (n,), 'Prenatal smoking', binary=True)
        censor = _numeric(c.censor_age, (n,), 'Censor ages')
        events = _numeric(c.tb_event, (n,), 'First-TB events', binary=True)
        tb_age = c.tb_age.to_numpy(float)
        _require(np.all(censor > 0), 'follow-up must be positive.')
        _require(np.array_equal(np.isfinite(tb_age), events.astype(bool)) and not np.isinf(tb_age).any(),
                 'first-TB ages must be finite exactly for observed events.')
        observed = events.astype(bool)
        _require(np.all((tb_age[observed] >= 0) & (tb_age[observed] < censor[observed])
                        & (tb_age[observed] < 10.5)), 'first-TB events must precede censoring and age10.5.')
        _require(c.dob.notna().all() and c.censor_date.notna().all(), 'follow-up dates are missing.')
        _require(np.array_equal(censor, (c.censor_date-c.dob).dt.days.to_numpy(float)/365.25),
                 'censor age and exact dates differ.')
        date_event = (c.tb_date.notna() & c.tb_date.ge(c.dob) & c.tb_date.lt(c.censor_date)
                      & ((c.tb_date-c.dob).dt.days/365.25).lt(10.5)).to_numpy()
        _require(np.array_equal(events, date_event)
                 and np.array_equal(tb_age[observed], ((c.tb_date-c.dob).dt.days/365.25).to_numpy()[observed]),
                 'first-TB dates and event ages differ.')
        si, ei = _indices(s, c, 'Scheduled'), _indices(e, c, 'Episode interval')
        age = _numeric(s.age, (ns,), 'Scheduled ages')
        _require(np.all((age >= 0) & (age < 10.5)) and s.date.notna().all()
                 and not s.duplicated(['pid_child', 'date']).any(), 'scheduled dates/ages are invalid or duplicate.')
        birth = c.dob.to_numpy()[si]; end = c.censor_date.to_numpy()[si]
        _require(np.array_equal(age, (s.date.to_numpy()-birth)/np.timedelta64(1, 'D')/365.25)
                 and np.all(s.date.to_numpy() < end), 'scheduled follow-up differs from exact dates.')
        if 'visit_category' in s:
            _require(s.visit_category.eq(1).all(), 'scheduled process includes a non-scheduled contact.')
        y = _numeric(s.wheeze, (ns,), 'Scheduled wheeze', binary=True)
        xs = _numeric(s[spec['covariates']], (ns, k), 'Scheduled covariates')
        _require(np.array_equal(xs, xc[si]), 'scheduled covariates differ from child covariates.')
        sb = _numeric(s.effect_band, (ns,), 'Scheduled effect band', integer=True)
        _require(np.array_equal(sb, _band(age, EFFECT_BREAKS)), 'scheduled age bands differ.')
        spost = _numeric(s.post_tb, (ns,), 'Scheduled post-TB indicator', binary=True)
        expected_post = observed[si] & (s.date.to_numpy() >= c.tb_date.to_numpy()[si])
        _require(np.array_equal(spost, expected_post), 'scheduled exposure differs from same-day post-TB status.')
        spline = _numeric(contract['spline'], (ns, 5), 'Scheduled spline')
        spline_audit = audit['age_spline']
        _require(spline_audit['formula'] == SPLINE_FORMULA and spline_audit['columns'] == 5,
                 'spline specification differs.')
        import patsy
        raw = np.asarray(patsy.dmatrix(SPLINE_FORMULA, {'age': age}), float)
        center, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.
        _require(np.allclose(spline, (raw-center)/scale, rtol=1e-12, atol=1e-12)
                 and np.allclose(spline_audit['center'], center, rtol=1e-12, atol=1e-12)
                 and np.allclose(spline_audit['scale'], scale, rtol=1e-12, atol=1e-12),
                 'spline values or saved transformation differ from the scheduled sample.')
        start = _numeric(e.start, (ne,), 'Episode interval start')
        stop = _numeric(e.stop, (ne,), 'Episode interval stop')
        py = _numeric(e.person_years, (ne,), 'Episode person-years')
        ey = _numeric(e.events, (ne,), 'Episode counts', integer=True)
        epost = _numeric(e.post_tb, (ne,), 'Episode post-TB indicator', binary=True)
        _require(np.all(start >= 0) and np.all(stop > start) and np.all(ey >= 0)
                 and np.array_equal(py, stop-start), 'episode intervals, counts, or durations are invalid.')
        mid = (start+stop)/2
        _require(np.array_equal(e.effect_band.to_numpy(), _band(mid, EFFECT_BREAKS))
                 and np.array_equal(e.base_band.to_numpy(), _band(mid, BASE_BREAKS)),
                 'episode age bands differ.')
        _require(np.array_equal(epost, observed[ei] & (start >= tb_age[ei]-1e-10)),
                 'episode exposure differs from its exact split at TB.')
        _require(set(episodes.pid_child).issubset(set(c.pid_child)), 'an episode is outside the scenario cohort.')
        _require(not episodes.date.isna().any() and not episodes.duplicated(['pid_child', 'date']).any(),
                 'episode index dates are missing or duplicated.')
        episode_child = c.set_index('pid_child')
        episode_dob = episodes.pid_child.map(episode_child.dob)
        episode_censor = episodes.pid_child.map(episode_child.censor_date)
        episode_age = _numeric(episodes.age, (len(episodes),), 'Episode index ages')
        _require(np.array_equal(episode_age, (episodes.date-episode_dob).dt.days.to_numpy(float)/365.25)
                 and episodes.date.ge(episode_dob).all() and episodes.date.lt(episode_censor).all()
                 and np.all((episode_age >= 0) & (episode_age < 10.5)),
                 'episode index ages or dates differ from exact eligible child follow-up.')
        episode_gap = episodes.sort_values(['pid_child', 'date']).groupby('pid_child', observed=True).date.diff().dt.days
        _require(episode_gap.dropna().gt(30).all(), 'episode index contacts violate the consecutive-gap clustering window.')
        episode_by_child = {key: group.age.to_numpy(float) for key, group in episodes.groupby('pid_child', sort=False)}
        for child in range(n):
            rows = np.flatnonzero(ei == child)
            boundaries = [0., censor[child], *[v for v in BASE_BREAKS[1:-1] if v < censor[child]]]
            if observed[child] and 0 < tb_age[child] < censor[child]:
                boundaries.append(tb_age[child])
            bounds = np.unique(boundaries)
            _require(np.array_equal(start[rows], bounds[:-1]) and np.array_equal(stop[rows], bounds[1:]),
                     'episode intervals do not partition full source follow-up at exact TB and baseline boundaries.')
            child_episodes = episode_by_child.get(c.pid_child.iloc[child], np.empty(0))
            _require(np.isfinite(child_episodes).all() and np.all((child_episodes >= 0) & (child_episodes < min(censor[child], 10.5))),
                     'episode ages lie outside eligible follow-up.')
            counts = ((child_episodes[:, None] >= start[rows]) & (child_episodes[:, None] < stop[rows])).sum(axis=0)
            _require(np.array_equal(ey[rows], counts), 'episode counts do not reconcile to left-inclusive interval assignment.')
        hazard_stop = np.where(observed, tb_age, censor)
        breaks = np.asarray(BASE_BREAKS)
        expected_exposure = np.maximum(0., np.minimum(hazard_stop[:, None], breaks[1:])-breaks[:-1])
        expected_full = np.maximum(0., np.minimum(censor[:, None], breaks[1:])-breaks[:-1])
        expected_event = observed[:, None] & (tb_age[:, None] >= breaks[:-1]) & (tb_age[:, None] < breaks[1:])
        _require(np.array_equal(_numeric(survival['exposure'], (n, 8), 'TB exposure'), expected_exposure)
                 and np.array_equal(_numeric(survival['full_exposure'], (n, 8), 'Full baseline exposure'), expected_full)
                 and np.array_equal(_numeric(survival['event_interval'], (n, 8), 'TB event interval', binary=True), expected_event),
                 'first-TB survival arrays differ from attained-age follow-up.')
        sa, ea, da = audit['scheduled'], audit['episode_process'], audit['deduplication']
        counts = {'children': n, 'scheduled_observations': ns, 'scheduled_children': int(s.pid_child.nunique()),
                  'scheduled_positive': int(y.sum()), 'episode_intervals': ne,
                  'recurrent_episodes': int(ey.sum()), 'episode_children': int(episodes.pid_child.nunique()),
                  'person_years': float(py.sum()), 'tb_events': int(events.sum())}
        _require(audit['children'] == n and audit['tb_events'] == counts['tb_events']
                 and sa['records'] == ns and sa['children'] == counts['scheduled_children']
                 and sa['positive_records'] == counts['scheduled_positive']
                 and ea['episode_process_rows'] == ne and ea['retained_deduplicated_episodes'] == counts['recurrent_episodes']
                 and len(episodes) == counts['recurrent_episodes']
                 and ea['children_with_retained_episode'] == counts['episode_children']
                 and np.isclose(ea['person_years'], py.sum(), rtol=1e-12, atol=1e-12),
                 'process sample counts disagree with preparation audit.')
        _require(sa['lag_window_days'] == 0 and sa['records_excluded_in_post_tb_lag'] == 0
                 and ea['events_excluded_in_post_tb_lag'] == 0 and ea['person_years_excluded_in_post_tb_lag'] == 0
                 and da['dedup_window_days'] == 30 and da['deduplicated_episodes'] == len(episodes),
                 'lag or episode-clustering settings differ from the selected source.')
        return {'model_id': MODEL_ID, 'specification_version': SPEC_VERSION, 'target': target,
                'scenario': target, 'covariates': spec['covariates'], 'counts': counts,
                'joint_children': n, 'tb_events': counts['tb_events'], 'spline_columns': 5,
                'effect_bands': list(EFFECT_LABELS), 'baseline_intervals': list(BASE_LABELS)}
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError('Wheeze contract is missing or has invalid required preparation fields.') from exc


def build_model(contract, target):
    """Build the exact selected graph and retain all child and global variables."""
    validate_contract(contract, target)
    import pymc as pm
    import pytensor.tensor as pt
    children, sched, epi = contract['children'], contract['scheduled'], contract['episode_intervals']
    arrays = process_arrays(contract, target)
    coords = {'child': np.arange(len(children)), 'covariate': fit_spec(target)['covariates'],
              'spline_term': np.arange(5), 'effect_age_band': list(EFFECT_LABELS),
              'baseline_age_interval': list(BASE_LABELS)}
    with pm.Model(coords=coords) as model:
        latent = pm.Normal('latent_propensity', 0., 1., dims='child')
        sigma_sched = pm.HalfNormal('sigma_scheduled', 1.)
        sigma_episode = pm.HalfNormal('sigma_episode', 1.)
        intercept = pm.Normal('scheduled_intercept', -3., 1.5)
        age = pm.Normal('scheduled_age', 0., 1., dims='spline_term')
        scov = pm.Normal('scheduled_covariate', 0., .75, dims='covariate')
        sinf = pm.Normal('scheduled_infancy_post_log_or', 0., 1., dims='effect_age_band')
        sother = pm.Normal('scheduled_other_post_log_or', 0., 1., dims='effect_age_band')
        sb = sched.effect_band.to_numpy(int)
        seta = (intercept + pt.dot(contract['spline'], age) + pt.dot(arrays['X_sched'], scov)
                + sigma_sched*latent[sched.child_index.to_numpy(int)]
                + sinf[sb]*arrays['sched_inf'] + sother[sb]*arrays['sched_other'])
        pm.Bernoulli('scheduled_wheeze_observed', logit_p=seta, observed=sched.wheeze.to_numpy(int))
        rate = pm.Normal('episode_log_baseline_rate', np.log(.05), 1.5, dims='baseline_age_interval')
        ecov = pm.Normal('episode_covariate', 0., .75, dims='covariate')
        einf = pm.Normal('episode_infancy_post_log_rr', 0., 1., dims='effect_age_band')
        eother = pm.Normal('episode_other_post_log_rr', 0., 1., dims='effect_age_band')
        eb = epi.effect_band.to_numpy(int)
        eeta = (rate[epi.base_band.to_numpy(int)] + pt.dot(arrays['X_epi'], ecov)
                + sigma_episode*latent[epi.child_index.to_numpy(int)]
                + einf[eb]*arrays['epi_inf'] + eother[eb]*arrays['epi_other'])
        pm.Poisson('episode_count_observed', mu=epi.person_years.to_numpy(float)*pt.exp(eeta),
                   observed=epi.events.to_numpy(int))
        baseline = pm.Normal('tb_log_baseline_hazard', np.log(.012), 1.5, dims='baseline_age_interval')
        tcov = pm.Normal('tb_covariate', 0., .75, dims='covariate')
        assoc = pm.Normal('tb_latent_association', 0., .75)
        subject = pt.dot(arrays['X_child'], tcov) + assoc*latent
        logh = baseline[None, :] + subject[:, None]
        pm.Potential('first_tb_loglikelihood', pt.sum(
            pt.as_tensor_variable(contract['survival']['event_interval'])*logh
            - pt.as_tensor_variable(contract['survival']['exposure'])*pt.exp(logh)))
        pm.Deterministic('scheduled_infancy_post_or', pt.exp(sinf), dims='effect_age_band')
        pm.Deterministic('scheduled_other_post_or', pt.exp(sother), dims='effect_age_band')
        pm.Deterministic('episode_infancy_post_rr', pt.exp(einf), dims='effect_age_band')
        pm.Deterministic('episode_other_post_rr', pt.exp(eother), dims='effect_age_band')
        pm.Deterministic('tb_propensity_hr', pt.exp(assoc))
        pm.Deterministic('scheduled_or_per_propensity_sd', pt.exp(sigma_sched))
        pm.Deterministic('episode_rr_per_propensity_sd', pt.exp(sigma_episode))
    return model
