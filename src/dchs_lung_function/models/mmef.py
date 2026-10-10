"""Early-TB MMEF joint graphs, specification 2.0.0.

The target and four oriented mechanics indicators share a unit-variance child
factor. An independent target child intercept captures remaining target
persistence. LCI has its own observed process, with no LCI path in the target or
first-TB hazard. Builders are pure: no data access, fitting, or source loading.
"""
from __future__ import annotations
from typing import Mapping
import numpy as np
import pandas as pd

SPEC_VERSION = '2.0.0'
MECHANICS = {'resistance': -1., 'compliance': 1., 'xee': 1., 'ree': -1.}
TARGETS = {
    'mmef_unpaired': {'column':'mmef', 'contract_key':'mmef/unpaired', 'source_target':'mmef_native', 'outcome':'Pre-bronchodilator MMEF (FEF25–75)', 'model_id':'MMEF-UNPAIRED', 'random_seed':20260813},
    'mmef_pre_paired': {'column':'mmef_pre_paired', 'contract_key':'mmef/pre_paired', 'source_target':'mmef_pre_paired', 'outcome':'Pre-bronchodilator MMEF (FEF25–75), paired sample', 'model_id':'MMEF-PAIRED-JOINT', 'random_seed':20260903},
    'mmef_post_paired': {'column':'mmef_post_paired', 'contract_key':'mmef/post_paired', 'source_target':'mmef_post_paired', 'outcome':'Post-bronchodilator MMEF (FEF25–75), paired sample', 'model_id':'MMEF-PAIRED-JOINT', 'random_seed':20260903},
}
DEFAULT_TARGETS = tuple(TARGETS)
CONTRASTS = {
    'First TB <5 years, child-weighted': ('total_ratio_lt5_child_weighted','total'),
    'First TB <1 year': ('total_ratio_infancy','total'),
    'First TB 1-<5 years': ('total_ratio_1to5','total'),
    'Infancy minus age 1-<5 years': ('total_ratio_infancy_difference','interaction'),
    'Direct component: first TB 1-<5 years': ('direct_ratio','component'),
    'Additional direct infancy component': ('direct_ratio_infancy','component'),
}
POSTERIOR_VARIABLES = ('mechanics','lci_raw','tau_lci','lci_process','target_raw','tau_target','target_process','loading_mechanics','sigma_mechanics','sigma_lci','shift_mechanics','shift_mechanics_infancy','shift_lci','shift_lci_infancy','loading_ratio_mechanics','direct_ratio','direct_ratio_infancy','sigma_ratio','log_baseline_hazard','assoc_mechanics','hazard_male','total_ratio_1to5','total_ratio_infancy','total_ratio_lt5_child_weighted','total_ratio_infancy_difference')


def fit_spec(target):
    if target not in TARGETS:
        raise ValueError('Unsupported MMEF joint target.')
    chosen = TARGETS[target]
    return {**chosen, 'target':target, 'family':'mmef', 'specification_version':SPEC_VERSION,
        'threshold':5, 'unit':'L/s', 'orientation':1., 'result_rows':6, 'hazard_rows':2,
        'mechanics_indicators':list(MECHANICS), 'mechanics_orientation':dict(MECHANICS),
        'nuisance_formula':'1+C(visit_label)+height_z+male',
        'nuisance_training':'pre/no-TB observations, separately for each process',
        'scale_method':'training residual SD with ddof equal to nuisance design column count',
        'hazard_breaks':[0.,1.,3.,5.], 'hazard_labels':['0-<1','1-<3','3-<5'],
        'log_hazard_prior_mean':np.log([.045,.018,.006]).tolist(), 'log_hazard_prior_sd':1.25,
        'priors':{'unit_normal_child':['mechanics','lci_raw','target_raw'], 'half_normal_sd1':['tau_lci','tau_target','loading_mechanics','sigma_mechanics','sigma_lci','sigma_ratio'], 'normal_sd1':['loading_ratio_mechanics'], 'normal_sd0_75':['shift_mechanics','shift_mechanics_infancy','shift_lci','shift_lci_infancy','direct_ratio','direct_ratio_infancy','assoc_mechanics','hazard_male']},
        'target_mean':'loading_ratio_mechanics*(mechanics + shift_mechanics*post_before5 + shift_mechanics_infancy*post_infancy) + target_process + direct_ratio*post_before5 + direct_ratio_infancy*post_infancy',
        'target_process':'tau_target*target_raw; independent of mechanics and LCI; no hazard coupling',
        'hazard_log_mean':'log_baseline_hazard + assoc_mechanics*mechanics + hazard_male*male',
        'lci_process':'tau_lci*lci_raw; independent of mechanics; no target/hazard loading',
        'contrast_weights':'distinct post-TB target children diagnosed before one or at ages one to before five',
        'contrasts':{k:list(v) for k,v in CONTRASTS.items()},
        'posterior_variables':list(POSTERIOR_VARIABLES), 'participant_posterior_variables':['mechanics','lci_raw','lci_process','target_raw','target_process'],
        'sampling':{'draws':1600,'tune':2200,'chains':4,'cores':4,'target_accept':.92,'random_seed':chosen['random_seed'],'init':'jitter+adapt_diag','initvals':None},
        'sampling_provenance':'Source wrapper defaults; selected saved arrays confirm four chains and 1600 retained draws. Historical seed, tune and target acceptance require separately retained execution evidence.',
        'sampling_authority':'Non-authoritative historical defaults; supporting_sampling.resolved_sampling and the active configuration determine executed controls and initialization.',
        'fresh_sampling':'Configured separately with full-posterior diagnostic gates and a recorded seed offset relative to the wrapper default.'}


def _array(value,shape,label,*,binary=False,integer=False):
    try:
        a=np.asarray(value,dtype=float)
    except (TypeError,ValueError):
        raise ValueError(label+' must be numeric.') from None
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError(label+' has invalid shape or nonfinite values.')
    if binary and not np.isin(a,[0.,1.]).all():
        raise ValueError(label+' must be binary.')
    if integer and not np.equal(a,np.floor(a)).all():
        raise ValueError(label+' must contain integers.')
    return a


def _validate_shared_contract(contract,spec):
    """Validate preserved M03 definitions; return aggregate counts and scales."""
    try:
        audit=contract['audit']; children=contract['children']; frames=contract['frames']
        x6=spec['family']=='x6'; threshold=spec['threshold']
        family='x6' if x6 else 'early'
        expected={'target',*MECHANICS} | (set() if x6 else {'lci'})
        if audit['family']!=family or audit['target']!=spec['column'] or audit['covariates']:
            raise ValueError('Prepared family, target or covariates differ from the selected model.')
        if set(frames)!=expected or set(contract['designs'])!=expected or set(audit['outcomes'])!=expected:
            raise ValueError('Prepared process inventory differs from the selected model.')
        n=len(children)
        if n<1 or children.pid_child.isna().any() or children.pid_child.duplicated().any():
            raise ValueError('Joint child linkage is empty, missing or duplicated.')
        if not np.array_equal(_array(children.child_index,(n,),'Child indices',integer=True),np.arange(n)):
            raise ValueError('Child coordinate order differs from the prepared linkage.')
        male=_array(children.male,(n,),'Child sex',binary=True)
        if list(contract['hazard_terms'])!=['male'] or not np.array_equal(_array(contract['hazard_design'],(n,1),'Hazard design')[:,0],male):
            raise ValueError('Hazard design must be the prepared child sex covariate.')
        post_terms=['post_any'] if x6 else ['post_before5','post_infancy']
        if list(contract['post_diagnosis_terms'])!=post_terms:
            raise ValueError('Post-TB design terms differ from the selected model.')
        counts={};scales={}
        for key in frames:
            f=frames[key];m=len(f);a=audit['outcomes'][key]
            if m<1 or f.pid_child.isna().any():
                raise ValueError('An observed process has missing or empty child linkage.')
            idx=_array(f.child_index,(m,),'Measurement child indices',integer=True).astype(int)
            if (idx<0).any() or (idx>=n).any() or not np.array_equal(children.pid_child.to_numpy()[idx],f.pid_child.to_numpy()):
                raise ValueError('Measurement indexing differs from child linkage.')
            post=_array(f.post_any,(m,),'Post-TB indicator',binary=True)
            actual_post=(f.tb_first_dx.notna() & f.test_date.ge(f.tb_first_dx)).to_numpy(int)
            if f.test_date.isna().any() or not np.array_equal(post,actual_post):
                raise ValueError('Post-TB indicators disagree with exact measurement/diagnosis dates.')
            if (f.exit_date.notna() & f.test_date.gt(f.exit_date)).any():
                raise ValueError('An observed process extends beyond child follow-up.')
            age=np.asarray(f.tb_age_first_years,dtype=float)
            if (post.astype(bool)&(~np.isfinite(age)|(age>=threshold)|(age<0))).any():
                raise ValueError('A post-TB observation violates the strict first-TB cutoff.')
            if not np.array_equal(_array(f.post_target_tb,(m,),'Target TB',binary=True),post):
                raise ValueError('Target TB indicators disagree with prepared post-TB status.')
            z=_array(f[post_terms],(m,len(post_terms)),'Post-TB design',binary=True)
            if not np.array_equal(z,_array(contract['post_diagnosis_designs'][key],z.shape,'Prepared post-TB matrix',binary=True)):
                raise ValueError('Post-TB matrix differs from prepared frame.')
            if not x6 and (not np.array_equal(z[:,0],post) or not np.array_equal(z[:,1],post*(age<1))):
                raise ValueError('Early-TB and infancy indicators disagree.')
            if x6 and key=='target':
                if not f.age_years.between(3.,17.,inclusive='both').all():
                    raise ValueError('X6 target ages must lie within three to seventeen inclusive.')
            column=spec['column'] if key=='target' else ('i_lci_2_5' if key=='lci' else key)
            orientation=1. if key=='target' else (-1. if key=='lci' else MECHANICS[key])
            values=_array(f.outcome_value,(m,),'Oriented outcome')
            if not np.array_equal(values,_array(f[column],(m,),'Native outcome')*orientation):
                raise ValueError('Prepared outcome orientation differs from its source definition.')
            terms=list(a['nuisance_terms']); design=contract['designs'][key]
            if list(design.columns)!=terms:
                raise ValueError('Nuisance terms disagree with the preparation audit.')
            x=_array(design,(m,len(terms)),'Nuisance design')
            beta=_array(a['nuisance_coefficients'],(len(terms),),'Nuisance coefficients')
            scale=float(a['scale'])
            if not np.isfinite(scale) or scale<=0 or not np.allclose(_array(f.y_std,(m,),'Standardized outcome'),(values-x@beta)/scale,rtol=1e-12,atol=1e-12):
                raise ValueError('Standardized outcome does not reconcile with its residual fit.')
            counts[key]={'n_observations':m,'n_children':int(np.unique(idx).size)}
            if any(a[k]!=v for k,v in counts[key].items()) or a['n_training']!=int((post==0).sum()):
                raise ValueError('Observed process counts differ from the preparation audit.')
            scales[key]=scale
        event=contract['event'];breaks=np.asarray(spec['hazard_breaks']);j=len(breaks)-1
        if not np.array_equal(event['breaks'],breaks):
            raise ValueError('First-TB hazard breaks differ from the selected cutoff.')
        events=_array(event['event'],(n,),'First-TB events',binary=True)
        stop=_array(event['stop'],(n,),'Hazard stop ages')
        endpoint = pd.Timestamp(event['ascertainment_end'])
        if pd.isna(endpoint):
            raise ValueError('Hazard ascertainment endpoint is missing.')
        censor = pd.concat([children.exit_date, pd.Series(endpoint, index=children.index)], axis=1).min(axis=1)
        censor_age = (censor - children.child_dob).dt.days.to_numpy(float) / 365.25
        tb_age = children.tb_age_first_years.to_numpy(float)
        diagnosed = children.tb_first_dx.notna().to_numpy()
        if (children.child_dob.isna().any() or not np.isfinite(censor_age).all() or (censor_age <= 0).any()
                or children.tb_first_dx.lt(children.child_dob).any()
                or (diagnosed & (~np.isfinite(tb_age) | (tb_age < 0))).any()
                or (~diagnosed & ~np.isnan(tb_age)).any()):
            raise ValueError('Hazard diagnosis chronology or follow-up is invalid.')
        expected_event = (diagnosed & (tb_age < threshold) & children.tb_first_dx.le(censor).to_numpy()).astype(int)
        expected_stop = np.minimum(censor_age, threshold)
        expected_stop = np.where(expected_event, np.minimum(tb_age, expected_stop), expected_stop)
        if not np.array_equal(events, expected_event) or not np.array_equal(stop, expected_stop) or (stop <= 0).any():
            raise ValueError('Hazard events or censoring disagree with observed follow-up and the strict age cutoff.')
        exposure=_array(event['exposure'],(n,j),'Hazard exposure')
        interval=_array(event['event_interval'],(n,j),'Event intervals',binary=True)
        if not np.allclose(exposure,np.maximum(0.,np.minimum(stop[:,None],breaks[1:])-breaks[:-1]),rtol=0,atol=1e-12):
            raise ValueError('Hazard exposures do not reconcile with follow-up.')
        if not np.array_equal(interval,events[:,None]*((stop[:,None]>=breaks[:-1])&(stop[:,None]<breaks[1:]))):
            raise ValueError('Hazard event intervals do not reconcile.')
        target_frame=frames['target'];post=target_frame.loc[target_frame.post_any.eq(1)].drop_duplicates('pid_child')
        age=post.tb_age_first_years
        weights=np.array([age.lt(1).sum(),age.between(1,5,inclusive='left').sum(),age.ge(5).sum()],dtype=int)
        if weights.sum()<1 or not np.array_equal(weights,_array(contract['weights'],(3,),'Exposed-child weights',integer=True)) or not np.array_equal(weights,audit['weights']):
            raise ValueError('Exposed-child weights do not reconcile with target support.')
        if audit['joint_children']!=n or audit['tb_events']!=int(events.sum()):
            raise ValueError('Joint model counts disagree with preparation audit.')
        return {'model_id':spec['model_id'],'specification_version':SPEC_VERSION,'target':spec['target'],
                'threshold':threshold,'joint_children':n,'tb_events':int(events.sum()),
                'process_counts':counts,'process_scales':scales,'target_scale':scales['target'],
                'orientation':1.,'weights':weights.tolist(),'n_exposed_target_children':int(weights.sum())}
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('Supporting contract lacks required M03 fields.') from exc


def validate_contract(contract,target):
    return _validate_shared_contract(contract,fit_spec(target))

def build_model(prepared: Mapping, target: str):
    """Build the explicit selected model; retain original posterior variable names."""
    validated = validate_contract(prepared, target)
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
        target_raw = pm.Normal("target_raw", 0.0, 1.0, dims="child")
        tau_target = pm.HalfNormal("tau_target", 1.0)
        target_process = pm.Deterministic("target_process", tau_target * target_raw, dims="child")

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
            + target_process[ratio_idx]
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
