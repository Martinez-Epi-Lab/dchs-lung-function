"""Computed sensitivity summaries and selected table inputs, without rendering."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .primary_results import (_scalar_draws, _sample_counts, _summarize, _summarize_ratio,
                              derive_primary_contrast_draws)
from .attenuation_precision import attenuation_precision, MAX_MCSE_BOUND_PP
from ..models.sensitivity import DEFAULT_TARGETS, fit_spec, validate_contract

SUMMARY_COLUMNS = ['mean','median','lower','upper','probability_below_zero',
                   'probability_above_zero','probability_equal_zero','mcse_mean']
ESTIMATE_COLUMNS = ['target','model_id','family','observation_window','outcome','contrast',
                    'estimand_type','unit','residual_scale',*SUMMARY_COLUMNS]
HAZARD_COLUMNS = ['target','model_id','contrast','unit',
                 *[name.replace('_zero', '_one') for name in SUMMARY_COLUMNS]]
COUNT_COLUMNS = ['target','process','group','measurements','children']
FIT_TABLE_COLUMNS = {'estimates':ESTIMATE_COLUMNS,'hazard':HAZARD_COLUMNS,'sample_counts':COUNT_COLUMNS}
S1_TARGETS = ('early_core','early_maternal_hiv','early_maternal_smoking','early_birthweight','early_all_baseline')
S3_TARGETS = (*DEFAULT_TARGETS[:8],'ratio_pre','window_ratio_pre','ratio_post','window_ratio_post')
WHEEZE_TARGETS = ('primary_sex','secondary_sex_smoking_birthweight')
LABELS = {'early_core':'Core early-TB model','early_maternal_hiv':'Core + maternal HIV',
    'early_maternal_smoking':'Core + prenatal maternal smoking','early_birthweight':'Core + birthweight',
    'early_all_baseline':'Core + all three baseline covariates','early_ever_lrti':'Core + ever-recorded LRTI (exploratory)',
    'early_all_four':'Core + baseline covariates + LRTI (exploratory)',
    'early_core_window':'Core early-TB model; restricted follow-up',
    'ratio_pre':'Expanded pre-BD model; full follow-up','window_ratio_pre':'Expanded pre-BD model; restricted follow-up',
    'ratio_post':'Expanded post-BD model; full follow-up','window_ratio_post':'Expanded post-BD model; restricted follow-up'}
CONTRAST_ORDER = ('First TB <5 years','First TB <1 year')
SPIROMETRY_TABLE_COLUMNS = ['target','analysis','model_family','observation_window','outcome','contrast',
    'unit','interval_type','mean','lower','upper','mcse_mean','joint_children','target_children','target_observations',
    'fit_fingerprint','validation_fingerprint']
S1_COLUMNS = [*SPIROMETRY_TABLE_COLUMNS,'attenuation_percent','attenuation_status',
    'attenuation_mcse_bound_pp','attenuation_mcse_independent_pp','attenuation_precision_passed']
S4_COLUMNS = ['scenario','analysis','process','age_band','exposure','estimand','measure','interval_type',
    'posterior_median','lower','upper','joint_children','scheduled_children','scheduled_observations',
    'episode_children','episode_intervals','episode_events','fit_fingerprint','validation_fingerprint']
TABLE_COLUMNS = {'table_s1':S1_COLUMNS,'table_s3':SPIROMETRY_TABLE_COLUMNS,'table_s4':S4_COLUMNS}


@dataclass
class SensitivityResults:
    estimates: pd.DataFrame
    hazard: pd.DataFrame
    sample_counts: pd.DataFrame
    metadata: dict


@dataclass
class SensitivityTables:
    table_s1: pd.DataFrame
    table_s3: pd.DataFrame
    table_s4: pd.DataFrame
    metadata: dict


def validate_posterior(posterior, contract, target):
    spec, info = fit_spec(target), validate_contract(contract,target)
    if not isinstance(posterior,dict) or set(posterior) != set(spec['posterior_variables']):
        raise ValueError('Sensitivity posterior must contain the complete selected variable inventory.')
    a=np.asarray(posterior['direct_ratio'])
    if a.dtype.kind not in 'iuf':raise ValueError('Sensitivity posterior must be real numeric arrays.')
    shape = _scalar_draws(posterior,'direct_ratio').shape
    if np.prod(shape) < 2:
        raise ValueError('Sensitivity summaries require at least two posterior draws.')
    dimensions = {n:() for n in spec['posterior_variables']}
    for name in spec['participant_posterior_variables']:dimensions[name]=(info['joint_children'],)
    for name in ('loading_mechanics','sigma_mechanics'):dimensions[name]=(4,)
    dimensions['log_baseline_hazard']=(3 if spec['family']=='early' else 6,)
    for name,tail in dimensions.items():
        a=np.asarray(posterior[name])
        if a.dtype.kind not in 'iuf' or a.shape != (*shape,*tail) or not np.isfinite(a).all():
            raise ValueError('Sensitivity posterior has invalid numeric values or coordinate shapes.')
    for name in ('tau_lci','tau_target','loading_mechanics','sigma_mechanics','sigma_lci','sigma_ratio'):
        if np.any(posterior[name] <= 0):
            raise ValueError('Sensitivity scale/loading draws must be positive.')
    expected=posterior['tau_lci'][...,None]*posterior['lci_raw']
    if not np.allclose(posterior['lci_process'],expected,rtol=1e-11,atol=1e-12):
        raise ValueError('Stored LCI latent transformation disagrees with primitive draws.')
    expected_target=posterior['tau_target'][...,None]*posterior['target_raw']
    if not np.allclose(posterior['target_process'],expected_target,rtol=1e-11,atol=1e-12):
        raise ValueError('Stored target latent transformation disagrees with primitive draws.')
    return spec,info,shape


def derive_contrast_draws(posterior, contract, target):
    """Native percentage-point effects, computed before any summary statistic."""
    spec,info,_=validate_posterior(posterior,contract,target)
    if spec['family']=='window':
        return derive_primary_contrast_draws(posterior,contract)
    names=['direct_ratio','direct_ratio_infancy','loading_ratio_mechanics','shift_mechanics','shift_mechanics_infancy']
    d={n:_scalar_draws(posterior,n) for n in names}
    middle=d['direct_ratio']+d['loading_ratio_mechanics']*d['shift_mechanics']
    infant=d['direct_ratio']+d['direct_ratio_infancy']+d['loading_ratio_mechanics']*(d['shift_mechanics']+d['shift_mechanics_infancy'])
    weight=info['weights'][0]/sum(info['weights'][:2])
    values=[weight*infant+(1-weight)*middle,infant,middle,infant-middle,d['direct_ratio'],d['direct_ratio_infancy']]
    result={}
    for (label,(variable,_)),value in zip(spec['contrasts'].items(),values):
        if not np.allclose(posterior[variable],value,rtol=1e-11,atol=1e-12):
            raise ValueError('Stored sensitivity contrast disagrees with primitive draws.')
        native=value*info['target_scale']
        if not np.isfinite(native).all():raise ValueError('Rescaled sensitivity effect is nonfinite.')
        result[label]=native
    return result


def derive_hazard_draws(posterior, target):
    result={}
    for variable,label in fit_spec(target)['hazard_variables'].items():
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            value=np.exp(_scalar_draws(posterior,variable))
        if not np.isfinite(value).all() or (value<=0).any():
            raise ValueError('Sensitivity hazard ratios must be finite and positive.')
        result[label]=value
    return result


def summarize_sensitivity(posterior, contract, target, *, compute_mcse=True):
    spec,info,shape=validate_posterior(posterior,contract,target)
    contrasts=derive_contrast_draws(posterior,contract,target)
    rows=[]
    for label,value in contrasts.items():
        kind=spec['contrasts'][label][1] if spec['family']=='early' else ('total' if label in ('TB <1','TB 1-<5','TB >=5','TB <5 weighted','Any TB weighted') else 'interaction')
        rows.append({'target':target,'model_id':spec['model_id'],'family':spec['family'],
            'observation_window':spec['observation_window'],'outcome':spec['outcome'],'contrast':label,
            'estimand_type':kind,'unit':spec['unit'],'residual_scale':info['target_scale'],**_summarize(value,compute_mcse)})
    hazards=[{'target':target,'model_id':spec['model_id'],'contrast':label,'unit':'hazard ratio',
              **_summarize_ratio(value,compute_mcse)} for label,value in derive_hazard_draws(posterior,target).items()]
    counts=_sample_counts(contract);counts.insert(0,'target',target)
    metadata={**info,'family':spec['family'],'observation_window':spec['observation_window'],
        'outcome':spec['outcome'],'unit':spec['unit'],'contrasts':list(contrasts),
        'hazards':list(spec['hazard_variables'].values()),'interval':'equal-tailed 95% posterior',
        'quantile_method':'linear','mcse_computed':bool(compute_mcse),
        'chains':shape[0],'draws_per_chain':shape[1],'posterior_variables':len(posterior),
        'hazard_covariates':list(spec['hazard_covariates']),
        'lrti_role':'nuisance regressions only; exploratory ever-recorded covariate',
        'reference_group':('pre/no-TB observations excluding post-TB measurements following first TB at or after age five'
                           if spec['family']=='early' else 'pre/no-prior-TB observations'),
        'acceptance':'complete posterior diagnostics and independent scientific comparison are separate'}
    return SensitivityResults(pd.DataFrame(rows,columns=ESTIMATE_COLUMNS),
        pd.DataFrame(hazards,columns=HAZARD_COLUMNS),counts[COUNT_COLUMNS],metadata)


def attenuation(adjusted_mean, core_mean):
    """Displayed attenuation uses posterior means; zero reference is undefined."""
    if any(not isinstance(v,(int,float,np.integer,np.floating)) or isinstance(v,(bool,np.bool_)) or not np.isfinite(v) for v in (adjusted_mean,core_mean)):
        raise ValueError('Attenuation requires finite numeric posterior means.')
    if core_mean == 0:return np.nan,'undefined_zero_core'
    value=100*(1-abs(float(adjusted_mean))/abs(float(core_mean)))
    if not np.isfinite(value):raise ValueError('Attenuation calculation is nonfinite.')
    return value,('zero_adjusted_mean' if adjusted_mean == 0 else
                  'sign_reversal' if np.sign(adjusted_mean) != np.sign(core_mean) else 'estimated')


def _binding(entry):
    if not isinstance(entry,dict) or not {'estimates','metadata','summary'}.issubset(entry):
        raise ValueError('Sensitivity table input requires estimates, metadata and verified fit summary.')
    summary=entry['summary']
    if not isinstance(summary,dict) or not isinstance(entry['metadata'],dict) or not isinstance(entry['estimates'],pd.DataFrame):
        raise ValueError('Sensitivity table input fields have invalid types.')
    if summary.get('diagnostics_passed') is not True or summary.get('status')!='diagnostics_passed':
        raise ValueError('Sensitivity table inputs require accepted complete-posterior diagnostics.')
    if summary.get('target') not in WHEEZE_TARGETS and summary.get('covariance_ppc_passed') is not True:
        raise ValueError('Sensitivity table inputs require accepted target covariance adequacy.')
    for name in ('fit_fingerprint','validation_fingerprint'):
        value=summary.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Sensitivity table fit identities must be complete SHA256 values.')
    return summary


def _build_tables(new_results, reused_primary, reused_wheeze):
    """Select S1/S3/S4 rows from full-precision, verified model summaries.

    Entries contain freshly reconstructed estimates, aggregate metadata and a
    verified diagnostic completion. Execution checks their receipt identities;
    this pure function performs no file access or fitting.
    """
    if set(new_results)!=set(DEFAULT_TARGETS) or set(reused_primary)!={'ratio_pre','ratio_post'} or set(reused_wheeze)!=set(WHEEZE_TARGETS):
        raise ValueError('Sensitivity tables require all ten selected fits and four reused fits.')
    entries={**new_results,**reused_primary};rows=[]
    for target in S3_TARGETS:
        entry=entries[target];summary=_binding(entry);meta=entry['metadata'];est=entry['estimates']
        if summary.get('target')!=target or not isinstance(est,pd.DataFrame):
            raise ValueError('Sensitivity table selector differs from fit completion.')
        early=target.startswith('early_');restricted=target.startswith('window_') or target=='early_core_window'
        selected=('First TB <5 years, child-weighted','First TB <1 year') if early else ('TB <5 weighted','TB <1')
        _check_spirometry_entry(entry,target)
        label_column='contrast'
        for label,display in zip(selected,CONTRAST_ORDER):
            chosen=est.loc[est[label_column].eq(label)]
            if len(chosen)!=1:raise ValueError('Sensitivity estimate selection must identify exactly one row.')
            row=chosen.iloc[0]
            values=np.asarray([row['mean'],row['lower'],row['upper'],row['mcse_mean']],float)
            if not np.isfinite(values).all() or values[1]>values[2] or values[3]<0 or row['unit']!='percentage points':
                raise ValueError('Sensitivity table estimate has invalid interval or units.')
            counts=summary['counts'];target_counts=counts['process_counts']['target']
            rows.append({'target':target,'analysis':LABELS[target],
                'model_family':'early-TB joint' if early else 'all-age joint',
                'observation_window':'restricted' if restricted else 'full',
                'outcome':'Post-BD FEV1/FVC' if target in ('ratio_post','window_ratio_post') else 'Pre-BD FEV1/FVC',
                'contrast':display,'unit':'percentage points','interval_type':'equal-tailed 95% credible interval',
                'mean':float(values[0]),'lower':float(values[1]),'upper':float(values[2]),'mcse_mean':float(values[3]),
                'joint_children':counts['joint_children'],'target_children':target_counts['n_children'],
                'target_observations':target_counts['n_observations'],
                'fit_fingerprint':summary['fit_fingerprint'],'validation_fingerprint':summary['validation_fingerprint']})
    s3=pd.DataFrame(rows,columns=SPIROMETRY_TABLE_COLUMNS)
    s1=s3.loc[s3.target.isin(S1_TARGETS)].copy().reset_index(drop=True)
    precision_rows=[]
    for row in s1.itertuples():
        if row.target=='early_core':
            precision={'attenuation_percent':np.nan,'attenuation_status':'reference',
                'attenuation_mcse_bound_pp':np.nan,'attenuation_mcse_independent_pp':np.nan,
                'attenuation_precision_passed':True}
        else:
            core=s1.loc[s1.target.eq('early_core') & s1.contrast.eq(row.contrast)].iloc[0]
            precision=attenuation_precision(row.mean,float(core['mean']),row.mcse_mean,float(core['mcse_mean']))
        precision_rows.append(precision)
    for name in precision_rows[0]:
        s1[name]=[np.nan if row[name] is None else row[name] for row in precision_rows]
    wheeze=[]
    for target in WHEEZE_TARGETS:
        entry=reused_wheeze[target];summary=_binding(entry)
        if summary.get('target')!=target:raise ValueError('Wheeze table scenario differs from verified fit.')
        _check_wheeze_entry(entry,target)
        effects=entry['estimates'];counts=summary['counts']['counts']
        if len(effects)!=6 or set(effects.scenario)!={target}:raise ValueError('Wheeze sensitivity table requires six conditional effects per scenario.')
        for row in effects.to_dict('records'):
            wheeze.append({**{k:row[k] for k in ('scenario','process','age_band','exposure','estimand','measure','posterior_median')},
                'analysis':'Primary sex-adjusted' if target=='primary_sex' else 'Sex, prenatal smoking and birthweight adjusted',
                'interval_type':'equal-tailed 95% credible interval','lower':row['cri_2_5'],'upper':row['cri_97_5'],
                'joint_children':summary['counts']['joint_children'],
                'scheduled_children':counts['scheduled_children'],'scheduled_observations':counts['scheduled_observations'],
                'episode_children':counts['episode_children'],'episode_intervals':counts['episode_intervals'],
                'episode_events':counts['recurrent_episodes'],'fit_fingerprint':summary['fit_fingerprint'],
                'validation_fingerprint':summary['validation_fingerprint']})
    s4=pd.DataFrame(wheeze,columns=S4_COLUMNS)
    return SensitivityTables(s1[S1_COLUMNS],s3,s4,{
        'schema_version':1,'new_model_targets':list(DEFAULT_TARGETS),
        'reused_primary_targets':['ratio_pre','ratio_post'],'reused_wheeze_targets':list(WHEEZE_TARGETS),
        'rows':{'table_s1':10,'table_s3':24,'table_s4':12},
        'attenuation':'100*(1-abs(adjusted posterior mean)/abs(early core posterior mean))',
        'attenuation_interval':'none; this is a transformation of posterior means',
        'attenuation_mcse':'first-order covariance-agnostic upper bound from marginal posterior-mean MCSEs',
        'attenuation_max_mcse_bound_pp':MAX_MCSE_BOUND_PP,
        'attenuation_precision_passed':bool(s1.attenuation_precision_passed.all()),
        'zero_core':'undefined_zero_core; no epsilon substitution',
        'negative_attenuation':'retained unchanged; increased effect magnitude',
        'reused_fit_count':4,'new_fit_count':10,'rendered_manuscript':False})


def _integer_count(value):
    return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_)) and value>=0


def _check_spirometry_entry(entry,target):
    summary,meta,est=entry['summary'],entry['metadata'],entry['estimates']
    counts=summary['counts'];selected=counts['process_counts']['target']
    values=[counts['joint_children'],selected['n_children'],selected['n_observations']]
    if not all(_integer_count(v) for v in values) or not 0<values[1]<=min(values[0],values[2]):
        raise ValueError('Sensitivity sample counts must be consistent nonnegative integer aggregates.')
    if meta.get('target')!=target or meta.get('joint_children')!=values[0] or meta.get('unit')!='percentage points':
        raise ValueError('Sensitivity metadata differs from selected target and aggregate counts.')
    if 'target' not in est or not est.target.eq(target).all() or not est.unit.eq('percentage points').all():
        raise ValueError('Sensitivity estimate rows differ from selected target or unit.')
    if target in DEFAULT_TARGETS:
        spec=fit_spec(target)
        for key in ('model_id','family','observation_window','outcome'):
            if meta.get(key)!=spec[key] or not est[key].eq(spec[key]).all():
                raise ValueError('Sensitivity estimate family or outcome differs from the selected specification.')
        if any(meta.get(k)!=v for k,v in counts.items()):
            raise ValueError('Sensitivity aggregate metadata and counts disagree.')
        from .primary_results import CONTRAST_VARIABLES
        labels=list(spec['contrasts']) if spec['family']=='early' else list(CONTRAST_VARIABLES)
        if list(est.contrast)!=labels:
            raise ValueError('Sensitivity estimate inventory is incomplete or duplicated.')
    else:
        from .primary_results import TARGETS, CONTRAST_VARIABLES
        if any(meta.get(k)!=v for k,v in TARGETS[target].items()) or list(est.contrast)!=list(CONTRAST_VARIABLES):
            raise ValueError('Reused primary metadata differs from the selected outcome.')
        if (not est.outcome.eq(TARGETS[target]['outcome']).all() or not est.stage.eq(TARGETS[target]['stage']).all()
                or not est.n_outcome_children.eq(values[1]).all() or not est.n_outcome_measurements.eq(values[2]).all()):
            raise ValueError('Reused primary row metadata differs from selected aggregate counts.')


def _check_wheeze_entry(entry,target):
    from .wheeze_results import EFFECT_COLUMNS, PROCESSES
    from ..models.wheeze import EFFECT_LABELS
    summary,meta,effects=entry['summary'],entry['metadata'],entry['estimates']
    info=summary['counts'];counts=info['counts']
    if meta.get('target')!=target or meta!=info:
        raise ValueError('Reused wheeze metadata differs from verified scenario and counts.')
    if (list(effects.columns)!=EFFECT_COLUMNS or len(effects)!=6 or not effects.scenario.eq(target).all()
            or list(zip(effects.process,effects.age_band))!=[(p,EFFECT_LABELS[b]) for p in PROCESSES for b in (1,2,3)]
            or effects.measure.tolist()!=['conditional odds ratio']*3+['conditional incidence rate ratio']*3
            or not effects.exposure.eq('TB during infancy').all()
            or not effects.estimand.eq('Post-infancy-TB versus pre/no-TB time within attained-age band').all()):
        raise ValueError('Wheeze sensitivity table requires exactly the selected six conditional effects.')
    numeric=effects[EFFECT_COLUMNS[6:]].to_numpy()
    if numeric.dtype.kind not in 'iuf' or not np.isfinite(numeric).all() or (numeric[:,:4]<=0).any():
        raise ValueError('Wheeze effect estimates must be finite positive numbers.')
    if (np.any(numeric[:,2]>numeric[:,1]) or np.any(numeric[:,1]>numeric[:,3])
            or np.any(numeric[:,4:]<0) or np.any(numeric[:,4:]>1) or np.any(numeric[:,4:].sum(axis=1)>1+1e-12)):
        raise ValueError('Wheeze posterior intervals or probabilities are inconsistent.')
    names=('children','scheduled_children','scheduled_observations','episode_children','episode_intervals','recurrent_episodes')
    if (not _integer_count(info['joint_children']) or not all(_integer_count(counts[n]) for n in names)
            or counts['children']!=info['joint_children']
            or not 0<counts['scheduled_children']<=min(info['joint_children'],counts['scheduled_observations'])
            or not 0<=counts['episode_children']<=min(info['joint_children'],counts['episode_intervals'])):
        raise ValueError('Wheeze aggregate sample counts are inconsistent.')


def build_tables(new_results,reused_primary,reused_wheeze):
    """Validate selected aggregate inputs and build full-precision S1/S3/S4 rows."""
    try:
        if not all(isinstance(v,dict) for v in (new_results,reused_primary,reused_wheeze)):
            raise ValueError()
        return _build_tables(new_results,reused_primary,reused_wheeze)
    except (KeyError,AttributeError,TypeError,ValueError,IndexError,OverflowError):
        raise ValueError('Sensitivity table input schema, selected identities or aggregate values are invalid.') from None
