"""Selected wheeze contrasts and reproducible aggregate predictive checks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2
from ..models.wheeze import EFFECT_LABELS, POSTERIOR_VARIABLES, process_arrays, validate_contract
from .wheeze_standardization import (_posterior, _draws, RR_COLUMNS,
                                     summarize_standardized_rr)

PROCESSES = ('Scheduled current wheeze', 'Participant-initiated recurrent wheeze episodes')
SUMMARY_COLUMNS = ['posterior_mean','posterior_median','cri_2_5','cri_97_5',
                   'posterior_probability_above_1','posterior_probability_below_1']
EFFECT_COLUMNS = ['scenario','process','age_band','exposure','estimand','measure',*SUMMARY_COLUMNS]
CONTRAST_COLUMNS = ['scenario','process','contrast','measure',*SUMMARY_COLUMNS,
                    'posterior_probability_later_effect_smaller']
GLOBAL_COLUMNS = ['scenario','process','null_hypothesis','posterior_wald_chi_square',
                  'degrees_of_freedom','reference_p_value',
                  'posterior_probability_1to3_greater_than_both_later_bands',
                  'posterior_probability_monotonic_decline']
SUPPORT_COLUMNS = ['scenario','process','age_band','exposure_group','children',
                   'observations_or_intervals','events_or_positive','person_years']
PPC_COLUMNS = ['scenario','process','age_band','exposure_group','observed_denominator',
               'observed_events','observed_statistic','replicated_mean','replicated_2_5',
               'replicated_97_5','units']
TABLE_COLUMNS = {'effects':EFFECT_COLUMNS,'contrasts':CONTRAST_COLUMNS,'global_tests':GLOBAL_COLUMNS,
                 'support_counts':SUPPORT_COLUMNS,'posterior_predictive_cells':PPC_COLUMNS,
                 'standardized_rr':RR_COLUMNS}
PPC_REPLICATE_KEYS = ('draw_indices','counts','statistics','conditional_means','conditional_variances')
SUPPORT_GROUPS = ('post-TB after infancy TB','post-TB after TB at age >=1','pre/no-TB')
PPC_GROUPS = ('post infancy TB','post other TB','pre/no TB')
# Under age one, post-TB after later TB is structurally impossible. Source
# sorting differs between these two label sets; preserve each source order.
SUPPORT_CELL_KEYS = tuple(sorted((band,SUPPORT_GROUPS[group]) for band in range(4)
                                for group in range(3) if band or group!=1))
PPC_CELL_KEYS = tuple(sorted((band,PPC_GROUPS[group]) for band in range(4)
                            for group in range(3) if band or group!=1))
TRANSFORMS = {
    'scheduled_infancy_post_or':'scheduled_infancy_post_log_or',
    'scheduled_other_post_or':'scheduled_other_post_log_or',
    'episode_infancy_post_rr':'episode_infancy_post_log_rr',
    'episode_other_post_rr':'episode_other_post_log_rr',
    'tb_propensity_hr':'tb_latent_association',
    'scheduled_or_per_propensity_sd':'sigma_scheduled',
    'episode_rr_per_propensity_sd':'sigma_episode'}

@dataclass
class WheezeResults:
    effects: pd.DataFrame
    contrasts: pd.DataFrame
    global_tests: pd.DataFrame
    support_counts: pd.DataFrame
    posterior_predictive_cells: pd.DataFrame
    standardized_rr: pd.DataFrame
    metadata: dict
    ppc_replicates: dict


def validate_posterior(posterior, info):
    """Require the complete source graph and consistent stored transforms."""
    posterior=_posterior(posterior)
    if set(posterior)!=set(POSTERIOR_VARIABLES):
        raise ValueError('Wheeze posterior variable inventory differs from the selected model.')
    tails={name:() for name in POSTERIOR_VARIABLES}
    tails['latent_propensity']=(info['joint_children'],)
    tails['scheduled_age']=(5,)
    for name in ('scheduled_covariate','episode_covariate','tb_covariate'):
        tails[name]=(len(info['covariates']),)
    for name in ('episode_log_baseline_rate','tb_log_baseline_hazard'):
        tails[name]=(8,)
    for name in ('scheduled_infancy_post_log_or','scheduled_other_post_log_or',
                 'episode_infancy_post_log_rr','episode_other_post_log_rr',
                 'scheduled_infancy_post_or','scheduled_other_post_or',
                 'episode_infancy_post_rr','episode_other_post_rr'):
        tails[name]=(4,)
    values={name:_draws(posterior,name,tail) for name,tail in tails.items()}
    shape=values['scheduled_intercept'].shape
    if np.prod(shape)<2 or any(a.shape[:2]!=shape for a in values.values()):
        raise ValueError('Wheeze posterior draw dimensions are inconsistent or insufficient.')
    if any((values[name]<0).any() for name in ('sigma_scheduled','sigma_episode')):
        raise ValueError('Wheeze propensity scales must be nonnegative.')
    errors={}
    for ratio,log in TRANSFORMS.items():
        with np.errstate(over='ignore',under='ignore'):
            expected=np.exp(values[log])
        if not np.isfinite(expected).all() or (expected<=0).any() or not np.allclose(values[ratio],expected,rtol=1e-11,atol=1e-12):
            raise ValueError('Wheeze stored deterministic transform is inconsistent: '+ratio)
        errors[ratio]=float(np.max(np.abs(values[ratio]-expected)))
    return values, {'chains':shape[0],'draws_per_chain':shape[1],
                    'posterior_variables':len(values),'maximum_transform_errors':errors}


def ratio_summary(draws):
    a=np.asarray(draws,float)
    if not a.size or not np.isfinite(a).all() or (a<=0).any():
        raise ValueError('Wheeze ratio draws must be finite and positive.')
    return dict(zip(SUMMARY_COLUMNS,map(float,(a.mean(),np.median(a),np.quantile(a,.025),
        np.quantile(a,.975),np.mean(a>1),np.mean(a<1)))))


def summarize_effects(posterior,target):
    rows=[]
    for name,process,measure in zip(('scheduled_infancy_post_or','episode_infancy_post_rr'),
            PROCESSES,('conditional odds ratio','conditional incidence rate ratio')):
        a=_draws(_posterior(posterior),name,(4,)).reshape(-1,4)
        for band in (1,2,3):
            rows.append({'scenario':target,'process':process,'age_band':EFFECT_LABELS[band],
                'exposure':'TB during infancy','estimand':'Post-infancy-TB versus pre/no-TB time within attained-age band',
                'measure':measure,**ratio_summary(a[:,band])})
    return pd.DataFrame(rows,columns=EFFECT_COLUMNS)


def summarize_contrasts(posterior,target):
    rows,global_rows=[],[]
    for name,process,measure in zip(('scheduled_infancy_post_log_or','episode_infancy_post_log_rr'),
                                  PROCESSES,('OR ratio','IRR ratio')):
        beta=_draws(_posterior(posterior),name,(4,)).reshape(-1,4)[:,[1,2,3]]
        if len(beta)<2:
            raise ValueError('Global wheeze contrasts require at least two posterior draws.')
        for later,earlier in ((2,1),(3,1),(3,2)):
            delta=beta[:,later-1]-beta[:,earlier-1]
            with np.errstate(over='ignore',under='ignore'):
                ratio=np.exp(delta)
            rows.append({'scenario':target,'process':process,
                'contrast':f'{EFFECT_LABELS[later]} versus {EFFECT_LABELS[earlier]}',
                'measure':measure,**ratio_summary(ratio),
                'posterior_probability_later_effect_smaller':float(np.mean(delta<0))})
        differences=np.column_stack((beta[:,1]-beta[:,0],beta[:,2]-beta[:,0]))
        mean=differences.mean(axis=0);cov=np.cov(differences,rowvar=False)
        stat=float(mean@np.linalg.pinv(cov)@mean)
        global_rows.append({'scenario':target,'process':process,
            'null_hypothesis':'Equal infancy-TB log effects at ages 1-<3, 3-<5, and 5-10.5',
            'posterior_wald_chi_square':stat,'degrees_of_freedom':2,'reference_p_value':float(chi2.sf(stat,2)),
            'posterior_probability_1to3_greater_than_both_later_bands':float(np.mean((beta[:,0]>beta[:,1])&(beta[:,0]>beta[:,2]))),
            'posterior_probability_monotonic_decline':float(np.mean((beta[:,0]>beta[:,1])&(beta[:,1]>beta[:,2])))})
    return pd.DataFrame(rows,columns=CONTRAST_COLUMNS),pd.DataFrame(global_rows,columns=GLOBAL_COLUMNS)


def _cells(contract,arrays,is_scheduled,labels,keys):
    frame=contract['scheduled' if is_scheduled else 'episode_intervals']
    prefix='sched' if is_scheduled else 'epi'
    inf,other=arrays[prefix+'_inf'],arrays[prefix+'_other']
    category=np.select((inf==1,other==1),labels[:2],default=labels[2])
    bands=frame.effect_band.to_numpy(int)
    return frame,[(key,(bands==key[0])&(category==key[1])) for key in keys]


def support_counts(contract,target,arrays=None):
    if arrays is None:arrays=process_arrays(contract,target)
    rows=[]
    for scheduled,process in zip((True,False),PROCESSES):
        frame,cells=_cells(contract,arrays,scheduled,SUPPORT_GROUPS,SUPPORT_CELL_KEYS)
        observed=frame['wheeze' if scheduled else 'events'].to_numpy(int)
        for (band,group),mask in cells:
            rows.append({'scenario':target,'process':process,'age_band':EFFECT_LABELS[band],
                'exposure_group':group,'children':int(frame.loc[mask,'child_index'].nunique()),
                'observations_or_intervals':int(mask.sum()),'events_or_positive':int(observed[mask].sum()),
                'person_years':np.nan if scheduled else float(frame.loc[mask,'person_years'].sum())})
    return pd.DataFrame(rows,columns=SUPPORT_COLUMNS)


def posterior_predictive_cells(posterior,contract,target,*,ppc_seed,arrays=None):
    """Source-order simulations with only aggregate cell replicates retained."""
    if isinstance(ppc_seed,(bool,np.bool_)) or not isinstance(ppc_seed,(int,np.integer)) or ppc_seed<0:
        raise ValueError('Wheeze predictive seed must be a nonnegative integer.')
    if arrays is None:arrays=process_arrays(contract,target)
    posterior=_posterior(posterior)
    flat={name:np.asarray(a).reshape((-1,)+np.asarray(a).shape[2:]) for name,a in posterior.items()}
    keep=np.linspace(0,len(flat['scheduled_intercept'])-1,min(200,len(flat['scheduled_intercept'])),dtype=int)
    counts=np.zeros((len(keep),22),dtype=np.int64)
    statistics=np.full(counts.shape,np.nan);means=statistics.copy();variances=statistics.copy()
    rng=np.random.default_rng(int(ppc_seed));rows=[];cellmeta=[]
    for process_idx,(scheduled,process) in enumerate(zip((True,False),PROCESSES)):
        frame,cells=_cells(contract,arrays,scheduled,PPC_GROUPS,PPC_CELL_KEYS)
        prefix='sched' if scheduled else 'epi';band=frame.effect_band.to_numpy(int)
        child=frame.child_index.to_numpy(int);inf=arrays[prefix+'_inf'];other=arrays[prefix+'_other']
        observed=frame['wheeze' if scheduled else 'events'].to_numpy(int)
        denominators=[int(mask.sum()) if scheduled else float(frame.loc[mask,'person_years'].sum()) for _,mask in cells]
        for j,draw in enumerate(keep):
            latent=flat['latent_propensity'][draw,child]
            if scheduled:
                eta=(flat['scheduled_intercept'][draw]+contract['spline']@flat['scheduled_age'][draw]
                     +arrays['X_sched']@flat['scheduled_covariate'][draw]+flat['sigma_scheduled'][draw]*latent
                     +flat['scheduled_infancy_post_log_or'][draw,band]*inf
                     +flat['scheduled_other_post_log_or'][draw,band]*other)
                mu=expit(eta);variance=mu*(1-mu)
                if not np.isfinite(eta).all():raise ValueError('Wheeze scheduled predictive linear predictor is not finite.')
                sim=rng.binomial(1,mu)
            else:
                eta=(flat['episode_log_baseline_rate'][draw,frame.base_band.to_numpy(int)]
                     +arrays['X_epi']@flat['episode_covariate'][draw]+flat['sigma_episode'][draw]*latent
                     +flat['episode_infancy_post_log_rr'][draw,band]*inf
                     +flat['episode_other_post_log_rr'][draw,band]*other)
                with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                    mu=frame.person_years.to_numpy(float)*np.exp(eta)
                if not np.isfinite(eta).all() or not np.isfinite(mu).all() or (mu<0).any():
                    raise ValueError('Wheeze episode predictive intensity is invalid.')
                variance=mu
                try:sim=rng.poisson(mu)
                except ValueError:raise ValueError('Wheeze episode predictive intensity exceeds numerical sampling limits.') from None
            for i,((_,mask),denom) in enumerate(zip(cells,denominators)):
                col=process_idx*11+i;counts[j,col]=sim[mask].sum()
                if denom:
                    scale=(1. if scheduled else 1000.)/denom
                    statistics[j,col]=counts[j,col]*scale
                    means[j,col]=float(mu[mask].sum())*scale
                    variances[j,col]=float(variance[mask].sum())*scale**2
        for i,(((band,group),mask),denom) in enumerate(zip(cells,denominators)):
            col=process_idx*11+i;events=int(observed[mask].sum());rep=statistics[:,col]
            rows.append({'scenario':target,'process':process,'age_band':EFFECT_LABELS[band],
                'exposure_group':group,'observed_denominator':denom,'observed_events':events,
                'observed_statistic':events*(1. if scheduled else 1000.)/denom if denom else np.nan,
                'replicated_mean':float(rep.mean()) if denom else np.nan,
                'replicated_2_5':float(np.quantile(rep,.025)) if denom else np.nan,
                'replicated_97_5':float(np.quantile(rep,.975)) if denom else np.nan,
                'units':'proportion positive' if scheduled else 'episodes per 1000 child-years'})
            error=abs(float(rep.mean()-means[:,col].mean())) if denom else None
            mcse=float(np.sqrt(variances[:,col].sum())/len(keep)) if denom else None
            cellmeta.append({'process':process,'age_band':EFFECT_LABELS[band],'exposure_group':group,
                             'empty':not bool(denom),'simulation_mean_error':error,
                             'conditional_simulation_mcse':mcse,
                             'within_four_conditional_mcse':bool(error<=4*mcse) if denom else None})
    replicas=dict(zip(PPC_REPLICATE_KEYS,(keep,counts,statistics,means,variances)))
    return pd.DataFrame(rows,columns=PPC_COLUMNS),replicas,{
        'seed':int(ppc_seed),'replicates':len(keep),'selection':'evenly spaced chain-major flattened draws',
        'rng':'numpy.default_rng; scheduled draws then episode draws, source row order',
        'cells':22,'empty_cell_semantics':'zero aggregate counts; normalized statistics are NaN',
        'conditional_mean_checks':cellmeta,
        'note':'Predictive intervals describe aggregate replicated data; visual coverage is not a convergence diagnostic.'}


def summarize_wheeze(posterior,contract,target,*,ppc_seed):
    info=validate_contract(contract,target)
    values,posterior_meta=validate_posterior(posterior,info)
    arrays=process_arrays(contract,target)
    effects=summarize_effects(values,target)
    contrasts,globals_=summarize_contrasts(values,target)
    support=support_counts(contract,target,arrays)
    ppc,replicates,ppc_meta=posterior_predictive_cells(values,contract,target,ppc_seed=ppc_seed,arrays=arrays)
    rr,rr_meta=summarize_standardized_rr(values,contract,target,validated=True)
    return WheezeResults(effects,contrasts,globals_,support,ppc,rr,
        {'model_id':'WHEEZE-JOINT','target':target,'counts':info['counts'],
         'posterior':posterior_meta,'posterior_predictive':ppc_meta,'standardization':rr_meta,
         'global_test':'Posterior Wald reference chi-square on two correlated log-effect contrasts; no multiplicity adjustment.',
         'privacy':'Only aggregate cells and aggregate predictive replicates are returned.'},replicates)
