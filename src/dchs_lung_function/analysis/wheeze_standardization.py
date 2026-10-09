"""Population-standardized scheduled-wheeze probability ratios.

Both exposure states integrate the same standard-normal propensity and cohort
sex distribution. Age-band probabilities are averaged before their ratio.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import patsy
from scipy.special import expit, roots_hermitenorm
from numpy.polynomial.legendre import leggauss
from ..models.wheeze import EFFECT_LABELS, SPLINE_FORMULA, validate_contract

LATENT_NODES = 80
AGE_NODES = 48
CONVERGENCE_NODES = 120
CONVERGENCE_LIMIT = 1e-4
BANDS = ((1, 1., 3., '1-<3'), (2, 3., 5., '3-<5'), (3, 5., 10.5, '5-10.5'))
RR_COLUMNS = ['scenario', 'figure', 'axis', 'age', 'age_band', 'measure'] + [
    prefix+'_'+stat for prefix in ('or','rr','no_tb_percent','tb_percent','difference_pp')
    for stat in ('median','low','high')] + ['probability_above_1']


def _posterior(posterior):
    return posterior.posterior if hasattr(posterior, 'posterior') else posterior


def _draws(posterior, name, tail):
    try:
        value = np.asarray(posterior[name], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError('Wheeze posterior lacks a required numeric variable: '+name) from None
    if value.ndim != 2+len(tail) or value.shape[2:] != tail or min(value.shape[:2]) < 1 or not np.isfinite(value).all():
        raise ValueError('Wheeze posterior shape or values are invalid: '+name)
    return value


def baseline_transform(contract):
    """Rebuild and reuse the scheduled training spline and its saved scaling."""
    meta = contract['audit']['age_spline']
    if meta['formula'] != SPLINE_FORMULA:
        raise ValueError('Wheeze standardization requires the selected training spline.')
    raw = patsy.dmatrix(meta['formula'], {'age':contract['scheduled'].age.to_numpy(float)})
    center, scale = np.asarray(meta['center'],float), np.asarray(meta['scale'],float)
    if center.shape != (5,) or scale.shape != (5,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Wheeze spline scaling is invalid.')
    rebuilt = (np.asarray(raw)-center)/scale
    expected = np.asarray(contract['spline'],float)
    if rebuilt.shape != expected.shape or not np.isfinite(rebuilt).all():
        raise ValueError('Wheeze spline dimensions are inconsistent.')
    error = float(np.max(np.abs(rebuilt-expected)))
    if error >= 1e-10:
        raise ValueError('Wheeze training spline reconstruction exceeds its numerical limit.')
    def transform(ages):
        age=np.asarray(ages,float)
        if age.ndim!=1 or not np.isfinite(age).all() or ((age<0)|(age>10.5)).any():
            raise ValueError('Wheeze standardization age is outside the selected range.')
        basis=patsy.build_design_matrices([raw.design_info],{'age':age})[0]
        return (np.asarray(basis)-center)/scale
    return transform, error


def integrated_probabilities(draws, basis, log_or, male_fraction, *, nodes=LATENT_NODES):
    """Integrate point-specific scheduled probabilities, retaining draw order."""
    if type(nodes) is not int or nodes<2 or not np.isfinite(male_fraction) or not 0<=male_fraction<=1:
        raise ValueError('Wheeze quadrature nodes or sex fraction are invalid.')
    basis, log_or = np.asarray(basis,float),np.asarray(log_or,float)
    n=len(draws['scheduled_intercept'])
    if basis.ndim!=2 or basis.shape[1]!=5 or log_or.shape!=(n,len(basis)) or not np.isfinite(basis).all() or not np.isfinite(log_or).all():
        raise ValueError('Wheeze quadrature design and effects are inconsistent.')
    z, weight=roots_hermitenorm(nodes);weight=weight/np.sqrt(2*np.pi)
    p0,p1=np.empty((n,len(basis))),np.empty((n,len(basis)))
    for start in range(0,n,200):
        sl=slice(start,min(n,start+200))
        eta=draws['scheduled_intercept'][sl,None]+draws['scheduled_age'][sl]@basis.T
        latent=eta[:,:,None]+draws['sigma_scheduled'][sl,None,None]*z[None,None,:]
        no,yes=np.zeros_like(eta),np.zeros_like(eta)
        for sex,sex_weight in ((0,1-male_fraction),(1,male_fraction)):
            predictor=latent+sex*draws['scheduled_covariate'][sl,0,None,None]
            no+=sex_weight*(expit(predictor)@weight)
            yes+=sex_weight*(expit(predictor+log_or[sl,:,None])@weight)
        p0[sl],p1[sl]=no,yes
    return p0,p1


def _ratio_checks(p0,p1,log_or):
    if not np.isfinite(p0).all() or not np.isfinite(p1).all() or not ((p0>0)&(p0<1)&(p1>0)&(p1<1)).all():
        raise ValueError('Standardized wheeze probabilities must be strictly between zero and one.')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        rr,odds=p1/p0,np.exp(log_or)
    if not np.isfinite(rr).all() or not np.isfinite(odds).all() or (odds<=0).any():
        raise ValueError('Standardized wheeze ratios must be finite and positive.')
    if not np.array_equal(rr>1,log_or>0) or (rr>np.maximum(odds,1)+1e-10).any() or (rr<np.minimum(odds,1)-1e-10).any():
        raise ValueError('Standardized probability ratios violate direction or odds-ratio bounds.')
    return rr,odds


def standardized_draws(posterior, contract, target, *, nodes=LATENT_NODES, validated=False):
    """Return three global draw-wise probability contrasts, never child effects."""
    if not validated:
        validate_contract(contract,target)
    if target!='primary_sex':
        return {}, {'applicable':False,'reason':'Selected probability ratios use the primary sex-adjusted model only.'}
    posterior=_posterior(posterior)
    values={name:_draws(posterior,name,tail) for name,tail in {
        'scheduled_intercept':(), 'scheduled_age':(5,), 'scheduled_covariate':(1,),
        'sigma_scheduled':(), 'scheduled_infancy_post_log_or':(4,)}.items()}
    shape=values['scheduled_intercept'].shape
    if any(v.shape[:2]!=shape for v in values.values()) or (values['sigma_scheduled']<0).any():
        raise ValueError('Wheeze standardization has incompatible draw shapes or propensity scales.')
    flat={name:a.reshape((-1,)+a.shape[2:]) for name,a in values.items()}
    transform,error=baseline_transform(contract)
    male=float(contract['children'].male.mean())
    gx,gw=leggauss(AGE_NODES);out={}
    for band,left,right,_ in BANDS:
        ages=(gx+1)*(right-left)/2+left
        lor=flat['scheduled_infancy_post_log_or'][:,band]
        p0,p1=integrated_probabilities(flat,transform(ages),np.repeat(lor[:,None],AGE_NODES,axis=1),male,nodes=nodes)
        no,yes=p0@(gw/2),p1@(gw/2)
        rr,odds=_ratio_checks(no,yes,lor)
        out[band]={key:value.reshape(shape) for key,value in {'p0':no,'p1':yes,'rr':rr,'or':odds}.items()}
    return out, {'applicable':True,'scenario':target,'cohort_children':len(contract['children']),
        'male_fraction':male,'latent_nodes':nodes,'age_nodes':AGE_NODES,'baseline_reconstruction_error':error,
        'latent_standardization':'N(0,1), same distribution under both states',
        'sex_standardization':'primary-cohort child-weighted sex distribution',
        'age_standardization':'uniform within each attained-age band; average probabilities before ratio',
        'other_post_effects':0.,'interpretation':'scheduled current-wheeze probability ratio, not cumulative incidence or a causal effect'}


def _quantiles(values,prefix):
    low,median,high=np.quantile(values,[.025,.5,.975],method='linear')
    return {prefix+'_median':float(median),prefix+'_low':float(low),prefix+'_high':float(high)}


def summarize_standardized_rr(posterior,contract,target,*,validated=False):
    """Primary-only source RR table plus mandatory 80-versus-120-node audit."""
    draws,meta=standardized_draws(posterior,contract,target,validated=validated)
    if not draws:
        return pd.DataFrame(columns=RR_COLUMNS),meta
    check,_=standardized_draws(posterior,contract,target,nodes=CONVERGENCE_NODES,validated=True)
    errors={str(band):float(np.max(np.abs(draws[band]['rr']-check[band]['rr']))) for band,_,_,_ in BANDS}
    maximum=max(errors.values())
    if maximum>=CONVERGENCE_LIMIT:
        raise ValueError('Standardized wheeze RR quadrature has not met the selected convergence criterion.')
    rows=[]
    for band,_,_,label in BANDS:
        d=draws[band]
        rows.append({'scenario':target,'figure':'Age-localized','axis':'attained_age_band','age':label,
            'age_band':EFFECT_LABELS[band],'measure':'population-standardized scheduled current-wheeze probability ratio',
            **_quantiles(d['or'],'or'),**_quantiles(d['rr'],'rr'),**_quantiles(d['p0']*100,'no_tb_percent'),
            **_quantiles(d['p1']*100,'tb_percent'),**_quantiles((d['p1']-d['p0'])*100,'difference_pp'),
            'probability_above_1':float(np.mean(d['rr']>1))})
    meta.update(convergence_nodes=CONVERGENCE_NODES,convergence_limit=CONVERGENCE_LIMIT,
                maximum_draw_level_RR_quadrature_change_80_vs_120=maximum,
                band_quadrature_changes=errors,convergence_passed=True,direction_and_attenuation_checks_passed=True)
    return pd.DataFrame(rows,columns=RR_COLUMNS),meta
