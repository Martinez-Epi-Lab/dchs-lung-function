"""Eleven selected FeNO Gaussian-independence GEE models on log FeNO."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Independence

_VISITS = [('12months','12 months','12 months',1.),('24months','24 months','24 months',2.),
           ('8years','96 months','8 years',8.),('9years','FeNO','9 years',9.),
           ('10years','10 years','10 years',10.)]
TARGETS = {}
for _family, _visits in [('main', _VISITS), ('supplement', [('6to10weeks','6-10 weeks','6–10 weeks',.15)]+_VISITS)]:
    for _key, _visit, _display, _age in _visits:
        TARGETS[_family+'_'+_key] = {'family':_family,'visit':_visit,'display_visit':_display,'nominal_age_years':_age}
DEFAULT_TARGETS = tuple(TARGETS)
MODEL_VERSION = '1.0.0'
ESTIMATE_COLUMNS = ('target','family','visit','display_visit','nominal_age_years','scenario','plotted_term','contrast',
    'observations','children','exposed_observations','exposed_children','reference_observations','reference_children',
    'other_observations','other_children','age_min','age_max','age_median','coefficient','robust_se',
    'geometric_mean_ratio','ci_low','ci_high','percent_difference','p_value','status','accepted','reason','converged')
COEFFICIENT_COLUMNS = ('target','family','visit','term','coefficient','robust_se','naive_se',
    'exponentiated_coefficient','ci_low','ci_high','p_value','status','accepted','reason')
COVARIANCE_COLUMNS = ('target','covariance_type','row_term','column_term','value','status','accepted')
COUNT_COLUMNS = ('target','family','visit','group','observations','children')
TABLE_COLUMNS = {'estimates':ESTIMATE_COLUMNS,'coefficients':COEFFICIENT_COLUMNS,
                 'covariance':COVARIANCE_COLUMNS,'counts':COUNT_COLUMNS}


@dataclass
class FenoFit:
    estimates: pd.DataFrame
    coefficients: pd.Series
    covariance: pd.DataFrame
    naive_covariance: pd.DataFrame
    counts: pd.DataFrame
    metadata: dict


def contract_key(target):
    if target not in TARGETS:
        raise ValueError('Unsupported selected FeNO target.')
    return target


def artifact_path(target):
    contract_key(target)
    spec = TARGETS[target]
    return ('M03','respiratory','main_feno' if spec['family']=='main' else 'supplementary_feno','samples',spec['visit'])


def fit_spec(target):
    contract_key(target)
    spec = TARGETS[target]
    main = spec['family']=='main'
    return {'target':target, **spec, 'model_version':MODEL_VERSION,
        'model_id':'FENO-MAIN' if main else 'FENO-SUPPLEMENT', 'artifact_path':list(artifact_path(target)),
        'scenario':('nominal12_sex_age_height' if spec['visit']=='12 months' else 'sex_age_height') if main else 'visit_specific',
        'family_distribution':'Gaussian','link':'identity','working_correlation':'Independence',
        'response':'log(positive FeNO)', 'covariance':'child-cluster robust sandwich',
        'finite_sample_multiplier':1., 'confidence_multiplier':1.96, 'multiplicity_adjustment':'none',
        'scaling_stage':'after complete-case selection' if main else 'before complete-case selection',
        'fit_controls':{'maxiter':60 if main else 250,'ctol':1e-6,'start_params':None,
                        'params_niter':1,'first_dep_update':0,'cov_type':'robust',
                        'ddof_scale':None,'scaling_factor':1.,'scale':None},
        'coefficient_transform':'exp(beta): exposure terms are adjusted geometric mean ratios; age/height terms are per one source-standardized SD; male is male versus female; intercept is the geometric mean at zero-valued model covariates'}

COUNT_GROUPS = {'main':('Infancy TB','Later TB already occurred','No prior TB'),
                'supplement':('TB before FeNO measurement','TB after FeNO measurement','No recorded TB')}
FAILURE_REASONS = ('singular_design','insufficient_residual_degrees_of_freedom',
    'optimizer_did_not_converge','nonfinite_fit','score_not_converged','invalid_covariance','numerical_fit_failure')
DIAGNOSTIC_KEYS = ('rank','condition_number','score_norm','scale','df_resid','n_clusters','iterations',
                   'robust_min_eigenvalue','naive_min_eigenvalue','robust_max_asymmetry')


def contrast_label(target, plotted_term):
    spec = fit_spec(target)
    if not plotted_term:
        return ''
    if spec['family']=='main':
        return 'Post-infancy-TB versus no prior TB'
    return ('TB before/on measurement versus no recorded TB' if plotted_term=='tb_before_measurement'
            else 'Future TB versus no recorded TB')


def _require(condition, message):
    if not condition:
        raise ValueError('FeNO contract '+message)


def _groups(frame, family):
    first, second = (('infancy_post','later_post') if family=='main'
                     else ('tb_before_measurement','tb_after_measurement'))
    a, b = frame[first].eq(1), frame[second].eq(1)
    return {name:mask for name,mask in zip(COUNT_GROUPS[family],(a,b,~(a|b)))}


def validate_contract(contract, target):
    """Validate source-specific matrices and eligibility without modifying them.

    Supplementary standardized columns intentionally retain their preparation
    scaling, which precedes complete-case selection. Their raw-source scale is
    independently checked upstream; recomputing it on retained rows is invalid.
    """
    spec = fit_spec(target)
    _require(isinstance(contract, Mapping), 'must be a mapping.')
    try:
        f, x, y, audit = contract['frame'], contract['design'], np.asarray(contract['response'],float), contract['audit']
        _require(isinstance(f,pd.DataFrame) and isinstance(x,pd.DataFrame) and isinstance(audit,dict), 'is incomplete.')
        n = len(f);terms = list(x.columns);family=spec['family']
        _require(n > 0 and len(x)==n and x.index.equals(f.index), 'has empty or misaligned frames.')
        clean_ids=f.pid_child.astype(str).str.strip()
        _require(not f.pid_child.isna().any() and clean_ids.ne('').all()
                 and clean_ids.eq(f.pid_child.astype(str)).all()
                 and not f.duplicated(['pid_child','date']).any(), 'child/date linkage is missing, unnormalized, or duplicated.')
        _require(f.visit_label.eq(spec['visit']).all() and f.date.notna().all(), 'visit or measurement dates differ from the selected target.')
        _require(terms==audit['terms'] and len(set(terms))==len(terms), 'design terms differ from the preparation audit.')
        matrix=x.to_numpy(float)
        _require(matrix.shape==(n,len(terms)) and np.isfinite(matrix).all() and y.shape==(n,) and np.isfinite(y).all(),
                 'model response or design is nonfinite or has the wrong shape.')
        _require(np.isfinite(f.feno.to_numpy(float)).all() and f.feno.gt(0).all(), 'logarithms require finite positive FeNO.')
        _require(np.array_equal(y,f.log_feno.to_numpy(float))
                 and np.array_equal(f.log_feno.to_numpy(),np.log(f.feno).to_numpy()), 'log FeNO response differs from its preserved source values.')
        _require(np.isin(f.male.to_numpy(float),[0.,1.]).all(), 'sex must be binary.')
        if family=='main':
            first,second='infancy_post','later_post';age_name='age'
            exposure_terms=['infancy_post']+(['later_post'] if 'later_post' in terms else [])
            expected_terms=['Intercept',*exposure_terms,'male','age','height']
            _require(terms==expected_terms and contract['target']=='infancy_post', 'main exposure or adjustment terms differ.')
            age=(f.date-f.dob).dt.days.to_numpy(float)/365.25
            _require(np.array_equal(f.age.to_numpy(float),age) and f.date.lt(f.censor_date).all(), 'main exact age or censoring rule differs.')
            expected_first=f.tb_age.lt(1)&f.date.ge(f.tb_date)
            expected_second=f.tb_age.ge(1)&f.date.ge(f.tb_date)
            if spec['visit']=='12 months':
                _require((~f.tb_age.lt(1)|f.infancy_post.eq(1)).all(), 'nominal twelve-month infancy-TB observations must be post diagnosis.')
            else:
                _require(np.all((age>=1)&(age<10.5)), 'main selected ages must lie in [1,10.5).')
            target_term='infancy_post'
            expected=x.copy()
            expected['Intercept']=1.
            for term in terms[1:]:
                values=f[term].astype(float)
                expected[term]=(values-values.mean())/values.std(ddof=1) if term in ('age','height') else values
            _require(np.allclose(matrix,expected.to_numpy(float),rtol=1e-12,atol=1e-12), 'main matrix does not preserve complete-case standardization.')
            _require(audit['infancy_post']==int(f.infancy_post.sum()) and audit['later_post']==int(f.later_post.sum()), 'main exposure counts disagree.')
        else:
            first,second='tb_before_measurement','tb_after_measurement';age_name='age_years'
            exposure_terms=[t for t in (first,second) if t in terms]
            expected_terms=['Intercept',*exposure_terms,'age_z','height_z','male']
            _require(terms==expected_terms, 'supplementary exposure or adjustment terms differ.')
            age=(f.date-f.child_dob).dt.days.to_numpy(float)/365.25
            _require(np.array_equal(f.age_years.to_numpy(float),age), 'supplementary exact ages differ.')
            expected_first=f.tb_first_dx.notna()&f.date.ge(f.tb_first_dx)
            expected_second=f.tb_first_dx.notna()&f.date.lt(f.tb_first_dx)
            target_term=first if f[first].sum()>0 else second if f[second].sum()>0 else ''
            _require((contract['target'] or '')==target_term, 'supplementary contrast selector differs.')
            expected=np.column_stack([np.ones(n),f[terms[1:]].to_numpy(float)])
            _require(np.array_equal(matrix,expected), 'supplementary matrix differs from preserved pre-complete-case standardized columns.')
            _require(audit['tb_before']==int(f[first].sum()) and audit['tb_after']==int(f[second].sum()), 'supplementary exposure counts disagree.')
            _require(audit['status']==('ready' if target_term else 'no_exposure_contrast'), 'supplementary preparation status differs.')
        _require(np.array_equal(f[first].to_numpy(float),expected_first.to_numpy(float))
                 and np.array_equal(f[second].to_numpy(float),expected_second.to_numpy(float)), 'exposure status differs from diagnosis timing.')
        _require(np.isin(f[[first,second]].to_numpy(float),[0.,1.]).all()
                 and (f[first]+f[second]).le(1).all(), 'exposure states must be mutually exclusive binary indicators.')
        _require(all(not f[term].eq(1).any() or term in terms for term in (first,second)), 'an observed exposure group is absent from the model matrix.')
        _require(np.isfinite(f[age_name].to_numpy(float)).all(), 'exact ages must be finite.')
        masks=_groups(f,family)
        selected=f[target_term].eq(1) if target_term else pd.Series(False,index=f.index)
        reference=list(masks.values())[2]
        other=~(selected|reference)
        counts={'observations':n,'children':int(f.pid_child.nunique()),
                'exposed_observations':int(selected.sum()),'exposed_children':int(f.loc[selected,'pid_child'].nunique()),
                'reference_observations':int(reference.sum()),'reference_children':int(f.loc[reference,'pid_child'].nunique()),
                'other_observations':int(other.sum()),'other_children':int(f.loc[other,'pid_child'].nunique())}
        _require(audit['observations']==n and audit['children']==counts['children'], 'sample counts disagree with the preparation audit.')
        return {'target':target,'family':family,'visit':spec['visit'],'model_version':MODEL_VERSION,
                'artifact_path':spec['artifact_path'],'terms':terms,'plotted_term':target_term,'counts':counts}
    except (KeyError,TypeError,AttributeError,IndexError) as exc:
        raise ValueError('FeNO contract lacks required prepared fields.') from exc


def count_rows(contract,target):
    """Retain every exposure category, including exact zero and small counts."""
    spec=fit_spec(target);f=contract['frame']
    return pd.DataFrame([{'target':target,'family':spec['family'],'visit':spec['visit'],'group':name,
                          'observations':int(mask.sum()),'children':int(f.loc[mask,'pid_child'].nunique())}
                         for name,mask in _groups(f,spec['family']).items()],columns=COUNT_COLUMNS)


def _valid_covariance(cov, *, positive_definite=False):
    return bool(np.isfinite(cov).all() and np.allclose(cov,cov.T,rtol=1e-10,atol=1e-12)
                and np.all(np.diag(cov)>0)
                and (np.linalg.eigvalsh(cov).min()>0 if positive_definite
                     else np.linalg.eigvalsh(cov).min()>=-1e-12*max(1.,np.linalg.norm(cov,ord=2))))


def _result(contract,target,metadata,beta=None,robust=None,naive=None):
    terms=metadata['terms'];p=len(terms);estimated=metadata['status']=='estimated'
    beta=np.asarray(beta,float) if estimated else np.full(p,np.nan)
    robust=np.asarray(robust,float) if estimated else np.full((p,p),np.nan)
    naive=np.asarray(naive,float) if estimated else np.full((p,p),np.nan)
    rows=[]
    for i,term in enumerate(terms):
        b=float(beta[i]);se=float(np.sqrt(robust[i,i]));naive_se=float(np.sqrt(naive[i,i]))
        rows.append({'target':target,'family':metadata['family'],'visit':metadata['visit'],'term':term,
                     'coefficient':b,'robust_se':se,'naive_se':naive_se,'exponentiated_coefficient':float(np.exp(b)),
                     'ci_low':float(np.exp(b-1.96*se)),'ci_high':float(np.exp(b+1.96*se)),
                     'p_value':float(2*norm.sf(abs(b/se))) if estimated else np.nan,
                     'status':metadata['status'],'accepted':metadata['accepted'],'reason':metadata['reason']})
    return FenoFit(pd.DataFrame(rows,columns=COEFFICIENT_COLUMNS),pd.Series(beta,index=terms,name='coefficient'),
        pd.DataFrame(robust,index=terms,columns=terms),pd.DataFrame(naive,index=terms,columns=terms),
        count_rows(contract,target),metadata)


def fit_contract(contract,target):
    """Fit once with the selected source GEE controls and complete diagnostics."""
    info=validate_contract(contract,target);spec=fit_spec(target)
    x=contract['design'].to_numpy(float);y=np.asarray(contract['response'],float);n,p=x.shape
    rank=int(np.linalg.matrix_rank(x));condition=float(np.linalg.cond(x))
    diagnostics={name:None for name in DIAGNOSTIC_KEYS}
    diagnostics.update(rank=rank,condition_number=condition if np.isfinite(condition) else None,
                       df_resid=n-p,n_clusters=info['counts']['children'])
    info.update(status='fit_failed',accepted=False,reason='',converged=False,warnings=[],
                fit_controls=spec['fit_controls'],covariance_correction=1.,diagnostics=diagnostics,
                coefficient_transform=spec['coefficient_transform'])
    if spec['family']=='supplement' and not info['plotted_term']:
        info.update(status='non_estimable',accepted=True,reason='source_no_exposure_contrast')
        return _result(contract,target,info)
    if rank<p:
        info['reason']='singular_design';return _result(contract,target,info)
    if n<=p:
        info['reason']='insufficient_residual_degrees_of_freedom';return _result(contract,target,info)
    # Main keeps the source string labels; supplement factorizes the labels as
    # its original implementation did. Both cluster by the same child identity.
    groups=contract['frame'].pid_child if spec['family']=='main' else pd.factorize(contract['frame'].pid_child,sort=True)[0]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit=GEE(y,x,groups=groups,family=Gaussian(),cov_struct=Independence()).fit(**spec['fit_controls'])
        info['warnings']=sorted(set(type(w.message).__name__ for w in caught))
        beta=np.asarray(fit.params,float);robust=np.asarray(fit.cov_robust,float);naive=np.asarray(fit.cov_naive,float)
        score=float(np.linalg.norm(x.T@(y-x@beta)))
        scale=float(fit.scale)
        info['converged']=bool(fit.converged)
        diagnostics.update(score_norm=score if np.isfinite(score) else None,scale=scale if np.isfinite(scale) else None,
                           iterations=len(fit.fit_history['params']))
        if not fit.converged:
            info['reason']='optimizer_did_not_converge'
        elif not np.isfinite(beta).all() or not np.isfinite(scale) or scale<=0 or not np.isfinite(score):
            info['reason']='nonfinite_fit'
        elif score>=spec['fit_controls']['ctol']:
            info['reason']='score_not_converged'
        elif not _valid_covariance(robust) or not _valid_covariance(naive,positive_definite=True):
            info['reason']='invalid_covariance'
        else:
            diagnostics.update(robust_min_eigenvalue=float(np.linalg.eigvalsh(robust).min()),
                naive_min_eigenvalue=float(np.linalg.eigvalsh(naive).min()),
                robust_max_asymmetry=float(np.max(abs(robust-robust.T))))
            se=np.sqrt(np.diag(robust))
            with np.errstate(over='ignore'):
                exponentials=np.exp(np.r_[beta,beta-1.96*se,beta+1.96*se])
            if not np.isfinite(exponentials).all() or np.any(exponentials<=0):
                info['reason']='nonfinite_fit'
            else:
                info.update(status='estimated',accepted=True)
                return _result(contract,target,info,beta,robust,naive)
    except (ValueError,ArithmeticError,np.linalg.LinAlgError):
        info['reason']='numerical_fit_failure'
    return _result(contract,target,info)


def fit_selected(contracts):
    """Fit exactly the five main and six supplementary selected visit models."""
    if not isinstance(contracts,Mapping) or set(contracts)!=set(DEFAULT_TARGETS):
        raise ValueError('FeNO selection requires exactly the eleven selected target contracts.')
    fits=[fit_contract(contracts[t],t) for t in DEFAULT_TARGETS]
    estimates=[];covariance=[]
    for target,fit in zip(DEFAULT_TARGETS,fits):
        spec=fit_spec(target);m=fit.metadata;f=contracts[target]['frame']
        row=fit.estimates.set_index('term').loc[m['plotted_term']] if m['plotted_term'] else None
        val=lambda name:float(row[name]) if row is not None else np.nan
        age=f['age' if spec['family']=='main' else 'age_years']
        b=val('coefficient');gmr=val('exponentiated_coefficient')
        estimates.append({'target':target,'family':spec['family'],'visit':spec['visit'],'display_visit':spec['display_visit'],
            'nominal_age_years':spec['nominal_age_years'],'scenario':spec['scenario'],'plotted_term':m['plotted_term'],
            'contrast':contrast_label(target,m['plotted_term']),**m['counts'],'age_min':float(age.min()),
            'age_max':float(age.max()),'age_median':float(age.median()),'coefficient':b,'robust_se':val('robust_se'),
            'geometric_mean_ratio':gmr,'ci_low':val('ci_low'),'ci_high':val('ci_high'),
            'percent_difference':100*(gmr-1),'p_value':val('p_value'),'status':m['status'],
            'accepted':m['accepted'],'reason':m['reason'],'converged':m['converged']})
        for kind,matrix in (('robust',fit.covariance),('naive',fit.naive_covariance)):
            for term1 in m['terms']:
                for term2 in m['terms']:
                    covariance.append({'target':target,'covariance_type':kind,'row_term':term1,'column_term':term2,
                        'value':float(matrix.loc[term1,term2]),'status':m['status'],'accepted':m['accepted']})
    accepted=all(f.metadata['accepted'] for f in fits)
    return {'tables':{'estimates':pd.DataFrame(estimates,columns=ESTIMATE_COLUMNS),
                      'coefficients':pd.concat([f.estimates for f in fits],ignore_index=True),
                      'covariance':pd.DataFrame(covariance,columns=COVARIANCE_COLUMNS),
                      'counts':pd.concat([f.counts for f in fits],ignore_index=True)},
            'metadata':{'schema_version':1,'accepted':accepted,'status':'accepted' if accepted else 'fit_failed',
                        'selection':list(DEFAULT_TARGETS),'models':[f.metadata for f in fits]}}
