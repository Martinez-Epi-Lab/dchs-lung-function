"""Longitudinal linkage and PFT analysis sample contracts.

Returned frames contain confidential records and belong only in configured
private work storage. ``audit`` mappings contain aggregate metadata.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import patsy
from .reference_scores import pft_male

ID='pid_child'
VISITS=['6-10 weeks','12 months','24 months','36 months','48 months','60 months','72 months','84 months','96 months','FeNO','10 years']
RAW_VISIT_LABELS={1:'6-10 weeks',2:'12 months',3:'24 months',4:'36 months',5:'48 months',6:'60 months',7:'Pneumonia visit',8:'Pneumonia follow-up visit',9:'72 months',10:'84 months',11:'96 months',12:'FeNO',13:'10 years',14:'12 year'}
BASE_TB={'TB':'base_tb_ever','tbcat':'base_tbcat1','tbtrt':'base_tbtrt1','date_tb_diagnosed':'base_tb_dx1','date_tb_investigated':'base_tb_inv1','age_tb':'base_age_tb_days1','TB2':'base_tb2','tbcat2':'base_tbcat2','tbtrt2':'base_tbtrt2','date_tb_diagnosed2':'base_tb_dx2','timeTBneg':'base_timetb_neg','timeTBpos':'base_timetb_pos','survivaltb':'base_survivaltb','survival_tb_new':'base_survival_tb_new','survivaltb_final':'base_survivaltb_final','agetb_final':'base_agetb_final'}
MECHANICS={'resistance':('resistance','date_of_test_ilf2',-1.),'compliance':('compliance','date_of_test_ilf2',1.),'xee':('xee','date_of_test_ilf2',1.),'ree':('ree','date_of_test_ilf2',-1.)}
PRIMARY_TARGETS={f'{name}_{stage}{scale}':(f'{col}{suffix}{z}',100. if name=='ratio' and not z else 1.) for stage,suffix in [('pre',''),('post','_post')] for name,col in [('ratio','fev1_fvc'),('fev1','fev1'),('fvc','fvc')] for scale,z in ([('', ''),('_z','_gli_z')] if name=='ratio' else [('_native',''),('_z','_gli_z')])}


def _key(frame):
    values=frame[ID].astype('string').str.strip()
    if values.isna().any() or values.eq('').any():raise ValueError('Missing child linkage key.')
    return values


def _unique(frame):
    if _key(frame).duplicated().any():raise ValueError('Child-level input contains duplicate linkage keys.')


def _required(frame, columns):
    if not set(columns).issubset(frame):raise ValueError('Input is missing required preparation fields.')


def link_longitudinal(pft,baseline,tb_events,*,ascertainment_end):
    """Retain PFT rows and append baseline and date-specific episode histories."""
    _required(pft,[ID,'date_of_test','date_of_test_ilf1','date_of_test_ilf2'])
    _required(baseline,[ID,*BASE_TB]);_unique(baseline)
    _required(tb_events,[ID,'date_tb_diagnosed','date_tb_investigated','childdob','tbcat','tbtrt'])
    result=pft.copy().reset_index(drop=True);result['_link_key']=_key(result)
    base=baseline[[ID,*BASE_TB]].copy();base['_link_key']=_key(base)
    base=base.drop(columns=ID).rename(columns=BASE_TB);base['base_record_match']=1
    result=result.merge(base,on='_link_key',how='left',sort=False,validate='many_to_one')
    if result.base_record_match.isna().any():raise ValueError('PFT child missing from baseline linkage.')
    events=tb_events.copy();events['_link_key']=_key(events)
    for col in ['date_tb_diagnosed','date_tb_investigated','childdob']:events[col]=pd.to_datetime(events[col])
    if events.date_tb_diagnosed.isna().any():raise ValueError('Episode history contains an undated diagnosis.')
    if events.date_tb_diagnosed.lt(events.childdob).any():raise ValueError('Episode diagnosis precedes birth.')
    events=events.sort_values(['_link_key','date_tb_diagnosed'],kind='stable')
    groups={k:g for k,g in events.groupby('_link_key',sort=False)}
    summaries=[]
    for key in result['_link_key'].drop_duplicates():
        g=groups.get(key)
        row={'_link_key':key,'tb_ever':int(g is not None),'tb_episode_total':0 if g is None else len(g),'tb_confirmed_ever':0 if g is None else int(g.tbcat.eq('Confirmed tb').any()),'tb_first_dx':pd.NaT,'tb_last_dx':pd.NaT,'tb_first_inv':pd.NaT,'tb_first_cat':None,'tb_first_trt':None,'tb_age_first_days':np.nan,'tb_age_first_years':np.nan}
        if g is not None:
            first=g.iloc[0];days=(first.date_tb_diagnosed-first.childdob).days
            row.update(tb_first_dx=first.date_tb_diagnosed,tb_last_dx=g.date_tb_diagnosed.max(),tb_first_inv=first.date_tb_investigated,tb_first_cat=first.tbcat,tb_first_trt=first.tbtrt,tb_age_first_days=days,tb_age_first_years=days/365.25)
        summaries.append(row)
    result=result.merge(pd.DataFrame(summaries),on='_link_key',how='left',sort=False,validate='many_to_one')
    if 'visit' in result:result['visit_label']=result.visit.map(RAW_VISIT_LABELS)
    for tag,datecol in [('visit','date_of_test'),('ilf1','date_of_test_ilf1'),('ilf2','date_of_test_ilf2')]:
        rows=[]
        for key,date in zip(result['_link_key'],pd.to_datetime(result[datecol])):
            g=groups.get(key)
            row={'state':np.nan,'post':np.nan,'days_since':np.nan,'prior_n':np.nan,'prior_conf_n':np.nan,'post_conf':np.nan,'latest_dx':pd.NaT,'latest_cat':None,'latest_trt':None,'recurrent':np.nan}
            if pd.notna(date):
                prior=None if g is None else g.loc[g.date_tb_diagnosed.le(date)]
                n=0 if prior is None else len(prior)
                conf=0 if n==0 else int(prior.tbcat.eq('Confirmed tb').sum())
                row.update(state=0 if g is None else (2 if n else 1),post=int(n>0),prior_n=n,prior_conf_n=conf,post_conf=int(conf>0),recurrent=int(n>=2))
                if n:
                    last=prior.iloc[-1];row.update(latest_dx=last.date_tb_diagnosed,days_since=(date-last.date_tb_diagnosed).days,latest_cat=last.tbcat,latest_trt=last.tbtrt)
            rows.append(row)
        for field in (list(rows[0]) if tag=='visit' and rows else ['state','post','days_since']):
            result[f'tb_{field}_{tag}']=[r[field] for r in rows]
    result['tb_years_since_visit']=result.tb_days_since_visit/365.25
    states=[]
    for key,date in zip(result['_link_key'],pd.to_datetime(result.date_of_test)):
        g=groups.get(key)
        inv=None if g is None else g.date_tb_investigated.dropna()
        # Retain the specified investigation-date sensitivity behavior.
        states.append(0 if g is None else (np.nan if inv.empty else (2 if inv.le(date).any() else 1)))
    result['tb_state_inv_visit']=states
    result['tb_ascertain_complete']=pd.to_datetime(result.date_of_test).le(pd.Timestamp(ascertainment_end)).astype('int8')
    return result.drop(columns='_link_key')


def enrich_covariates(pft,baseline):
    """Add baseline variables using baseline-wide birthweight standardization."""
    _unique(baseline)
    mapping={'Prenatal_smoking_composite':'maternal_smoking','lrti':'ever_lrti','birth_weight':'birthweight_g_baseline'}
    base=baseline[[ID,'maternal_hiv',*mapping]].rename(columns=mapping).copy();base[ID]=_key(base)
    for col in ['maternal_hiv','maternal_smoking','ever_lrti','birthweight_g_baseline']:base[col]=pd.to_numeric(base[col],errors='coerce')
    sd=base.birthweight_g_baseline.std(ddof=1)
    if not np.isfinite(sd) or sd<=0:raise ValueError('Baseline birthweight has no finite positive scale.')
    base['birthweight_z']=(base.birthweight_g_baseline-base.birthweight_g_baseline.mean())/sd
    base['prenatal_smoking']=base.maternal_smoking
    base=base.drop(columns='birthweight_g_baseline')
    result=pft.copy();result[ID]=_key(result)
    collisions=set(base.columns)&set(result.columns)-{ID}
    if collisions:raise ValueError('Covariate enrichment would overwrite existing columns.')
    return result.merge(base,on=ID,how='left',sort=False,validate='many_to_one')


def normalize_pft(pft,baseline=None,*,sex_labels=None):
    p=pft.copy();p[ID]=_key(p)
    p['male']=pft_male(p.sex,sex_labels)
    label=p.visit.astype('string')
    numeric=pd.to_numeric(label,errors='coerce')
    mapping=RAW_VISIT_LABELS
    if 'visit_label' not in p:p['visit_label']=numeric.map(mapping).where(numeric.notna(),label).astype(str)
    if numeric.notna().any() and (numeric.notna() & ~numeric.isin(mapping)).any():raise ValueError('Unsupported scheduled visit code.')
    for c in ['child_dob','tb_first_dx','date_of_test','date_of_test_ilf1','date_of_test_ilf2']:p[c]=pd.to_datetime(p[c])
    p['height']=pd.to_numeric(p.height,errors='coerce')
    if baseline is not None:
        _unique(baseline);exits=pd.Series(pd.to_datetime(baseline.etq1).to_numpy(),index=_key(baseline))
        p['exit_date']=p[ID].map(exits)
    return p


def exposure_frame(p,column,datecol='date_of_test',*,orientation=1.,observation_end=None,respect_exit=False,threshold=None,age_range=None,required_covariates=()):
    """One outcome's exact-date, available-case frame and exposure indicators."""
    f=p.copy();f['test_date']=pd.to_datetime(f[datecol]);f['outcome_value']=pd.to_numeric(f[column],errors='coerce')*orientation
    keep=f.outcome_value.notna() & f.test_date.notna() & f.height.notna() & f.male.notna()
    for c in required_covariates:keep &= f[c].notna()
    if observation_end is not None:keep &= f.test_date.le(pd.Timestamp(observation_end))
    if respect_exit:keep &= f.exit_date.isna() | f.test_date.le(f.exit_date)
    age=(f.test_date-f.child_dob).dt.days/365.25
    if age_range is not None:keep &= age.between(*age_range)
    post=f.tb_first_dx.notna() & f.test_date.ge(f.tb_first_dx)
    if threshold is not None:keep &= ~(post & f.tb_age_first_years.ge(threshold))
    f=f.loc[keep].copy();post=post.loc[f.index]
    f['age_years']=age.loc[f.index]
    f['future_tb']=(f.tb_first_dx.notna() & ~post).astype(float)
    f['post_any']=post.astype(float)
    f['post_before5']=(post & f.tb_age_first_years.lt(5)).astype(float)
    f['post_infancy']=(post & f.tb_age_first_years.lt(1)).astype(float)
    f['post_age1_before5']=(post & f.tb_age_first_years.ge(1) & f.tb_age_first_years.lt(5)).astype(float)
    f['post_at_after5']=(post & f.tb_age_first_years.ge(5)).astype(float)
    f['post_tb_lt5']=f.post_before5;f['post_tb_infancy']=f.post_infancy;f['post_tb_late']=f.post_at_after5
    return f


def standardized(values,*,allow_constant=False):
    x=pd.to_numeric(values,errors='raise').astype(float);sd=x.std(ddof=1)
    if not np.isfinite(sd) or sd<=0:
        if allow_constant:return pd.Series(np.zeros(len(x)),index=x.index)
        raise ValueError('A model covariate has no finite positive scale.')
    return (x-x.mean())/sd


def residualize(frame,*,covariates=(),scale_method='std'):
    f=frame.copy();f['height_z']=(f.height-f.height.mean())/f.height.std(ddof=1)
    formula='1+C(visit_label)+height_z+male'+''.join('+'+c for c in covariates)
    x=patsy.dmatrix(formula,f,return_type='dataframe',NA_action='raise')
    train=f.post_any.eq(0)
    if train.sum()<=x.shape[1]:raise ValueError('Insufficient pre/no-TB observations for residual scaling.')
    beta=np.linalg.lstsq(x.loc[train].to_numpy(),f.loc[train,'outcome_value'].to_numpy(),rcond=None)[0]
    residual=f.outcome_value.to_numpy()-x.to_numpy()@beta
    scale=float(np.std(residual[train.to_numpy()],ddof=x.shape[1])) if scale_method=='std' else float(np.sqrt(np.sum(residual[train.to_numpy()]**2)/(train.sum()-x.shape[1])))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid outcome residual scale.')
    f['y_std']=residual/scale
    return f,x,{'n_observations':len(f),'n_children':int(f[ID].nunique()),'n_training':int(train.sum()),'scale':scale,'nuisance_terms':list(x.columns),'nuisance_coefficients':beta.tolist()}


def event_arrays(children,*,threshold=None,event_end=None):
    """First-TB risk stops at observed follow-up, optionally capped by age.

    Diagnosis on the exit/admin date is observed. A threshold is a strict
    diagnosis-age estimand, not evidence of surveillance after study exit.
    Known diagnoses after observed follow-up do not become threshold events.
    """
    if event_end is None:raise ValueError('Event preparation needs an explicit endpoint.')
    endpoint=pd.Timestamp(event_end)
    if pd.isna(endpoint):raise ValueError('Event endpoint is missing.')
    censor=pd.concat([children.exit_date,pd.Series(endpoint,index=children.index)],axis=1).min(axis=1)
    censor_age=(censor-children.child_dob).dt.days.to_numpy(float)/365.25
    age=children.tb_age_first_years.to_numpy(float)
    diagnosed=children.tb_first_dx.notna().to_numpy()
    if (children.child_dob.isna().any() or (~np.isfinite(censor_age)).any()
            or (censor_age<=0).any()):raise ValueError('Invalid event follow-up duration.')
    if (children.tb_first_dx.lt(children.child_dob).any()
            or (diagnosed & (~np.isfinite(age) | (age<0))).any()
            or (~diagnosed & ~np.isnan(age)).any()):
        raise ValueError('First diagnosis chronology or age is malformed.')
    if threshold is not None:
        if isinstance(threshold,(bool,np.bool_)) or threshold not in (1,5):
            raise ValueError('Threshold must be one or five years.')
        event=(diagnosed & (age<threshold) & children.tb_first_dx.le(censor).to_numpy()).astype(int)
        stop=np.minimum(censor_age,float(threshold))
        stop=np.where(event,np.minimum(age,stop),stop)
        breaks=np.array([x for x in [0.,1.,3.,5.] if x<=threshold])
    else:
        event=diagnosed.astype(int)
        valid=children.tb_first_dx.isna() | (children.tb_first_dx.ge(children.child_dob)&children.tb_first_dx.le(censor))
        if not valid.all():raise ValueError('First diagnosis lies outside configured event follow-up.')
        stop=np.where(event,age,censor_age)
        breaks=np.array([0.,1.,3.,5.,8.,10.,20.])
    if (stop<=0).any() or (stop>breaks[-1]).any():raise ValueError('Invalid event follow-up duration.')
    exposure=np.maximum(0.,np.minimum(stop[:,None],breaks[1:])-breaks[:-1])
    interval=event[:,None]*((stop[:,None]>=breaks[:-1])&(stop[:,None]<breaks[1:]))
    if not np.allclose(exposure.sum(1),stop) or not np.array_equal(interval.sum(1),event):raise ValueError('Event intervals do not reconcile.')
    result={'event':event,'stop':stop,'exposure':exposure,'event_interval':interval,'breaks':breaks}
    if threshold is not None:result['ascertainment_end']=endpoint.date().isoformat()
    return result


def joint_inputs(p,target,*,family='primary',event_end=None,observation_end=None,covariates=(),threshold=5):
    """Construct linked Gaussian processes, first-event arrays, and contrast weights."""
    p=p.dropna(subset=list(covariates)).copy()
    children=p.sort_values(ID).drop_duplicates(ID).copy();children['child_index']=np.arange(len(children))
    indices=children.set_index(ID).child_index
    allage=family in ['primary','mechanics'];specs={}
    if family=='mechanics':
        native,orientation=target;specs={'target':(native,'date_of_test_ilf2',orientation)}
    elif family=='x6':specs={**MECHANICS,'target':(target,'date_of_test_ilf2',1.)}
    else:
        col,orientation=target
        specs={'lci':('i_lci_2_5','date_of_test_ilf1',-1.),**MECHANICS,'target':(col,'date_of_test',orientation)}
    frames={};designs={};audits={}
    for key,(column,date,orientation) in specs.items():
        f=exposure_frame(p,column,date,orientation=orientation,observation_end=observation_end,respect_exit=True,threshold=None if allage else threshold,age_range=(3,17) if family=='x6' and key=='target' else None,required_covariates=covariates)
        f,x,a=residualize(f,covariates=covariates,scale_method='sse' if family=='x6' else 'std')
        f['child_index']=f[ID].map(indices).astype(int)
        frames[key]=f;designs[key]=x;audits[key]=a
    f=frames['target'];post=f.loc[f.post_any.eq(1)].drop_duplicates(ID)
    weights=[int(post.tb_age_first_years.lt(1).sum()),int(post.tb_age_first_years.between(1,5,inclusive='left').sum()),int(post.tb_age_first_years.ge(5).sum())]
    events=event_arrays(children,threshold=None if allage else threshold,event_end=event_end)
    hazard_terms=['male',*[c for c in covariates if c!='ever_lrti']]
    post_terms=['post_any'] if family=='x6' else (['post_before5','post_infancy','post_at_after5'] if allage else ['post_before5','post_infancy'])
    for key,f in frames.items():
        f['post_target_tb']=f.post_any
        f['is_pre_own_tb']=f.future_tb.astype(bool)
    return {'children':children,'frames':frames,'designs':designs,'event':events,'weights':np.array(weights),'hazard_design':children[hazard_terms].to_numpy(float),'hazard_terms':hazard_terms,'post_diagnosis_designs':{key:f[post_terms].to_numpy(float) for key,f in frames.items()},'post_diagnosis_terms':post_terms,'audit':{'family':family,'joint_children':len(children),'tb_events':int(events['event'].sum()),'target':specs['target'][0],'weights':weights,'outcomes':audits,'covariates':list(covariates),'hazard_terms':hazard_terms,'post_diagnosis_terms':post_terms}}


def drop_design_columns(x,names,required=()):
    kept=[];terms=[];dropped=[]
    for j,name in enumerate(names):
        col=x[:,j]
        if name!='Intercept' and np.std(col)<1e-12:
            if name in required:raise ValueError('A required exposure term is constant.')
            dropped.append(name+' (constant)');continue
        candidate=np.column_stack(kept+[col])
        if np.linalg.matrix_rank(candidate,tol=1e-10)<candidate.shape[1]:
            if name in required:raise ValueError('A required exposure term is collinear.')
            dropped.append(name+' (collinear)');continue
        kept.append(col);terms.append(name)
    return np.column_stack(kept),terms,dropped


def repeated_design(frame,exposure_terms):
    columns=[np.ones(len(frame))]+[frame[t].to_numpy(float) for t in exposure_terms]+[standardized(frame.height,allow_constant=True).to_numpy(),frame.male.to_numpy(float)]
    terms=['Intercept',*exposure_terms,'height_z','sex:Male']
    present=[v for v in VISITS if frame.visit_label.eq(v).any()]
    present+=sorted(set(frame.visit_label)-set(present))
    for visit in present[1:]:columns.append(frame.visit_label.eq(visit).to_numpy(float));terms.append('visit:'+visit)
    # Zero-support categories intentionally yield no coefficient and are marked
    # nonestimable by the downstream model. A supported exposure is required;
    # a constant nonzero or collinear column must not disappear silently.
    required={term for term in exposure_terms if frame[term].ne(0).any()}
    return drop_design_columns(np.column_stack(columns),terms,required)


def binary_inputs(p,*,stage='pre',outcome='ratio_LLN',observation_end=None):
    suffix='' if stage=='pre' else '_post'
    columns={'ratio_LLN':'fev1_fvc','FEV1_LLN':'fev1','FVC_LLN':'fvc','obstructive_pattern':'fev1_fvc','possible_restriction':'fev1_fvc','any_abnormal':'fev1_fvc'}
    f=exposure_frame(p,columns[outcome]+suffix+'_gli_z',observation_end=observation_end,respect_exit=True)
    if outcome in ['obstructive_pattern','possible_restriction','any_abnormal']:
        f=f.loc[f['fvc'+suffix+'_gli_z'].notna()].copy()
        ratio=f['fev1_fvc'+suffix+'_gli_z'].lt(-1.645);fvc=f['fvc'+suffix+'_gli_z'].lt(-1.645)
        y={'obstructive_pattern':ratio&~fvc,'possible_restriction':fvc&~ratio,'any_abnormal':ratio|fvc}[outcome]
    else:y=f.outcome_value.lt(-1.645)
    f['outcome_value']=y.astype(int)
    designs={}
    for key,terms in [('before5',['post_before5','post_at_after5','future_tb']),('age_groups',['post_infancy','post_age1_before5','post_at_after5','future_tb'])]:
        x,names,dropped=repeated_design(f,terms);designs[key]={'matrix':x,'terms':names,'dropped_terms':dropped}
    return {'frame':f,'designs':designs,'audit':{'family':'binary','stage':stage,'outcome':outcome,'n_observations':len(f),'n_children':int(f[ID].nunique()),'events':int(y.sum()),'designs':{k:{'terms':v['terms'],'dropped_terms':v['dropped_terms']} for k,v in designs.items()}}}


def trajectory_inputs(p,*,outcome='fev1_fvc',threshold=1,x6=False):
    date='date_of_test_ilf2' if x6 else 'date_of_test';bounds=(3.,10.) if x6 else (5.,10.)
    f=exposure_frame(p,outcome,date,orientation=100. if outcome=='fev1_fvc' else 1.,age_range=bounds)
    if not x6:f=f.loc[f.fev1_fvc.notna() & f.fev1_fvc_gli_z.notna()].copy()
    f=f.sort_values([ID,'test_date']).reset_index(drop=True)
    formula=f'cr(age, df=3, constraints="center", lower_bound={bounds[0]:g}, upper_bound={bounds[1]:g}) - 1'
    basis=patsy.dmatrix(formula,{'age':f.age_years},return_type='dataframe')
    target=f.post_any.eq(1)&f.tb_age_first_years.lt(threshold);later=f.post_any.eq(1)&f.tb_age_first_years.ge(threshold)
    cols=[np.ones(len(f)),target.to_numpy(float),later.to_numpy(float),f.future_tb.to_numpy(float),standardized(f.height).to_numpy(),f.male.to_numpy(float)]
    names=['Intercept','target','later','future_tb','height_z','sex:Male']
    for i in range(3):cols.append(basis.iloc[:,i].to_numpy());names.append(f'age_spline_{i+1}')
    for i in range(3):cols.append(target.to_numpy()*basis.iloc[:,i].to_numpy());names.append(f'target:age_spline_{i+1}')
    x,names,dropped=drop_design_columns(np.column_stack(cols),names,{'target',*[f'target:age_spline_{i+1}' for i in range(3)]})
    if dropped:raise ValueError('Trajectory design unexpectedly dropped terms.')
    reduced=[i for i,n in enumerate(names) if ':age_spline_' not in n]
    return {'frame':f,'design':x,'terms':names,'reduced_design':x[:,reduced],'reduced_terms':[names[i] for i in reduced],'basis':np.asarray(basis),'spline_formula':formula,'audit':{'family':'trajectory','outcome':outcome,'threshold':threshold,'n_observations':len(f),'n_children':int(f[ID].nunique()),'spline_formula':formula,'terms':names}}


def pretb_cox_inputs(p,column,orientation,reference_z,*,observation_end,window_years=2.):
    """Construct selected counting-process risk intervals; no survival fitting."""
    f=p.copy();f['test_date']=f.date_of_test_ilf2;f['test_age']=(f.test_date-f.child_dob).dt.days/365.25
    valid=pd.to_numeric(f[column],errors='coerce').notna()&f.test_date.notna()&f.child_dob.notna()&f.height.notna()&f.male.notna()&f.site.notna()&f.test_date.le(pd.Timestamp(observation_end))&f.test_age.between(.08,12.)&(f.tb_first_dx.isna()|f.test_date.lt(f.tb_first_dx))&pd.to_numeric(f.base_survivaltb_final,errors='coerce').gt(f.test_age)
    f=f.loc[valid].sort_values([ID,'test_date']).drop_duplicates([ID,'test_date']).copy()
    y=pd.to_numeric(f[column],errors='coerce').to_numpy()*orientation
    if reference_z:f['lf_adverse_sd']=y
    else:
        # Numeric scheduled-visit codes determine the reference-category order here.
        visit_codes=f.visit_label.map({v:i for i,v in enumerate(VISITS)})
        if visit_codes.isna().any():raise ValueError('Cox reference standardization has an unsupported visit.')
        visits=pd.get_dummies(visit_codes.astype('Int64').astype(str),drop_first=True,dtype=float)
        x=np.column_stack([np.ones(len(f)),visits.to_numpy(),standardized(f.height),f.male])
        b=np.linalg.lstsq(x,y,rcond=None)[0];residual=y-x@b;scale=np.std(residual,ddof=x.shape[1])
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid pre-TB reference residual scale.')
        f['lf_adverse_sd']=residual/scale
    f['mbekweni']=pd.to_numeric(f.site,errors='coerce').eq(1).astype(float)
    rows=[]
    for child,g in f.groupby(ID,sort=False):
        ages=g.test_age.to_numpy();stop_age=float(g.base_survivaltb_final.iloc[0]);tb_age=float(g.tb_age_first_years.iloc[0])
        for j,(_,row) in enumerate(g.iterrows()):
            start=float(row.test_age);stop=min(stop_age,ages[j+1] if j+1<len(g) else np.inf,start+window_years)
            event=int(np.isfinite(tb_age) and start<tb_age<=stop)
            if event:stop=tb_age
            if stop>start:rows.append({ID:child,'start':start,'stop':stop,'event':event,'lf_adverse_sd':row.lf_adverse_sd,'male':row.male,'mbekweni':row.mbekweni})
    intervals=pd.DataFrame(rows,columns=[ID,'start','stop','event','lf_adverse_sd','male','mbekweni'])
    return {'frame':intervals,'design':intervals[['lf_adverse_sd','male','mbekweni']].to_numpy(float),'terms':['lf_adverse_sd','male','mbekweni'],'audit':{'family':'pretb_cox','column':column,'n_children':int(intervals[ID].nunique()),'n_intervals':len(intervals),'events':int(intervals.event.sum()),'person_years':float((intervals.stop-intervals.start).sum()),'carry_forward_years':window_years}}


def prepare_model_inputs(pft,baseline,*,event_end,observation_end=None):
    """Build all selected PFT-family preparation variants, returning private frames."""
    p=normalize_pft(pft,baseline)
    samples={}
    for key,target in PRIMARY_TARGETS.items():samples['primary/'+key]=joint_inputs(p,target,event_end=event_end,observation_end=observation_end)
    for stage in ['pre','post']:
        samples['window/'+stage]=joint_inputs(p,PRIMARY_TARGETS['ratio_'+stage],event_end=event_end,observation_end=event_end)
        for outcome in ['ratio_LLN','FEV1_LLN','FVC_LLN','obstructive_pattern','possible_restriction','any_abnormal']:
            samples[f'binary/{stage}/{outcome}']=binary_inputs(p,stage=stage,outcome=outcome,observation_end=observation_end)
    for key,column,orient in [('R','resistance',1.),('C','compliance',-1.),('ReE','ree',1.),('ReI','rei',1.),('XeE','xee',-1.),('XeI','xei',-1.)]:samples['mechanics/'+key]=joint_inputs(p,(column,orient),family='mechanics',event_end=event_end,observation_end=observation_end)
    for key,cov in [('core',()),('maternal_hiv',('maternal_hiv',)),('maternal_smoking',('maternal_smoking',)),('birthweight',('birthweight_z',)),('all_baseline',('maternal_hiv','maternal_smoking','birthweight_z')),('ever_lrti',('ever_lrti',)),('all_four',('maternal_hiv','maternal_smoking','birthweight_z','ever_lrti'))]:samples['sensitivity/'+key]=joint_inputs(p,('fev1_fvc',100.),family='early',covariates=cov,event_end=event_end)
    samples['sensitivity/core_window']=joint_inputs(p,('fev1_fvc',100.),family='early',event_end=event_end,observation_end=event_end)
    for key,column in [('unpaired','mmef'),('pre_paired','mmef_pre_paired'),('post_paired','mmef_post_paired')]:samples['mmef/'+key]=joint_inputs(p,(column,1.),family='early',event_end=event_end)
    f=exposure_frame(p,'mmef');f=f.loc[f.mmef_post.notna()].copy();f['outcome_value']=f.mmef_post-f.mmef
    if f.duplicated([ID,'date_of_test']).any():raise ValueError('Paired MMEF has duplicate child/test-date records.')
    designs={}
    for key,terms in [('before5',['post_before5','post_at_after5','future_tb']),('infancy',['post_infancy','post_age1_before5','post_at_after5','future_tb'])]:
        x,names,dropped=repeated_design(f,terms);designs[key]={'matrix':x,'terms':names,'dropped_terms':dropped}
    samples['mmef/change']={'frame':f,'designs':designs,'audit':{'family':'mmef_change','n_observations':len(f),'n_children':int(f[ID].nunique())}}
    for threshold in [1,5]:
        for outcome in ['fev1_fvc','fev1_fvc_gli_z']:samples[f'trajectory/{outcome}/{threshold}']=trajectory_inputs(p,outcome=outcome,threshold=threshold)
        samples[f'trajectory/x6/{threshold}']=trajectory_inputs(p,outcome='x6_sa_z',threshold=threshold,x6=True)
        for outcome in ['x6_sa_z','reactance_6hz']:
            samples[f'x6_joint/{outcome}/{threshold}']=joint_inputs(p,outcome,family='x6',threshold=threshold,event_end=event_end)
            f=exposure_frame(p,outcome,'date_of_test_ilf2',age_range=(3,17));f['target']=(f.post_any.eq(1)&f.tb_age_first_years.lt(threshold)).astype(float);f['later']=(f.post_any.eq(1)&f.tb_age_first_years.ge(threshold)).astype(float)
            x,names,dropped=repeated_design(f,['target','later','future_tb'])
            samples[f'x6_repeated/{outcome}/{threshold}']={'frame':f,'design':x,'terms':names,'audit':{'family':'x6_repeated','n_observations':len(f),'n_children':int(f[ID].nunique()),'terms':names,'dropped_terms':dropped}}
    for col,orient,z in [('resistance',1,False),('compliance',-1,False),('ree',1,False),('rei',1,False),('xee',-1,False),('xei',-1,False),('dr_mean',1,False),('dx_mean',1,False),('r6_sa_z',1,True),('r8_sa_z',1,True),('r10_sa_z',1,True),('x6_sa_z',-1,True),('x8_sa_z',-1,True),('x10_sa_z',-1,True),('fres_sa_z',1,True),('r6_r20_sa_z',1,True),('ax_sa_z',1,True)]:samples['pretb_cox/'+col]=pretb_cox_inputs(p,col,orient,z,observation_end=event_end)
    return {'samples':samples,'audit':{'sample_contracts':len(samples),'samples':{key:value['audit'] for key,value in samples.items()}}}
