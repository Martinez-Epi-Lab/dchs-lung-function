"""Deterministic reference scores and paired spirometry columns.

The caller supplies the GLI Global lookup and retains its version/hash. No
network access, archived-code imports, model fitting, or file writes occur.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def pft_male(values: pd.Series, labels: dict | None = None) -> pd.Series:
    """Decode PFT sex: numeric 0 is male and 1 is female; labels are explicit."""
    text = values.map(labels) if labels is not None else values
    if labels is not None and (values.notna() & text.isna()).any():
        raise ValueError('PFT sex contains a code missing from its label mapping.')
    text = text.astype('string').str.strip().str.lower()
    mapping = {'male': 1., 'female': 0., '0': 1., '0.0': 1., '1': 0., '1.0': 0.}
    bad = text.notna() & ~text.isin(mapping)
    if bad.any():
        raise ValueError('PFT sex contains an unsupported code or label.')
    return text.map(mapping).astype(float)


def load_gli_lookup(path: str | Path) -> pd.DataFrame:
    """Load the externally configured rspiro GLI Global R-data lookup."""
    import rdata
    lookup = rdata.read_rda(path)['GLIgl_lookup'].copy()
    return validate_gli_lookup(lookup)


def validate_gli_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    required = ['agebound', 'gender', 'f', 'Mspline', 'Sspline']
    if not set(required).issubset(lookup.columns):
        raise ValueError('GLI lookup is missing required coefficients or keys.')
    result = lookup.copy()
    for column in ['agebound', 'gender', 'Mspline', 'Sspline']:
        result[column] = pd.to_numeric(result[column], errors='raise')
    if result[required].isna().any().any() or result.duplicated(['agebound', 'gender', 'f']).any():
        raise ValueError('GLI lookup contains missing values or duplicate keys.')
    return result


# Coefficients are (M intercept, log-height slope, log-age slope,
# S intercept, log-age slope, L intercept, L log-age slope).
_GLI = {
    'fev1': {1: (-11.399108,2.462664,-.011394,-2.256278,.080729,1.22703,0),
             2: (-10.901689,2.385928,-.076386,-2.364047,.129402,1.21388,0)},
    'fvc': {1: (-12.629131,2.727421,.009174,-2.195595,.068466,.9346,0),
            2: (-12.055901,2.621579,-.035975,-2.310148,.120428,.899,0)},
    'fev1_fvc': {1: (1.022608,-.218592,-.027586,-2.882025,.068889,3.8243,-.3328),
                 2: (.9189568,-.1840671,-.0461306,-3.171582,.144358,6.6490,-.9920)},
}


def add_gli_global_zscores(frame: pd.DataFrame, lookup: pd.DataFrame,
                          *, suffix: str = '', sex_labels: dict | None = None) -> pd.DataFrame:
    """Append pre or post LMS scores without changing measured source values."""
    lookup = validate_gli_lookup(lookup)
    result = frame.copy()
    # Positional working index prevents duplicate caller index labels multiplying rows.
    work = frame.reset_index(drop=True)
    age = (pd.to_datetime(work.date_of_test)-pd.to_datetime(work.child_dob)).dt.days/365.25
    height = pd.to_numeric(work.height, errors='coerce')
    male = pft_male(work.sex, sex_labels)
    gender = pd.Series(np.where(male.eq(1),1,np.where(male.eq(0),2,np.nan)))
    bound = np.floor(age*4)/4
    result['spiro_age_years'] = age.to_numpy()
    result['gli_global_agebound'] = bound.to_numpy()
    result['gli_global_gender'] = gender.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        logage, logheight = np.log(age), np.log((height / 100.0) * 100.0)
        for stem, table in _GLI.items():
            obs = pd.to_numeric(work[stem+suffix], errors='coerce')
            keys = pd.DataFrame({'agebound':bound,'gender':gender,'f':{'fev1':'FEV1','fvc':'FVC','fev1_fvc':'FEV1FVC'}[stem]})
            joined = keys.merge(lookup,on=['agebound','gender','f'],how='left',sort=False,validate='many_to_one')
            M, S, L = (pd.Series(np.nan,index=work.index) for _ in range(3))
            for sex, c in table.items():
                mask = gender.eq(sex)
                M.loc[mask] = np.exp(c[0]+c[1]*logheight+c[2]*logage+joined.Mspline).loc[mask]
                S.loc[mask] = np.exp(c[3]+c[4]*logage+joined.Sspline).loc[mask]
                L.loc[mask] = (c[5]+c[6]*logage).loc[mask]
            eligible = age.between(3,95) & height.gt(0) & gender.isin([1,2]) & obs.gt(0)
            if (eligible & (joined.Mspline.isna() | joined.Sspline.isna())).any():
                raise ValueError('GLI lookup does not cover an eligible measurement.')
            valid = eligible & M.gt(0) & S.gt(0) & L.notna()
            z = (((obs/M)**L)-1)/(L*S)
            lln = M*(1-1.645*L*S)**(1/L)
            for field, values in [('gli_z',z),('gli_pred',M),('gli_lln',lln)]:
                result[f'{stem}{suffix}_{field}'] = values.where(valid).to_numpy()
    return result


@dataclass(frozen=True)
class OscEquation:
    column: str
    z_column: str
    transform: str
    intercept: float
    slope: float
    see: float
    multiplier: float = 1.


OSC_EQUATIONS = (
    OscEquation('resistance_6hz','r6_sa_z','log',4.34,-.0189,.214),
    OscEquation('resistance_8hz','r8_sa_z','log',4.29,-.0190,.210),
    OscEquation('resistance_10hz','r10_sa_z','log',4.27,-.0191,.204),
    OscEquation('reactance_6hz','x6_sa_z','inverse',3.46,-727.,1.168),
    OscEquation('reactance_8hz','x8_sa_z','inverse',3.31,-647.,.995),
    OscEquation('reactance_10hz','x10_sa_z','inverse',2.73,-531.,.938),
    OscEquation('fres_final','fres_sa_z','log',3.74,-.0062,.211),
    OscEquation('r6_r20','r6_r20_sa_z','linear',5.67,-.0311,1.227),
    OscEquation('ax_mean','ax_sa_z','log',6.35,-.0287,.624),
    OscEquation('resistance','resistance_sa_z','log',4.16,-.0187,.205,.980665),
    OscEquation('compliance','compliance_sa_z','log',.099,.0168,.295,1000/.980665),
    OscEquation('ree','ree_sa_z','log',4.36,-.0204,.225),
    OscEquation('rei','rei_sa_z','log',4.32,-.0208,.233),
    OscEquation('xee','xee_sa_z','inverse',2.17,-409.,1.178),
    OscEquation('xei','xei_sa_z','inverse',3.22,-577.,.943),
    OscEquation('dr_mean','dr_sa_z','linear',2.26,-.0144,1.272),
    OscEquation('dx_mean','dx_sa_z','linear',1.80,-.0117,.928),
)


def add_oscillometry_zscores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    age = (pd.to_datetime(frame.date_of_test_ilf2)-pd.to_datetime(frame.child_dob)).dt.days/365.25
    height = pd.to_numeric(frame.height,errors='coerce')
    eligible = age.between(3,17) & height.gt(0)
    result['osc_age_years'] = age
    result['osc_sa_reference_eligible'] = eligible.astype(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        for eq in OSC_EQUATIONS:
            obs = pd.to_numeric(frame[eq.column],errors='coerce')*eq.multiplier
            if eq.transform == 'log':
                expected = eq.intercept + eq.slope * height
                values=(np.log(obs)-expected)/eq.see
                valid=eligible & obs.gt(0)
            else:
                expected=eq.intercept+eq.slope/(height) if eq.transform=='inverse' else eq.intercept+eq.slope*height
                values=(obs-expected)/eq.see
                valid=eligible & obs.notna()
            result[eq.z_column]=values.where(valid)
    return result


def add_reference_scores(frame: pd.DataFrame, lookup: pd.DataFrame,
                         *, sex_labels: dict | None = None) -> pd.DataFrame:
    result=add_gli_global_zscores(frame,lookup,sex_labels=sex_labels)
    result=add_oscillometry_zscores(result)
    result=add_gli_global_zscores(result,lookup,suffix='_post',sex_labels=sex_labels)
    for stem in ['fev1','fvc','fev1_fvc']:
        result[f'{stem}_pre_paired']=pd.to_numeric(result[stem],errors='coerce').where(pd.to_numeric(result[f'{stem}_post'],errors='coerce').notna())
    pre,post=(pd.to_numeric(result[c],errors='coerce') for c in ['mmef','mmef_post'])
    paired=pre.notna() & post.notna()
    result['mmef_pre_paired']=pre.where(paired)
    result['mmef_post_paired']=post.where(paired)
    return result
