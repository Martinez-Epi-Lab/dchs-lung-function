"""One keyed interface for full-precision, participant-free aggregate results.

The source adapters name row coordinates and the meaning of interval fields.
Rows retain their original numerical values. Tables and prose reference the
same result key even when they select a row using different equivalent filters.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from urllib.parse import quote

from .formatting import ReportingError


def _schema(source):
    p = source.split('/')
    family, table = p[0], p[-1]
    if family == 'preparation':
        keys = {'table1_statistics':['variable','group'], 'flow':['node'],
                'cohort_ledger':['population','outcome','step'],
                'modality_schedule':['modality','visit']}[table]
    elif family == 'facts':
        keys = {'early_tb_age':['population','diagnosis_group'],
                'early_tb_distribution':['fact'], 'unavailable_flow':['node']}[table]
    elif family in {'primary','mechanics','supporting','sensitivity_model'}:
        keys = (['process','group'] if table == 'sample_counts' else
                ['estimand'] if family == 'supporting' and table == 'hazard' else ['contrast'])
    elif family == 'binary':
        keys = ['model_id','term'] if table == 'estimates' else ['stage','exposure_state','pattern']
    elif family == 'trajectory':
        keys = ['model_id']+({'predictions':['age_years','group'],
                             'age_contrasts':['age_years']}.get(table,[]))
    elif family == 'shared_factor':
        keys = ['target']
    elif family == 'supporting_repeated':
        keys = (['group'] if table == 'mmef_descriptive' else
                ['model_id']+({'x6_predictions':['age_years','group'],
                              'x6_age_contrasts':['age_years']}.get(table,[])))
    elif family == 'pretb':
        keys = ['target']
    elif family == 'feno':
        keys = ['target']+(['group'] if table == 'counts' else [])
    elif family == 'wheeze':
        keys = {'effects':['process','age_band'], 'contrasts':['process','contrast'],
                'global_tests':['process'], 'standardized_rr':['age_band'],
                'support_counts':['process','age_band','exposure_group'],
                'posterior_predictive_cells':['process','age_band','exposure_group']}[table]
    elif family == 'sensitivity':
        keys = ['scenario','process','age_band'] if table=='table_s4' else ['target','contrast']
    else:
        raise ReportingError('Unknown aggregate result family.')
    return keys


def _semantics(source, row):
    family, table = source.split('/')[0], source.split('/')[-1]
    value = {'unit':row.get('unit'), 'intervals':[], 'point_estimators':{},
             'denominator_fields':{}, 'limitations':[]}
    support_suffixes = ('children','observations','measurements','intervals','events','pairs',
                        'positive','person_years','subsequent_tb','nonmissing','missing')
    for name in row:
        if (name.endswith(support_suffixes) or name in {'count','numerator','denominator','n_group',
                'records_before','records_excluded','records_retained','children_retained','observed_denominator'}) and isinstance(row[name],(int,float)):
            value['denominator_fields'][name] = ('Time at risk in person-years' if name.endswith('person_years')
                else 'Source-defined support for '+name.replace('_',' ')+'; not a common model denominator')
    if family in {'primary','supporting','shared_factor','sensitivity_model'} and 'mean' in row:
        value['point_estimators'] = {'mean':'posterior mean','median':'posterior median'}
        value['intervals'] = [{'lower':'lower','upper':'upper','level':0.95,'type':'equal-tailed credible interval'}]
    if family == 'mechanics' and 'mean' in row:
        value['unit'] = row.get('reported_unit',row.get('unit'))
        value['point_estimators'] = {'mean':'posterior mean on adverse-oriented standardized scale',
             'reported_mean':'posterior mean in reported native units', 'native_mean':'posterior mean in native units'}
        value['intervals'] = [{'lower':a+'lower','upper':a+'upper','level':0.95,
                               'type':'equal-tailed credible interval'} for a in ('','native_','reported_')
                              if a+'lower' in row]
        if table == 'hazard':
            value['point_estimators']['mean'] = 'posterior mean hazard ratio'
    if family in {'primary','mechanics','supporting','sensitivity_model'} and table == 'hazard':
        value['limitations'].append('Probability fields ending below_zero/above_zero use zero, not the hazard-ratio null of one.')
    if family == 'sensitivity':
        field = 'posterior_median' if table=='table_s4' else 'mean'
        value['point_estimators'] = {field:'posterior median' if table=='table_s4' else 'posterior mean'}
        value['intervals'] = [{'lower':'lower','upper':'upper','level':0.95,'type':'equal-tailed credible interval'}]
    if family == 'wheeze':
        value['point_estimators'] = {k:'posterior median' for k in row if k.endswith('_median')}
        value['point_estimators'].update({k:'posterior mean' for k in row if k.endswith('_mean')})
        value['unit'] = row.get('measure',row.get('units'))
        for low,high in [('cri_2_5','cri_97_5')]+[(p+'_low',p+'_high') for p in ('or','rr','no_tb_percent','tb_percent','difference_pp')]:
            if low in row:
                value['intervals'].append({'lower':low,'upper':high,'level':0.95,'type':'equal-tailed credible interval'})
        if table == 'global_tests':
            value['limitations'].append('Reference p value is a posterior Wald chi-square comparison, not a posterior probability.')
        if table == 'posterior_predictive_cells':
            value['intervals'].append({'lower':'replicated_2_5','upper':'replicated_97_5','level':0.95,
                                      'type':'posterior predictive interval'})
    if family in {'binary','trajectory','supporting_repeated','pretb','feno'}:
        descriptions={'rr':'fitted risk ratio','beta':'fitted log risk ratio',
            'hr_per_adverse_sd':'fitted hazard ratio per adverse SD',
            'geometric_mean_ratio':'fitted geometric mean ratio',
            'coefficient':'fitted log ratio coefficient',
            'estimate':'fitted adjusted difference',
            'parallel_trajectory_difference':'fitted parallel-trajectory contrast',
            'difference_tb_minus_reference':'fitted age-specific contrast',
            'predicted_mean':'fitted mean',
            'standardized_reference_change':'standardized reference mean change',
            'standardized_exposed_change':'standardized exposed mean change'}
        value['point_estimators']={k:v for k,v in descriptions.items() if k in row}
        if family == 'binary' and table=='estimates':value['unit']='dimensionless risk ratio'
        if family == 'feno' and table=='estimates':value['unit']='dimensionless geometric mean ratio'
        if family == 'pretb':value['unit']='dimensionless hazard ratio per adverse SD'
        for low,high in [('lower','upper'),('ci_low','ci_high'),('parallel_trajectory_ci_low','parallel_trajectory_ci_high')]:
            if low in row:
                value['intervals'].append({'lower':low,'upper':high,'level':0.95,
                    'type':'robust normal Wald confidence interval','normal_multiplier':1.96})
        if family == 'pretb':
            value['limitations'].append('G08: selected source covariance reproduced; pre-entry score-residual membership remains unresolved before Figure S3 and release.')
        if family == 'feno':
            value['limitations'].append('Uncorrected child-cluster sandwich and normal Wald inference reproduce the selected source method.')
        if family == 'supporting_repeated' and table == 'mmef_change':
            for prefix in ('standardized_reference','standardized_exposed'):
                value['intervals'].append({'lower':prefix+'_ci_low','upper':prefix+'_ci_high','level':0.95,
                    'type':'robust normal Wald confidence interval for standardized mean change','normal_multiplier':1.96})
            value['intervals'].append({'lower':'bootstrap_ci_low','upper':'bootstrap_ci_high','level':0.95,
                                      'type':'child-cluster bootstrap percentile confidence interval'})
    if family == 'facts' and table == 'early_tb_age':
        value['point_estimators'] = {'median':'descriptive sample median'}
        value['intervals'] = [{'lower':'q1','upper':'q3','level':0.5,'type':'descriptive interquartile range'}]
    value['point_estimators']={k:v for k,v in value['point_estimators'].items() if k in row}
    return value


def _clean(v):
    if hasattr(v,'item'):
        v = v.item()
    if isinstance(v,float) and math.isnan(v):
        return None
    if isinstance(v,float) and not math.isfinite(v):
        raise ReportingError('Infinite aggregate values are invalid; use explicit null and status.')
    if v is None or type(v) in (str,int,float,bool):
        return v
    raise ReportingError('Aggregate rows must contain only scalar values.')


class ResultRegistry:
    def __init__(self):
        self._records = {}
        self._sources = {}
        self._schema_contract = json.loads((Path(__file__).resolve().parents[3]/'config/reporting_schemas.json').read_text())

    def add(self, source, frame, provenance=None):
        """Register a whitelisted aggregate table, with unique semantic row keys."""
        if source in self._sources:
            raise ReportingError('An aggregate source was registered twice.')
        contract = self._schema_contract
        if source not in contract['sources']:
            raise ReportingError('Unknown aggregate source or target.')
        if list(frame.columns) != contract['schemas'][contract['sources'][source]]:
            raise ReportingError('Aggregate column inventory differs from the approved schema.')
        try:
            keys = _schema(source)
        except (KeyError,IndexError):
            raise ReportingError('Unknown aggregate table.') from None
        if not set(keys) <= set(frame.columns) or frame.columns.duplicated().any():
            raise ReportingError('Aggregate table lacks unique declared key columns.')
        prohibited = {'pid_child','child_id','participant_id','dob','date_of_birth','name','address'}
        if prohibited & {str(c).lower() for c in frame.columns}:
            raise ReportingError('Participant fields cannot enter the aggregate reporting registry.')
        records, pending = [],{}
        for raw in frame.to_dict('records'):
            row = {str(k):_clean(v) for k,v in raw.items()}
            selector = {k:row[k] for k in keys}
            if any(v is None for v in selector.values()):
                raise ReportingError('Aggregate keys cannot be missing.')
            key = source+'#'+'&'.join(k+'='+quote(json.dumps(v,ensure_ascii=False,separators=(',',':')),safe='') for k,v in selector.items())
            if key in self._records or key in pending:
                raise ReportingError('Aggregate semantic result keys must be unique.')
            status = row.get('status') or ('not_estimable' if row.get('estimable') is False else 'available')
            record = {'key':key,'source':source,'selector':selector,'values':row,'status':status,
                      'semantics':_semantics(source,row),'provenance':copy.deepcopy(provenance or {})}
            pending[key] = record
            records.append(record)
        optional_empty = {'supporting/'+t+'/hazard' for t in ('x6_z_before1','x6_z_before5','x6_native_before1','x6_native_before5')}
        optional_empty |= {'wheeze/secondary_sex_smoking_birthweight/standardized_rr',
                           'sensitivity_model/window_ratio_pre/hazard',
                           'sensitivity_model/window_ratio_post/hazard'}
        if not records and source not in optional_empty:
            raise ReportingError('Required aggregate source is empty.')
        self._records.update(pending)
        self._sources[source] = records

    def select(self, source, **where):
        if source not in self._sources or not where:
            raise ReportingError('A known aggregate source and explicit selector are required.')
        rows = self._sources[source]
        if not rows:
            raise ReportingError('This source has no selected estimands.')
        if any(k not in rows[0]['values'] for k in where):
            raise ReportingError('Aggregate selector names an unknown field.')
        def equal(a,b):
            return a==b and (not isinstance(a,bool) and not isinstance(b,bool) or type(a) is type(b))
        chosen = [r for r in rows if all(equal(r['values'][k],v) for k,v in where.items())]
        if len(chosen) != 1:
            raise ReportingError('Aggregate selector must match exactly one row.')
        return self._with_aliases(chosen[0])

    def _with_aliases(self, record):
        result = copy.deepcopy(record)
        source, row = result['source'], result['values']
        if source in {'sensitivity/table_s1','sensitivity/table_s3'}:
            if row['contrast'] not in {'First TB <1 year','First TB <5 years'}:
                raise ReportingError('Unknown sensitivity-table contrast.')
            target = row['target']
            primary = target in {'ratio_pre','ratio_post'}
            expanded = primary or target.startswith('window_ratio_')
            contrast = ('TB <1' if row['contrast']=='First TB <1 year' else 'TB <5 weighted') if expanded else (
                'First TB <1 year' if row['contrast']=='First TB <1 year' else 'First TB <5 years, child-weighted')
            destination = ('primary/' if primary else 'sensitivity_model/')+target+'/estimates'
            canonical = self.select(destination,contrast=contrast)
            fields = {v:v for v in ('mean','lower','upper')}
            if row['unit'] != canonical['values']['unit']:
                raise ReportingError('A repeated estimate has conflicting units.')
        elif source == 'sensitivity/table_s4':
            canonical = self.select('wheeze/'+row['scenario']+'/effects',process=row['process'],age_band=row['age_band'])
            fields = {'posterior_median':'posterior_median','lower':'cri_2_5','upper':'cri_97_5'}
            if row['measure'] != canonical['values']['measure']:
                raise ReportingError('A repeated wheeze estimate has a conflicting estimand.')
        else:
            return result
        if (row['fit_fingerprint'] != canonical['provenance'].get('fit_fingerprint')
                or row['validation_fingerprint'] != canonical['provenance'].get('fingerprint')
                or row['interval_type'] != 'equal-tailed 95% credible interval'):
            raise ReportingError('A repeated estimate has conflicting fit provenance or uncertainty semantics.')
        for old,new in fields.items():
            if row[old] != canonical['values'][new]:
                raise ReportingError('A repeated table estimate differs from its canonical model result.')
        result['aliases'] = {old:{'result_key':canonical['key'],'field':new} for old,new in fields.items()}
        return result

    def get(self, key):
        if key not in self._records:
            raise ReportingError('Unknown result key.')
        return self._with_aliases(self._records[key])

    def export(self):
        return {'schema_version':1,'records':[self._with_aliases(self._records[k]) for k in sorted(self._records)],
                'sources':sorted(self._sources),'record_count':len(self._records)}
