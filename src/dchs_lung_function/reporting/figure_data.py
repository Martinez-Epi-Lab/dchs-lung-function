"""Resolve publication panels from full-precision aggregate results."""
from __future__ import annotations

import copy
import csv
import io
import json
import math
from pathlib import Path
from string import Formatter
from urllib.parse import quote

from .formatting import ReportingError, cell, number
from .results import _schema, _semantics

FIGURE_IDS = ('main_figure_1', 'main_figure_2', 'main_figure_3',
              'appendix_figure_s1', 'appendix_figure_s2', 'appendix_figure_s3',
              'appendix_figure_s4', 'appendix_figure_s5', 'appendix_figure_s6')
EXPECTED_PANELS = (4, 4, 2, 1, 1, 1, 2, 1, 1)


def require(condition, message):
    if not condition:
        raise ReportingError('Figure data '+message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal(a, b):
    return a == b and (not isinstance(a, bool) and not isinstance(b, bool) or type(a) is type(b))


def object_fields(value, required, optional=()):
    require(isinstance(value,dict) and set(required)<=set(value)<=set(required)|set(optional),
            'object fields differ from the declared publication schema.')


def strings(value):
    return isinstance(value,list) and all(isinstance(x,str) and x for x in value)


def validate_specification(specification):
    """Reject undeclared nested content before any specification is exported."""
    object_fields(specification,('schema_version','figures'))
    require(type(specification['schema_version']) is int and specification['schema_version']==1
            and isinstance(specification['figures'],list), 'specification schema differs.')
    def axis(value):
        object_fields(value,('label','scale','null'),('ticks','limits','shared_group'))
        require(isinstance(value['label'],str) and value['scale'] in ('linear','log')
                and (value['null'] is None or finite(value['null'])), 'axis schema differs.')
        if 'shared_group' in value:
            require(isinstance(value['shared_group'],str) and value['shared_group'], 'shared axis group must be named.')
        for name in ('ticks','limits'):
            if name in value:
                vals=value[name]
                require(isinstance(vals,list) and vals and all(finite(v) for v in vals)
                        and vals==sorted(set(vals)) and (value['scale']!='log' or min(vals)>0), 'axis positions are invalid.')
                if name=='limits':require(len(vals)==2,'axis limits require two ordered bounds.')
    def fields(value, expected):
        object_fields(value,expected)
        require(all(isinstance(v,str) and v for v in value.values()), 'source fields must be named.')
    def series(panel):
        order=panel['series_order']
        require(strings(order) and order and len(order)==len(set(order)), 'series order differs.')
        object_fields(panel['series_labels'],order);object_fields(panel['series_styles'],order)
        require(all(isinstance(v,str) for v in panel['series_labels'].values()),'series labels must be text.')
        for style in panel['series_styles'].values():
            object_fields(style,('label','color','line') if panel['kind']=='trajectory' else ('label','color','marker','fill'))
            require(all(isinstance(v,str) for v in style.values()), 'series styles must be text.')
    for figure in specification['figures']:
        object_fields(figure,('id','title','kind','panels','caption','limitations'))
        require(all(isinstance(figure[k],str) for k in ('id','title','kind'))
                and strings(figure['limitations']) and isinstance(figure['panels'],list), 'figure schema differs.')
        for panel in figure['panels']:
            require(isinstance(panel,dict) and panel.get('kind')==figure['kind'], 'panel kind differs.')
            kind=panel['kind']
            common=('id','title','kind','estimate','interval','series_order','series_labels','series_styles')
            if kind=='forest':
                object_fields(panel,(*common,'rows','row_order','orientation','value_axis','annotation'),('support_annotation',))
                axis(panel['value_axis']);series(panel)
                object_fields(panel['annotation'],('show','style','digits'))
                require(type(panel['annotation']['show']) is bool and panel['annotation']['style']=='interval'
                        and type(panel['annotation']['digits']) is int and 0<=panel['annotation']['digits']<=6,
                        'estimate annotation is invalid.')
                if 'support_annotation' in panel:
                    object_fields(panel['support_annotation'],('template','heading','digits'))
                    require(panel['support_annotation']['template']=='{children}/{events}'
                            and isinstance(panel['support_annotation']['heading'],str)
                            and type(panel['support_annotation']['digits']) is int and panel['support_annotation']['digits']==0,
                            'support annotation schema differs.')
                require(isinstance(panel['rows'],list) and strings(panel['row_order']), 'forest rows must be lists.')
                for row in panel['rows']:
                    object_fields(row,('id','label','series','source','where','fields'),('support_fields','unit'))
                    require(all(isinstance(row[k],str) for k in ('id','label','series','source'))
                            and isinstance(row['where'],dict), 'forest row schema differs.')
                    fields(row['fields'],('point','lower','upper'))
                    if 'support_fields' in row:
                        fields(row['support_fields'],('children','events'))
                        require('support_annotation' in panel,'support annotation is required.')
                    if 'unit' in row:require(isinstance(row['unit'],str),'row unit must be text.')
            elif kind=='trajectory':
                object_fields(panel,(*common,'source','where','row_key','row_order','series_key','fields','x_axis','y_axis','population','annotations'))
                axis(panel['x_axis']);axis(panel['y_axis']);series(panel);fields(panel['fields'],('x','point','lower','upper'))
                require(panel['annotations']==[] and panel['row_order']=='ascending'
                        and all(isinstance(panel[k],str) for k in ('source','row_key','series_key','population'))
                        and isinstance(panel['where'],dict), 'trajectory schema differs.')
            elif kind=='flow':
                object_fields(panel,('id','kind','nodes','edges','unavailable_label','unavailable_nodes','outcome_samples_overlap'))
                require(isinstance(panel['nodes'],list) and isinstance(panel['edges'],list)
                        and all(strings(e) and len(e)==2 for e in panel['edges'])
                        and strings(panel['unavailable_nodes']) and isinstance(panel['unavailable_label'],str)
                        and type(panel['outcome_samples_overlap']) is bool, 'flow schema differs.')
                for node in panel['nodes']:
                    object_fields(node,('id','label','role','template','bindings','unit'))
                    require(all(isinstance(node[k],str) for k in ('id','label','role','template','unit'))
                            and isinstance(node['bindings'],dict), 'flow node schema differs.')
            elif kind=='schedule':
                object_fields(panel,('id','kind','source','where','row_key','column_key','row_order','column_order','row_labels','column_labels','fields','column_groups'))
                fields(panel['fields'],('represented','records','children'))
                require(all(isinstance(panel[k],str) for k in ('source','row_key','column_key'))
                        and isinstance(panel['where'],dict) and isinstance(panel['column_groups'],list), 'schedule schema differs.')
                for name in ('row','column'):
                    order=panel[name+'_order'];require(strings(order) and len(order)==len(set(order)), 'schedule order differs.')
                    object_fields(panel[name+'_labels'],order)
                    require(all(isinstance(v,str) for v in panel[name+'_labels'].values()),'schedule labels must be text.')
                for group in panel['column_groups']:
                    object_fields(group,('label','columns'))
                    require(isinstance(group['label'],str) and strings(group['columns'])
                            and set(group['columns'])<=set(panel['column_order']), 'schedule group schema differs.')
            else:raise ReportingError('Unknown publication panel kind.')
            require(isinstance(panel['id'],str),'panel identifier must be text.')
            for name in ('title','estimate','interval'):
                if name in panel:require(isinstance(panel[name],str),'panel labels must be text.')


class AggregateView:
    """Select unique rows from the verified M13 result registry."""
    def __init__(self, exported, corrected):
        require(isinstance(exported, dict) and set(exported) == {'schema_version','sources','record_count','records'}
                and type(exported['schema_version']) is int and exported['schema_version'] == 1
                and type(exported['record_count']) is int
                and isinstance(exported['records'],list) and strings(exported['sources'])
                and exported['record_count'] == len(exported['records']) == 3196,
                'requires the complete current M13 aggregate registry.')
        records = copy.deepcopy(exported['records'])
        for r in records:
            object_fields(r,('key','source','selector','values','status','semantics','provenance'),('aliases',))
            require(all(isinstance(r[k],str) for k in ('key','source','status'))
                    and all(isinstance(r[k],dict) for k in ('selector','values','semantics','provenance')),
                    'a result row is malformed.')
            if 'aliases' in r:
                require(isinstance(r['aliases'],dict), 'result aliases are malformed.')
                for alias in r['aliases'].values():
                    object_fields(alias,('field','result_key'))
                    require(all(isinstance(v,str) for v in alias.values()), 'result aliases must name aggregate fields.')
        contract=json.loads((Path(__file__).resolve().parents[3]/'config/reporting_schemas.json').read_text())
        optional_empty={'supporting/'+t+'/hazard' for t in ('x6_z_before1','x6_z_before5','x6_native_before1','x6_native_before5')}
        optional_empty|={'wheeze/secondary_sex_smoking_birthweight/standardized_rr',
                        'sensitivity_model/window_ratio_pre/hazard','sensitivity_model/window_ratio_post/hazard'}
        require(len(exported['sources']) == len(set(exported['sources'])) == 143
                and set(exported['sources'])==set(contract['sources'])
                and {r['source'] for r in records}<=set(exported['sources'])
                and set(exported['sources'])-{r['source'] for r in records}<=optional_empty, 'source coverage differs.')
        require(len({r['key'] for r in records}) == len(records), 'result keys must be unique.')
        for r in records:
            require(isinstance(r,dict) and {'key','source','selector','values','status','semantics','provenance'} <= set(r)
                    and isinstance(r['values'],dict) and isinstance(r['selector'],dict)
                    and all(k in r['values'] and equal(v,r['values'][k]) for k,v in r['selector'].items()),
                    'a result row or selector is malformed.')
            require(all(v is None or type(v) in (str,bool,int,float) and (not isinstance(v,float) or math.isfinite(v))
                        for v in r['values'].values()), 'aggregate values must be finite scalars or explicit nulls.')
            require(not {'pid_child','participant_id','child_id','dob','name','address'} & {k.lower() for k in r['values']},
                    'participant fields cannot enter plotting data.')
            require(set(r['values'])==set(contract['schemas'][contract['sources'][r['source']]])
                    and set(r['selector'])==set(_schema(r['source']))
                    and r['semantics']==_semantics(r['source'],r['values']), 'aggregate schema or scientific meaning differs.')
            require(r['status']==(r['values'].get('status') or ('not_estimable' if r['values'].get('estimable') is False else 'available')),
                    'result status differs from its source row.')
            key=r['source']+'#'+'&'.join(k+'='+quote(json.dumps(r['values'][k],ensure_ascii=False,separators=(',',':')),safe='') for k in _schema(r['source']))
            require(r['key']==key, 'semantic result key differs from its row coordinates.')
        self.rows = records
        require(corrected.metadata.get('accepted') is True and corrected.metadata.get('status') == 'accepted'
                and corrected.metadata.get('point_estimates_preserved_exactly') is True
                and corrected.metadata.get('model_covariance_preserved_exactly') is True
                and type(corrected.metadata.get('model_fits_performed')) is int and corrected.metadata['model_fits_performed'] == 0,
                'requires accepted corrected pre-TB uncertainty without changed point estimates.')
        original = {r['values']['target']:r for r in records if r['source'] == 'pretb/estimates'}
        require(all(r['provenance'].get('fingerprint')==corrected.metadata.get('source_fingerprint') for r in original.values()),
                'corrected uncertainty does not match the original pre-TB publication.')
        new = corrected.estimates.to_dict('records')
        require(len(new) == 17 and {r['target'] for r in new} == set(original), 'corrected pre-TB targets differ.')
        for raw in new:
            old = original[raw['target']]
            values = {k:(None if isinstance(v,float) and math.isnan(v) else v) for k,v in raw.items()}
            for key in ('coefficient','hr_per_adverse_sd','n_children','n_intervals','n_subsequent_tb','person_years'):
                require(equal(values[key],old['values'][key]), 'corrected uncertainty changed a point estimate or its sample.')
            r = copy.deepcopy(old)
            r.update(source='pretb_corrected/estimates', values=values,
                     key='pretb_corrected/estimates#target='+quote(json.dumps(raw['target'],separators=(',',':')),safe=''))
            r['semantics']['limitations'] = list(corrected.metadata['remaining_limitations'])
            r['semantics']['intervals'] = [{'lower':'ci_low','upper':'ci_high','level':0.95,
                'type':'actual-risk-membership child-cluster robust normal Wald confidence interval','normal_multiplier':1.96}]
            r['provenance'] = {'source_result_key':old['key'],'source_fingerprint':corrected.metadata['source_fingerprint'],
                               'correction':'actual event-time risk membership; accepted coefficients unchanged'}
            self.rows.append(r)

    def select(self, source, where, *, many=False):
        require(isinstance(source,str) and isinstance(where,dict), 'source selectors must be objects.')
        rows = [r for r in self.rows if r['source'] == source]
        require(rows and all(k in rows[0]['values'] for k in where), 'an aggregate source or selector field is unknown.')
        chosen = [r for r in rows if all(equal(r['values'][k],v) for k,v in where.items())]
        require(bool(chosen) and (many or len(chosen) == 1), 'a selector has missing or ambiguous matches.')
        return copy.deepcopy(chosen if many else chosen[0])


def binding(record, fields):
    require(isinstance(fields,dict) and fields and all(k in record['values'] for k in fields.values()),
            'a plotted field is absent from its selected result.')
    return {'result_key':record['key'], 'source':record['source'], 'selector':record['selector'],
            'fields':dict(fields), 'values':{k:record['values'][v] for k,v in fields.items()},
            'semantics':record['semantics'], 'provenance':record['provenance']}


def text_binding(spec, view):
    require(isinstance(spec,dict) and set(spec)=={'source','where','fields','style','digits'}, 'text binding schema differs.')
    require(strings(spec['fields']) and bool(spec['fields']) and isinstance(spec['style'],str)
            and type(spec['digits']) is int and 0<=spec['digits']<=8, 'text binding display controls differ.')
    record=view.select(spec['source'],spec['where'])
    result=cell(record,spec['fields'],spec['style'],spec['digits'])
    return {**result,'source':record['source'],'selector':record['selector'],
            'values':[record['values'][f] for f in spec['fields']],
            'semantics':record['semantics'],'provenance':record['provenance']}


def template(spec, view):
    require(isinstance(spec,dict) and set(spec)=={'template','bindings'}
            and isinstance(spec['template'],str) and isinstance(spec['bindings'],dict), 'template schema differs.')
    fields=[]
    try:parts=list(Formatter().parse(spec['template']))
    except ValueError:raise ReportingError('Figure data template braces are malformed.') from None
    for _,field,fmt,conversion in parts:
        if field is not None:
            require(field.isidentifier() and not fmt and conversion is None, 'template placeholders must be simple names.')
            fields.append(field)
    require(set(fields)==set(spec['bindings']), 'template placeholders and computed bindings differ.')
    result={k:text_binding(v,view) for k,v in spec['bindings'].items()}
    return {'text':spec['template'].format(**{k:v['text'] for k,v in result.items()}),'bindings':result}


def interval(record, fields, scale):
    require(set(fields) in ({'point','lower','upper'},{'x','point','lower','upper'}), 'interval field schema differs.')
    bound=binding(record,fields);values=bound['values']
    known=record['semantics']['intervals']
    require(any(x['lower']==fields['lower'] and x['upper']==fields['upper'] for x in known)
            and fields['point'] in record['semantics']['point_estimators'], 'point estimator or interval meaning differs.')
    status='available'
    selected=[values[k] for k in ('point','lower','upper')]
    if all(v is None for v in selected):
        require(record['values'].get('estimable') is False or 'estimab' in record['status'], 'missing estimates need an explicit non-estimable status.')
        status='not_estimable'
    else:
        require(record['status'] in ('available','estimated','estimable','derived')
                and record['values'].get('accepted',True) is True and record['values'].get('estimable',True) is True,
                'finite estimates require an accepted estimable source status.')
        require(all(finite(v) for v in selected) and values['lower'] <= values['upper'], 'interval values are incomplete or invalid.')
        if scale=='log':require(all(v>0 for v in selected), 'logarithmic axes require positive estimates and intervals.')
    return {**values,'binding':bound,'status':status}


def check_meaning(panel, record, fields):
    source=record['source'];family=source.split('/')[0]
    meanings={
        'primary':('mean','posterior mean','95% credible interval'),
        'shared_factor':('mean','posterior mean','95% credible interval'),
        'binary':('rr','adjusted risk ratio','95% confidence interval'),
        'feno':('geometric_mean_ratio','adjusted geometric mean ratio','95% confidence interval'),
        'pretb_corrected':('hr_per_adverse_sd','adjusted hazard ratio','95% confidence interval'),
        'trajectory':('predicted_mean','model-predicted mean','95% confidence interval')}
    if family=='wheeze':
        expected=(('rr_median','posterior median standardized risk ratio','95% credible interval')
                  if source.endswith('/standardized_rr') else
                  ('posterior_median','posterior median conditional incidence rate ratio','95% credible interval'))
        if source.endswith('/effects'):
            require(record['values']['process']=='Participant-initiated recurrent wheeze episodes',
                    'conditional episode figures require the recurrent process.')
    else:
        require(family in meanings,'an unselected model family cannot enter a publication panel.')
        expected=meanings[family]
    require((fields['point'],panel['estimate'],panel['interval'])==expected,
            'panel point-estimator or interval label does not match its scientific source.')


def build_data(registry, specification, correction):
    """Resolve estimates, labels, and counts for all nine figures before rendering."""
    validate_specification(specification)
    require(isinstance(specification,dict) and set(specification)=={'schema_version','figures'}
            and type(specification['schema_version']) is int and specification['schema_version']==1,
            'specification schema differs.')
    figures=copy.deepcopy(specification['figures'])
    require(tuple(f['id'] for f in figures)==FIGURE_IDS, 'requires all nine figures in their declared order.')
    require(tuple(len(f['panels']) for f in figures)==EXPECTED_PANELS, 'figure panel coverage differs.')
    view=AggregateView(registry,correction)
    totals={'figures':9,'panels':17,'forest_estimates':0,'trajectory_points':0,'schedule_cells':0,
            'flow_nodes':0,'flow_count_bindings':0,'caption_count_bindings':0}
    used=set()
    for figure in figures:
        require(set(figure)=={'id','title','kind','panels','caption','limitations'}, 'figure schema differs.')
        figure['caption']=template(figure['caption'],view)
        totals['caption_count_bindings']+=len(figure['caption']['bindings'])
        used.update(b['result_key'] for b in figure['caption']['bindings'].values())
        require(len({p['id'] for p in figure['panels']})==len(figure['panels']), 'panel identifiers must be unique.')
        for panel in figure['panels']:
            kind=panel['kind']
            if kind=='forest':
                require(panel['value_axis']['scale'] in ('linear','log') and panel['orientation'] in ('horizontal','vertical'), 'forest axis is invalid.')
                rows=panel['rows'];ids=[r['id'] for r in rows]
                require(ids==panel['row_order'] and len(ids)==len(set(ids)), 'forest row order differs or repeats.')
                for row in rows:
                    require({'id','label','series','source','where','fields'} <= set(row)
                            and set(row) <= {'id','label','series','source','where','fields','support_fields','unit'}, 'forest row schema differs.')
                    require(row['series'] in panel['series_order'], 'forest series is not declared.')
                    record=view.select(row['source'],row['where'])
                    check_meaning(panel,record,row['fields'])
                    values=interval(record,row['fields'],panel['value_axis']['scale'])
                    row.update({k:v for k,v in values.items() if k!='binding'})
                    row['bindings']=values['binding'];used.add(record['key'])
                    display=cell(record,[row['fields'][k] for k in ('point','lower','upper')],
                                 panel['annotation']['style'],panel['annotation']['digits'])
                    row['display']=display['text']
                    row['support']={};row['support_display']=''
                    if 'support_fields' in row:
                        support=binding(record,row['support_fields']);row['support_bindings']=support
                        row['support']=support['values']
                        require(all(finite(v) and v>=0 and int(v)==v for v in row['support'].values()), 'support counts must be nonnegative integers.')
                        row['support_display']=panel['support_annotation']['template'].format(**{k:number(v,0) for k,v in row['support'].items()})
                totals['forest_estimates']+=len(rows)
            elif kind=='trajectory':
                records=view.select(panel['source'],panel['where'],many=True)
                points=[];seen=set();grids={s:[] for s in panel['series_order']}
                for record in records:
                    check_meaning(panel,record,panel['fields'])
                    series=record['values'][panel['series_key']]
                    require(series in grids, 'trajectory series differs.')
                    result=interval(record,panel['fields'],panel['y_axis']['scale'])
                    require(result['status']=='available' and finite(result['x']), 'trajectory predictions must be complete.')
                    pair=(series,result['x']);require(pair not in seen, 'trajectory points repeat.');seen.add(pair)
                    grids[series].append(result['x']);points.append({**result,'series':series});used.add(record['key'])
                require(len(points)==202 and all(len(g)==101 and min(g)==5 and max(g)==10 for g in grids.values())
                        and len({tuple(sorted(g)) for g in grids.values()})==1, 'trajectory grid or group coverage differs.')
                panel['points']=sorted(points,key=lambda r:(panel['series_order'].index(r['series']),r['x']))
                totals['trajectory_points']+=len(points)
            elif kind=='flow':
                ids=[n['id'] for n in panel['nodes']]
                require(len(ids)==len(set(ids)) and all(len(e)==2 and e[0] in ids and e[1] in ids and e[0]!=e[1] for e in panel['edges']), 'flow nodes or edges are invalid.')
                for node in panel['nodes']:
                    rendered=template({'template':node['template'],'bindings':node['bindings']},view)
                    node.update(rendered)
                    missing=any(any(v is None for v in b['values']) for b in rendered['bindings'].values())
                    node['status']='unavailable' if missing else 'available'
                    require(missing == (node['id'] in panel['unavailable_nodes']), 'unavailable flow-node inventory differs.')
                    totals['flow_count_bindings']+=len(node['bindings'])
                    used.update(b['result_key'] for b in node['bindings'].values())
                totals['flow_nodes']+=len(ids)
            elif kind=='schedule':
                records=view.select(panel['source'],panel['where'],many=True);cells=[];seen=set()
                for record in records:
                    values=record['values'];row=values[panel['row_key']];column=values[panel['column_key']]
                    require(row in panel['row_order'] and column in panel['column_order'] and (row,column) not in seen, 'schedule coordinates differ or repeat.')
                    seen.add((row,column));bound=binding(record,panel['fields']);v=bound['values']
                    require(type(v['represented']) is bool and all(finite(v[k]) and v[k]>=0 and int(v[k])==v[k] for k in ('records','children'))
                            and v['children']<=v['records'] and v['represented']==(v['records']>0), 'schedule counts or representation status are inconsistent.')
                    cells.append({'row':row,'column':column,**v,'status':'available','bindings':bound});used.add(record['key'])
                require(seen=={(r,c) for r in panel['row_order'] for c in panel['column_order']}, 'schedule inventory is incomplete.')
                panel['cells']=sorted(cells,key=lambda c:(panel['row_order'].index(c['row']),panel['column_order'].index(c['column'])))
                totals['schedule_cells']+=len(cells)
            else:
                raise ReportingError('Unknown publication panel kind.')
    require(totals=={'figures':9,'panels':17,'forest_estimates':56,'trajectory_points':808,'schedule_cells':44,
                     'flow_nodes':20,'flow_count_bindings':28,'caption_count_bindings':12}, 'complete numerical figure coverage differs.')
    return {'schema_version':1,'figures':figures,'coverage':totals,'result_keys':sorted(used),
            'uncertainty_selection':'actual-risk-membership corrected child-cluster covariance for Figure S3',
            'scientific_values_modified_by_rendering':False}


def points_csv(data):
    """A long-form, aggregate-only audit table for every plotted number."""
    stream=io.StringIO(newline='');fields=('figure','panel','kind','item','series','field','value','status','result_key','source_field')
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    def emit(f,p,kind,item,series,b,status):
        for key,value in b['values'].items():
            writer.writerow(dict(zip(fields,(f,p,kind,item,series,key,value,status,b['result_key'],b['fields'][key]))))
    for f in data['figures']:
        for p in f['panels']:
            if p['kind']=='forest':
                for row in p['rows']:
                    emit(f['id'],p['id'],'forest',row['id'],row['series'],row['bindings'],row['status'])
                    if row['support']:emit(f['id'],p['id'],'support',row['id'],row['series'],row['support_bindings'],row['status'])
            elif p['kind']=='trajectory':
                for row in p['points']:emit(f['id'],p['id'],'trajectory',str(row['x']),row['series'],row['binding'],row['status'])
            elif p['kind']=='schedule':
                for row in p['cells']:emit(f['id'],p['id'],'schedule',row['row'],row['column'],row['bindings'],row['status'])
            elif p['kind']=='flow':
                for row in p['nodes']:
                    for name,b in row['bindings'].items():
                        emit(f['id'],p['id'],'flow',row['id'],name,{'result_key':b['result_key'],
                             'fields':{field:field for field in b['fields']},'values':dict(zip(b['fields'],b['values']))},row['status'])
        for name,b in f['caption']['bindings'].items():
            emit(f['id'],'caption','caption',name,'',{'result_key':b['result_key'],
                 'fields':{field:field for field in b['fields']},'values':dict(zip(b['fields'],b['values']))},b['status'])
    return stream.getvalue()
