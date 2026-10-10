"""Render accepted aggregate snapshots without opening study inputs or fitting.

Original analytical identities remain immutable provenance. The current source,
fonts and rendering runtime receive a separate identity. A source-compatible
layout change may produce new artwork; changed scientific payloads are rejected.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import wraps
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .execution import canonical_bytes, file_hash, preparation_locks, write_json, write_pointer
from .pipeline import ValidationError
from .reporting.sources import digest, no_symlinks, verify_files
from .reporting.results import ResultRegistry
from .reporting.tables import build_tables
from .reporting.numerical_text import build_checks, compare_references
from .reporting.figure_data import AggregateView, build_data, points_csv
from .reporting.render_document import _font
from .reporting_execution import (RESULT_FILES as TABLE_FILES, assert_loaded, source_hashes,
                                  document_runtime, render_payloads as render_tables)
from .figures_execution import (RESULT_FILES as FIGURE_FILES, IMAGE_FILES, CORRECTION_FILES,
                                figure_runtime, verify_pdf_fonts)
from .reporting.figures import render_figures

RESULT_FILES = tuple('tables/'+name for name in TABLE_FILES) + tuple(
    'figures/'+name for name in FIGURE_FILES) + ('metadata.json','summary.json')
TERMS = ('lf_adverse_sd','male','mbekweni')
TARGETS = ('resistance','compliance','ree','rei','xee','xei','dr_mean','dx_mean',
           'r6_sa_z','r8_sa_z','r10_sa_z','x6_sa_z','x8_sa_z','x10_sa_z',
           'fres_sa_z','r6_r20_sa_z','ax_sa_z')
CORRECTION_SPECIFICATION = {
    'specification_version':'1.0.0','method_id':'actual_risk_child_sandwich',
    'likelihood':'accepted Efron Cox partial likelihood; no refitting',
    'risk_membership':'start <= event_age <= stop',
    'score_method':'Breslow weighted covariate means and d/S0 martingale increments',
    'bread':'unchanged accepted inverse Efron observed information','cluster':'child',
    'finite_correction':'G/(G-1)*(N-1)/(N-3)','confidence_multiplier':1.96,
    'multiplicity':'Benjamini-Hochberg over the selected 17 finite target p-values',
    'source_minimum_events':3,'correction_scope':'Only score-residual risk-set membership changes.',
    'remaining_conventions':['Efron-fit/Breslow-score combination retained',
                             'Inclusive entry/stop event-risk boundaries retained']}
COEFFICIENT_COLUMNS = ('target','term','coefficient','robust_se','model_se','hazard_ratio',
                       'ci_low','ci_high','p_value','status','accepted','reason')
COVARIANCE_COLUMNS = ('target','covariance_type','row_term','column_term','value','status','accepted')
HEX = re.compile('[0-9a-f]{64}')


def require(condition, reason):
    if not condition:
        raise ValidationError('Render-only '+reason)


def _object(value, fields):
    require(isinstance(value,dict) and set(value)==set(fields), 'object fields differ from the required schema.')


def _input_boundary(function):
    """Translate malformed external aggregate structures into a closed failure."""
    @wraps(function)
    def checked(*args,**kwargs):
        try:return function(*args,**kwargs)
        except ValidationError:raise
        except (OSError,KeyError,TypeError,ValueError,IndexError,OverflowError):
            raise ValidationError('Render-only input structure or numeric metadata is invalid.') from None
    return checked


def read_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'JSON objects cannot contain duplicate keys.')
            result[key]=value
        return result
    def invalid(value):
        raise ValidationError('Render-only JSON must contain finite values or explicit nulls.')
    def finite_float(value):
        result=float(value)
        if not math.isfinite(result):invalid(value)
        return result
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,
                          parse_constant=invalid,parse_float=finite_float)
    except (OSError,UnicodeError,json.JSONDecodeError):
        raise ValidationError('Render-only input must be readable strict JSON.') from None


def _hex(value):
    return isinstance(value,str) and HEX.fullmatch(value) is not None


def _same(a,b):
    return canonical_bytes(a)==canonical_bytes(b)


def _close(a,b):
    return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)


def _overlap(a,b):
    return a==b or a in b.parents or b in a.parents


def render_config(path):
    path=Path(path).absolute();no_symlinks(path)
    value=read_json(path)
    _object(value,('schema_version','reporting_publication','reporting_receipt_sha256',
                  'figures_publication','figures_receipt_sha256','work_directory','output_directory',
                  'document_python','pdf_fonts_tool','fonts'))
    require(type(value['schema_version']) is int and value['schema_version']==1,'configuration version must be one.')
    _object(value['fonts'],('arial','arial_bold'))
    for key in ('reporting_receipt_sha256','figures_receipt_sha256'):
        require(_hex(value[key]),'each accepted receipt requires an explicit trusted SHA-256.')
    result={'schema_version':1,**{key:value[key] for key in ('reporting_receipt_sha256','figures_receipt_sha256')}}
    def resolve(text,executable=False):
        require(isinstance(text,str) and text.strip(),'paths must be nonempty strings.')
        p=Path(text).expanduser();p=p if p.is_absolute() else path.parent/p
        if not executable:no_symlinks(p)
        p=p.resolve()
        if executable:require(p.is_file() and os.access(p,os.X_OK),'configured executable is unavailable.')
        return p
    for key in ('reporting_publication','figures_publication','work_directory','output_directory',
                'document_python','pdf_fonts_tool'):
        result[key]=resolve(value[key],key in ('document_python','pdf_fonts_tool'))
    result['fonts']={key:resolve(text) for key,text in value['fonts'].items()}
    return result


def _publication(directory,expected_receipt,files,through):
    no_symlinks(directory)
    require(directory.is_dir(),'accepted aggregate directory is missing.')
    receipt_path=directory/'receipt.json';no_symlinks(receipt_path)
    require(receipt_path.is_file() and file_hash(receipt_path)==expected_receipt,'trusted aggregate receipt hash differs.')
    receipt=read_json(receipt_path)
    _object(receipt,('schema_version','status','fingerprint','identity','summary','files','completed_at'))
    require(type(receipt['schema_version']) is int and receipt['schema_version']==1
            and receipt['status']=='accepted' and isinstance(receipt['identity'],dict)
            and _hex(receipt['fingerprint']) and receipt['fingerprint']==digest(receipt['identity'])
            and isinstance(receipt['summary'],dict),'aggregate receipt is not accepted or its identity is invalid.')
    summary=receipt['summary']
    require(summary.get('accepted') is True and summary.get('status')=='accepted'
            and summary.get('through')==through and summary.get('fingerprint')==receipt['fingerprint']
            and type(summary.get('new_model_fits')) is int and summary['new_model_fits']==0,
            'aggregate receipt has contradictory completion metadata.')
    verify_files(directory,receipt['files'],files)
    require(_same(read_json(directory/'summary.json'),summary),'aggregate summary differs from its receipt.')
    source=receipt['identity'].get('source')
    require(isinstance(source,dict) and bool(source) and all(isinstance(k,str) and not Path(k).is_absolute()
            and '..' not in Path(k).parts and '\\' not in k and _hex(v) for k,v in source.items()),
            'producer source manifest is malformed.')
    return receipt


@_input_boundary
def registry_from_export(exported):
    """Recreate the defensive registry and require its complete canonical export."""
    _object(exported,('schema_version','records','sources','record_count'))
    require(type(exported['schema_version']) is int and exported['schema_version']==1
            and isinstance(exported['records'],list) and type(exported['record_count']) is int
            and exported['record_count']==len(exported['records'])==3196
            and isinstance(exported['sources'],list) and len(exported['sources'])==143,
            'requires the complete accepted aggregate registry.')
    registry=ResultRegistry();contract=registry._schema_contract
    require(exported['sources']==sorted(contract['sources']),'aggregate source inventory differs.')
    for record in exported['records']:
        require(isinstance(record,dict) and set(record) in (
            {'key','source','selector','values','status','semantics','provenance'},
            {'key','source','selector','values','status','semantics','provenance','aliases'})
            and isinstance(record.get('values'),dict) and isinstance(record.get('provenance'),dict),
            'aggregate record schema differs.')
    for source in exported['sources']:
        rows=[r for r in exported['records'] if r['source']==source]
        columns=contract['schemas'][contract['sources'][source]]
        require(all(set(r['values'])==set(columns) for r in rows),'aggregate field inventory differs.')
        provenance=rows[0]['provenance'] if rows else {}
        require(all(_same(r['provenance'],provenance) for r in rows),'source rows have conflicting provenance.')
        frame=pd.DataFrame([r['values'] for r in rows],columns=columns,dtype=object)
        registry.add(source,frame,provenance)
    require(_same(registry.export(),exported),'aggregate values, statuses, aliases or meanings differ from the approved registry.')
    return registry


def _csv(path,columns):
    text={'target','measure','measure_label','source_column','scale','status','reason','warning',
          'term','covariance_type','row_term','column_term'}
    with path.open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        require(reader.fieldnames==list(columns),'corrected aggregate columns differ.')
        rows=[]
        for raw in reader:
            require(set(raw)==set(columns) and all(v is not None for v in raw.values()),'corrected aggregate row width differs.')
            row={}
            for key,value in raw.items():
                if key=='accepted':
                    require(value in ('True','False'),'acceptance flags must be typed booleans.')
                    row[key]=value=='True'
                elif key in text:row[key]=value
                elif value=='':row[key]=None
                else:
                    try:number=float(value)
                    except ValueError:raise ValidationError('Render-only corrected numeric field is invalid.') from None
                    require(math.isfinite(number),'corrected numeric fields must be finite or blank.')
                    if key.startswith('n_'):
                        require(number>=0 and number.is_integer(),'corrected counts must be nonnegative integers.')
                        number=int(number)
                    row[key]=number
            rows.append(row)
    return pd.DataFrame(rows,columns=columns,dtype=object)


@_input_boundary
def load_correction(directory,registry,figure_receipt):
    """Validate saved aggregate inference; never recompute risk-set scores."""
    estimate_columns=registry._schema_contract['schemas'][registry._schema_contract['sources']['pretb/estimates']]
    frames={name:_csv(directory/('corrected_pretb_'+name+'.csv'),columns) for name,columns in
            [('estimates',estimate_columns),('coefficients',COEFFICIENT_COLUMNS),('covariance',COVARIANCE_COLUMNS)]}
    metadata=read_json(directory/'corrected_pretb_metadata.json')
    _object(metadata,('accepted','contracts','inference_provenance','model_covariance_preserved_exactly',
        'model_fits_performed','models','multiplicity','point_estimates_preserved_exactly','preparation_fingerprint',
        'remaining_limitations','rows','schema_version','selection','source_files','source_fingerprint',
        'source_outputs_modified','specification','status'))
    require(type(metadata['schema_version']) is int and metadata['schema_version']==1
            and metadata['accepted'] is True and metadata['status']=='accepted'
            and metadata['point_estimates_preserved_exactly'] is True
            and metadata['model_covariance_preserved_exactly'] is True
            and metadata['source_outputs_modified'] is False
            and type(metadata['model_fits_performed']) is int and metadata['model_fits_performed']==0,
            'saved corrected uncertainty has invalid acceptance flags.')
    original={r['values']['target']:r for r in registry.export()['records'] if r['source']=='pretb/estimates'}
    selected=metadata['selection']
    require(selected==list(TARGETS) and set(selected)==set(original),
            'corrected target inventory differs.')
    _object(metadata['source_files'],('estimates.csv','coefficients.csv','covariance.csv',
                                     'diagnostics.json','summary.json','receipt.json'))
    require(all(_hex(v) for v in metadata['source_files'].values())
            and isinstance(metadata['contracts'],dict)
            and set(metadata['contracts'])=={'pretb_cox/'+t for t in TARGETS}
            and all(_hex(v) for v in metadata['contracts'].values())
            and _hex(metadata['source_fingerprint']) and _hex(metadata['preparation_fingerprint']),
            'saved correction provenance hashes are invalid.')
    require(metadata['source_fingerprint']==figure_receipt['identity']['correction_source_fingerprint']
            and metadata['preparation_fingerprint']==figure_receipt['identity']['preparation_fingerprint']
            and metadata['source_files'].get('receipt.json')==figure_receipt['identity']['correction_source_receipt_sha256']
            and all(r['provenance'].get('fingerprint')==metadata['source_fingerprint'] for r in original.values()),
            'saved correction belongs to a different accepted source.')
    require(metadata['rows']=={'estimates':17,'coefficients':51,'covariance':306}
            and all(len(frames[n])==metadata['rows'][n] for n in frames),'corrected aggregate row counts differ.')
    require(_same(metadata['specification'],CORRECTION_SPECIFICATION)
            and metadata['remaining_limitations']==CORRECTION_SPECIFICATION['remaining_conventions'],
            'saved correction method or retained conventions differ.')
    require(isinstance(metadata['models'],list) and len(metadata['models'])==17
            and all(isinstance(m,dict) for m in metadata['models'])
            and [m.get('target') for m in metadata['models']]==selected,'correction diagnostic target coverage differs.')
    models={m['target']:m for m in metadata['models']};pvalues=[]
    for target in selected:
        erows=frames['estimates'].loc[frames['estimates'].target.eq(target)].to_dict('records')
        cr=frames['coefficients'].loc[frames['coefficients'].target.eq(target)].to_dict('records')
        vr=frames['covariance'].loc[frames['covariance'].target.eq(target)].to_dict('records')
        require(len(erows)==1 and len(cr)==3 and {r['term'] for r in cr}==set(TERMS)
                and len(vr)==18 and {(r['covariance_type'],r['row_term'],r['column_term']) for r in vr}==
                {(kind,a,b) for kind in ('robust','model') for a in TERMS for b in TERMS},
                'complete corrected coefficients/covariance cells are required.')
        estimate=erows[0];old=original[target]['values'];model=models[target]
        _object(model['counts'],('n_children','n_intervals','n_subsequent_tb','person_years'))
        require(model.get('contract_key')=='pretb_cox/'+target
                and all(type(model['counts'][k]) is int and model['counts'][k]>=0
                        for k in ('n_children','n_intervals','n_subsequent_tb'))
                and _close(model['counts']['person_years'],estimate['person_years']),
                'correction diagnostic counts or contract identity differ.')
        require(all(r['accepted'] is True and r['status']==estimate['status'] for r in [estimate,*cr,*vr])
                and model.get('accepted') is True and model.get('status')==estimate['status'],
                'corrected component statuses disagree.')
        for key in set(estimate)-{'robust_se','ci_low','ci_high','p_value','fdr_q_value'}:
            # The M13 CSV adapter serializes empty textual annotations as null;
            # the corrected CSV preserves its source's empty strings.
            empty_annotation=key in {'reason','warning'} and estimate[key]=='' and old[key] is None
            require(estimate[key]==old[key] or empty_annotation,
                    'saved correction changed a source point estimate or support field.')
        require(all(model['counts'][k]==estimate[k] for k in ('n_children','n_intervals','n_subsequent_tb','person_years')),
                'correction diagnostic sample counts disagree.')
        if estimate['status']=='non_estimable':
            require(estimate['n_subsequent_tb']<3 and estimate['reason']=='source_minimum_event_rule'
                    and model.get('correction_performed') is False,'non-estimable correction status is inconsistent.')
            require(all(estimate[k] is None for k in ('coefficient','robust_se','hr_per_adverse_sd','ci_low','ci_high','p_value','fdr_q_value'))
                    and all(r[k] is None for r in cr for k in ('coefficient','robust_se','model_se','hazard_ratio','ci_low','ci_high','p_value'))
                    and all(r['value'] is None for r in vr),'non-estimable aggregates must retain explicit null inference.')
            continue
        require(estimate['status']=='estimated' and estimate['n_subsequent_tb']>=3
                and model.get('correction_performed') is True and model.get('outside_risk_scores_exactly_zero') is True,
                'saved corrected estimates require accepted correction diagnostics.')
        require(model.get('accepted_coefficients_and_hazard_ratios_preserved') is True
                and all(_hex(model.get(k)) for k in ('risk_membership_sha256','event_membership_sha256'))
                and type(model.get('event_time_count')) is int and model['event_time_count']>0
                and model.get('risk_membership_shape')==[model['event_time_count'],estimate['n_intervals']],
                'saved correction risk-set provenance is invalid.')
        n=estimate['n_intervals'];g=estimate['n_children']
        require(n>3 and g>1 and _close(model.get('finite_sample_correction'),g/(g-1)*(n-1)/(n-3)),
                'saved correction finite-sample factor differs from its aggregate counts.')
        matrices={}
        for kind in ('robust','model'):
            cells={(r['row_term'],r['column_term']):r['value'] for r in vr if r['covariance_type']==kind}
            matrix=np.array([[cells[a,b] for b in TERMS] for a in TERMS],float)
            require(np.isfinite(matrix).all() and np.allclose(matrix,matrix.T,rtol=1e-10,atol=1e-12)
                    and np.all(np.diag(matrix)>0) and np.linalg.eigvalsh(matrix).min()>=-1e-12*max(1.,np.linalg.norm(matrix,2)),
                    'saved covariance must be finite, symmetric and positive semidefinite.')
            matrices[kind]=matrix
        require(_close(model.get('robust_covariance_min_eigenvalue'),float(np.linalg.eigvalsh(matrices['robust']).min()))
                and _close(model.get('robust_covariance_max_asymmetry'),float(np.abs(matrices['robust']-matrices['robust'].T).max())),
                'saved covariance diagnostics disagree with complete saved cells.')
        byterm={r['term']:r for r in cr}
        for i,term in enumerate(TERMS):
            row=byterm[term];beta=row['coefficient'];se=row['robust_se']
            require(_close(se,math.sqrt(matrices['robust'][i,i]))
                    and _close(row['model_se'],math.sqrt(matrices['model'][i,i]))
                    and _close(row['hazard_ratio'],math.exp(beta))
                    and _close(row['ci_low'],math.exp(beta-1.96*se)) and _close(row['ci_high'],math.exp(beta+1.96*se))
                    and _close(row['p_value'],math.erfc(abs(beta/se)/math.sqrt(2))),
                    'saved coefficient inference disagrees with the full covariance.')
        target_row=byterm[TERMS[0]]
        require(all(estimate[a]==target_row[b] for a,b in [('coefficient','coefficient'),('robust_se','robust_se'),
            ('hr_per_adverse_sd','hazard_ratio'),('ci_low','ci_low'),('ci_high','ci_high'),('p_value','p_value')]),
            'target inference differs from its coefficient row.')
        require(_close(model.get('corrected_target_robust_se'),estimate['robust_se'])
                and _close(model.get('corrected_target_p'),estimate['p_value'])
                and model.get('corrected_target_ci')==[estimate['ci_low'],estimate['ci_high']]
                and _close(model.get('source_target_robust_se'),old['robust_se'])
                and _close(model.get('source_target_p'),old['p_value'])
                and model.get('source_target_ci')==[old['ci_low'],old['ci_high']]
                and _close(model.get('target_relative_se_change'),estimate['robust_se']/old['robust_se']-1),
                'saved correction diagnostics differ from target inference.')
        pvalues.append((target,estimate['p_value'],estimate['fdr_q_value']))
    ordered=sorted(pvalues,key=lambda r:r[1]);minimum=1.
    for i in range(len(ordered)-1,-1,-1):
        target,p,q=ordered[i];minimum=min(minimum,p*len(ordered)/(i+1))
        require(_close(q,minimum),'saved family-adjusted values disagree with selected target p-values.')
    require(metadata['multiplicity']=={'method':'Benjamini-Hochberg','family':'selected_17','finite_tests':len(pvalues)},
            'saved multiplicity metadata differs.')
    correction=SimpleNamespace(**frames,metadata=metadata)
    AggregateView(registry.export(),correction)
    return correction


def _comparison(checks,saved):
    if saved.get('status')=='reference_not_supplied':
        current=compare_references(checks)
    else:
        require(saved.get('status')=='compared' and isinstance(saved.get('cells'),list)
                and isinstance(saved.get('sentences'),list),'saved comparison schema differs.')
        current=compare_references(checks,[{'id':r['id'],'displayed_text':r['reference_display']} for r in saved['cells']],
                                   [{'id':r['id'],'text':r['reference_text']} for r in saved['sentences']])
    require(_same(current,saved),'reconstructed historical display comparison differs.')
    return current


def _inventory(directory):
    no_symlinks(directory)
    for path in directory.rglob('*'):
        no_symlinks(path)
        require(path.is_file() or path.is_dir(),'publication contains an unsupported file type.')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_dir()}=={'tables','figures'},
            'rendered publication has unexpected directories.')
    return {str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file() and p!=directory/'receipt.json'}


def _verify_render(directory,identity,summary,files):
    receipt=read_json(directory/'receipt.json')
    _object(receipt,('schema_version','status','fingerprint','identity','summary','files','completed_at'))
    require(type(receipt['schema_version']) is int and receipt['schema_version']==1
            and receipt['status']=='rendered' and receipt['fingerprint']==digest(identity)
            and _same(receipt['identity'],identity) and _same(receipt['summary'],summary)
            and receipt['files']==files and _inventory(directory)==files,
            'cached render differs from freshly regenerated complete outputs.')


@_input_boundary
def run_render_only(root,render_config_file,*,resume=False):
    """Regenerate presentation from two trusted, self-contained aggregate sets."""
    require(type(resume) is bool,'resume must be a boolean.')
    root=Path(root).absolute();no_symlinks(root);root=root.resolve();assert_loaded(root)
    config_path=Path(render_config_file).absolute();no_symlinks(config_path);cfg=render_config(config_path)
    report=cfg['reporting_publication'];figure=cfg['figures_publication']
    work_root=cfg['work_directory'];output_root=cfg['output_directory']
    require(not _overlap(work_root,output_root) and not _overlap(report,figure)
            and not any(_overlap(p,root) for p in (work_root,output_root,report,figure)),
            'source, aggregate inputs and generated roots must be separate.')
    protected=[config_path,cfg['document_python'],cfg['pdf_fonts_tool'],*cfg['fonts'].values()]
    require(not any(_overlap(d,p) for d in (work_root,output_root) for p in (*protected,report,figure)),
            'generated roots cannot overlap protected inputs.')
    for key,bold in (('arial',False),('arial_bold',True)):_font(cfg['fonts'][key].read_bytes(),bold)
    rp=_publication(report,cfg['reporting_receipt_sha256'],TABLE_FILES,'M13')
    fp=_publication(figure,cfg['figures_receipt_sha256'],FIGURE_FILES,'M14')
    require(fp['identity'].get('reporting_fingerprint')==rp['fingerprint']
            and fp['identity'].get('reporting_receipt_sha256')==cfg['reporting_receipt_sha256']
            and fp['identity'].get('preparation_fingerprint')==rp['identity'].get('preparation_fingerprint')
            and fp['identity'].get('uncertainty_selection')=='corrected_actual_risk_membership'
            and fp['summary'].get('uncertainty_selection')=='corrected_actual_risk_membership',
            'M13/M14 provenance linkage or corrected uncertainty selection differs.')
    require(fp['summary'].get('pretb_point_estimates_preserved') is True,'saved corrected point-estimate preservation is missing.')
    inputs={p:file_hash(p) for p in protected}
    for directory,receipt in ((report,rp),(figure,fp)):
        inputs.update({directory/name:h for name,h in receipt['files'].items()})
        inputs[directory/'receipt.json']=file_hash(directory/'receipt.json')
    registry=registry_from_export(read_json(report/'registry.json'))
    tables=build_tables(registry)
    checks=build_checks(registry,tables,read_json(root/'config/manuscript_reporting.json'))
    require(not checks['wording_review_required'] and _same({'schema_version':1,'tables':tables},read_json(report/'tables.json'))
            and _same(checks,read_json(report/'numerical_text.json')),
            'current table/text code changed the accepted scientific payload; obtain a new accepted snapshot.')
    comparison=_comparison(checks,read_json(report/'reference_comparison.json'))
    correction=load_correction(figure,registry,fp)
    data=build_data(registry.export(),read_json(root/'config/publication_figures.json'),correction)
    require(_same(data,read_json(figure/'plot_data.json'))
            and points_csv(data)==(figure/'plot_points.csv').read_text(encoding='utf-8'),
            'current figure specification changed accepted scientific selections or values.')
    document=document_runtime(cfg['document_python']);runtime=figure_runtime(cfg['pdf_fonts_tool'])
    identity={'schema_version':1,'mode':'render_only','source':source_hashes(root),
        'aggregate_inputs':{'reporting':{'fingerprint':rp['fingerprint'],'receipt_sha256':cfg['reporting_receipt_sha256'],'files':rp['files']},
                            'figures':{'fingerprint':fp['fingerprint'],'receipt_sha256':cfg['figures_receipt_sha256'],'files':fp['files']}},
        'document_runtime':document,'figure_runtime':runtime,
        'render_python_executable_sha256':file_hash(Path(sys.executable).resolve()),
        'fonts':{k:file_hash(v) for k,v in cfg['fonts'].items()},
        'scientific_payload_policy':'exact accepted tables/text/figure-data equality; no analytical revalidation'}
    fingerprint=digest(identity)
    summary={'schema_version':1,'mode':'render_only','status':'rendered','accepted':True,'fingerprint':fingerprint,
        'source_reporting_fingerprint':rp['fingerprint'],'source_figures_fingerprint':fp['fingerprint'],
        'source_publications_verified':2,'analytical_revalidation_performed':False,'new_model_fits':0,
        'correction_recomputed':False,'participant_data_read':False,'raw_posteriors_read':False,
        'table_coverage':checks['coverage'],'figure_coverage':data['coverage'],
        'visual_review':'Required separately when rendering code, fonts or runtime changes.'}
    metadata={'schema_version':1,'mode':'render_only','source_analytical_identities':{'reporting':rp['identity'],'figures':fp['identity']},
        'current_render_identity':identity,'analytical_revalidation_performed':False,
        'scientific_values_changed':False,'participant_records_exported':False,'small_counts_suppressed':False,
        'inherited_limitations':fp['summary']['inherited_limitations']}
    work=work_root/'render_only';output=output_root/'render_only'
    for path in (work,output):no_symlinks(path);path.mkdir(parents=True,exist_ok=True,mode=0o700)
    destination=output/fingerprint
    def unchanged():
        assert_loaded(root)
        require(source_hashes(root)==identity['source'],'render source identity changed during publication.')
        for path in (work_root,output_root,work,output,destination,report,figure):no_symlinks(path)
        for path,h in inputs.items():
            no_symlinks(path);require(path.is_file() and file_hash(path)==h,'a protected aggregate, configuration, font or executable changed.')
        require(_same(document_runtime(cfg['document_python']),document) and _same(figure_runtime(cfg['pdf_fonts_tool']),runtime),
                'rendering runtime changed during publication.')
    with preparation_locks(work,output):
        no_symlinks(destination);reused=destination.exists()
        require(not reused or resume,'identical render exists; use --resume to regenerate and compare.')
        stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=output));published=False
        try:
            table_stage=stage/'tables';figure_stage=stage/'figures';table_stage.mkdir();figure_stage.mkdir()
            table_metadata={**read_json(report/'metadata.json'),'mode':'render_only','document_runtime':document,
                            'document_fonts':identity['fonts'],'analytical_revalidation_performed':False}
            render_tables(table_stage,root,registry,tables,checks,comparison,table_metadata,
                          {**summary,'component':'manuscript_tables'},cfg,cfg['fonts'])
            for name in CORRECTION_FILES:shutil.copyfile(figure/name,figure_stage/name)
            write_json(figure_stage/'plot_data.json',data)
            (figure_stage/'plot_points.csv').write_text(points_csv(data),encoding='utf-8')
            (figure_stage/'figure_captions.txt').write_text('\n\n'.join(f['title']+'\n'+f['caption']['text'] for f in data['figures'])+'\n',encoding='utf-8')
            rendered=render_figures(data,figure_stage,cfg['fonts'],dpi=600)
            require(set(rendered.get('files',{}))==set(IMAGE_FILES)
                    and all(file_hash(figure_stage/n)==h for n,h in rendered['files'].items()),'rendered image inventory differs.')
            write_json(figure_stage/'render_metadata.json',rendered)
            write_json(figure_stage/'font_validation.json',verify_pdf_fonts(figure_stage,cfg['pdf_fonts_tool'],cfg['fonts']))
            write_json(figure_stage/'metadata.json',{**read_json(figure/'metadata.json'),'mode':'render_only',
                'runtime':runtime,'fonts':identity['fonts'],'analytical_revalidation_performed':False,'correction_recomputed':False})
            write_json(figure_stage/'summary.json',{**summary,'component':'publication_figures'})
            write_json(stage/'metadata.json',metadata);write_json(stage/'summary.json',summary)
            files=_inventory(stage);require(set(files)==set(RESULT_FILES),'rendered publication has an incomplete or unexpected inventory.')
            unchanged()
            if reused:_verify_render(destination,identity,summary,files)
            else:
                write_json(stage/'receipt.json',{'schema_version':1,'status':'rendered','fingerprint':fingerprint,
                    'identity':identity,'summary':summary,'files':files,'completed_at':datetime.now(timezone.utc).isoformat()})
                _verify_render(stage,identity,summary,files);no_symlinks(destination)
                require(not destination.exists(),'render destination appeared during validation.')
                os.replace(stage,destination);published=True
            unchanged()
            completion={**summary,'resumed':reused};write_pointer(work_root/'render_current.json',completion)
        except Exception:
            if stage.exists():os.replace(stage,work/('failed-'+stage.name.removeprefix('.staging-')))
            if published and destination.exists():
                no_symlinks(destination);os.replace(destination,work/('failed-publication-'+stage.name.removeprefix('.staging-')))
            raise
        finally:
            if stage.exists():shutil.rmtree(stage)
    return completion
