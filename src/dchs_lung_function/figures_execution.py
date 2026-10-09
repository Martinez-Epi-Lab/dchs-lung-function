"""Cumulative M14 validation, corrected pre-TB inference and figure publication."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import zlib

from .execution import (canonical_bytes, file_hash, preparation_locks, resolve_paths,
                        verify_artifacts, write_json, write_pointer)
from .pipeline import ValidationError, read_json
from .reporting.sources import digest, no_symlinks, require, verify_files, verify_receipt
from .reporting_execution import (assert_loaded, source_hashes, run_reporting,
                                  reporting_config, document_runtime,
                                  RESULT_FILES as REPORTING_FILES)
from .reporting.figure_data import FIGURE_IDS, build_data, points_csv

IMAGE_FILES = tuple(n+ext for n in FIGURE_IDS for ext in ('.pdf','.png'))+('publication_figures.pdf',)
CORRECTION_FILES = ('corrected_pretb_estimates.csv','corrected_pretb_coefficients.csv',
                    'corrected_pretb_covariance.csv','corrected_pretb_metadata.json')
RESULT_FILES = IMAGE_FILES+CORRECTION_FILES+('plot_data.json','plot_points.csv','figure_captions.txt',
                'font_validation.json','render_metadata.json','metadata.json','summary.json')


def figures_config(path):
    path=Path(path).absolute();no_symlinks(path)
    value=read_json(path,'Local figure configuration')
    require(isinstance(value,dict) and set(value)=={'schema_version','pdf_fonts_tool','pretb_uncertainty'}
            and type(value.get('schema_version')) is int and value['schema_version']==1
            and value['pretb_uncertainty']=='corrected_actual_risk_membership',
            'M14 requires the explicitly selected corrected actual-risk-membership uncertainty.')
    require(isinstance(value['pdf_fonts_tool'],str) and value['pdf_fonts_tool'].strip(), 'PDF font inspection executable is missing.')
    tool=Path(value['pdf_fonts_tool']).expanduser()
    tool=tool if tool.is_absolute() else path.parent/tool
    # Installed command wrappers may be symlinks; the resolved executable is bound.
    tool=tool.resolve()
    require(tool.is_file() and os.access(tool,os.X_OK), 'PDF font inspector must be an executable file.')
    return {'pdf_fonts_tool':tool,'pretb_uncertainty':value['pretb_uncertainty']}


def font_tool_identity(tool):
    try:
        result=subprocess.run([str(tool),'-v'],check=True,capture_output=True,text=True,timeout=20)
        lines=(result.stdout+'\n'+result.stderr).strip().splitlines()
    except (OSError,subprocess.SubprocessError):
        raise ValidationError('PDF font inspection preflight failed.') from None
    require(bool(lines) and lines[0].startswith('pdffonts version '), 'PDF font inspector is not pdffonts.')
    return {'version':lines[0],'executable_sha256':file_hash(tool)}


def figure_runtime(tool):
    from matplotlib import ft2font
    return {'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),
            'packages':{n:importlib.metadata.version(n) for n in ('matplotlib','numpy','pandas','pillow','fonttools','contourpy')},
            'freetype_version':ft2font.__freetype_version__,
            'freetype_build_type':ft2font.__freetype_build_type__,
            'zlib_compiled':zlib.ZLIB_VERSION,'zlib_runtime':zlib.ZLIB_RUNTIME_VERSION,
            'font_inspector':font_tool_identity(tool)}


def verify_pdf_fonts(directory, tool, fonts):
    from fontTools.ttLib import TTFont
    from .reporting.render_document import _font
    names=set();font_info={}
    for name,bold in (('arial',False),('arial_bold',True)):
        p=Path(fonts[name]);data=p.read_bytes();_font(data,bold)
        with TTFont(p) as font:
            postscript=font['name'].getDebugName(6)
        require(isinstance(postscript,str) and postscript, 'Arial PostScript name is missing.')
        names.add(postscript);font_info[name]={'sha256':file_hash(p),'postscript_name':postscript}
    inspected=[]
    for name in (n for n in IMAGE_FILES if n.endswith('.pdf')):
        try:
            result=subprocess.run([str(tool),str(Path(directory)/name)],check=True,
                                  capture_output=True,text=True,timeout=30)
        except (OSError,subprocess.SubprocessError):
            raise ValidationError('A publication PDF could not pass font inspection.') from None
        lines=result.stdout.splitlines()
        require(len(lines)>=3 and 'emb' in lines[0] and 'uni' in lines[0], 'PDF font inventory is empty or malformed.')
        emb=lines[0].index('emb');sub=lines[0].index('sub');uni=lines[0].index('uni');obj=lines[0].index('object ID')
        records=[]
        for line in lines[2:]:
            if not line.strip():continue
            face=line.split()[0];family=re.sub(r'^[A-Z]{6}\+','',face)
            record={'name':face,'postscript_name':family,'embedded':line[emb:sub].strip(),
                    'subset':line[sub:uni].strip(),'unicode':line[uni:obj].strip()}
            require(family in names and record['embedded']=='yes' and record['unicode']=='yes'
                    and 'Type 3' not in line, 'PDF uses an unembedded, fallback or non-Unicode font.')
            records.append(record)
        require(bool(records) and {r['postscript_name'] for r in records}==names,
                'each PDF must embed both actual Arial regular and Arial Bold.')
        inspected.append({'file':name,'fonts':records})
    return {'schema_version':1,'status':'passed','actual_fonts':font_info,'pdfs':inspected,
            'pdf_count':len(inspected),'only_actual_Arial':True}


def _reporting_publication(root, directories, prior):
    expected={'numbered_tables':8,'physical_tables':10,'numeric_cells':304,'scientific_cells':261,
              'cell_classifications':{'estimate':186,'count':75,'header':26,'context':17},
              'sentences':42,'value_tokens':95,'qualitative_bindings':52}
    upstream={'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
              'mechanics_models_verified':6,'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
              'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,'wheeze_models_verified':2,'feno_models_verified':11}
    require(isinstance(prior,dict) and prior.get('accepted') is True and prior.get('status')=='accepted'
            and prior.get('through')=='M13' and canonical_bytes(prior.get('coverage'))==canonical_bytes(expected)
            and type(prior.get('registry_records')) is int and prior['registry_records']==3195
            and type(prior.get('aggregate_sources')) is int and prior['aggregate_sources']==142
            and type(prior.get('new_model_fits')) is int and prior['new_model_fits']==0
            and type(prior.get('resumed')) is bool
            and canonical_bytes(prior.get('upstream_models'))==canonical_bytes(upstream)
            and type(prior.get('source_publications_verified')) is int and prior['source_publications_verified']==45
            and type(prior.get('preceding_modules_verified')) is int and prior['preceding_modules_verified']==12
            and isinstance(prior.get('fingerprint'),str) and re.fullmatch('[0-9a-f]{64}',prior['fingerprint'])
            and isinstance(prior.get('preparation_fingerprint'),str) and re.fullmatch('[0-9a-f]{64}',prior['preparation_fingerprint']),
            'M14 requires complete successful cumulative M13 validation.')
    directory=directories['outputs']/'manuscript_reporting'/prior['fingerprint'];no_symlinks(directory)
    receipt=read_json(directory/'receipt.json','Current manuscript reporting receipt')
    require(set(receipt)=={'schema_version','status','fingerprint','identity','summary','files','completed_at'}
            and type(receipt['schema_version']) is int and receipt['schema_version']==1
            and receipt['status']=='accepted' and receipt['fingerprint']==prior['fingerprint']==digest(receipt['identity'])
            and receipt['identity']['source']==source_hashes(root)
            and receipt['identity']['preparation_fingerprint']==prior['preparation_fingerprint']
            and canonical_bytes(receipt['summary'])==canonical_bytes({k:v for k,v in prior.items() if k!='resumed'}),
            'M13 publication identity does not match its current cumulative completion.')
    verify_files(directory,receipt['files'],REPORTING_FILES)
    return directory,receipt


def verify_publication(directory, identity, files, summary):
    r=read_json(Path(directory)/'receipt.json','Figure publication receipt')
    require(isinstance(r,dict) and set(r)=={'schema_version','status','fingerprint','identity','summary','files','completed_at'}
            and type(r['schema_version']) is int and r['schema_version']==1
            and r['status']=='accepted' and r['fingerprint']==digest(identity)
            and canonical_bytes(r['identity'])==canonical_bytes(identity)
            and canonical_bytes(r['summary'])==canonical_bytes(summary) and r['files']==files,
            'cached figures differ from completely regenerated publication contents.')
    verify_files(directory,r['files'],RESULT_FILES)


def render_payloads(stage, root, data, correction, fonts, tool, metadata, summary):
    from .reporting.figures import render_figures
    write_json(stage/'plot_data.json',data)
    (stage/'plot_points.csv').write_text(points_csv(data),encoding='utf-8')
    (stage/'figure_captions.txt').write_text('\n\n'.join(f['title']+'\n'+f['caption']['text'] for f in data['figures'])+'\n',encoding='utf-8')
    for name in ('estimates','coefficients','covariance'):
        getattr(correction,name).to_csv(stage/('corrected_pretb_'+name+'.csv'),index=False,lineterminator='\n')
    write_json(stage/'corrected_pretb_metadata.json',correction.metadata)
    rendered=render_figures(data,stage,fonts,dpi=600)
    require(set(rendered.get('files',{}))==set(IMAGE_FILES)
            and all(file_hash(stage/n)==h for n,h in rendered['files'].items()), 'renderer image inventory differs.')
    write_json(stage/'render_metadata.json',rendered)
    write_json(stage/'font_validation.json',verify_pdf_fonts(stage,tool,fonts))
    write_json(stage/'metadata.json',metadata);write_json(stage/'summary.json',summary)
    require({p.name for p in stage.iterdir()}==set(RESULT_FILES), 'figure publication artifact inventory differs.')


def run_figures(root, config_file, analysis_file, primary_sampling_file, mechanics_sampling_file,
                supporting_sampling_file, wheeze_sampling_file, sensitivity_sampling_file,
                reporting_file, figures_file, *, jobs=1, resume=False):
    from .execution import run_preparation
    from . import pretb_execution
    from .analysis.pretb_uncertainty import calculate
    from .reporting.render_document import _font
    root=Path(root).absolute();no_symlinks(root);root=root.resolve();assert_loaded(root)
    require(type(resume) is bool and type(jobs) is int and 1<=jobs<=3, 'M14 jobs/resume arguments are invalid.')
    paths=[Path(p).absolute() for p in (config_file,analysis_file,primary_sampling_file,mechanics_sampling_file,
           supporting_sampling_file,wheeze_sampling_file,sensitivity_sampling_file,reporting_file,figures_file)]
    for p in paths:no_symlinks(p)
    paths=[p.resolve() for p in paths]
    local=read_json(paths[0],'Local input configuration');_,directories=resolve_paths(local,paths[0])
    cfg=figures_config(paths[-1]);report_cfg=reporting_config(paths[-2]);fonts={}
    for name,value in local['fonts'].items():
        p=Path(value).expanduser();p=p if p.is_absolute() else paths[0].parent/p
        no_symlinks(p);fonts[name]=p.resolve()
    require(set(fonts)=={'arial','arial_bold'}, 'both actual Arial font files are required.')
    for name,bold in (('arial',False),('arial_bold',True)):_font(fonts[name].read_bytes(),bold)
    font_tool_identity(cfg['pdf_fonts_tool'])
    def overlap(a,b):return a==b or a in b.parents or b in a.parents
    for value in local['directories'].values():
        p=Path(value).expanduser();no_symlinks(p if p.is_absolute() else paths[0].parent/p)
    require(not overlap(directories['work'],directories['outputs']) and not any(overlap(p,root) for p in directories.values()),
            'M14 generated roots must be separate from source and one another.')
    protected=[*paths,*fonts.values(),cfg['pdf_fonts_tool'],*report_cfg.values()]
    require(not any(p==d or d in p.parents for p in protected for d in directories.values()), 'M14 generated roots cannot contain protected inputs.')
    inputs={p:file_hash(p) for p in protected}
    doc_runtime=document_runtime(report_cfg['document_python'])
    prior=run_reporting(root,*paths[:-1],jobs=jobs,resume=resume)
    publication,report=_reporting_publication(root,directories,prior)
    require(report['identity']['document_runtime']==doc_runtime
            and report['identity']['references']=={k:inputs[p] for k,p in report_cfg.items() if k.startswith('reference_')},
            'M13 document runtime or reference inputs differ from the M14 preflight.')
    for name in (*REPORTING_FILES,'receipt.json'):inputs[publication/name]=file_hash(publication/name)
    prep=directories['work']/'preparation_runs'/prior['preparation_fingerprint'];no_symlinks(prep)
    manifest=verify_artifacts(prep)
    for name,h in manifest['files'].items():inputs[prep/name]=h
    for name in ('receipt.json','artifacts.json'):inputs[prep/name]=file_hash(prep/name)
    descriptor=pretb_execution.contract_descriptor(prep)
    fp,source_identity=pretb_execution.fit_identity(root,descriptor)
    source=directories['outputs']/'pretb_cox'/fp
    original=verify_receipt(source,source_identity,pretb_execution.RESULT_FILES,status=None)
    bound=[x for x in report['identity']['source_bindings'] if x['family']=='pretb']
    require(len(bound)==1 and bound[0]['fingerprint']==fp and bound[0]['receipt_sha256']==file_hash(source/'receipt.json'),
            'corrected uncertainty must use the same Cox source as the verified M13 registry.')
    for name in (*original['files'],'receipt.json'):inputs[source/name]=file_hash(source/name)
    correction=calculate(prep,source)
    data=build_data(read_json(publication/'registry.json','Current aggregate registry'),
                    read_json(root/'config/publication_figures.json','Publication figure specification'),correction)
    runtime=figure_runtime(cfg['pdf_fonts_tool'])
    identity={'schema_version':1,'source':source_hashes(root),'reporting_fingerprint':prior['fingerprint'],
              'reporting_receipt_sha256':inputs[publication/'receipt.json'],
              'preparation_fingerprint':prior['preparation_fingerprint'],
              'correction_source_fingerprint':fp,'correction_source_receipt_sha256':inputs[source/'receipt.json'],
              'uncertainty_selection':cfg['pretb_uncertainty'],'runtime':runtime,
              'fonts':{k:file_hash(p) for k,p in fonts.items()}}
    fingerprint=digest(identity)
    summary={'through':'M14','accepted':True,'status':'accepted','fingerprint':fingerprint,
             'reporting_fingerprint':prior['fingerprint'],'preparation_fingerprint':prior['preparation_fingerprint'],
             'coverage':data['coverage'],'preceding_modules_verified':13,'new_model_fits':0,
             'corrected_pretb_models':17,'corrected_pretb_coefficients':51,'corrected_pretb_covariance_cells':306,
             'pretb_point_estimates_preserved':True,'uncertainty_selection':cfg['pretb_uncertainty'],
             'G08_status':'actual event-time score membership corrected for Figure S3; source-reproduction artifacts retained',
             'inherited_limitations':['Four upstream antenatal flow counts remain unavailable',
                                      *correction.metadata['remaining_limitations']]}
    metadata={'schema_version':1,'coverage':data['coverage'],'fonts':identity['fonts'],'runtime':runtime,
              'exports':{'individual_figures':9,'combined_pdf_pages':9,'PNG_dpi':600,'PDF':'vector with embedded actual Arial'},
              'uncertainty_selection':cfg['pretb_uncertainty'],'manuscripts_modified':False,
              'participant_records_exported':False,'small_counts_suppressed':False,
              'display_policy':'Full precision for positions; centralized half-up rounding for labels.',
              'visual_review':'Rendered-page inspection is a separate required delivery check.',
              'limitations':summary['inherited_limitations']}
    work=directories['work']/'publication_figures';output=directories['outputs']/'publication_figures'
    for p in (work,output):
        no_symlinks(p);require(not overlap(p,prep), 'M14 generated roots cannot overlap prepared inputs.');p.mkdir(parents=True,exist_ok=True,mode=0o700)
    destination=output/fingerprint
    def unchanged():
        assert_loaded(root)
        for p in (work,output,destination,prep,publication,source):no_symlinks(p)
        for p,h in inputs.items():
            no_symlinks(p);require(p.is_file() and file_hash(p)==h,'an M14 input, font, configuration or upstream artifact changed.')
        verified=run_preparation(root,paths[0],paths[1],'M03',resume=True)
        require(verified['fingerprint']==prior['preparation_fingerprint'],'M14 preparation changed during publication.')
        require(figure_runtime(cfg['pdf_fonts_tool'])==runtime,'M14 rendering runtime changed during publication.')
        require(document_runtime(report_cfg['document_python'])==doc_runtime,'M13 document runtime changed during M14 publication.')
    with preparation_locks(work,output):
        no_symlinks(destination);reused=destination.exists()
        require(not reused or resume,'identical figures already exist; use --resume to regenerate and compare.')
        staged=Path(tempfile.mkdtemp(prefix='.staging-',dir=output));published=False
        try:
            render_payloads(staged,root,data,correction,fonts,cfg['pdf_fonts_tool'],metadata,summary)
            files={n:file_hash(staged/n) for n in RESULT_FILES};unchanged()
            if reused:verify_publication(destination,identity,files,summary)
            else:
                write_json(staged/'receipt.json',{'schema_version':1,'status':'accepted','fingerprint':fingerprint,
                           'identity':identity,'summary':summary,'files':files,'completed_at':datetime.now(timezone.utc).isoformat()})
                verify_publication(staged,identity,files,summary);no_symlinks(destination)
                require(not destination.exists(),'M14 publication destination appeared during validation.')
                os.replace(staged,destination);published=True
            unchanged()
            completion={**summary,'resumed':reused};write_pointer(directories['work']/'figures_current.json',completion)
        except Exception:
            if staged.exists():os.replace(staged,work/('failed-'+staged.name.removeprefix('.staging-')))
            if published and destination.exists():
                no_symlinks(destination);os.replace(destination,work/('failed-publication-'+staged.name.removeprefix('.staging-')))
            raise
        finally:
            if staged.exists():shutil.rmtree(staged)
    return completion
