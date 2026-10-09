"""Cumulative M13 validation and atomic manuscript-table publication."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

from .execution import canonical_bytes, file_hash, preparation_locks, resolve_paths, write_json, write_pointer
from .pipeline import ValidationError, read_json
from .reporting.sources import collect_sources, digest, no_symlinks, require, verify_files
from .reporting.tables import build_tables
from .reporting.numerical_text import build_checks, compare_references, TABLE_IDS

RESULT_FILES = ('registry.json','tables.json','numerical_text.json','numerical_text.txt',
                'reference_comparison.json','metadata.json','summary.json',
                'manuscript_tables.html','manuscript_tables.docx',
                *(table+'.csv' for table in TABLE_IDS))


def source_hashes(root):
    root=Path(root)
    paths=[]
    for folder,pattern in [('src','*.py'),('config','*.json'),('docs','*.md'),('tests','*.py'),
                           ('tools','*.py'),('.github/workflows','*.yml'),('.github/workflows','*.yaml')]:
        paths.extend(p for p in (root/folder).rglob(pattern) if '.local.' not in p.name)
    paths.extend(root/n for n in ('run.py','README.md','requirements-models.lock.txt','requirements-preparation.lock.txt'))
    if (root/'.gitignore').is_file():paths.append(root/'.gitignore')
    for p in paths:
        no_symlinks(p)
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(paths)}


_IMPORTED_ROOT = Path(__file__).resolve().parents[2]
_LOADED = source_hashes(_IMPORTED_ROOT)


def assert_loaded(root):
    require(source_hashes(root)==_LOADED and source_hashes(_IMPORTED_ROOT)==_LOADED,
            'source files changed after import or differ from the supplied source tree; restart.')


def reporting_config(path):
    path=Path(path).absolute();no_symlinks(path)
    value=read_json(path,'Local reporting configuration')
    require(isinstance(value,dict) and set(value) in (
        {'schema_version','document_python'},
        {'schema_version','document_python','reference_cells','reference_sentences'})
        and type(value.get('schema_version')) is int and value['schema_version']==1,
        'local configuration requires schema_version, document_python and an optional pair of reference inventories.')
    resolved={}
    for key in set(value)-{'schema_version'}:
        require(isinstance(value[key],str) and value[key].strip(), 'configuration paths must be nonempty strings.')
        p=Path(value[key]).expanduser()
        p=p if p.is_absolute() else path.parent/p
        if key!='document_python':no_symlinks(p)
        p=p.resolve()
        require(p.is_file(), 'configured document runtime and references must be readable files.')
        resolved[key]=p
    require(os.access(resolved['document_python'],os.X_OK), 'document Python must be executable.')
    return resolved


def document_runtime(executable):
    try:
        run=subprocess.run([str(executable),'-I','-c',
            'import json,platform,sys,zlib; print(json.dumps({"python":platform.python_version(),"implementation":sys.implementation.name,"system":platform.system(),"machine":platform.machine(),"zlib":zlib.ZLIB_VERSION}))'],
            check=True,capture_output=True,text=True,timeout=30)
        identity=json.loads(run.stdout)
    except (OSError,ValueError,subprocess.SubprocessError):
        raise ValidationError('Document runtime preflight failed.') from None
    require(isinstance(identity,dict) and set(identity)=={'python','implementation','system','machine','zlib'}
            and all(isinstance(v,str) and v for v in identity.values()), 'document runtime preflight returned invalid metadata.')
    return {**identity,'executable_sha256':file_hash(executable)}


def _html(tables):
    out=['<!doctype html><html lang="en"><meta charset="utf-8"><title>DCHS lung function manuscript tables</title>',
         '<style>body{font-family:Arial,sans-serif;margin:2rem;line-height:1.35;color:#000}section{max-width:1100px;margin:0 auto 3rem}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d9d9d9;padding:8px;text-align:center;white-space:pre-line}td:first-child,th:first-child{text-align:left}th{background:#e8eef3}.section,.category_heading{background:#f3f3f3;font-weight:bold}p{font-size:14px}@media print{section{break-before:page}thead{display:table-header-group}tr{break-inside:avoid}}</style>',
         '<h1>DCHS lung function manuscript tables</h1>']
    for table in tables:
        out.append('<section><h2>'+escape(table['title'])+'</h2><table contenteditable="true"><thead>')
        in_head=True
        for row in table['rows']:
            if row['kind']=='spacer':continue
            if in_head and row['kind']!='header':
                out.append('</thead><tbody>');in_head=False
            out.append('<tr class="'+escape(row['kind'],quote=True)+'">')
            tag='th' if row['kind']=='header' else 'td'
            for c in row['cells']:
                out.append(f'<{tag} colspan="{c.get("colspan",1)}">'+escape(c['text'])+f'</{tag}>')
            out.append('</tr>')
        out.append('</tbody></table>')
        out.extend('<p>'+escape(note)+'</p>' for note in table['footnotes'])
        out.append('</section>')
    out.append('</html>')
    return '\n'.join(out)+'\n'


def render_payloads(stage, root, registry, tables, checks, comparison, metadata, summary, config, fonts):
    for name,value in [('registry',registry.export()),('tables',{'schema_version':1,'tables':tables}),
                       ('numerical_text',checks),('reference_comparison',comparison),
                       ('metadata',metadata),('summary',summary)]:
        write_json(stage/(name+'.json'),value)
    (stage/'numerical_text.txt').write_text('\n\n'.join(s['id']+' ['+s['status']+']\n'+s['text'] for s in checks['sentences'])+'\n')
    (stage/'manuscript_tables.html').write_text(_html(tables),encoding='utf-8')
    for table in tables:
        stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
        for row in table['rows']:
            expanded=[]
            for c in row['cells']:expanded.extend([c['text']]+['']*(c.get('colspan',1)-1))
            writer.writerow(expanded)
        (stage/(table['id']+'.csv')).write_text(stream.getvalue(),encoding='utf-8')
    command=[str(config['document_python']),'-I',str(root/'src/dchs_lung_function/reporting/render_document.py'),
             '--tables',str(stage/'tables.json'),'--output',str(stage/'manuscript_tables.docx'),
             '--arial',str(fonts['arial']),'--arial-bold',str(fonts['arial_bold'])]
    try:
        subprocess.run(command,check=True,capture_output=True,timeout=60)
    except (OSError,subprocess.SubprocessError):
        raise ValidationError('Editable document rendering failed; no manuscript publication was accepted.') from None
    require({p.name for p in stage.iterdir()}==set(RESULT_FILES), 'renderer produced an incomplete or unexpected file inventory.')


def verify_publication(directory, identity, expected_hashes, summary):
    receipt=read_json(Path(directory)/'receipt.json','Manuscript publication receipt')
    require(isinstance(receipt,dict) and set(receipt)=={'schema_version','status','fingerprint','identity','summary','files','completed_at'}
            and type(receipt['schema_version']) is int and receipt['schema_version']==1
            and receipt['status']=='accepted' and receipt['fingerprint']==digest(identity)
            and canonical_bytes(receipt['identity'])==canonical_bytes(identity)
            and canonical_bytes(receipt['summary'])==canonical_bytes(summary)
            and receipt['files']==expected_hashes,
            'cached publication differs from the complete freshly computed contents.')
    verify_files(directory,receipt['files'],RESULT_FILES)


def run_reporting(root, config_file, analysis_file, primary_sampling_file,
                  mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file,
                  sensitivity_sampling_file, reporting_file, *, jobs=1, resume=False):
    """Revalidate M01–M12, derive every reporting input, then publish M13."""
    from .sensitivity_execution import run_sensitivities
    from .execution import run_preparation
    from .sampling import validate_sampling as vp
    from .mechanics_sampling import validate_sampling as vm
    from .supporting_sampling import validate_sampling as vs
    from .wheeze_sampling import validate_sampling as vw
    from .sensitivity_sampling import validate_sampling as vx
    root=Path(root).absolute();no_symlinks(root);root=root.resolve()
    require(type(resume) is bool and type(jobs) is int and 1<=jobs<=3, 'resume/jobs arguments are invalid.')
    assert_loaded(root)
    paths=[Path(p).absolute() for p in (config_file,analysis_file,primary_sampling_file,
           mechanics_sampling_file,supporting_sampling_file,wheeze_sampling_file,sensitivity_sampling_file,reporting_file)]
    for p in paths:no_symlinks(p)
    local=read_json(paths[0],'Local input configuration')
    _,directories=resolve_paths(local,paths[0])
    def overlap(a,b):return a==b or a in b.parents or b in a.parents
    for value in local['directories'].values():
        p=Path(value).expanduser();no_symlinks(p if p.is_absolute() else paths[0].parent/p)
    require(not overlap(directories['work'],directories['outputs'])
            and not any(overlap(p,root) for p in directories.values()), 'generated roots must be separate from one another and source.')
    cfg=reporting_config(paths[-1]);fonts={}
    for name,value in local['fonts'].items():
        p=Path(value).expanduser();p=p if p.is_absolute() else paths[0].parent/p
        no_symlinks(p);fonts[name]=p.resolve()
    protected=[*paths,*cfg.values(),*fonts.values()]
    require(not any(p==d or d in p.parents for p in protected for d in directories.values()),
            'generated roots cannot contain configuration, reference, runtime or font files.')
    input_hashes={p:file_hash(p) for p in protected}
    runtime=document_runtime(cfg['document_python'])
    reference_cells=read_json(cfg['reference_cells'],'Reference cells') if 'reference_cells' in cfg else None
    reference_sentences=read_json(cfg['reference_sentences'],'Reference sentences') if 'reference_sentences' in cfg else None
    settings={k:validate(read_json(p,'Sampling settings')) for k,validate,p in zip(
        ('primary','mechanics','supporting','wheeze','sensitivity_model'),(vp,vm,vs,vw,vx),paths[2:7])}
    prior=run_sensitivities(root,*paths[:7],jobs=jobs,resume=resume)
    require(isinstance(prior,dict) and prior.get('accepted') is True and prior.get('status')=='accepted'
            and prior.get('through')=='M12' and type(prior.get('selected_models')) is int
            and prior['selected_models']==10 and type(prior.get('reused_models')) is int and prior['reused_models']==4
            and isinstance(prior.get('preparation_fingerprint'),str)
            and re.fullmatch('[0-9a-f]{64}',prior['preparation_fingerprint']),
            'requires successful cumulative validation through M12.')
    expected={'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
              'mechanics_models_verified':6,'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
              'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,'wheeze_models_verified':2,'feno_models_verified':11}
    require(all(type(prior.get(k)) is int and prior[k]==v for k,v in expected.items()),
            'cumulative predecessor coverage is incomplete.')
    prep=directories['work']/'preparation_runs'/prior['preparation_fingerprint']
    registry,bindings,upstream_hashes=collect_sources(root,prep,directories,settings)
    input_hashes.update(upstream_hashes)
    spec=read_json(root/'config/manuscript_reporting.json','Manuscript specification')
    tables=build_tables(registry);checks=build_checks(registry,tables,spec)
    require(not checks['wording_review_required'], 'a qualitative numerical statement requires scientific review before publication.')
    comparison=compare_references(checks,reference_cells,reference_sentences)
    identity={'schema_version':1,'source':source_hashes(root),'preparation_fingerprint':prior['preparation_fingerprint'],
              'source_bindings':bindings,'document_runtime':runtime,
              'fonts':{k:file_hash(p) for k,p in fonts.items()},
              'references':{k:file_hash(p) for k,p in cfg.items() if k.startswith('reference_')}}
    fp=digest(identity)
    summary={'through':'M13','fingerprint':fp,'preparation_fingerprint':prior['preparation_fingerprint'],
             'accepted':True,'status':'accepted','coverage':checks['coverage'],
             'registry_records':registry.export()['record_count'],'aggregate_sources':len(registry.export()['sources']),
             'source_publications_verified':len(bindings)+1,'preceding_modules_verified':12,
             'upstream_models':expected,'new_model_fits':0,
             'reference_comparison_status':comparison['status'],
             'scientific_reference_comparison':'mapped values checked against accepted computed results; retained displays are informational',
             'inherited_limitations':prior['inherited_limitations']}
    metadata={'schema_version':1,'coverage':checks['coverage'],'display_policy':
              'Decimal half-up; fixed precision; no negative displayed zero; exact small counts retained.',
              'reference_policy':comparison.get('comparison_policy','Reference display inventory not supplied.'),
              'document_fonts':identity['fonts'],'document_runtime':runtime,
              'inherited_limitations':prior['inherited_limitations'],
              'manuscript_files_modified':False,'participant_records_exported':False,
              'visual_review':'Separate rendered-page inspection is required for delivery; CLI checks structure and exact regeneration.'}
    work=directories['work']/'manuscript_reporting';output=directories['outputs']/'manuscript_reporting'
    for p in (work,output):
        no_symlinks(p);require(not overlap(p,prep),'reporting paths cannot overlap preparation.');p.mkdir(parents=True,exist_ok=True,mode=0o700)
    destination=output/fp

    def unchanged():
        assert_loaded(root)
        for p in (work,output,destination,prep):no_symlinks(p)
        for p,h in input_hashes.items():
            no_symlinks(p)
            require(p.is_file() and not p.is_symlink() and file_hash(p)==h,
                    'an input, configuration, font or upstream artifact changed during publication.')
        verified=run_preparation(root,paths[0],paths[1],'M03',resume=True)
        require(verified['fingerprint']==prior['preparation_fingerprint'],'preparation changed during publication.')
        require(document_runtime(cfg['document_python'])==runtime,'document runtime changed during publication.')

    with preparation_locks(work,output):
        no_symlinks(destination)
        reused=destination.exists()
        require(not reused or resume,'identical reporting results exist; use --resume to recheck them.')
        staged=Path(tempfile.mkdtemp(prefix='.staging-',dir=output))
        published=False
        try:
            render_payloads(staged,root,registry,tables,checks,comparison,metadata,summary,cfg,fonts)
            files={n:file_hash(staged/n) for n in RESULT_FILES}
            unchanged()
            if reused:
                verify_publication(destination,identity,files,summary)
            else:
                write_json(staged/'receipt.json',{'schema_version':1,'status':'accepted','fingerprint':fp,
                    'identity':identity,'summary':summary,'completed_at':datetime.now(timezone.utc).isoformat(),'files':files})
                verify_publication(staged,identity,files,summary)
                no_symlinks(destination)
                require(not destination.exists(), 'publication destination appeared during verification.')
                os.replace(staged,destination)
                published=True
            unchanged()
            completion={**summary,'resumed':reused}
            write_pointer(directories['work']/'reporting_current.json',completion)
        except Exception:
            if staged.exists():
                failed=work/('failed-'+staged.name.removeprefix('.staging-'))
                os.replace(staged,failed)
            if published and destination.exists():
                no_symlinks(destination)
                os.replace(destination,work/('failed-publication-'+staged.name.removeprefix('.staging-')))
            raise
        finally:
            if staged.exists():shutil.rmtree(staged)
    return completion
