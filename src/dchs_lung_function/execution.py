"""Run and store deterministic preparation in private, fingerprinted directories."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

from .pipeline import ValidationError, read_json, validate_config
from .lifecycle import directory_locks, rename_noreplace, sync_directory, sync_tree


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    Path(path).write_bytes(canonical_bytes(value)+b"\n")


def runtime_identity():
    from .runtime_bootstrap import bootstrap
    controls=bootstrap()
    import numpy as np
    packages = ["numpy", "pandas", "scipy", "patsy", "statsmodels", "rdata", "pyarrow"]
    return {"python":platform.python_version(), "system":platform.system(), "machine":platform.machine(),
            "packages":{name:importlib.metadata.version(name) for name in packages},
            "os_release":platform.release(), "os_version":platform.version(),
            "macos_version":platform.mac_ver()[0], "requested_thread_environment":controls,
            "numpy_blas":np.show_config(mode="dicts").get("Build Dependencies",{}).get("blas",{})}


def resolve_paths(configuration, local_path):
    base = Path(local_path).resolve().parent
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base/path).resolve()
    return ({name:resolve(spec["path"]) for name,spec in configuration["inputs"].items()},
            {name:resolve(value) for name,value in configuration["directories"].items()})


def validate_analysis(settings):
    if not isinstance(settings, dict) or set(settings)!={"schema_version", "event_end", "respiratory_end"}:
        raise ValidationError("Preparation settings require schema_version, event_end and respiratory_end.")
    if type(settings["schema_version"]) is not int or settings["schema_version"]!=1:
        raise ValidationError("Preparation settings require schema_version 1.")
    from .inputs import normalize_dates
    import pandas as pd
    for field in ("event_end", "respiratory_end"):
        normalize_dates(pd.Series([settings[field]]), field, allow_missing=False)
    return settings


def preparation_identity(root, configuration, paths, settings, through):
    source = {}
    for folder, pattern in [("src", "*.py"), ("config", "*.json")]:
        for path in sorted((root/folder).rglob(pattern)):
            if ".local." not in path.name:
                source[str(path.relative_to(root))] = file_hash(path)
    for name in ("run.py", "requirements-preparation.lock.txt"):
        source[name] = file_hash(root/name)
    inputs = {role:{"sha256":file_hash(path), "bytes":path.stat().st_size} for role,path in sorted(paths.items())}
    for role, definition in configuration["inputs"].items():
        expected = definition.get("sha256")
        if expected and inputs[role]["sha256"]!=expected.lower():
            raise ValidationError("An input no longer matches its configured checksum.")
    identity = {"schema_version":1, "through":through, "settings":settings,
                "configuration_sha256":hashlib.sha256(canonical_bytes(configuration)).hexdigest(),
                "source":source, "inputs":inputs, "runtime":runtime_identity()}
    return hashlib.sha256(canonical_bytes(identity)).hexdigest(), identity


class ArtifactWriter:
    """Serialize typed private frames/arrays without executable pickle objects."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.counter = 0
        self.files = {}

    def allocate(self, extension):
        self.counter += 1
        name = "artifact_{:05d}.{}".format(self.counter, extension)
        return name, self.directory/name

    def encode(self, value):
        import numpy as np
        import pandas as pd
        if is_dataclass(value) and not isinstance(value, type):
            value = {field.name:getattr(value, field.name) for field in fields(value)}
        if isinstance(value, pd.DataFrame):
            name, path = self.allocate("parquet")
            value.to_parquet(path, engine="pyarrow", index=True, compression="zstd")
            self.files[name] = file_hash(path)
            return {"type":"dataframe", "file":name, "rows":len(value), "columns":len(value.columns)}
        if isinstance(value, pd.Series):
            return {"type":"series", "name":self.encode(value.name), "data":self.encode(value.to_frame("value"))}
        if isinstance(value, pd.Index):
            return {"type":"index", "name":self.encode(value.name), "data":self.encode(value.to_numpy())}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {"type":"object_array", "shape":list(value.shape), "items":self.encode(value.tolist())}
            name,path = self.allocate("npy")
            np.save(path, value, allow_pickle=False)
            self.files[name] = file_hash(path)
            return {"type":"ndarray", "file":name, "shape":list(value.shape), "dtype":str(value.dtype)}
        if isinstance(value, dict):
            return {"type":"mapping", "items":[[self.encode(k),self.encode(v)] for k,v in value.items()]}
        if isinstance(value, (list,tuple)):
            return {"type":"tuple" if isinstance(value,tuple) else "list", "items":[self.encode(v) for v in value]}
        if value is pd.NA or value is pd.NaT:
            return {"type":"missing"}
        if isinstance(value, (pd.Timestamp, datetime)):
            return {"type":"timestamp", "value":value.isoformat()}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return {"type":"nonfinite", "value":str(value)}
        if value is None or isinstance(value,(str,int,float,bool)):
            return {"type":"scalar", "value":value}
        raise TypeError("Unsupported private artifact type: "+type(value).__name__)

    def finish(self, value):
        tree = self.encode(value)
        manifest = {"schema_version":1, "tree":tree, "files":self.files}
        write_json(self.directory/"artifacts.json", manifest)
        return manifest


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory/"artifacts.json", "Artifact manifest")
    if manifest.get("schema_version")!=1 or not isinstance(manifest.get("files"),dict):
        raise ValidationError("Invalid artifact manifest.")
    for name, expected in manifest["files"].items():
        if Path(name).name!=name or not isinstance(expected,str):
            raise ValidationError("Invalid artifact location or digest.")
        path = directory/name
        if path.is_symlink() or not path.is_file() or file_hash(path)!=expected:
            raise ValidationError("A private preparation artifact is missing or changed.")
    return manifest


def artifact_subtree(manifest, keys=()):
    """Select a mapping subtree without loading unrelated participant artifacts."""
    node = manifest['tree']
    for key in keys:
        if node.get('type')!='mapping':
            raise ValidationError('Requested preparation artifact is not a mapping.')
        matches = [value for item,value in node['items'] if item=={'type':'scalar','value':key}]
        if len(matches)!=1:
            raise ValidationError('Requested preparation artifact is missing or duplicated.')
        node = matches[0]
    return node


def load_artifacts(directory, keys=()):
    """Read only typed artifacts whose file identities have been verified."""
    import numpy as np
    import pandas as pd
    directory = Path(directory)
    manifest = verify_artifacts(directory)
    def decode(node):
        kind = node["type"]
        if kind in {"dataframe","ndarray"}:
            name = node["file"]
            if name not in manifest["files"]:
                raise ValidationError("Artifact tree refers to an unverified file.")
            return (pd.read_parquet(directory/name,engine="pyarrow") if kind=="dataframe"
                    else np.load(directory/name,allow_pickle=False))
        if kind=="mapping":return {decode(k):decode(v) for k,v in node["items"]}
        if kind in {"list","tuple"}:
            values = [decode(v) for v in node["items"]]
            return tuple(values) if kind=="tuple" else values
        if kind=="series":return decode(node["data"])["value"].rename(decode(node["name"]))
        if kind=="index":return pd.Index(decode(node["data"]),name=decode(node["name"]))
        if kind=="object_array":return np.array(decode(node["items"]),dtype=object).reshape(node["shape"])
        if kind=="timestamp":return pd.Timestamp(node["value"])
        if kind=="missing":return pd.NA
        if kind=="nonfinite":return float(node["value"])
        if kind=="scalar":return node["value"]
        raise ValidationError("Unknown artifact type.")
    return decode(artifact_subtree(manifest,keys))


def compute_preparation(configuration, local_path, paths, settings, through):
    from .inputs import load_inputs
    imported = load_inputs(configuration["inputs"],base_dir=Path(local_path).resolve().parent)
    bundle = {"M01":{role:{"data":table.data,"metadata":table.metadata} for role,table in imported.items()}}
    summary = {"through":through, "imports":{role:{"rows":len(table.data),"columns":len(table.data.columns)} for role,table in imported.items()}}
    aggregates = {}
    if through=="M01":return bundle,summary,aggregates
    from .preparation import link_longitudinal,enrich_covariates,normalize_pft,prepare_model_inputs
    from .reference_scores import add_reference_scores,load_gli_lookup
    from .cohorts import build_cohorts,build_flow_summary
    from .descriptive import build_table1,build_modality_schedule
    baseline = imported["baseline"].data
    linked = link_longitudinal(imported["pulmonary_function"].data,baseline,imported["tb_events"].data,
                               ascertainment_end=settings["event_end"])
    scored = add_reference_scores(linked,load_gli_lookup(paths["gli_lookup"]))
    prepared = normalize_pft(enrich_covariates(scored,baseline),baseline)
    cohorts = build_cohorts(prepared,baseline,ascertainment_end=settings["event_end"])
    descriptive = build_table1(cohorts.descriptive_children)
    flow = build_flow_summary(cohorts,baseline)
    schedule = build_modality_schedule(prepared)
    bundle["M02"] = {"pft":prepared,"cohorts":cohorts,"descriptive":descriptive,"flow":flow,"schedule":schedule}
    aggregates.update({"table1":descriptive.table,"table1_statistics":descriptive.statistics,
                       "table1_missingness":descriptive.missingness,"cohort_ledger":cohorts.ledger,
                       "flow":flow,"modality_schedule":schedule})
    summary["cohorts"] = cohorts.audit
    summary["descriptive"] = descriptive.audit
    summary["flow_unavailable_rows"] = int(flow["status"].str.startswith("not_reconstructable").sum())
    if through=="M02":return bundle,summary,aggregates
    from .respiratory_preparation import prepare_respiratory_inputs
    pft_models = prepare_model_inputs(prepared,baseline,event_end=settings["event_end"])
    respiratory = prepare_respiratory_inputs(prepared,baseline,imported["respiratory_tb_dates"].data,
        imported["wheeze_assessments"].data,cohorts.descriptive_children["pid_child"],administrative_end=settings["respiratory_end"])
    bundle["M03"] = {"pft":pft_models,"respiratory":respiratory}
    summary["pft_preparation"] = pft_models["audit"]
    summary["respiratory_preparation"] = respiratory["audit"]
    return bundle,summary,aggregates


@contextmanager
def preparation_locks(work, output):
    with ExitStack() as stack:
        try:
            stack.enter_context(directory_locks((work, output), legacy_name='.preparation.lock'))
        except (BlockingIOError, ValueError) as error:
            raise ValidationError('Another preparation run holds a work or output lock, or lock ownership is unsafe.') from error
        yield


def _verify_preparation_pair(work, output, fingerprint, identity, through, receipt_hash=None):
    if work.is_symlink() or output.is_symlink():
        raise ValidationError('Preparation directories cannot be symlinks.')
    receipt_path = work/'receipt.json'
    if receipt_path.is_symlink() or (receipt_hash and file_hash(receipt_path)!=receipt_hash):
        raise ValidationError('Preparation transaction receipt changed.')
    receipt = read_json(receipt_path, 'Preparation receipt')
    if (receipt.get('fingerprint')!=fingerprint or receipt.get('identity')!=identity
            or receipt.get('status')!='prepared' or receipt.get('through')!=through):
        raise ValidationError('Preparation cache identity does not match this run.')
    if (work/'artifacts.json').is_symlink() or file_hash(work/'artifacts.json')!=receipt.get('artifact_manifest_sha256'):
        raise ValidationError('Preparation artifact manifest changed.')
    manifest = verify_artifacts(work)
    if {p.name for p in work.iterdir()}!={'receipt.json','artifacts.json',*manifest['files']}:
        raise ValidationError('Preparation private file inventory changed.')
    outputs = receipt.get('aggregate_files')
    if not isinstance(outputs,dict) or 'summary.json' not in outputs:
        raise ValidationError('Preparation receipt lacks aggregate output identities.')
    if {p.name for p in output.iterdir()}!=set(outputs):
        raise ValidationError('Preparation aggregate file inventory changed.')
    for name,expected in outputs.items():
        path = output/name
        if Path(name).name!=name or path.is_symlink() or not path.is_file() or file_hash(path)!=expected:
            raise ValidationError('Aggregate preparation output is missing or changed.')
    summary = read_json(output/'summary.json','Preparation summary')
    if receipt.get('summary')!=summary:
        raise ValidationError('Preparation receipt and verified aggregate summary disagree.')
    return summary


def _recover_preparation(journal, work, output, fingerprint, identity, through, unchanged):
    if journal.is_symlink(): raise ValidationError('Preparation journal cannot be a symlink.')
    state = read_json(journal, 'Preparation transaction journal')
    keys = {'schema_version','fingerprint','stage_work','stage_output','receipt_sha256'}
    if (not isinstance(state,dict) or set(state)!=keys or type(state['schema_version']) is not int
            or state['schema_version']!=1 or state['fingerprint']!=fingerprint):
        raise ValidationError('Invalid preparation recovery journal.')
    candidates=[]
    for parent, key in ((work,'stage_work'),(output,'stage_output')):
        name=state[key]
        if not isinstance(name,str) or Path(name).name!=name or not name.startswith('.staging-'):
            raise ValidationError('Invalid preparation recovery staging location.')
        stage, final = parent/name, parent/fingerprint
        if stage.is_symlink() or final.is_symlink() or stage.exists()==final.exists():
            raise ValidationError('Preparation recovery has ambiguous or unsafe directories.')
        candidate = final if final.exists() else stage
        if not candidate.is_dir(): raise ValidationError('Preparation recovery staging is missing.')
        candidates.append((candidate,final))
    _verify_preparation_pair(candidates[0][0],candidates[1][0],fingerprint,identity,through,state['receipt_sha256'])
    unchanged()
    for candidate,final in reversed(candidates):
        if candidate!=final:
            rename_noreplace(candidate,final)
            sync_directory(final.parent)
    _verify_preparation_pair(work/fingerprint,output/fingerprint,fingerprint,identity,through,state['receipt_sha256'])
    unchanged()
    journal.unlink();sync_directory(work)


def write_pointer(pointer, value):
    handle,name = tempfile.mkstemp(prefix=".preparation-current-",suffix=".tmp",dir=pointer.parent)
    os.close(handle)
    temp = Path(name)
    try:
        write_json(temp,value)
        with temp.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(temp,pointer)
        sync_directory(pointer.parent)
    finally:
        temp.unlink(missing_ok=True)


def run_preparation(root, config_file, analysis_file, through, *, resume=False):
    """Commit one cumulative M01-M03 run only after every requested check succeeds."""
    root = Path(root).resolve()
    config_file = Path(config_file).resolve()
    analysis_file = Path(analysis_file).resolve()
    configuration = read_json(config_file,"Local configuration")
    configuration_file_hash = file_hash(config_file)
    contract = read_json(root/"config/input_contract.json","Input contract")
    validate_config(configuration,contract,config_file,root)
    if through not in {"M01","M02","M03"}:
        raise ValidationError("Only preparation modules M01-M03 are implemented.")
    settings = validate_analysis(read_json(analysis_file,"Preparation settings"))
    analysis_file_hash = file_hash(analysis_file)
    paths,directories = resolve_paths(configuration,config_file)
    for directory in directories.values():
        if directory==analysis_file or directory in analysis_file.parents:
            raise ValidationError("Generated directories cannot contain preparation configuration.")
    fingerprint,identity = preparation_identity(root,configuration,paths,settings,through)
    work = directories["work"]/"preparation_runs"
    out = directories["outputs"]/"preparation_runs"
    work.mkdir(parents=True,exist_ok=True,mode=0o700)
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    final_work = work/fingerprint
    final_output = out/fingerprint
    pointer = directories["work"]/"preparation_current.json"
    def assert_unchanged():
        if (file_hash(config_file)!=configuration_file_hash or file_hash(analysis_file)!=analysis_file_hash
                or preparation_identity(root,configuration,paths,settings,through)[0]!=fingerprint):
            raise ValidationError("Source, inputs or settings changed while preparation was running.")
    with preparation_locks(work,out):
        journal = work/('.transaction-'+fingerprint+'.json')
        if journal.exists() or journal.is_symlink():
            if not resume: raise ValidationError('An interrupted preparation transaction exists; use --resume.')
            _recover_preparation(journal,work,out,fingerprint,identity,through,assert_unchanged)
        if resume and final_work.is_dir() and final_output.is_dir():
            verified_summary = _verify_preparation_pair(final_work,final_output,fingerprint,identity,through)
            assert_unchanged()
            write_pointer(pointer,{'fingerprint':fingerprint,'through':through,'status':'prepared'})
            return {'fingerprint':fingerprint,'through':through,'resumed':True,'summary':verified_summary}
        stages = []
        published = []
        try:
            stage_work = Path(tempfile.mkdtemp(prefix=".staging-",dir=work));stages.append(stage_work)
            stage_output = Path(tempfile.mkdtemp(prefix=".staging-",dir=out));stages.append(stage_output)
            bundle,summary,aggregates = compute_preparation(configuration,config_file,paths,settings,through)
            ArtifactWriter(stage_work).finish(bundle)
            aggregate_files = {}
            for name,frame in aggregates.items():
                filename = name+".csv"
                frame.to_csv(stage_output/filename,index=False,float_format="%.17g")
                aggregate_files[filename] = file_hash(stage_output/filename)
            import numpy as np
            def clean(value):
                if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return [clean(v) for v in value]
                if isinstance(value,np.ndarray):return clean(value.tolist())
                if isinstance(value,np.generic):return clean(value.item())
                if isinstance(value,float) and not np.isfinite(value):return None
                return value
            summary = clean(summary)
            write_json(stage_output/"summary.json",summary)
            aggregate_files["summary.json"] = file_hash(stage_output/"summary.json")
            assert_unchanged()
            receipt = {"schema_version":1,"fingerprint":fingerprint,"identity":identity,
                       "status":"prepared","through":through,"summary":summary,
                       "artifact_manifest_sha256":file_hash(stage_work/"artifacts.json"),"aggregate_files":aggregate_files,
                       "completed_at":datetime.now(timezone.utc).isoformat()}
            write_json(stage_work/"receipt.json",receipt)
            _verify_preparation_pair(stage_work,stage_output,fingerprint,identity,through)
            if final_work.exists() or final_output.exists() or final_work.is_symlink() or final_output.is_symlink():
                raise ValidationError("An identical run exists; use --resume after verifying its artifacts, or choose fresh output directories.")
            sync_tree(stage_work);sync_tree(stage_output)
            write_pointer(journal,{'schema_version':1,'fingerprint':fingerprint,
                'stage_work':stage_work.name,'stage_output':stage_output.name,
                'receipt_sha256':file_hash(stage_work/'receipt.json')})
            sync_directory(work)
            for stage,final in [(stage_output,final_output),(stage_work,final_work)]:
                rename_noreplace(stage,final);published.append((final,stage))
                sync_directory(final.parent)
            assert_unchanged()
            write_pointer(pointer,{"fingerprint":fingerprint,"through":through,"status":"prepared"})
            journal.unlink();sync_directory(work)
            return {"fingerprint":fingerprint,"through":through,"resumed":False,"summary":summary}
        except BaseException:
            for final,stage in reversed(published):
                rename_noreplace(final,stage)
            journal.unlink(missing_ok=True)
            raise
        finally:
            for stage in stages:
                if stage.exists() and not journal.exists():shutil.rmtree(stage)
