"""Validate the manuscript workflow and execute implemented analysis modules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


MODULE_IDS = ["M{:02d}".format(index) for index in range(1, 16)]


class ValidationError(ValueError):
    """A plan or local configuration failed a documented check."""


def read_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, ValueError):
        raise ValidationError("{} must be a readable JSON file.".format(label)) from None


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValidationError("{} must be an object.".format(label))
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("{} must be a nonempty string.".format(label))
    return value


def _list(value, label, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        requirement = "a nonempty list" if nonempty else "a list"
        raise ValidationError("{} must be {}.".format(label, requirement))
    return value


def _strings(value, label, nonempty=False):
    values = _list(value, label, nonempty)
    for item in values:
        _string(item, label + " entries")
    if len(set(values)) != len(values):
        raise ValidationError("{} contains duplicates.".format(label))
    return values


def _known(values, allowed, label):
    if set(values) - set(allowed):
        raise ValidationError("{} contains an unknown reference.".format(label))


def _version(document, label):
    _mapping(document, label)
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValidationError("{} must use schema_version 1.".format(label))


def validate_plan(document):
    """Validate workflow coverage, ownership, and dependency ordering."""
    _version(document, "Workflow")
    models = _strings(document.get("model_families"), "Model families", True)
    preparations = _strings(document.get("preparation_families"), "Preparation families", True)
    if set(models) & set(preparations):
        raise ValidationError("Model and preparation family IDs must be distinct.")
    modules = _list(document.get("modules"), "Modules", True)
    targets = _list(document.get("targets"), "Targets", True)
    gaps = _list(document.get("gaps"), "Gaps", True)
    module_map = {}
    model_owners = {family: [] for family in models}
    preparation_owners = {family: [] for family in preparations}
    all_steps = set()
    for module in modules:
        _mapping(module, "Module")
        module_id = _string(module.get("id"), "Module ID")
        if module_id in module_map:
            raise ValidationError("Module IDs must be unique.")
        module_map[module_id] = module
        for field in ("title", "status"):
            _string(module.get(field), "Module " + field)
        stage = module.get("stage")
        if not ((type(stage) is int and stage > 0) or (isinstance(stage, str) and stage.strip())):
            raise ValidationError("Module stage must be a positive integer or nonempty string.")
        _strings(module.get("depends_on"), "Module dependencies")
        owned_models = _strings(module.get("model_families"), "Module model families")
        owned_preparations = _strings(module.get("preparation_families"), "Module preparation families")
        _known(owned_models, models, "Module model families")
        _known(owned_preparations, preparations, "Module preparation families")
        for family in owned_models:
            model_owners[family].append(module_id)
        for family in owned_preparations:
            preparation_owners[family].append(module_id)
        _strings(module.get("output_ids"), "Module output IDs")
        for code_target in _strings(module.get("code_targets"), "Module code targets", True):
            path = Path(code_target)
            if path.is_absolute() or ".." in path.parts or "\\" in code_target:
                raise ValidationError("Code targets must be relative paths inside the repository.")
        _strings(module.get("acceptance"), "Module acceptance checks", True)
        for step in _list(module.get("steps"), "Module steps", True):
            _mapping(step, "Step")
            for field in ("id", "title", "work", "acceptance"):
                _string(step.get(field), "Step " + field)
            if step["id"] in all_steps:
                raise ValidationError("Step IDs must be unique.")
            all_steps.add(step["id"])
    if list(module_map) != MODULE_IDS:
        raise ValidationError("Modules must contain M01 through M15 once, in order.")
    visited = set()
    active = set()

    def visit(module_id):
        if module_id in active:
            raise ValidationError("Module dependencies contain a cycle.")
        if module_id in visited:
            return
        active.add(module_id)
        dependencies = module_map[module_id]["depends_on"]
        _known(dependencies, module_map, "Module dependencies")
        for dependency in dependencies:
            visit(dependency)
        active.remove(module_id)
        visited.add(module_id)

    for module_id in MODULE_IDS:
        visit(module_id)
        if any(MODULE_IDS.index(dependency) >= MODULE_IDS.index(module_id)
               for dependency in module_map[module_id]["depends_on"]):
            raise ValidationError("Module dependencies must precede the consuming module.")
    for owners in (model_owners, preparation_owners):
        if any(len(module_ids) != 1 for module_ids in owners.values()):
            raise ValidationError("Every declared family must have exactly one owning module.")
    target_map = {}
    covered_models = set()
    covered_preparations = set()
    for target in targets:
        _mapping(target, "Target")
        target_id = _string(target.get("id"), "Target ID")
        if target_id in target_map:
            raise ValidationError("Target IDs must be unique.")
        target_map[target_id] = target
        _string(target.get("label"), "Target label")
        kind = _string(target.get("kind"), "Target kind")
        if kind not in {"table", "figure", "prose"}:
            raise ValidationError("Target kind must be table, figure, or prose.")
        target_models = _strings(target.get("model_ids"), "Target model IDs")
        target_preparations = _strings(target.get("preparation_ids", []), "Target preparation IDs")
        _known(target_models, models, "Target model IDs")
        _known(target_preparations, preparations, "Target preparation IDs")
        if not target_models and not target_preparations:
            raise ValidationError("Every target must identify a model or preparation family.")
        covered_models.update(target_models)
        covered_preparations.update(target_preparations)
        producers = _strings(target.get("producer_modules"), "Target producer modules", True)
        _known(producers, MODULE_IDS[:12], "Target computational producers")
        render_module = target.get("render_module")
        expected_renderer = "M14" if kind == "figure" else "M13"
        if render_module != expected_renderer:
            raise ValidationError("Figures must render in M14; tables and prose must render in M13.")
        for family in target_models:
            if model_owners[family][0] not in producers:
                raise ValidationError("A target omits its model family's owning producer.")
        for family in target_preparations:
            if preparation_owners[family][0] not in producers:
                raise ValidationError("A target omits its preparation family's owning producer.")
        for producer in producers + [render_module]:
            if target_id not in module_map[producer]["output_ids"]:
                raise ValidationError("A target is absent from a producer or renderer's output IDs.")
    if covered_models != set(models) or covered_preparations != set(preparations):
        raise ValidationError("Every declared family, including prose-only models, must map to a target.")
    for module_id, module in module_map.items():
        _known(module["output_ids"], target_map, "Module output IDs")
        for target_id in module["output_ids"]:
            target = target_map[target_id]
            if module_id not in target["producer_modules"] + [target["render_module"]]:
                raise ValidationError("A module claims an output without a producer or renderer assignment.")
    gap_ids = set()
    for gap in gaps:
        _mapping(gap, "Gap")
        for field in ("id", "title", "resolution", "status"):
            _string(gap.get(field), "Gap " + field)
        if gap["id"] in gap_ids:
            raise ValidationError("Gap IDs must be unique.")
        gap_ids.add(gap["id"])
        _known(_strings(gap.get("owners"), "Gap owners", True), MODULE_IDS, "Gap owners")
    return {"modules": len(modules), "models": len(models), "preparations": len(preparations),
            "targets": len(targets), "gaps": len(gaps), "steps": len(all_steps)}


def _resolve(value, base, label):
    value = _string(value, label)
    try:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    except (OSError, RuntimeError, ValueError):
        raise ValidationError("{} is not a valid path.".format(label)) from None


def _readable_file(path, label):
    try:
        if not path.is_file():
            raise OSError()
        with path.open("rb"):
            pass
    except (OSError, ValueError):
        raise ValidationError("{} must be a readable file.".format(label)) from None


def _contains(directory, path):
    return directory == path or directory in path.parents


def validate_config(document, contract, config_path, source_directory):
    """Check paths and optional hashes without loading study records or writing files."""
    _version(document, "Local configuration")
    _version(contract, "Input contract")
    if set(document) != {"schema_version", "inputs", "directories", "fonts"}:
        raise ValidationError("Local configuration must contain only schema_version, inputs, directories, and fonts.")
    inputs = _mapping(document.get("inputs"), "Configuration inputs")
    directories = _mapping(document.get("directories"), "Configuration directories")
    fonts = _mapping(document.get("fonts"), "Configuration fonts")
    definitions = _list(contract.get("inputs"), "Input contract definitions", True)
    aliases = {}
    for definition in definitions:
        _mapping(definition, "Input definition")
        alias = _string(definition.get("id"), "Input alias")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", alias):
            raise ValidationError("Input aliases must use letters, digits, underscores, or hyphens.")
        if alias in aliases:
            raise ValidationError("Input aliases must be unique.")
        if type(definition.get("required")) is not bool:
            raise ValidationError("Input required flags must be boolean.")
        aliases[alias] = definition
    _known(inputs, aliases, "Configuration inputs")
    if any(definition["required"] and alias not in inputs for alias, definition in aliases.items()):
        raise ValidationError("One or more required input aliases are missing.")
    try:
        local_path = Path(config_path).resolve()
        source_path = Path(source_directory).resolve()
    except (OSError, RuntimeError, ValueError):
        raise ValidationError("Configuration or source location is invalid.") from None
    base = local_path.parent
    protected_files = [local_path]
    hashes_checked = 0
    for alias, specification in inputs.items():
        specification = _mapping(specification, "Input specification")
        if set(specification) - {"path", "sha256"}:
            raise ValidationError("Input specifications accept only path and optional sha256.")
        input_path = _resolve(specification.get("path"), base, "Input path")
        _readable_file(input_path, "Input " + alias)
        protected_files.append(input_path)
        if "sha256" in specification:
            expected_hash = specification["sha256"]
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                raise ValidationError("Input SHA256 must contain 64 hexadecimal characters.")
            digest = hashlib.sha256()
            try:
                with input_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError:
                raise ValidationError("Input hash could not be read.") from None
            if digest.hexdigest() != expected_hash.lower():
                raise ValidationError("Input " + alias + " does not match its SHA256.")
            hashes_checked += 1
    if set(fonts) != {"arial", "arial_bold"}:
        raise ValidationError("Fonts must declare arial and arial_bold.")
    for name in ("arial", "arial_bold"):
        path = _resolve(fonts[name], base, "Font path")
        _readable_file(path, "Configured font")
        protected_files.append(path)
    if set(directories) != {"work", "outputs"}:
        raise ValidationError("Directories must declare work and outputs.")
    resolved_directories = []
    for name in ("work", "outputs"):
        path = _resolve(directories[name], base, "Output directory")
        try:
            if path.exists() and not path.is_dir():
                raise ValidationError("Configured work and output locations must be directories.")
            if any(parent.exists() and not parent.is_dir() for parent in path.parents):
                raise ValidationError("Configured directories cannot be nested inside a file.")
        except OSError:
            raise ValidationError("Configured directory could not be inspected.") from None
        if _contains(path, source_path) or _contains(source_path, path):
            raise ValidationError("Work and output directories must not overlap the source directory.")
        if any(_contains(path, protected) or _contains(protected, path) for protected in protected_files):
            raise ValidationError("Work and output directories must not contain protected input, font, or configuration files.")
        resolved_directories.append(path)
    if (_contains(resolved_directories[0], resolved_directories[1])
            or _contains(resolved_directories[1], resolved_directories[0])):
        raise ValidationError("Work and output directories must be separate and nonoverlapping.")
    return {"inputs": len(inputs), "hashes_checked": hashes_checked, "fonts": len(fonts)}


def main(argv=None, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--plan", action="store_true", help="show planned modules (default)")
    commands.add_argument("--check-plan", action="store_true", help="validate workflow coverage and dependencies")
    commands.add_argument("--check-config", metavar="FILE", help="check local input paths and optional hashes")
    commands.add_argument("--full", action="store_true", help="reconstruct all analyses from empty work/output roots, or resume that same full run")
    commands.add_argument("--render", action="store_true", help="render verified saved aggregates without loading study data or fitting models")
    commands.add_argument("--through", choices=["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12", "M13", "M14"], help="run cumulatively through this module")
    parser.add_argument("--config", metavar="FILE", help="local input and output configuration for preparation")
    parser.add_argument("--analysis", metavar="FILE", help="local preparation endpoint settings")
    parser.add_argument("--resume", action="store_true", help="reuse completed preparation and model runs only after artifact checks")
    parser.add_argument("--sampling", metavar="FILE", help="M04 sampling controls; defaults to config/primary_sampling.json")
    parser.add_argument("--mechanics-sampling", metavar="FILE", help="M07 mechanics sampling controls; defaults to config/mechanics_sampling.json")
    parser.add_argument("--supporting-sampling", metavar="FILE", help="M08 MMEF/X6 sampling controls; defaults to config/supporting_sampling.json")
    parser.add_argument("--wheeze-sampling", metavar="FILE", help="M10 wheeze sampling controls; defaults to config/wheeze_sampling.json")
    parser.add_argument("--sensitivity-sampling", metavar="FILE", help="M12 sensitivity sampling controls; defaults to config/sensitivity_sampling.json")
    parser.add_argument("--reporting", metavar="FILE", help="M13 private document-runtime and optional manuscript-reference configuration")
    parser.add_argument("--figures", metavar="FILE", help="M14 private PDF-font inspector and selected pre-TB uncertainty configuration")
    parser.add_argument("--render-config", metavar="FILE", help="saved aggregate receipts, rendering runtimes and separate render-only destinations")
    parser.add_argument("--targets", nargs="+", help="M04 targets; defaults to all twelve primary models")
    parser.add_argument("--jobs", type=int, default=1, help="simultaneous joint models, one to three")
    arguments = parser.parse_args(argv)
    try:
        if arguments.render:
            if (not arguments.render_config or arguments.config or arguments.analysis or arguments.sampling
                or arguments.mechanics_sampling or arguments.supporting_sampling or arguments.wheeze_sampling
                or arguments.sensitivity_sampling or arguments.reporting or arguments.figures
                or arguments.targets or arguments.jobs!=1):
                raise ValidationError("--render requires --render-config; only --resume is an additional rendering option.")
            from .render_execution import run_render_only
            result=run_render_only(root,arguments.render_config,resume=arguments.resume)
            print("Saved aggregates rendered successfully. No study inputs were loaded and no models were fitted.")
            return 0 if result['accepted'] else 2
        if arguments.render_config:
            raise ValidationError("--render-config requires --render.")
        if arguments.full:
            if not all((arguments.config,arguments.analysis,arguments.reporting,arguments.figures)) or arguments.targets:
                raise ValidationError("--full requires --config, --analysis, --reporting and --figures, and all selected models.")
            from .full_execution import run_full
            result=run_full(root,arguments.config,arguments.analysis,
                arguments.sampling or root/'config/primary_sampling.json',
                arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                arguments.supporting_sampling or root/'config/supporting_sampling.json',
                arguments.wheeze_sampling or root/'config/wheeze_sampling.json',
                arguments.sensitivity_sampling or root/'config/sensitivity_sampling.json',
                arguments.reporting,arguments.figures,jobs=arguments.jobs,resume=arguments.resume)
            print("Full reconstruction completed: {} joint fits verified, {} publication figures. GitHub delivery is checked separately.".format(
                result['joint_fits_verified'],result['coverage']['figures']))
            return 0 if result['accepted'] else 2
        if arguments.reporting and arguments.through not in {"M13", "M14"}:
            raise ValidationError("--reporting applies to M13 and M14 only.")
        if arguments.figures and arguments.through!="M14":
            raise ValidationError("--figures applies to M14 only.")
        if arguments.through:
            if not arguments.config or not arguments.analysis:
                raise ValidationError("Preparation requires --config and --analysis.")
            if arguments.mechanics_sampling and arguments.through not in {"M07", "M08", "M09", "M10", "M11", "M12", "M13", "M14"}:
                raise ValidationError("--mechanics-sampling applies to M07 through M14 only.")
            if arguments.supporting_sampling and arguments.through not in {"M08", "M09", "M10", "M11", "M12", "M13", "M14"}:
                raise ValidationError("--supporting-sampling applies to M08 through M14 only.")
            if arguments.wheeze_sampling and arguments.through not in {"M10", "M11", "M12", "M13", "M14"}:
                raise ValidationError("--wheeze-sampling applies to M10 through M14 only.")
            if arguments.sensitivity_sampling and arguments.through not in {"M12", "M13", "M14"}:
                raise ValidationError("--sensitivity-sampling applies to M12 through M14 only.")
            if arguments.through=="M14":
                if arguments.targets or not arguments.reporting or not arguments.figures:
                    raise ValidationError("Cumulative M14 requires --reporting, --figures and all selected models; --targets applies to M04 only.")
                from .figures_execution import run_figures
                result=run_figures(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    arguments.wheeze_sampling or root/'config/wheeze_sampling.json',
                    arguments.sensitivity_sampling or root/'config/sensitivity_sampling.json',
                    arguments.reporting,arguments.figures,jobs=arguments.jobs,resume=arguments.resume)
                print("Publication figures completed: {} figures and {} panels. Status: {}.".format(
                    result['coverage']['figures'],result['coverage']['panels'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M13":
                if arguments.targets or not arguments.reporting:
                    raise ValidationError("Cumulative M13 requires --reporting and all selected models; --targets applies to M04 only.")
                from .reporting_execution import run_reporting
                result=run_reporting(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    arguments.wheeze_sampling or root/'config/wheeze_sampling.json',
                    arguments.sensitivity_sampling or root/'config/sensitivity_sampling.json',arguments.reporting,
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Manuscript reporting completed: {} physical tables, {} classified cells and {} numerical sentences. Status: {}.".format(
                    result['coverage']['physical_tables'],result['coverage']['numeric_cells'],result['coverage']['sentences'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M12":
                if arguments.targets:
                    raise ValidationError("Cumulative M12 requires all selected models; --targets applies to M04 only.")
                from .sensitivity_execution import run_sensitivities
                result=run_sensitivities(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    arguments.wheeze_sampling or root/'config/wheeze_sampling.json',
                    arguments.sensitivity_sampling or root/'config/sensitivity_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Sensitivity fitting completed: {} new models and {} reused fits. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['selected_models'],result['reused_models'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M11":
                if arguments.targets:
                    raise ValidationError("Cumulative M11 requires all selected models; --targets applies to M04 only.")
                from .feno_execution import run_feno
                result=run_feno(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    arguments.wheeze_sampling or root/'config/wheeze_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("FeNO fitting completed: {} models. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['selected_models'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M10":
                if arguments.targets:
                    raise ValidationError("Cumulative M10 requires all selected models; --targets applies to M04 only.")
                from .wheeze_execution import run_wheeze
                result=run_wheeze(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    arguments.wheeze_sampling or root/'config/wheeze_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Age-specific wheeze fitting completed: {} models. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['selected_models'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M09":
                if arguments.targets:
                    raise ValidationError("Cumulative M09 requires all selected models; --targets applies to M04 only.")
                from .pretb_execution import run_pretb
                result=run_pretb(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Pre-TB survival fitting completed: {} models. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['selected_models'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M08":
                if arguments.targets:
                    raise ValidationError("Cumulative M08 requires all selected models; --targets applies to M04 only.")
                from .supporting_execution import run_supporting
                result=run_supporting(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    arguments.supporting_sampling or root/'config/supporting_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Supporting fitting completed: {} joint models and {} deterministic fits. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['selected_models'],result['deterministic_fits'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M07":
                if arguments.targets:
                    raise ValidationError("Cumulative M07 requires all selected models; --targets applies to M04 only.")
                from .mechanics_execution import run_mechanics
                result=run_mechanics(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    arguments.mechanics_sampling or root/'config/mechanics_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Mechanics fitting completed: {} models and {} shared-factor rows. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['selected_models'],result['shared_factor_rows'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M06":
                if arguments.targets:
                    raise ValidationError("Cumulative M06 requires all primary models; --targets applies to M04 only.")
                from .trajectory_execution import run_trajectories
                result=run_trajectories(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Trajectory fitting completed: {} analyses, {} fitted models and {} prediction rows. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['analyses'],result['fitted_models'],result['prediction_rows'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M05":
                if arguments.targets:
                    raise ValidationError("Cumulative M05 requires all twelve primary models; --targets applies to M04 only.")
                from .binary_execution import run_binary
                result=run_binary(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    jobs=arguments.jobs,resume=arguments.resume)
                print("Binary fitting completed: {} models; {} estimable and {} explicitly non-estimable contrasts. Status: {}. Independent scientific comparison is a separate check.".format(
                    result['models'],result['estimable_rows'],result['non_estimable_rows'],result['status']))
                return 0 if result['accepted'] else 2
            if arguments.through=="M04":
                from .primary_execution import run_primary
                result=run_primary(root,arguments.config,arguments.analysis,
                    arguments.sampling or root/'config/primary_sampling.json',
                    targets=arguments.targets,jobs=arguments.jobs,resume=arguments.resume)
                passed=sum(model['diagnostics_passed'] for model in result['models'])
                print("Primary fitting completed: {}/{} models passed diagnostics. Independent scientific comparison is a separate check.".format(passed,result['selected_models']))
                return 0 if result['diagnostics_passed'] else 2
            if arguments.sampling or arguments.targets or arguments.jobs!=1:
                raise ValidationError("Sampling options require --through M04 through M14; --targets applies to M04 only.")
            from .execution import run_preparation
            result = run_preparation(root,arguments.config,arguments.analysis,arguments.through,resume=arguments.resume)
            state = "Verified cached" if result["resumed"] else "Completed"
            print("{} preparation through {}; {}. No outcome models fitted.".format(state,result["through"],result["fingerprint"][:16]))
            missing = result["summary"].get("flow_unavailable_rows",0)
            if missing:
                print("{} upstream antenatal flow counts remain unavailable; this is recorded in the aggregate flow output.".format(missing))
            return 0
        if arguments.config or arguments.analysis or arguments.resume or arguments.sampling or arguments.mechanics_sampling or arguments.supporting_sampling or arguments.wheeze_sampling or arguments.sensitivity_sampling or arguments.targets or arguments.jobs!=1:
            raise ValidationError("--config, --analysis and --resume require --through.")
        if arguments.check_config is not None:
            configuration = read_json(arguments.check_config, "Local configuration")
            contract = read_json(root / "config" / "input_contract.json", "Input contract")
            summary = validate_config(configuration, contract, arguments.check_config, root)
            print("Configuration checks passed: {inputs} input bindings, {hashes_checked} hashes, {fonts} font files. "
                  "No records loaded or files written.".format(**summary))
        else:
            workflow = read_json(root / "config" / "workflow.json", "Workflow")
            summary = validate_plan(workflow)
            if arguments.check_plan:
                print("Plan checks passed: {modules} modules, {models} model families, {preparations} preparation families, "
                      "{targets} targets, {steps} steps, {gaps} gaps.".format(**summary))
            else:
                for module in workflow["modules"]:
                    print("{id} [{status}] {title}".format(**module))
                print("M01-M14 execution, clean full reconstruction and aggregate-only rendering are available; this command displays status without executing analysis.")
        return 0
    except ValidationError as error:
        print("Validation failed: {}".format(error), file=sys.stderr)
        return 2
    except (ValueError,TypeError,KeyError,OSError,ImportError,RuntimeError) as error:
        # Third-party exception text can contain source values or local paths.
        print("Execution failed ({}); the requested cumulative run did not complete.".format(type(error).__name__),file=sys.stderr)
        return 2
