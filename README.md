# DCHS lung function

Reproducible Python code for the DCHS lung-function manuscript and appendix. The workflow imports authorized study extracts, rebuilds the selected samples, fits all 37 joint models and the deterministic analyses, and generates eight numbered tables, numerical text and nine publication figures. Study data, local configuration, raw posteriors and generated study results remain outside this source package.

The current [corrective specification](docs/CORRECTIVE_REVISION.md) adds independent target child intercepts, bounds threshold-model TB follow-up by observed exit, and requires conditional covariance checks and explicit Table S1 Monte Carlo precision. Prior source-reproduction receipts describe the previous models. Acceptance of this revision requires its own complete reconstruction and independent review; old numerical agreement alone is not a model-adequacy test.

## Prerequisites

The reference scientific environment uses Python 3.12.14 on macOS ARM64, with 41 exact dependency pins in `requirements-models.lock.txt`. Bayesian sampling requires a working C++ compiler. Supply the authorized study extracts and reference lookup listed in [INPUTS.md](docs/INPUTS.md), licensed actual Arial/Arial Bold font files, and a separately installed `pdffonts` executable. The standard-library DOCX renderer needs no `python-docx`; its configured interpreter can be the scientific interpreter.

Keep the environment and private configuration outside the reviewed source tree:

```sh
python3.12 -m venv ../dchs-runtime
../dchs-runtime/bin/python -m pip install -r requirements-models.lock.txt
```

Copy `config/paths.example.json` and `config/analysis.example.json` to a private directory such as `../dchs-config/`, then fill in the authorized paths and selected analysis endpoints. Name separate work and output directories that are absent or truly empty, including hidden files; neither may overlap the source, inputs or dependencies.

Create private reporting and figures configurations as described in [reporting methods](docs/MANUSCRIPT_REPORTING.md) and [figure methods](docs/PUBLICATION_FIGURES.md). Reporting specifies `schema_version: 1` and `document_python`. Figures specifies `schema_version: 1`, `pdf_fonts_tool` and `pretb_uncertainty: "corrected_actual_risk_membership"`. Actual font paths belong in the private paths configuration. Optional private manuscript-reference files are not needed to compute the results.

## Full reconstruction

Run from the reviewed source directory with the nine selected configuration files:

```sh
../dchs-runtime/bin/python run.py --full \
  --config ../dchs-config/paths.local.json \
  --analysis ../dchs-config/analysis.local.json \
  --sampling config/primary_sampling.json \
  --mechanics-sampling config/mechanics_sampling.json \
  --supporting-sampling config/supporting_sampling.json \
  --wheeze-sampling config/wheeze_sampling.json \
  --sensitivity-sampling config/sensitivity_sampling.json \
  --reporting ../dchs-config/reporting.local.json \
  --figures ../dchs-config/figures.local.json \
  --jobs 3
```

The five sampling flags select the repository defaults shown above and may be omitted. `--jobs` allows one to three concurrent joint models. Full reconstruction always runs every selected target, creates a paired clean-origin record and verifies complete raw-fit and publication receipts. It rebuilds preparation, twelve primary, six mechanics, seven supporting, two wheeze and ten sensitivity joint fits, the deterministic models, and all tables/figures. Sensitivity tables reuse the appropriate fits produced within the same run.

Repeat the identical command with `--resume` to continue or verify that full run. Both roots must retain matching `full_run_origin.json` records with unchanged source, input, configuration, runtime and destination identities. Old caches without that origin cannot be adopted. Changing a bound dependency requires a new pair of empty roots.

`--through M01` through `--through M14` support incremental execution with the applicable flags. `--through M14 --resume` validates existing cumulative results but does not establish a clean full-run origin. See [RELEASE.md](docs/RELEASE.md) for complete commands, independent same-seed comparisons and acceptance requirements.

## Render accepted results

Copy `config/render.example.json` to a private configuration. Supply complete accepted M13/M14 aggregate-publication directories, trusted receipt SHA-256 values, separate render destinations, the interpreter/font inspector and actual Arial files:

```sh
../dchs-runtime/bin/python run.py --render --render-config ../dchs-config/render.local.json
```

Only `--resume` may be added. This regenerates editable tables, numerical text, nine vector PDFs, nine 600-dpi PNGs and a combined figure PDF without reading study inputs or raw posteriors, fitting models or revalidating analysis. Compatible presentation/runtime changes get a new render identity; structured scientific payloads must remain identical to the accepted aggregates. Inspect newly rendered pages and embedded fonts when presentation dependencies change.

## Checks and outputs

```sh
../dchs-runtime/bin/python run.py --check-plan
../dchs-runtime/bin/python run.py --check-config ../dchs-config/paths.local.json
../dchs-runtime/bin/python -B tests/run_tests.py
```

Tests use fabricated data and reject empty discovery. The smaller preparation lock supports `tests/run_tests.py --preparation-only`. Hosted macOS/Linux synthetic checks are separate from confidential-data reconstruction; local acceptance and actual remote CI outcomes have distinct receipts.

Private work holds typed participant-level intermediates, complete posterior arrays, manifests and pointers. The separate output root holds aggregate results and publications. Full completion is recorded in `full_reconstruction/<fingerprint>/receipt.json`; render-only results use `render_only/<fingerprint>/`. Input/source/runtime hashes, locks, staged writes and complete inventories govern reuse. Independent scientific checks remain additional to command-level diagnostic acceptance.

## Scientific methods

| Analysis | Documentation |
|---|---|
| Inputs, linkage, cohorts and model-specific samples | [Inputs](docs/INPUTS.md), [preparation](docs/PREPARATION.md) |
| Continuous primary spirometry | [Primary models](docs/PRIMARY_MODELS.md) |
| Binary spirometry and patterns | [Binary models](docs/BINARY_MODELS.md) |
| School-age ratio trajectories | [Trajectory models](docs/TRAJECTORY_MODELS.md) |
| Mechanics and shared-factor associations | [Mechanics models](docs/MECHANICS_MODELS.md) |
| MMEF and X6 supporting analyses | [Supporting models](docs/SUPPORTING_MODELS.md) |
| Pre-TB survival and corrected publication uncertainty | [Pre-TB models](docs/PRETB_MODELS.md) |
| Wheeze and standardized probability ratios | [Wheeze models](docs/WHEEZE_MODELS.md) |
| Visit-specific FeNO | [FeNO models](docs/FENO_MODELS.md) |
| Designated sensitivity analyses | [Sensitivity models](docs/SENSITIVITY_MODELS.md) |
| Revised covariance, censoring and precision requirements | [Corrective revision](docs/CORRECTIVE_REVISION.md) |
| Tables, numerical prose and publication figures | [Reporting](docs/MANUSCRIPT_REPORTING.md), [figures](docs/PUBLICATION_FIGURES.md) |

Four upstream antenatal counts and the combined antenatal/pregnancy-loss fact remain explicitly unavailable. They are not inferred or treated as zero. Exact aggregate values remain reportable when N is 1–9; participant-level material stays private.

M09 preserves its original covariance outputs. Figure S3 uses the separately accepted actual-risk-membership correction while retaining coefficients, hazard ratios and samples. The retained Efron-fit/Breslow-score and inclusive-boundary conventions are documented in the pre-TB methods. Release and rendering commands do not edit the selected manuscripts.

## Source release

The explicit `config/source_allowlist.json` controls export. From a clean reviewed source tree:

```sh
python3 tools/export_source.py export --root . --output ../dchs-lung-function-source.zip
python3 tools/export_source.py verify ../dchs-lung-function-source.zip \
  --manifest-sha256 REPLACE_WITH_REVIEWED_MANIFEST_SHA256
```

The externally trusted manifest hash identifies the reviewed release; archive integrity alone does not. [RELEASE.md](docs/RELEASE.md) covers safe extraction, complete source/content checks, clean reconstruction and verification of the chosen private GitHub repository and hosted checks. Source export performs no upload and selects no reuse license. Study inputs/results, private bindings/reviews, reference lookup data and proprietary fonts are excluded.

The [implementation plan](docs/IMPLEMENTATION_PLAN.md) and `config/workflow.json` retain the full module, coverage and acceptance specification. Final validation and delivery receipts stay private.
