# Reconstruction, rendering and source release

M15 provides separate commands for a complete analytical reconstruction, rendering accepted aggregate results, and exporting reviewed source. Each operation has its own identity and receipt. A successful render or archive check does not establish analytical reproducibility or GitHub delivery. Final validation evidence, source manifests, authorized local configuration and delivery receipts remain private.

M01–M14 have independent numerical and presentation checks, a completed 868-test baseline, and successful final-source original and copied cumulative replays. The original M15 release passed 946 synthetic tests. Corrective release acceptance additionally requires the complete current synthetic suite, the clean reconstruction and comparisons below, aggregate-only rendering checks, extracted-package verification, and remote source/hosted-CI verification. The private acceptance receipts establish which release and executions passed these requirements.

## Environment and external dependencies

The revised runtime, early thread-control bootstrap and crash recovery protocol are documented in [RUNTIME_RECOVERY.md](RUNTIME_RECOVERY.md). Historical accepted runs retain their original identities.

The scientific reference environment uses Python 3.12.14 on macOS ARM64. `requirements-models.lock.txt` contains 41 exact dependency pins, including the plotting stack. Install it into a local environment and run nonempty synthetic test discovery:

```sh
python3.12 -m venv ../dchs-runtime
../dchs-runtime/bin/python -m pip install -r requirements-models.lock.txt
../dchs-runtime/bin/python -B tests/run_tests.py
../dchs-runtime/bin/python run.py --check-plan
```

Verify the interpreter version before reconstruction. Bayesian sampling requires a working C++ compiler. Install `pdffonts` separately and provide its executable path. Arial and Arial Bold are externally supplied, user-licensed font files; the code loads their actual bytes and verifies PDF embedding. Neither the fonts nor their licenses are included in the source export. The user must have the rights needed for their use and embedding.

The DOCX renderer uses the Python standard library and does not require `python-docx`. Its configured `document_python` can be the same pinned interpreter as the scientific environment. The separate configuration records and checks that interpreter's identity. LibreOffice and PDF page-rendering tools are optional independent visual-QA dependencies, rather than scientific or DOCX-authoring dependencies. `requirements-preparation.lock.txt` is sufficient for preparation-only work, not the complete workflow.

The hosted macOS/Linux workflow runs source and fabricated-data checks without study inputs or proprietary fonts. A hosted synthetic pass does not certify numerical equivalence of a confidential-data reconstruction on another platform. Record actual hosted outcomes separately; the existence of a workflow file is not a passing run.

## Clean full reconstruction

Supply the authorized input roles and reference lookup described in [INPUTS.md](INPUTS.md). Configure nine files: private paths, analysis endpoints, reporting, figures, and the five selected sampling-control files. The repository sampling files are the defaults; explicit flags make the chosen controls visible. Place the private files in a separate directory such as `../dchs-config/`. The environment, local files and all work/output directories stay outside the exported source tree. Any method examples using in-tree local configurations are development examples; a release tree must contain only its reviewed inventory.

The paths configuration must name work and output directories that are absent or truly empty, including hidden files. They must be separate from each other, the source tree, inputs, configurations, interpreters, font files and font-inspection executable. Do not copy preparation artifacts, posterior draws, aggregate caches or previous completion files into these directories. Begin from the reviewed extracted source and run:

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

`--full` requires paths, analysis, reporting and figures configurations and always runs all selected targets. The five sampling flags are optional because their repository defaults are explicit. `--jobs` accepts one to three concurrent joint models. `--targets` is not permitted for a full reconstruction. See the model methods for within-model chain controls and diagnostic gates.

The runner rebuilds preparation, deterministic models, aggregate reporting and all publication figures. It creates all 37 joint fits: twelve primary, six mechanics, seven supporting, two wheeze and ten sensitivity fits. M12 reuses four fits produced within this reconstruction; it does not import those fits from the earlier accepted run. M14 recomputes the separate Figure S3 uncertainty correction while retaining the M09 source calculation.

Under locks, the runner creates identical `full_run_origin.json` records in both roots before calculation. These bind source, authorized inputs, all nine configurations, runtimes, external fonts/references and exact destination paths. Completion verifies the 37 raw posterior receipts, accepted M13 and M14 publications, unchanged source/dependencies and complete payload inventories. It writes `output_directory/full_reconstruction/<fingerprint>/receipt.json` and the private `work_directory/full_current.json` pointer. GitHub delivery remains a separate check.

To resume the same full reconstruction, repeat the identical command with `--resume`. Both origin records must match and retain the original identity; a matching pre-analysis origin journal can safely complete a missing second record. Changed source, inputs, configurations, runtimes or destinations require a new empty pair of roots. Existing analysis caches without a matching full-run origin cannot be adopted. By comparison, `--through M14 --resume` validates an existing cumulative analysis; it does not establish an empty-root origin.

## Independent reconstruction acceptance

For reproduction of the original M15 specification, the accepted 37-fit run is the comparison baseline. That reconstruction used the same recorded seeds and selected controls, so it was a same-seed reconstruction, not an independent random sample. Its review compared complete posterior and non-timing sampler arrays; timing differences were not scientific differences.

All posterior scalars, including child effects and deterministic variables, must pass the existing diagnostic gates: R-hat at most 1.01, bulk and tail effective sample size at least 400, BFMI at least 0.3, and no divergences or maximum-depth hits. The original specification's 924 posterior mean/quantile comparisons used a four-combined-MCSE rule and quantile-specific MCSEs, with exact agreement at zero combined MCSE. Those historical comparisons do not establish acceptance of changed models.

The [corrective revision](CORRECTIVE_REVISION.md) instead requires a fresh 37-fit reconstruction, all 35 continuous covariance screens, the eight Table S1 precision bounds, and independent comparisons of revised estimates, intervals and null inclusion. Target intercepts, threshold follow-up and the declared S1 draw counts intentionally differ from the original specification. A separately documented primary diagnostic-remediation amendment fixes 4,800 retained draws after initial R-hat failures and requires exact original 1,200-draw prefixes. Explain every corresponding sample/design or result change; do not force numerical agreement with the superseded models. Unchanged deterministic analyses and the approved Figure S3 correction retain their exact or previously declared scale-specific comparison requirements. No threshold widening, seed selection or repeated unplanned sampling extension is permitted to force acceptance.

Full posterior checks and independent scientific comparisons remain distinct from a command's diagnostic acceptance. Record the final source manifest, test receipt, clean origin, full-run receipt, scientific comparisons, render checks and all five review bindings before delivery.

## Render accepted aggregates without refitting

Copy `config/render.example.json` to a private local configuration outside the source tree, such as `../dchs-config/render.local.json`. Supply complete accepted M13 and M14 publication directories, their receipt SHA-256 values from trusted acceptance records, separate render work/output roots, `document_python`, `pdf_fonts_tool`, and actual Arial/Arial Bold files. The two publications must have matching accepted lineage. The render configuration does not contain study inputs, model controls or analysis endpoints.

```sh
../dchs-runtime/bin/python run.py --render --render-config ../dchs-config/render.local.json
```

Only `--resume` may be added. Analysis flags, target selections and model parallelism are rejected. The command verifies the anchored receipts and complete file inventories, reconstructs the aggregate registry, and requires exact equality of structured tables, numerical prose and figure data to the accepted payloads. It preserves the saved corrected Figure S3 uncertainty; it does not recalculate risk scores, fit models, read study records or load raw posteriors.

Compatible renderer layout/style or runtime changes receive a new render identity. They must preserve the accepted scientific data, selectors, estimands and structured reporting content. A scientific or schema/specification change requires a newly accepted aggregate bundle. Render-only records historical analysis identities as provenance; it does not revalidate the current analytical code. Inspect the new pages and font embedding whenever rendering code, runtime or fonts change.

Outputs are written under `output_directory/render_only/<fingerprint>/`: `tables/` contains 19 M13 payloads, `figures/` contains 30 M14 payloads, and top-level metadata, summary and receipt record render provenance. `work_directory/render_current.json` is written only on success. Resume regenerates and compares the complete payloads. Preserve the accepted source publications separately.

## Export and verify the source package

`config/source_allowlist.json` explicitly enumerates release files. It includes source, synthetic tests, methods, example configurations, dependency locks, CI and these release instructions. It excludes study extracts, reference lookup data, private local configuration, derived participant records, raw posteriors, aggregate results, manuscripts, internal reviews and font files. The exporter checks exact inventory, safe paths, file types and content hashes; reviewers must separately inspect the listed content for sensitive material. `.gitignore` alone is insufficient.

Use the standard-library exporter from the reviewed tree. The output archive and extraction destination must not already exist; their parent directories must exist:

```sh
python3 tools/export_source.py export --root . --output ../dchs-lung-function-source.zip
python3 tools/export_source.py verify ../dchs-lung-function-source.zip \
  --manifest-sha256 REPLACE_WITH_REVIEWED_MANIFEST_SHA256
python3 tools/export_source.py extract ../dchs-lung-function-source.zip ../dchs-lung-function-extracted \
  --manifest-sha256 REPLACE_WITH_REVIEWED_MANIFEST_SHA256
python3 tools/export_source.py verify-tree ../dchs-lung-function-extracted \
  --manifest-sha256 REPLACE_WITH_REVIEWED_MANIFEST_SHA256
```

The archive contains the allowlisted files plus `SOURCE_MANIFEST.json`, which records file sizes and SHA-256 values. Export is deterministic for identical reviewed bytes. The externally recorded manifest hash establishes reviewed release identity; an internally consistent manifest by itself establishes only archive integrity. The exporter rejects unsafe paths, unexpected files, symlinks, hardlinks, special files and changed content. Extraction validates before committing a new destination. Source export neither chooses a reuse license nor uploads anything.

Verify the extracted tree against the trusted manifest, install the locked environment, run the full nonempty synthetic suite, and use that reviewed tree for the clean reconstruction. Keep all generated files and private configuration outside the release tree. Record the allowlist inventory and archive/manifest hashes in private acceptance evidence.

## Remote delivery and retained boundaries

Record the chosen repository and private visibility before upload. Upload only reviewed source files, verify the remote tree/commit against the accepted manifest, and check the hosted macOS/Linux synthetic workflow. Record remote identity and actual CI outcomes separately from local source export and confidential-data reconstruction. No reuse license is selected automatically.

Four upstream antenatal recruitment/disposition counts are unavailable from the supplied sources. The combined antenatal/pregnancy-loss fact is also missing because its components are unavailable; it is never treated as zero. Exact aggregate values remain reportable when N is 1–9. Participant-level material stays private.

M09's original covariance outputs remain preserved. Figure S3 uses the separately accepted actual-risk-membership correction, retaining coefficients, hazard ratios, samples and model covariance while recomputing robust covariance, standard errors, confidence intervals, p-values and selected-family BH q-values. The Efron-fit/Breslow-score and inclusive-boundary conventions remain explicit. No selected manuscript is edited by release commands.
