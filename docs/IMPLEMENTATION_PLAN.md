# DCHS lung function implementation plan

The deliverable is a reviewed, reproducible, code-only repository for the accepted lung-function manuscript and appendix. The existing Stage 1 inventory remains the reference baseline. This document replaces the forward implementation outline; it does not replace or rerun that inventory.

Stages are delivery phases. M01-M15 are implementation modules, comparable to the ordered modules in the AMPATH package. Each module below contains explicit coding pieces and a completion check. No separate stage reports are required.

## Current state

Stage 1 is complete and unchanged. M01-M03 preparation is implemented and independently checked in a clean pinned environment. Four antenatal flow counts remain unavailable from the supplied inputs. M04 is implemented and independently validated across twelve fresh fits and 288 prespecified posterior comparisons. M05 is implemented and independently validated across 28 selected GEE fits. M06 is implemented and independently validated across four full/parallel trajectory pairs. M07 is implemented and independently validated across six individual mechanics fits and six shared-factor summaries. M08 is implemented and independently validated across seven joint fits and ten deterministic component fits. M09 source reproduction is implemented and independently validated across 17 Cox fits; the inherited score-residual covariance limitation remains explicit under G08. M10 is implemented and independently validated across both selected joint fits, conditional and age-comparison summaries, and three standardized probability ratios. M11 is implemented and independently validated across all eleven selected FeNO regressions, with successful original-source and copied-source cumulative execution. M12 is implemented and independently validated across ten new fits and four reused fits; final original-source and copied-source cumulative replays passed. M13 table and numerical-text generation is implemented and independently validated; all 802 tests and actual cumulative M01-M13 execution pass. Final-source original/copied M13 cumulative replays passed. M14 figure publication and the approved Figure S3 uncertainty correction are independently validated, including all 868 tests and final-source original/copied cumulative replays. M15 provides clean full reconstruction, aggregate-only rendering and source-release controls; final reconstruction, package and remote-delivery acceptance are established by their private receipts. Both preparation and primary sampling environments have exact dependency locks.

| Stage | Scope | Status |
|---|---|---|
| S1 | Inventory and reference selection | complete |
| S2 | Repository foundation and manuscript workflow | implemented and independently validated through M14 |
| S3 | Independent verification and clean rebuild | M01–M14 validated; final clean reconstruction evidence recorded separately |
| S4 | Code-only GitHub delivery | M15 controls implemented; delivery requires package/remote receipts |

## Repository organization

```text
run.py                              # one entry point
config/workflow.json                # ordered modules, targets and acceptance checks
config/input_contract.json          # neutral external-input roles
config/paths.example.json           # placeholders only
config/analysis.example.json        # private endpoint settings template
requirements-preparation.lock.txt   # pinned M01-M03 environment
requirements-models.lock.txt        # pinned scientific and plotting environment
src/dchs_lung_function/pipeline.py   # plan, cumulative, full and render entry points
src/dchs_lung_function/execution.py  # fingerprints, typed storage and restart controls
src/dchs_lung_function/inputs.py     # source import and harmonization
src/dchs_lung_function/cohorts.py    # cohort memberships and flow ledgers
src/dchs_lung_function/descriptive.py # Table 1 and modality availability
src/dchs_lung_function/preparation.py # PFT linkage and model contracts
src/dchs_lung_function/reference_scores.py # GLI and oscillometry scores
src/dchs_lung_function/respiratory_preparation.py # wheeze and FeNO contracts
src/dchs_lung_function/models/       # spirometry, mechanics, MMEF, X6 and pre-TB Cox models implemented
src/dchs_lung_function/analysis/     # primary, mechanics and supporting summaries implemented
src/dchs_lung_function/reporting/    # tables, prose, registry, DOCX and publication figure renderers
config/manuscript_reporting.json    # source-bound cell and prose coverage, without study estimates
config/publication_figures.json     # source-bound panels and caption templates
src/dchs_lung_function/figures_execution.py # cumulative figure publication and verification
src/dchs_lung_function/full_execution.py # clean origin and complete reconstruction
src/dchs_lung_function/render_execution.py # accepted aggregate-only rendering
config/render.example.json         # trusted publications and render dependencies
config/source_allowlist.json       # exact reviewed release inventory
tools/export_source.py             # source export, verification and safe extraction
.github/workflows/checks.yml       # hosted source and fabricated-data checks
docs/RELEASE.md                    # reconstruction, rendering and delivery procedure
tests/                              # fabricated-data and failure tests
docs/                               # concise setup, methods and development guidance
```

The selected statistical functions are explicit and implemented. Original scripts outside this repository remain unchanged; their source bindings and selected reference results are retained privately for independent verification. Python is retained because the existing models use PyMC and the Python statistical stack; no R translation is assumed.

The repository uses one dataset identity and neutral input roles. All input bindings remain accurate and private. Role-specific inputs and model-specific samples preserve the selected analysis definitions.

## Stage 2 foundation work

- **Available now:** M01-M03 deterministic preparation, selected input bindings and lookup provenance, a clean-installed pinned preparation environment, synthetic boundary/failure tests and private independent comparisons.
- **Execution controls available now:** fingerprints of inputs/code/configuration/runtime, verified typed artifacts, work/output locks, atomic preparation publication and checked resume. M15 adds strict clean-origin full reconstruction and separately identified aggregate-only rendering.
- **M04 implemented and validated:** explicit primary joint models, pinned sampling environment, recorded seeds/initialization/chains/tuning/draws, complete-posterior diagnostics, Monte Carlo comparison gates and separately fingerprinted sampling/validation. All twelve fresh fits passed full-posterior diagnostics and the 288 prespecified reference comparisons; five original review roles checked the implementation and evidence.
- **M05 implemented and validated:** 28 modified-Poisson GEE fits, strict LLN/pattern classification, exact small/zero counts, full coefficient/covariance outputs, stable evaluation of the selected robust sandwich, independent score/QR comparisons and checked aggregate resume.
- **M06 implemented and validated:** four native/GLI trajectory analyses, eight full/reduced REML fits, training-basis standardized predictions, raw Wald tests, complete coefficient/covariance artifacts, independent numerical and manuscript checks, and verified aggregate resume.
- **M07 implemented and validated:** six outcome-specific joint mechanics models with complete private posteriors, eight contrasts and a hazard summary per model, exact native/reporting scales and child support, six derived shared-factor associations from accepted M04 fits, and independent full-posterior/Monte Carlo checks.
- **M08 implemented and validated:** seven explicit MMEF/X6 joint fits, two paired-change GEE fits, four repeated X6 fits and four full/reduced trajectory fits; all posterior comparisons, deterministic numerical checks and sample-specific contracts independently verified. The documented initialization and precision amendments retain the original model and acceptance gates.
- **M09 source reproduction implemented and validated:** 17 selected Cox models, 51 coefficients and 306 model/robust covariance cells; independent likelihood, score-membership, uncertainty and manuscript comparisons. The inherited source residual-membership issue remains a statistical limitation under G08, with a separate quantitative uncertainty audit.
- **M10 implemented and validated:** both selected wheeze scenarios, conditional OR/IRR effects, age comparisons, exact support and predictive cells, and primary standardized probability ratios; 81 frozen scientific comparisons pass.
- **M11 implemented and independently validated:** five main and six supplementary FeNO Gaussian-independence GEE regressions, with distinct eligibility/reference definitions; all 2,368 numerical checks pass, including 63 coefficients and 726 full robust/naive covariance cells. Original-source and copied-source cumulative execution through M11 passed.
- **M12 implemented and independently validated:** ten designated new sensitivity fits, with all 267 frozen comparisons passing at the initial 3,200 draws per chain; verified reuse of two M04 ratio fits and both M10 wheeze scenarios; full-precision aggregate inputs for Tables S1/S3/S4 and attenuation from posterior means. The complete M12 suite has 729 passing tests. Final original-source and copied-source cumulative replays through M12 passed.
- **M13 implemented and independently validated:** eight numbered/ten physical tables, 304 classified digit-containing cells, 261 computed scientific cells and 42 sentence templates with 96 result tokens and 52 qualitative bindings. All 802 tests and actual cumulative M01-M13 execution pass. The published editable document matches the independently inspected ten-page artifact with verified Arial embedding. Final-source original/copied cumulative M13 replays passed with reporting payloads and prior science preserved.
- **M14 implemented and independently validated:** all nine figures in 17 panels, 56 forest estimates, 808 trajectory predictions, 44 schedule cells and ledger-derived flow labels. The authorized Figure S3 correction retains source coefficients, hazard ratios and samples while separately recomputing actual-risk-membership uncertainty. All 868 tests, final layout/font review and both final-source cumulative replays passed; private receipts bind the evidence.
- **M15 release controls:** paired empty-root origins, all 37 joint fits, complete analytical/publication receipts, trusted accepted-aggregate rendering, exact source allowlisting and safe verified extraction. [Release instructions](RELEASE.md) define the separate clean-run, synthetic, source-package and remote-delivery acceptance evidence.

File preflight checks readability and optional identity. Cumulative M01-M03 execution additionally verifies source schemas and constructs the selected scientific populations/matrices. Neither command fits the later outcome models.

## Ordered coding modules

### M01. Import and harmonize study extracts

**Status:** implemented and validated. **Dependencies:** verified configuration and authorized inputs.

**Implemented code:** `src/dchs_lung_function/inputs.py`.

- **M01.1 — Read study extracts:** Implement format-specific readers, required-column checks, explicit missing-value/date conversions and stable row identities. Completion: Reconcile source and imported rows; reject malformed rows, silent truncation and missing required fields.
- **M01.2 — Resolve input bindings:** Bind neutral input roles to the selected private files; verify wheeze candidates and reference lookup provenance before use. Completion: Every required source is identified and hashed; no unresolved candidate is silently selected; private paths stay outside shared source.

### M02. Construct cohorts and descriptive outputs

**Status:** implemented; upstream antenatal flow gap remains. **Dependencies:** M01.

**Implemented code:** `src/dchs_lung_function/cohorts.py`, `src/dchs_lung_function/descriptive.py`.

- **M02.1 — Build cohort ledgers:** Construct the joint, descriptive and outcome-specific populations, diagnosis groups and sequential exclusions. Completion: Independent membership comparisons and exact counts reproduce the selected populations; joins do not inflate records.
- **M02.2 — Calculate Table 1:** Calculate all category counts, nonmissing denominators, continuous summaries and the three parity median/IQR entries. Completion: Every Table 1 scientific cell is generated from the descriptive sample with the selected summary convention.
- **M02.3 — Calculate flow and testing schedule:** Derive participant-flow arithmetic and observed modality-by-visit availability from imported PFT records; use the provisional visit/TB linkage shared with M03. Completion: Figures S1/S2 use calculated aggregate ledgers; population counts are checked again against finalized M03 samples.

### M03. Prepare longitudinal model inputs

**Status:** implemented and validated. **Dependencies:** M01, M02.

**Implemented code:** `src/dchs_lung_function/preparation.py`, `src/dchs_lung_function/reference_scores.py`, `src/dchs_lung_function/respiratory_preparation.py`.

- **M03.1 — Link dates and construct exposure:** Implement child/visit linkage, source date precedence, same-day TB transitions and mutually exclusive diagnosis-age groups. Completion: Reference linkage and boundary fixtures agree, including missing dates and repeated visits; preserve one dataset identity with neutral role names.
- **M03.2 — Rebuild reference scores and transforms:** Recreate pre/post-bronchodilator GLI scores, X6 reference scores, native units and visit/height/sex residualization. Completion: Reference coefficients and lookup version are verified; model matrices and transformed values agree before fitting.
- **M03.3 — Freeze model-specific samples:** Create explicit available-case, paired, exact-age and nominal-visit sample rules with model covariates and child weights. Completion: All distinct eligibility rules have tested sample ledgers; source values are preserved and no sample is reused solely because its label is similar.

### M04. Fit primary spirometry joint models

**Status:** implemented and independently validated. **Dependencies:** M03.

**Implemented code:** `src/dchs_lung_function/models/spirometry.py`, `src/dchs_lung_function/sampling.py`, `src/dchs_lung_function/model_runtime.py`, `src/dchs_lung_function/diagnostics.py`, `src/dchs_lung_function/analysis/primary_results.py`, `src/dchs_lung_function/primary_execution.py`.

- **M04.1 — Make primary models explicit:** Replace runtime source rewriting with versioned PyMC model functions preserving joint likelihoods, priors, residual scales, TB process and shared child effects. Completion: An independent reviewer reconstructs the specification and compares the complete design matrices and prior/likelihood terms.
- **M04.2 — Fit continuous spirometry:** Run the 12 selected continuous fits and compute primary and diagnosis-age contrasts; retain private posterior draws and realized sampler metadata. Completion: Sample counts and deterministic transforms match exactly; convergence and prespecified Monte Carlo comparison gates pass for Table 2, Table S6 and shared primary rows.

### M05. Fit binary spirometry models

**Status:** implemented and independently validated. **Dependencies:** M03.

**Implemented code:** `src/dchs_lung_function/models/binary_spirometry.py`, `src/dchs_lung_function/binary_execution.py`.

- **M05.1 — Define binary outcomes:** Derive LLN indicators and mutually exclusive spirometric patterns separately for pre/post-bronchodilator samples. Completion: LLN equality, joint low-ratio/low-FVC states, missing stages and all small/zero aggregate counts are tested.
- **M05.2 — Fit binary spirometry:** Fit modified-Poisson GEE with the selected log link, working correlation and child-robust covariance. Completion: Independent coefficients, cluster counts, covariance and intervals agree; nonestimable terms receive an explicit status.

### M06. Fit spirometry trajectories

**Status:** implemented and independently validated. **Dependencies:** M03.

**Implemented code:** `src/dchs_lung_function/models/spirometry_trajectories.py`, `src/dchs_lung_function/trajectory_execution.py`.

- **M06.1 — Fit spirometry trajectories:** Implement the selected spline basis, random intercept, covariance calculation and reduced-model comparison for school-age ratio trajectories. Completion: Fit and interaction tests reproduce the selected specification and exact-age sample; X6 fits are owned only by M08.
- **M06.2 — Standardize trajectory predictions:** Produce the common age grid, sex/height standardization, uncertainty bands and native/z-score prose contrasts. Completion: Figure 1 plotting data and retained trajectory sentences are checked against independent predictions.

### M07. Fit mechanics and summarize the shared factor

**Status:** implemented and independently validated. **Dependencies:** M03, M04.

**Implemented code:** `src/dchs_lung_function/models/mechanics.py`, `src/dchs_lung_function/mechanics_sampling.py`, `src/dchs_lung_function/mechanics_execution.py`, `src/dchs_lung_function/analysis/mechanics_results.py`, `src/dchs_lung_function/analysis/shared_factor.py`, `src/dchs_lung_function/shared_factor_execution.py`.

- **M07.1 — Fit individual mechanics outcomes:** Fit the six selected joint mechanics models with their own samples, adverse-direction transforms and rescaling. Completion: Table S5 sample counts, contrasts, signs, units and intervals pass independent comparisons.
- **M07.2 — Summarize the shared mechanics factor:** Calculate the selected six factor-loading associations from M04 posterior draws, preserving the selected sign and outcome-scale conversion. Completion: Figure S5 summaries reproduce the loading transformation; this is postprocessing of primary fits, not a separate TB-effect model.

### M08. Fit supporting MMEF and X6 models

**Status:** implemented and independently validated. **Dependencies:** M03.

**Implemented code:** `src/dchs_lung_function/models/mmef.py`, `src/dchs_lung_function/models/x6.py`, `src/dchs_lung_function/models/supporting_repeated.py`, `src/dchs_lung_function/supporting_sampling.py`, `src/dchs_lung_function/analysis/supporting_results.py`, `src/dchs_lung_function/supporting_execution.py`, `src/dchs_lung_function/supporting_repeated_execution.py`.

- **M08.1 — Fit unpaired MMEF:** Implement available-case MMEF joint models for retained prose estimates. Completion: Unpaired sample counts and native-unit contrasts are verified separately from Table S2.
- **M08.2 — Fit paired MMEF:** Implement paired pre/post-bronchodilator joint MMEF models. Completion: Table S2 paired joint estimates and intervals match the selected model sample and reference group.
- **M08.3 — Fit MMEF change:** Fit the paired-change GEE using its own covariates and valid paired-change sample. Completion: Change-model membership, covariance and contrasts are verified independently; its denominator is not forced to equal the joint-model denominator.
- **M08.4 — Fit joint X6:** Implement the selected X6 joint model and reference-score contrast. Completion: Retained prose estimates, units and sample windows are checked even though they have no separate numbered display.
- **M08.5 — Fit repeated X6 and trajectories:** Implement the repeated X6 model, trajectory interaction and reduced-model differences once in this module. Completion: All retained X6 tests and native/z-score prose estimates are checked; M06 does not duplicate these fits.

### M09. Fit pre-TB lung-function survival models

**Status:** implemented; original-source reproduction independently validated. M14 separately implements the authorized G08 score-membership correction for publication; its independent numerical, visual and cumulative validation passed. **Dependencies:** M03.

**Implemented code:** `src/dchs_lung_function/models/pretb_cox.py`, `src/dchs_lung_function/pretb_execution.py`. See `docs/PRETB_MODELS.md` for the inherited score-membership behavior and the distinction between source reproduction and statistical covariance validation.

- **M09.1 — Construct pre-TB risk intervals:** Use attained age; carry each eligible measurement until the next measurement, two-year ceiling, first TB event or censoring, whichever occurs first. Completion: Independent risk intervals and risk sets agree; no post-TB predictor enters analysis and full counts survive nonestimable fits.
- **M09.2 — Fit and verify Cox uncertainty:** Fit the selected sex/site-adjusted Cox models with adverse orientation and the original robust covariance specification. Completion: Independent estimation verifies all 17 Figure S3 rows, covariance calculations and uncertainty intervals.

### M10. Fit age-specific wheeze models

**Status:** implemented and independently validated. **Dependencies:** M03.

**Code:** existing M03 `respiratory_preparation.py`, `models/wheeze.py`, `wheeze_sampling.py`, `wheeze_execution.py`, `analysis/wheeze_results.py`, and `analysis/wheeze_standardization.py` under `src/dchs_lung_function/`.

- **M10.1 — Build scheduled and episode outcomes:** Verify wheeze input bindings, separate scheduled assessments from participant-initiated contacts, form episodes by consecutive gaps of at most 30 days and construct risk time. Completion: Episode chaining, 30/31-day boundaries, assessment denominators, no-lag exposure and later-onset indicators pass independent checks.
- **M10.2 — Fit joint wheeze:** Implement primary and designated adjusted joint wheeze specifications through explicit configuration and a shared model function. Completion: Conditional ORs and recurrent-episode IRRs for Table S4/Figure S6 are validated on their original scales; M12 verifies and reuses the accepted scenarios without duplicating this engine.
- **M10.3 — Calculate standardized probability ratios:** Integrate the selected posterior predictive probabilities and standardization distribution to obtain the displayed risk ratios. Completion: Figure 3A uses standardized RR; independent calculations show that conditional ORs and episode IRRs were not substituted.

### M11. Fit FeNO regressions

**Status:** implemented and independently validated; final cumulative original-source and copied-source replays passed. **Dependencies:** M03.

**Code:** `src/dchs_lung_function/models/feno.py`, `src/dchs_lung_function/feno_execution.py`. See [FeNO methods](FENO_MODELS.md).

- **M11.1 — Fit main FeNO:** Fit the five age-specific log-FeNO regressions with selected exposure timing, sex, height, exact age and child-robust covariance. Completion: Figure 3B sample counts, GMRs and intervals match; nonpositive observations never enter a logarithm.
- **M11.2 — Fit supplementary FeNO:** Implement the six visit-specific supplementary contrasts, retaining the distinct earliest-visit future-TB comparison. Completion: Figure S4 selectors, comparator groups and intervals are verified separately from the main five-age analysis.

### M12. Run designated sensitivity analyses

**Status:** implemented and independently validated; final cumulative original-source and copied-source replays passed. **Dependencies:** M04, M10.

**Code:** `src/dchs_lung_function/models/sensitivity.py`, `src/dchs_lung_function/sensitivity_sampling.py`, `src/dchs_lung_function/sensitivity_execution.py`, `src/dchs_lung_function/sensitivity_tables_execution.py`, `src/dchs_lung_function/analysis/sensitivities.py`. See [sensitivity methods](SENSITIVITY_MODELS.md).

- **M12.1 — Run spirometry sensitivities:** Fit the eight selected early-TB specifications and two all-age observation-window variants with their distinct graphs and samples; verify and reuse the two accepted full-follow-up M04 ratio fits. Preserve configurable endpoints, baseline versus exploratory covariate placement and complete sampler metadata. Completion: All 34 selected Tables S1/S3 effect rows are selected from computed means and intervals; eight attenuation cells use full-precision posterior mean magnitudes, with no invented attenuation interval. The 267 frozen new-fit comparisons and complete posterior diagnostics must pass.
- **M12.2 — Coordinate wheeze sensitivities:** Verify and reuse both accepted M10 scenarios for Table S4, retaining conditional scheduled-wheeze ORs and recurrent-episode IRRs, posterior medians and the original model-specific samples. Completion: Verified fit identities prevent duplicate runs; all twelve Table S4 effects preserve accepted M10 draws and summaries on the conditional OR/IRR scales.

### M13. Generate tables and numerical text

**Status:** implemented and independently validated; final-source original and copied cumulative replays passed. **Dependencies:** M02, M04, M05, M06, M07, M08, M09, M10, M11, M12.

**Code:** `src/dchs_lung_function/reporting/` and `config/manuscript_reporting.json`. See [reporting methods](MANUSCRIPT_REPORTING.md) for `--through M13 --reporting FILE`, the separate standard-library document runtime, output inventory and acceptance requirements.

- **M13.1 — Create a common result interface:** Resolve current analytical identities and verify complete aggregate receipts/inventories; retain full-precision estimates, point estimators, interval types, denominators, units and explicit non-estimable statuses without participant records. Repeated sensitivity-table fields alias to their original model result keys with exact-value verification. Completion: Tables and prose use the same canonical results; estimates are never typed into production rendering code.
- **M13.2 — Render tables and numerical prose:** Produce editable DOCX/HTML tables, ten CSVs, generated numerical text and machine-readable lineage/comparison outputs with centralized display rules. Use a separate standard-library document interpreter and actual embedded Arial/Arial Bold, preserving the scientific locks. Completion: All eight numbered/ten physical tables, 304 digit-containing cells, 261 scientific cells and 42 sentences are classified and checked; 96 tokens and 52 qualitative bindings resolve. Complete-suite, document visual/font and final original-source/copied-source cumulative receipts must pass. Original reference comparison is informational and never forces a computed estimate to match an archived rounded literal.

### M14. Generate publication figures

**Status:** implemented and independently validated; final-source original/copied cumulative replays and the 868-test suite passed. **Dependencies:** M13.

**Code:** `config/publication_figures.json`, `src/dchs_lung_function/analysis/pretb_uncertainty.py`, `src/dchs_lung_function/reporting/figure_data.py`, `src/dchs_lung_function/reporting/figures.py`, `src/dchs_lung_function/figures_execution.py`. See [publication figure methods](PUBLICATION_FIGURES.md).

- **M14.1 — Resolve figure data and corrected uncertainty:** Verify accepted M13 inputs, preserve the source M09 archive, and separately calculate the approved actual-risk-membership robust covariance for Figure S3. Resolve all nine figures, 17 panels, 56 forest estimates, 808 trajectory rows and ledger-driven flow/schedule data. Completion: Source coefficients, hazard ratios, counts and model covariance remain unchanged; corrected standard errors, intervals, p-values, q-values and full robust covariance agree with independent calculations. All scientific labels and caption counts bind accepted aggregate records.
- **M14.2 — Render and verify publication figures:** Load actual Arial/Arial Bold, produce nine vector PDFs, nine 600-dpi PNGs and a combined nine-page PDF, with eleven data/metadata payloads plus receipt. Completion: All panels pass numerical, caption, visual and font-embedding checks. The full expanded suite and original-source/copied-source cumulative replays must pass before acceptance. Layout revisions must preserve plotted scientific values, and proprietary fonts remain external inputs.

### M15. Verify and deliver the code-only repository

**Status:** release controls implemented; final execution/package/remote acceptance is recorded separately. **Dependencies:** M13, M14.

**Code:** `src/dchs_lung_function/full_execution.py`, `src/dchs_lung_function/render_execution.py`, `src/dchs_lung_function/pipeline.py`, `config/render.example.json`, `config/source_allowlist.json`, `tools/export_source.py`, `.github/workflows/checks.yml`. See [RELEASE.md](RELEASE.md).

- **M15.1 — Verify clean full and render runs:** `--full` begins with genuinely empty, separate work/output roots and creates a paired origin bound to source, inputs, nine configurations, runtimes and destinations. It rebuilds all 37 selected joint fits and the deterministic models before M13/M14 publication. `--full --resume` requires that same origin. `--render --render-config FILE` instead verifies trusted accepted M13/M14 receipt hashes and exact structured aggregate payloads, without study inputs, raw draws, refits or analytical revalidation. Completion: The extracted reviewed source completes the clean reconstruction, unchanged diagnostics and prespecified scientific comparisons; aggregate-only rendering and its separate resume preserve scientific payloads.
- **M15.2 — Prepare source-only release:** Export the explicit reviewed allowlist with a deterministic SHA-256 manifest; verify and safely extract using the externally recorded reviewed manifest hash. Keep private inputs/configurations, participant-derived intermediates, draws, results, manuscripts, review evidence and proprietary fonts outside the package. Completion: Exact extracted inventory, content/privacy review, locked installation and nonempty complete synthetic tests pass. Hosted source/fabricated-data checks cover macOS/Linux; confidential-data cross-platform equivalence is not inferred.
- **M15.3 — Verify GitHub delivery:** Record the chosen private repository, upload only reviewed source, compare remote files/commit with the accepted manifest, and check hosted synthetic CI. Completion: Remote identity and actual CI outcomes have separate receipts; local export or full-run success is not recorded as remote delivery. Reuse licensing is not selected automatically.

## Manuscript coverage

The 17 numbered outputs comprise eight numbered tables and nine figures; the tables occupy ten physical table objects. Retained numerical prose is also in scope, including MMEF and X6 results without separate numbered displays.

| Target | Computational modules | Rendering module |
|---|---|---|
| Table 1 | M02 | M13 |
| Table 2 | M04 | M13 |
| Figure 1 | M06 | M14 |
| Figure 2 | M04, M05 | M14 |
| Figure 3 | M10, M11 | M14 |
| Table S1 | M12 | M13 |
| Table S2 | M08 | M13 |
| Table S3 | M04, M12 | M13 |
| Table S4 | M10, M12 | M13 |
| Table S5 | M07 | M13 |
| Table S6 | M04 | M13 |
| Figure S1 | M02 | M14 |
| Figure S2 | M02 | M14 |
| Figure S3 | M09; authorized uncertainty correction in M14 | M14 |
| Figure S4 | M11 | M14 |
| Figure S5 | M07 | M14 |
| Figure S6 | M10 | M14 |
| Numerical prose: unpaired MMEF | M08 | M13 |
| Numerical prose: joint X6 | M08 | M13 |
| Numerical prose: repeated X6 and trajectory tests | M08 | M13 |

**Prose-only ownership:** M08 owns unpaired MMEF, joint X6 and repeated X6/trajectory results. M13 maps all 42 numerical sentences and all scientific table cells to calculated results. Header digits and labels are classified rather than treated as estimates.

M06 owns spirometry trajectories; M08 owns X6 fits. M07 derives factor summaries from primary draws and separately fits the individual mechanics outcomes. M10 owns the wheeze engine and standardized RR; M12 verifies and reuses both accepted scenarios without implementing a second engine.

## Stage 3: verification

Five independent review roles cover (1) import/linkage/cohort reconstruction, (2) longitudinal preparation and statistical specifications, (3) independent model calculations and uncertainty, (4) complete manuscript and presentation coverage, and (5) execution, environment and release checks. Reviewers cross-check findings against one another and the lead implementation. Review is performed during coding and repeated on the final integrated result; summaries stay in the task and private validation evidence.

| Check family | Acceptance |
|---|---|
| V01 — Input identity | A changed input or code dependency invalidates downstream results; an absent required input fails before analysis. |
| V02 — Encoding and malformed rows | Non-ASCII fields work under UTF-8 and C locales; import warnings, silent truncation and missing required columns fail. |
| V03 — Keys and joins | Unexpected duplicate child/visit keys and many-to-many joins fail or follow a documented reviewed aggregation rule. |
| V04 — TB timing boundaries | A measurement immediately before diagnosis stays pre-TB; same-day and later measurements follow the selected inclusive rule. |
| V05 — Eligibility boundaries | Test exactly 1 and exactly 5 years at diagnosis; nominal visits, exact-age limits, study exit and administrative cutoffs are separate explicit rules. |
| V06 — Available and paired samples | Missing one bronchodilator stage excludes a paired measurement without excluding valid unpaired outcomes. |
| V07 — Outcome definitions | Test LLN equality versus strictly below, mixed low-ratio/low-FVC patterns, unit transforms and adverse-direction signs. |
| V08 — Wheeze episodes | Contacts 30 days apart remain one episode; contacts 31 days apart split; chained contacts use consecutive gaps; scheduled assessments are never episode contacts. |
| V09 — FeNO | Nonpositive values do not enter the log analysis; main and supplementary timing comparators are checked separately. |
| V10 — Cox follow-up | Carry-forward stops at the earliest next observation, two-year limit, TB event or censoring; no post-TB measurement enters the predictor. |
| V11 — Small and zero counts | All aggregate counts remain exact; zero-event/nonestimable coefficients receive an explicit status without fabricated estimates. |
| V12 — Model/scale correspondence | RR, conditional OR, IRR, GMR and mean differences remain distinct; native versus z-score outputs cannot be swapped. |
| V13 — No-op test prevention | Test discovery fails when zero tests or source files are found; deliberate malformed fixtures demonstrate that checks reject errors. |
| V14 — Rebuild and presentation | Two execution modes validate their own prerequisites; figures retain estimates and embedded Arial; tables/prose share one rounding policy. |

Deterministic counts, membership and classifications must agree exactly. Floating-point tolerances are specified by output scale before comparison. Bayesian comparisons require an identical specification/sample, prespecified diagnostic and Monte Carlo-error criteria, and independent posterior-summary checks; neither matching displayed rounding nor bitwise draws alone establishes scientific validity.

A full clean rebuild uses absent/empty, nonoverlapping work/output roots and only documented authorized inputs and dependencies. The accepted 37-fit run is the baseline for same-seed reconstruction with recorded controls. Complete posterior/non-timing sampler comparisons, unchanged full-posterior diagnostic gates and the 924 frozen mean/quantile MCSE comparisons assess reproduction; counts/membership remain exact and deterministic tolerances remain unchanged. This is not an independent random sample. Aggregate-only rendering has separate trusted-receipt prerequisites and cannot establish analytical reproduction. See RELEASE.md for origin/resume and evidence requirements.

## Assigned implementation gaps

The nine inventory findings retain their original scope. G01 and G02 are resolved. Selected engines and fresh-fit provenance through M12 are implemented and independently checked. All ten designated M12 new fits pass full-posterior diagnostics and all 267 frozen comparisons, and the four reused fits retain their accepted raw identities. Final M12 cumulative original-source and copied-source replays passed. M13 analytical lineage, numerical coverage, document presentation and final-source original/copied cumulative replays are verified. Outcome-specific preparation and implemented model samples are checked under G09. G04 retains four unavailable antenatal nodes; the computed M13 reporting layer and its complete numerical coverage are independently validated. G08 has an approved and independently validated actual-risk-membership correction for Figure S3, with completed numerical, visual and final-source replay evidence. The Efron-fit/Breslow-score and inclusive-boundary conventions remain explicit; original M09 outputs remain preserved. G07 table, prose and figure/caption correspondence is independently checked. G06 clean-release and platform-specific checks require their own receipts; synthetic CI does not establish cross-platform scientific equivalence. The machine-readable workflow records these statuses without treating numerical implementation as final release.

| Gap | Owners | Completion check |
|---|---|---|
| G01 — Required wheeze paths | M01, M10 | Resolved: selected binding and complete Stata/CSV comparison verified; hashes retained privately. |
| G02 — Parity summaries | M02 | Resolved: all three median/quartile entries generated and matched to selected cells. |
| G03 — Explicit model implementation | M03, M04, M12 | Selected engines through M12, including designated early/all-age sensitivity graphs, priors and results, are independently validated. |
| G04 — Calculated reporting inputs | M02, M12, M13 | Computed table/prose bindings are verified; preserve explicit unavailable antenatal facts and bind final-source replay receipts before delivery. |
| G05 — Realized sampler configuration | M04, M07, M08, M10, M12, M15 | Recover commands/logs where available; otherwise document the reconstruction specification and validate independent fits. |
| G06 — Portable environment and restart rules | M01, M15 | Lock dependencies, centralize paths, fingerprint code/inputs and invalidate stale descendants. |
| G07 — Model and caption correspondence | M07, M08, M13, M14 | Preserve estimates and document each model-specific reference group and factor interpretation in methods and generated captions. Manuscript editing is outside this coding task. |
| G08 — Pre-TB survival uncertainty | M09, M14, M15 | Preserve M09 source reproduction; independently verify the approved actual-risk-membership robust uncertainty used in Figure S3, without changing betas, HRs, counts or model covariance. Retain declared tie and boundary conventions. |
| G09 — Outcome-specific eligibility | M03, M06, M08 | Retain separate sample ledgers; check age bounds actually represented and compare all saved outcome-specific counts. |

## Stage 4: GitHub delivery

The final export contains reviewed code, concise README/methods/input documentation, example configuration, exact dependency specifications and synthetic tests. The explicit reviewed allowlist determines the complete source package, including this plan and release instructions. Stage reports, review conversations, manuscript files, private path bindings, study extracts, participant-derived intermediates, posterior draws and generated study results remain outside the shared package.

Release requires an explicit reviewed file allowlist, checks of the extracted source archive, a clean environment installation, nonempty test discovery and a complete authorized-data rebuild. GitHub Actions runs source and synthetic checks; confidential-data scientific validation remains local. The chosen private repository and visibility must be recorded before upload; preparing source does not itself select a reuse license.

GitHub completion means the uploaded source matches the reviewed release manifest and the hosted checks pass. Local preparation alone is not recorded as upload success.

## Release acceptance sequence

1. Freeze the reviewed source allowlist/manifest and prespecified clean-reconstruction protocol; retain completed M01–M14 evidence and all prior scientific results.
2. Verify/export/extract the package, run the complete synthetic suite, and reconstruct all analyses from genuinely empty roots with the selected runtime, inputs and controls.
3. Independently compare all 37 joint fits and deterministic/publication outputs, verify full-run resume and aggregate-only rendering, and bind all five review receipts to the exact source and run identities.
4. Preserve explicit missing antenatal facts, the original M09 calculation and the accepted separate Figure S3 correction; no inferred upstream counts or manuscript edits enter release.
5. Verify the chosen private remote tree/commit and hosted macOS/Linux synthetic outcomes against the reviewed package; record delivery separately from local acceptance.

## M01-M15 implementation update

M01 and M03 preparation checks pass. M02 reproduces Table 1, selected cohort memberships and empirical modality availability; four antenatal recruitment/disposition nodes are explicitly unresolved. See PREPARATION.md for the inherited descriptive recodes and upstream input boundary. Runtime fingerprints, artifact verification, two-directory locks and atomic preparation publication are implemented. All five original review roles contributed. Detailed review evidence remains private. M04 now uses those validated contracts for twelve independently checked fresh fits, all 288 reference comparisons, complete-posterior diagnostic recomputation and selected global trace review. M05 adds 28 selected modified-Poisson GEE fits with complete robust covariance and explicit boundary statuses; 8,006 independent current-output checks and 499 archived-reference checks pass. M06 adds four independently validated full/parallel trajectory pairs; 5,480 current-output checks, 3,412 historical-reference checks and 58 manuscript checks pass. The completed M14 baseline contains 868 passing synthetic tests, including 102 preparation regression tests; release acceptance requires the complete current suite. Cumulative execution and identical copied-source resume are required for each completed module; earlier primary draws and binary scientific outputs remain unchanged. M07 adds six independently validated joint mechanics fits and six derived shared-factor associations; all 180 prespecified scientific comparisons pass, with full-posterior diagnostics and exact source/sample checks. M08 adds seven independently validated joint MMEF/X6 fits and ten supporting component fits; all 108 unchanged Monte Carlo comparison gates and 6,995 independent deterministic checks pass. The documented initialization repair and single precision extension preserve the selected likelihoods and acceptance gates. M09 reproduces all 17 archived Cox models, all 51 coefficients and all 306 covariance cells. Independent review also quantifies an inherited score-membership limitation affecting uncertainty; computational agreement does not close G08 statistically. M10 adds two independently seeded wheeze joint fits and exact source graph parity checks, with all 81 frozen posterior comparisons passing. It retains 12 conditional effects, 12 age contrasts, four global age summaries, 44 support cells, 44 predictive cells and three primary standardized probability ratios. Complete posterior diagnostics cover 2,175 scalars, including child effects; all 44 conditional predictive simulation checks pass. M11 implements all eleven selected log-FeNO regressions: five main and six supplementary models. All 2,368 independent full-precision checks across 65,293 cells pass; outputs retain 11 selected estimates, 63 coefficients, 726 covariance cells and 33 support rows. Source selection, nominal-visit exceptions, comparison groups and manuscript rounding are independently checked. Original-source and copied-source cumulative execution through M11 passed. M12 is implemented and independently validated: ten new fits, two reused M04 ratios and both reused M10 scenarios, with distinct early/all-age estimands. All 267 frozen new-fit comparisons pass at 3,200 retained draws per chain; no precision extension was needed. Full-posterior diagnostics cover 31,910 scalars. The module retains 64 effect rows, 25 hazard rows, 300 support rows and 46 selected sensitivity table-input rows. Final M12 original-source and copied-source cumulative replays passed. M13 is implemented and independently validated across all 304 classified locations, 261 scientific cells and 42 sentence templates. The actual cumulative M01-M13 command passes, and its published DOCX is identical to the independently inspected ten-page artifact. Final-source original and copied-source M13 replays passed and all five review receipts bind that delivery. M14 implements all selected figures and the approved separate G08 uncertainty correction; the 868-test suite, independent visual/font review and final-source original/copied cumulative replays passed. M15 implements distinct full/render modes and source-release controls. Its final clean-reconstruction, extracted-package and remote-delivery results are established by separate private acceptance receipts, without changing scientific thresholds or the frozen Stage 1 baseline.
