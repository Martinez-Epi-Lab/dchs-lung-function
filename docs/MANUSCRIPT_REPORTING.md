# Manuscript tables and numerical text

M13 is implemented and independently validated. It assembles the selected
manuscript tables and numerical prose from verified M01–M12 aggregate results.
The M13 baseline suite has 802 passing tests, and actual cumulative M01–M13
execution passes. The published DOCX is identical to the independently
inspected ten-page document with verified Arial embedding. Final original-source and copied-source cumulative replay against the reviewed
M13 source manifest passed. Private receipts verify all reporting payloads and
preservation of prior scientific outputs and raw fits.

## Selected coverage

The eight numbered tables occupy ten physical tables: main Tables 1 and 2,
Tables S1, S2, S3 and S5, separate scheduled/recurrent sections of Table S4,
and separate pre/post-bronchodilator sections of Table S6. The public
`config/manuscript_reporting.json` maps the original one-based accepted-view
paragraph and unique-cell coordinates to source selectors. Merged cells retain
their spans; they are not duplicated into expanded-grid coordinates.

The 304 original digit-containing cells comprise 186 estimate cells, 75 count
cells, 26 numeric headers and 17 contextual labels. All 261 scientific cells
have computed result bindings. The count category includes the three cohort
size headers and the children/tests pairs in Table S3. Table 1 includes all
three parity median/IQR summaries. The digit inventory counts ASCII numerals;
the complete layouts also preserve labels containing only Unicode subscripts.

All 42 retained numerical sentences have explicit templates and classifications,
with 95 result tokens and 52 qualitative evidence bindings. Age thresholds,
interval coverage, references and contextual dates are distinguished from study
estimates. The prose file is a generated companion for review; it does not edit
either selected manuscript. Context-only statements remain classified, and
qualitative statements retain their supporting result selections. An interval
that no longer supports a null-compatibility statement requires wording review;
this does not introduce another model-significance or Monte Carlo acceptance
threshold. No numerical similarity threshold is invented for qualitative
sensitivity interpretation.

## Result identity and scientific meaning

The aggregate registry retains full-precision values, semantic row selectors,
source fingerprints, units, point estimators, interval definitions and support
fields. It admits only declared aggregate schemas and rejects participant
fields. Source identities are reconstructed from the current inputs, contracts,
settings and code; a latest-pointer file alone is not evidence of an accepted
result. Complete upstream receipts and artifact inventories must verify before
reporting. Repeated sensitivity-table estimates alias to the same canonical
model result used elsewhere, with exact-value agreement required.

Reporting preserves these selected distinctions:

- Tables 2 and S6 use expanded joint-model posterior means and equal-tailed
  credible intervals. Early-TB weighted and infancy contrasts remain distinct.
- Table S1 attenuation uses the magnitudes of full-precision posterior means;
  it has no invented posterior interval. Table S3 preserves separate early-TB,
  all-age and configured follow-up-window specifications and their samples.
- Table S2 separates paired pre/post joint-model credible intervals from
  cluster-robust paired-change confidence intervals. Numerical prose uses the
  separate available-case MMEF model.
- Table S4 reports conditional scheduled-wheeze odds ratios and recurrent-event
  incidence rate ratios. The primary wheeze prose uses standardized risk ratios,
  all summarized by their selected posterior medians and credible intervals.
- Table S5 and compliance prose use the already transformed reported native-unit
  fields. Compliance orientation and conversion to mL/cmH₂O are not applied twice.
- Spirometry trajectory prose uses raw three-degree-of-freedom interaction
  Wald tests and reduced-model parallel differences. X6 trajectory tests remain
  in their own selected age window and model family.
- Main FeNO statements use the five designated GMR estimates, with their
  selected measurement-time definitions, including the nominal twelve-month
  exception. They do not use supplementary future-TB comparisons.

Central display rules apply decimal half-up rounding at the last step, remove
negative displayed zero, preserve outcome-specific precision and distinguish
counts, percentages, mean/SD, median/IQR and interval displays. A displayed
interval retains its source-specific CI or CrI meaning. Exact small aggregate
counts remain visible; missing or non-estimable values have explicit statuses.

Optional private reference files compare all 304 original cells and 42 original
sentences with the generated displays. These comparisons are informational:
fresh accepted Bayesian results need not reproduce archived rounded strings,
and sentence wording or contextual numbers can change their numerical order.
An archived manuscript literal never replaces a computed value. Scientific
acceptance remains governed by the independently verified upstream definitions,
diagnostics and frozen numerical/Monte Carlo criteria.

## Command and local configuration

Use the pinned scientific environment and the same authorized input and analysis
configurations used for prior modules. Create a private reporting configuration
with a document-authoring Python interpreter:

```json
{
  "schema_version": 1,
  "document_python": "REPLACE_WITH_DOCUMENT_PYTHON"
}
```

For reference comparison, additionally supply both `reference_cells` and
`reference_sentences`, each pointing to its private JSON array. They are optional
as a pair; one alone is invalid. Each array must contain exactly the declared
original locator IDs. Paths, original manuscript literals and reference files
remain outside the shared source tree. Actual Arial and Arial Bold font paths
come from the private paths configuration.

```sh
.venv/bin/python run.py --through M13 --config config/paths.local.json --analysis config/analysis.local.json --reporting config/reporting.local.json --resume
```

The existing `--sampling`, `--mechanics-sampling`, `--supporting-sampling`,
`--wheeze-sampling` and `--sensitivity-sampling` options retain their meanings.
The cumulative command verifies the preceding analytical modules before
assembling their results. Resume verifies the current identities and regenerates
the reporting content for comparison; it cannot authorize stale or incomplete
artifacts. Unchanged accepted raw Bayesian arrays remain reusable under their
verified sampling identities.

Document creation runs as a separate process using the configured interpreter
and the standard-library-only renderer. It does not add document packages to or
change either scientific dependency lock. The renderer reads the actual font
bytes, verifies Arial regular/bold identity and embedding permission, and embeds
them in the editable DOCX. Font files remain external inputs and are excluded
from the source repository. Visual review of the rendered document is a separate
delivery check; successful DOCX generation alone does not establish layout quality.

## Outputs and acceptance

The aggregate output directory receives
`manuscript_reporting/<fingerprint>/`, containing:

- `registry.json` and `tables.json`, retaining full-precision lineage and the ten
  selected physical table layouts;
- `manuscript_tables.docx`, `manuscript_tables.html` and ten table CSV files;
- `numerical_text.json` and `numerical_text.txt`, with all computed sentences,
  token/evidence bindings and explicit status information;
- `reference_comparison.json`, including an explicit reference-not-supplied
  status when the optional pair is absent;
- `metadata.json`, `summary.json` and `receipt.json`, binding the publication to
  its inputs, code, configuration, runtime and complete output inventory.

Successful publication writes `reporting_current.json` in private work.
Generated tables, text, original reference comparisons, local bindings and
validation evidence stay outside the code-only repository. Full raw arrays are
never copied into the reporting artifacts.

The M13 complete test suite, coverage/lineage checks, actual cumulative execution
and document structure/font and visual checks pass. Final original-source and
copied-source cumulative execution against the reviewed source manifest also
passed, preserving prior scientific outputs and raw fits and reproducing the M13
payloads under recorded identities. These completed outcomes are bound in local
receipts. Subsequent modules require their own final-source replay evidence.

Four upstream flow nodes remain unavailable: antenatal enrollment, antenatal
loss to follow-up, pregnancy loss and exclusion before the child cohort. The
combined antenatal/pregnancy-loss prose fact is also unavailable because its
components are unavailable; they are never summed as zero. Neither missing
counts nor archived literals are invented to complete the flow.

The inherited [G08 pre-TB Cox covariance behavior](PRETB_MODELS.md) remains
explicit in the preserved M13 source registry. The user approved a separate
M14 actual-risk-membership uncertainty correction for Figure S3. It retains the
M09 coefficients, hazard ratios, sample and source files, while recomputing robust
covariance and derived inference. M13 does not overwrite or silently relabel its
source-reproduction rows. [M14 figure publication](PUBLICATION_FIGURES.md) is
independently validated, including final-source original/copied replays. M15
provides `--full` clean reconstruction and aggregate-only
`--render --render-config FILE`; [RELEASE.md](RELEASE.md) defines their distinct
identities, prerequisites and acceptance evidence, plus separate source/GitHub
delivery verification. No separate stage report is produced.
