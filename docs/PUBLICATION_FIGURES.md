# Publication figures

M14 implements all three main and six supplementary
figures from verified aggregate results, including the approved correction
of pre-TB score-residual risk membership for Figure S3. The original M14 baseline
passed independent numerical/caption checks, rendered-page and actual-font
inspection, its 868-test suite, and original/copied cumulative replays. Private
acceptance receipts bind that historical evidence. Corrective figures require
fresh numerical, layout and actual-font checks against their revised accepted
aggregates. [M15 release instructions](RELEASE.md) describe clean
full reconstruction, aggregate-only rendering and separate code-only delivery
acceptance.

## Scientific coverage

The public `config/publication_figures.json` specifies nine figures in 17 panels.
It binds 56 forest estimates, 808 trajectory predictions, 44 modality/visit cells,
28 flow count fields in 20 display nodes and 12 caption count tokens. Selectors,
field names, row order, series, scales, null values, units and interval labels are
explicit. No study estimate or private input path is typed into the specification.

| Figure | Selected input and meaning |
|---|---|
| 1 | Four native/GLI pre-bronchodilator FEV₁/FVC trajectory models, two groups and a 101-point exact-age 5–10 grid per model; predicted means with confidence bands |
| 2 | Eight joint-model posterior mean contrasts with credible intervals and eight binary adjusted risk ratios with confidence intervals; pre/post stages and infancy/before-five comparisons remain separate |
| 3 | Three primary standardized scheduled-wheeze probability ratios, summarized by posterior medians and credible intervals, plus five main FeNO geometric mean ratios and confidence intervals |
| S1 | Computed woman/child flow ledgers and overlapping outcome-specific samples; four upstream antenatal counts explicitly unavailable |
| S2 | Four modalities by eleven nominal visits; observed availability comes from the ledger, not a hardcoded testing schedule |
| S3 | Seventeen adverse-oriented hazard ratios with the authorized corrected robust confidence intervals, retaining exact children/event counts |
| S4 | Six supplementary FeNO geometric mean ratios: future TB at the earliest visit, then TB before/on measurement at later visits, each against no recorded TB |
| S5 | Six already oriented and scaled shared-factor associations, expressed as posterior mean GLI z-score differences per model-defined SD poorer mechanics |
| S6 | Three primary conditional recurrent-wheeze incidence rate ratios, summarized by posterior medians and credible intervals |

Ratios use logarithmic axes with a null of one; differences use linear axes with
a null of zero. Figure 1 shows mean trajectories rather than differences and has
no added p-value annotations. The no-TB reference uses a dashed line; paired panels for the same outcome use common y-axis limits derived from all selected intervals. Pointwise 95% confidence bands are not simultaneous bands and describe uncertainty in the mean,
not individual prediction intervals. Full-precision results determine positions
and interval endpoints; centralized decimal half-up rounding supplies labels.
Axis bounds include the complete selected intervals.

Captions preserve model-specific populations and reference states. In particular,
standardized wheeze risk ratios are not conditional odds ratios or recurrent-event
rate ratios; supplementary FeNO samples are not substituted for main FeNO samples;
and shared-factor associations are not TB exposure effects or independent pre-TB
predictions. Infancy and before-five contrasts overlap by definition. Separate
pre/post available-case models are not a formal paired bronchodilator test.
The FeNO age gap is not interpolated. Figure S4 prints all six GMRs and confidence intervals below the visit labels. Small aggregate counts remain exact.

## Figure S3 uncertainty

The accepted M09 Efron coefficients, hazard ratios, inverse observed information,
sample counts and original output files remain unchanged. M14 performs no Cox
refit. It restricts child-score contributions to actual event-time membership,
`start <= event age <= stop`, while retaining the source Breslow-style weighted
means and martingale increments. Scores are summed by child and retain the
`G/(G-1) * (N-1)/(N-3)` finite-sample multiplier. Intervals outside every event-time
risk set receive zero score contributions.

Separate outputs retain 17 selected estimates, all 51 coefficients and all 306
model/robust covariance cells. The robust covariance, its standard errors,
normal-Wald confidence intervals and p-values are recomputed. BH q-values are
recomputed over the same 17 selected lung-function tests. Model covariance and
all point estimates remain unchanged. Figure S3 uses the explicit
`pretb_corrected/estimates` source; the M13 registry retains original source
reproduction. See [pre-TB methods](PRETB_MODELS.md) for the inherited issue and
the retained Efron-fit/Breslow-score and inclusive-boundary conventions. This
correction does not replace those conventions or establish general small-sample
coverage.

## Command and local configuration

Use the existing pinned scientific environment, input/analysis configuration,
sampling settings and M13 document configuration. Create a private figure
configuration containing exactly:

```json
{
  "schema_version": 1,
  "pdf_fonts_tool": "REPLACE_WITH_PDFFONTS_EXECUTABLE",
  "pretb_uncertainty": "corrected_actual_risk_membership"
}
```

The font inspector must be an executable `pdffonts`; its version and resolved
executable hash are recorded. Relative paths resolve from the local configuration
directory. Actual Arial regular and Arial Bold paths remain in the private paths
configuration. Font files are external inputs, never bundled in the repository.

```sh
.venv/bin/python run.py --through M14 --config config/paths.local.json --analysis config/analysis.local.json --reporting config/reporting.local.json --figures config/figures.local.json --resume
```

Both `--reporting` and `--figures` are required. All selected upstream models must
verify; `--targets` remains restricted to M04. Existing sampling options retain
their module-specific meanings, and `--jobs` retains its cumulative-run meaning.
M14 adds no posterior sampling or new model fit. It verifies M01–M13 before
correcting uncertainty and rendering. M15 also supplies `--full` and aggregate-only
`--render --render-config FILE`; their distinct prerequisites are in [RELEASE.md](RELEASE.md).

The plotting runtime uses the existing locked scientific dependencies. Python,
platform, plotting packages, FreeType, zlib, font bytes and font-inspector identity
are bound in the publication fingerprint. M13's separate document interpreter
and optional reference inputs are also checked. Scientific dependency locks are
unchanged. Each PDF must embed both actual Arial faces with Unicode mapping,
without fallback or Type 3 fonts.

## Outputs, resume and acceptance

`publication_figures/<fingerprint>/` in the separate aggregate-output directory
contains 30 payloads plus `receipt.json`:

- Nine individual vector PDFs, nine 600-dpi PNGs and the combined nine-page
  `publication_figures.pdf` — nineteen image files.
- `corrected_pretb_estimates.csv`, `corrected_pretb_coefficients.csv`,
  `corrected_pretb_covariance.csv` and `corrected_pretb_metadata.json`.
- `plot_data.json`, `plot_points.csv` and `figure_captions.txt`, preserving
  full-precision values, source keys, semantics and resolved caption counts.
- `font_validation.json`, `render_metadata.json`, `metadata.json` and
  `summary.json`, binding fonts, layout evidence, coverage and runtime.

Successful cumulative publication writes `figures_current.json` in private work.
Output/work roots remain separate from each other, the source tree and protected
inputs. Locked staging and atomic publication prevent partial outputs from being
accepted. Resume regenerates the uncertainty correction and every figure payload,
checks their complete hashes and inventories, and verifies source/input/runtime
identity again before updating the pointer. A cached receipt alone is insufficient.

Wrapping and text bounds are measured on the actual 600-dpi canvas. Every text object must stay inside a 0.35-inch content box, and captions must retain at least half-inch horizontal margins. The layout receipt records the measured caption margin and DPI. Actual vector-PDF and 600-dpi raster caption bounds are checked independently; a whole-page low-resolution preview alone is insufficient.

Rendered-page inspection remains separate from successful generation and font
inspection. Independent review checks every panel, scale, interval, label,
caption and unavailable-count display. Final-source original/copied cumulative
replays must reproduce the publication and preserve all earlier scientific
outputs and raw fits. Generated figures, aggregate data, local configurations
and review evidence stay outside the code-only source export. No participant
records are exported, and the selected manuscripts remain unchanged.
