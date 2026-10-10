# Pre-TB lung-function survival models

M09 implements the 17 selected two-year carry-forward Cox models and preserves their source-reproduction outputs. M14 applies the approved robust-uncertainty correction for Figure S3, preserving the fitted coefficients, hazard ratios and samples. Each model uses its own verified M03 sample and estimates the hazard of subsequent first TB per one adverse-oriented SD of lung function, adjusting for child sex and recruitment site. The model has no intercept. Exploratory alternative carry-forward windows are outside this selected analysis.

## Measurements and risk intervals

Eligible measurements precede the first TB diagnosis date strictly; same-day and later measurements do not become predictors. Measurements require valid dates, height, sex and site, exact attained age from 0.08 through 12 years, a test date within the configured observation endpoint, and observed TB follow-up beyond the measurement age. Duplicate child/test-date records follow the selected preparation rule. Each measure has its own available-case sample.

For the eight measures without applicable reference equations, adverse-oriented values are regressed on scheduled visit, standardized height and sex using eligible pre-TB observations. Residuals are divided by their SD with degrees of freedom equal to the nuisance design column count. The nine frequency-domain outcomes use adverse-oriented South African reference z-scores directly. Higher resistance, resistance differences, within-breath reactance differences, resonant frequency and reactance area are adverse; compliance and the end-expiratory, end-inspiratory and frequency-domain reactances are reversed.

For the residualized outcomes, the Cox fit treats those estimated nuisance coefficients and residual scales as fixed. Reported uncertainty is conditional on that first stage. The omitted uncertainty does not have a proven uniform direction or correction factor across outcomes.

The analysis time scale is attained age. Each measurement contributes until the next eligible measurement of that outcome, two years after measurement, first TB, or the end of observed TB follow-up, whichever occurs first. Gaps after the two-year limit contribute no risk time. Only positive-length intervals are retained. Event assignment requires `start < TB age <= stop`.

The selected PHReg implementation includes intervals with `start <= event age <= stop` in an event-time risk set. Consequently, adjacent intervals can both contribute when their shared boundary equals an event age. This boundary convention is preserved and audited separately from the strictly pre-diagnosis measurement rule and event-assignment rule.

## Estimation and uncertainty

Cox coefficients and the observed information use the Efron tied-event partial likelihood. The selected source calculates child-cluster uncertainty from PHReg score residuals, which use Breslow-style martingale increments even when the likelihood uses Efron ties. This combination is retained explicitly in both source reproduction and the M14 membership correction. Replacing these residuals with a different tied-event score calculation would be a separate methodological change.

The source residual routine also has an inherited membership issue. Intervals wholly outside the retained event span receive undefined residuals and are assigned zero scores. Some other intervals contain no event time but lie within the retained span; the residual routine's reverse-loop bookkeeping retains these in additional score sets, including times before their entry. The likelihood risk sets remain correct under the inclusive boundary convention. Reproducing the archived standard errors therefore requires reproducing this residual behavior as well as the Efron/Breslow combination. Exact source reproduction does not establish that the inherited residual membership is statistically appropriate. The independent uncertainty audit compares it with the same calculation using actual event-time risk membership. That membership correction was approved for Figure S3. M09 files continue to preserve the original calculation for traceability.

Scores are summed by child. With inverse observed information `B`, child score vectors `U`, `G` children, `N` intervals and three covariates, the selected archived covariance is

```text
B @ (U.T @ U) @ B * G/(G-1) * (N-1)/(N-3)
```

The target hazard ratio is `exp(beta)`, its 95% interval is `exp(beta ± 1.96 * robust_SE)`, and its two-sided p-value uses the standard normal distribution. Both model and robust covariance matrices, all three coefficients, sample counts and aggregate risk-set support are retained. Benjamini–Hochberg q-values use the selected family of 17 two-year models; raw p-values remain available.

The source's fewer-than-three-event fitting guard is reported explicitly as a source-policy non-estimable status, with all sample and event counts retained. It is not a statement that every such model is mathematically non-identifiable. Numerical failure, singular information, invalid covariance and nonconvergence cannot produce accepted scientific inference. Small aggregate counts are never suppressed.

Above that source guard, a converged three-covariate fit can still have sparse event support. Robust covariance and the finite-sample multiplier do not establish accurate normal-Wald coverage in that setting. Exact event counts, estimates and intervals remain visible; no blanket percentage widening or automatic suppression is applied.

## Approved Figure S3 uncertainty correction

M14 reads the accepted M09 coefficients and inverse Efron observed information;
it performs no model refit. It reconstructs event-time membership directly as
`start <= event age <= stop`. Intervals outside every event-time risk set receive
exactly zero score contributions. For each event time, weighted covariate means
and martingale increments use the retained Breslow-style `d/S0` calculation,
restricted to the actual members of that risk set. Scores are summed by child
and use the same `G/(G-1) * (N-1)/(N-3)` finite-sample multiplier.

The corrected child-score sandwich replaces only the robust covariance in a
separate M14 output. Coefficients, hazard ratios, model covariance, adverse
orientation, interval samples, support counts, covariates and carry-forward
window are preserved. The correction recomputes robust standard errors,
normal-Wald confidence intervals and p-values for all three terms in each model,
then recomputes BH q-values over the same 17 selected lung-function tests. It does
not apply BH adjustment to all covariates or use q-values as plotted intervals.

This addresses the inherited score-membership defect. It retains the
Efron-fit/Breslow-score combination, the inclusive entry/stop risk boundaries and
the source finite-sample convention. It does not claim to replace these with a
different tie method or establish general small-sample coverage. The original
M09 archive remains unchanged; corrected Figure S3 rows have an explicit
`pretb_corrected/estimates` lineage. The corrective revision retains this
approved score-membership correction, rather than reverting to source
uncertainty or silently changing the tie/boundary conventions. Numerical,
caption, visual and cumulative replay validation are bound to each accepted
publication's private receipts.

## Execution and validation

```sh
.venv/bin/python run.py --through M09 --config config/paths.local.json --analysis config/analysis.local.json --resume
```

M09 first verifies the preceding M01–M08 analyses. Existing sampling options retain their meanings for those preceding Bayesian models; M09 itself is deterministic. Partial target selection does not complete a cumulative M09 run.

The module verifies prepared contracts, records source/configuration/runtime identities, and publishes three aggregate tables with complete coefficient and covariance inventories. Work and output directories remain outside the source repository. Every resume recomputes the selected Cox models and compares all saved tables and metadata with the new calculations. Changed or corrupt artifacts, mismatched source imports and incomplete prior modules block completion. Only successful cumulative execution writes `pretb_current.json` in private work.

Independent validation covers source-derived intervals and risk sets, likelihood derivatives, model and robust covariance, all 17 archived result rows, manuscript definitions, execution failures and copied-source reproducibility. M14 verifies those source results before computing the approved uncertainty correction and rendering Figure S3. See [publication figure methods](PUBLICATION_FIGURES.md) for its separate configuration, outputs and acceptance requirements.
