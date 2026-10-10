# Designated sensitivity models

M12 implements the selected sensitivity analyses. This module
produces aggregate inputs for Tables S1, S3 and S4 and the selected sensitivity
paragraph. Manuscript tables and figures are rendered in M13 and M14. The
selected scope uses ten new joint fits, two accepted M04 ratio fits and both
accepted M10 wheeze scenarios. Reused fits are verified rather than resampled.
The current corrective specification changes the target covariance model and
early-event censoring. Its acceptance is governed by the frozen
[corrective protocol](CORRECTIVE_REVISION.md); historical source-reproduction
results below describe the previous model, not validation of this revision.

## Selected analyses

| Family | New fits | Selected definition |
|---|---:|---|
| Early-TB, full lung-function follow-up | 7 | Core; maternal HIV; prenatal maternal smoking; birthweight; all three baseline covariates; ever-recorded LRTI; all three baseline covariates plus LRTI |
| Early-TB, restricted lung-function follow-up | 1 | Core model with observations on or before the configured endpoint |
| All-age TB, restricted lung-function follow-up | 2 | Pre- and post-bronchodilator native FEV1/FVC |
| Accepted M04 comparisons | 0 | Full-follow-up pre- and post-bronchodilator native FEV1/FVC |
| Accepted M10 comparisons | 0 | Sex-only and sex, prenatal smoking and birthweight wheeze scenarios |

The seven full-follow-up early models concern pre-bronchodilator FEV1/FVC in the
early-TB family. They are not additional covariate fits for the all-age model,
post-bronchodilator ratio, FEV1, FVC or reference z-scores. The two families
preserve their distinct likelihoods, eligibility and first-TB processes.

## Eligibility and covariates

The early family excludes every process measurement on or after first TB
diagnosed at age five or older. Measurements before a future diagnosis remain
in the pre/no-TB state. Same-day measurements are post-TB; a diagnosis exactly
at age one is in the age 1–<5 group, and one exactly at five is outside the
early-TB exposure. The first-TB process ends at eligible first TB, study exit,
the administrative endpoint or age five, whichever occurs first. A diagnosis
on exit or the endpoint counts; one at age five does not. It uses three hazard
intervals with boundaries 0, 1, 3 and 5 years. This observed-follow-up rule
replaces the source assumption of full follow-up to age five after exit;
continued post-exit TB ascertainment was not independently established.
All lung-process measurements are also bounded by study exit, separately
from the designated administrative measurement-window restriction.

The all-age family retains measurements after later-onset TB and includes
separate later-TB shifts and direct effects. Its M04 first-TB process
uses interval boundaries 0, 1, 3, 5, 8, 10 and 20 years, with event follow-up
limited by the configured event endpoint and study exit. Full-follow-up ratio
comparisons reuse M04. The two new window fits additionally require each
measurement to be on or before the endpoint and on or before study exit.

The restricted early core and the all-age window fits use the configurable
`event_end` in the private analysis configuration as their lung-function
observation endpoint. Equality is included. This restriction changes observed
follow-up, process-specific nuisance fits, residual scales and exposed-child
weights. It does not add a covariate or replace the selected first-TB family.

Each process requires its outcome, test date, height, sex and selected
covariates. Maternal HIV, prenatal smoking and birthweight enter both the
process-specific nuisance regressions and the first-TB hazard when selected.
Birthweight is standardized using the nonmissing baseline population before
the joint-model complete-case restriction. Missing birthweight changes the
joint and outcome-specific samples; counts are computed for each model.
Table S3 children/tests refer to FEV1/FVC contributors, not the entire joint
cohort. The reused adjusted wheeze scenario retains its separate M10
birthweight standardization and sample definition.

Ever-recorded lower respiratory tract infection (LRTI) enters the nuisance
regressions only. Its episode timing is unavailable and it may follow TB or
mediate an association; it is exploratory, not a baseline hazard confounder.
No ever-LRTI coefficient is added to the first-TB hazard.

## Joint likelihood and estimands

Each early model retains six Gaussian observation processes: resistance,
compliance, expiratory resistance, expiratory reactance, LCI and FEV1/FVC.
Resistance and expiratory resistance are reversed in the mechanics factor;
compliance and expiratory reactance retain their native orientation. LCI is
reversed and has its own child effect. Native FEV1/FVC is multiplied by 100
before residualization. Outcome summaries are therefore in percentage points,
with no second multiplication by 100.

For each process, the nuisance model includes scheduled visit, standardized
height, sex and the selected covariates. Height uses the complete process
sample mean and sample SD. Least squares is fitted to pre/no-TB observations,
then applied to the whole eligible process sample. The residual SD is
calculated on the training residuals, with degrees of freedom equal to the
number of nuisance-design columns. The selected M03 responses and matrices
are verified without importing historical scripts during production execution.

The shared mechanics child effect is standard normal. Its four positive
indicator loadings and residual SDs have HalfNormal(1) priors. LCI uses an
independent standard-normal child effect multiplied by a HalfNormal(1) scale,
a loading fixed at one and a HalfNormal(1) residual SD. The FEV1/FVC mechanics
loading is Normal(0, 1); its residual SD is HalfNormal(1). Post-early-TB and
additional infancy shifts, direct target effects and hazard coefficients have
Normal(0, 0.75) priors. The early log-baseline hazard prior has mean
log(0.045, 0.018, 0.006) and SD 1.25. The first-TB piecewise-exponential
likelihood links to the mechanics effect and selected baseline covariates.
The retained LCI process does not enter the FEV1/FVC mean or the hazard.
A separate target child intercept, `tau_target * target_raw`, is added to the
FEV1/FVC mean, with `target_raw ~ Normal(0,1)` and
`tau_target ~ HalfNormal(1)`. It is independent of mechanics in the prior
and does not enter any other process or the hazard. The all-age window fits
use the same revised target intercept in the [M04 graph](PRIMARY_MODELS.md).

Second-stage posterior intervals condition on the estimated nuisance
regressions and residual scales. They do not propagate first-stage estimation
uncertainty, whose effect need not have a uniform direction across estimands.
The hazard's higher-mechanics label means better mechanics on a unit-normal
factor. That factor is informed jointly by mechanics, target and first-TB
data, including later measurements; it is not a purely prospective pre-TB
measurement. Hazard probabilities are evaluated relative to a ratio of one.

Let s be the target residual scale, l its mechanics loading, m the early
mechanics shift, m1 the additional infancy shift, d the direct early effect
and d1 the additional direct infancy effect. Within each posterior draw:

- Age 1–<5 effect = s × (d + l × m).
- Infancy effect = s × (d + d1 + l × (m + m1)).
- Before-five effect = the weighted average of these effects using distinct
  target-measured children observed after TB in each diagnosis-age group.
- Infancy interaction = infancy effect minus age 1–<5 effect.
- Direct components = s × d and s × d1.

Weights are recomputed for each selected sample, including each observation
window. Infancy is nested within the before-five contrast. Both compare with
the modelled pre/no-TB state. The all-age family also retains later-TB,
any-TB weighted and later-versus-earlier contrasts. Spirometry and hazard
summaries use posterior means and equal-tailed 95% credible intervals;
hazard ratios exponentiate the appropriate coefficient without target scaling.

## Table inputs and attenuation

The new-fit outputs retain 64 spirometry contrast rows, 25 early-model hazard
rows and 300 process/exposure support rows. Selected table inputs are:

- Table S1: five early models × two contrasts = ten effect rows, with eight
  adjusted-versus-core attenuation values.
- Table S3: twelve model rows × two contrasts = 24 effect rows, including
  the two verified existing M04 full-follow-up comparisons.
- Table S4: two verified M10 scenarios × two processes × three attained-age
  bands = twelve effect rows.

Table S1 attenuation is calculated from full-precision posterior means as
`100 * (1 - abs(adjusted_mean) / abs(core_mean))`. It is a comparison of mean
magnitudes, not a mean of draw-level ratios. Negative attenuation is retained
as increased magnitude. Sign reversal is separately flagged rather than
hidden by the absolute values. A zero core mean is undefined; a zero adjusted
mean is a nondifferentiable point for the precision calculation. Neither can
pass the attenuation-precision gate. The core row is the reference.

For adjusted and core posterior means `a` and `c`, and their mean MCSEs
`SE_a` and `SE_c`, the conservative first-order MCSE bound is

```text
100 * SE_a / abs(c) + 100 * abs(a) * SE_c / abs(c)^2.
```

Cauchy's covariance bound makes this valid for any cross-run Monte Carlo
correlation within the first-order approximation; it does not presume that
different fits or seed labels are independent. It is an MCSE bound, not an
exact error bound or posterior interval. All eight adjusted comparisons must
have a bound at most one percentage point. Reporting uses approximate whole
percentages alongside the precision bound. Rounding alone does not establish
Monte Carlo stability. No attenuation credible interval is invented. These
mean-magnitude comparisons, potentially across different complete-case
samples, are not causal mediation estimates. Their sign is the reverse of
the historical source column describing percentage change in magnitude.

Table S4 retains posterior medians and equal-tailed 95% credible intervals
for scheduled-wheeze conditional odds ratios and recurrent-episode incidence
rate ratios after infancy TB, within attained-age bands 1–<3, 3–<5 and
5–10.5 years. The comparator is pre/no-TB time within each band; other
diagnosis ages are represented separately. These are the exact accepted
M10 estimates and samples, not the standardized probability ratios used for
Figure 3A. See [wheeze methods](WHEEZE_MODELS.md).

## Sampling, validation and execution

The revised initial specification is four chains, 2,200 tuning iterations,
target acceptance 0.95 and maximum tree depth 10. The five Table S1 fits
(core, maternal HIV, prenatal smoking, birthweight and all three baseline
covariates) each retain 12,800 draws per chain; the other five new fits retain
3,200. Seeds use the selected source seed or default plus the prespecified
offset 104729. `adapt_diag` starts the four mechanics-indicator and target
loadings at 0.25, 0.35, 0.45 and 0.55, with initial target residual SD 0.9.
These starts do not constrain the target-loading prior. All realized controls,
complete posterior arrays and numeric sampler statistics remain private.

For the early-model archives, saved arrays establish chains and retained
draws; realized seeds, tuning and initialization were not recovered. Source
defaults are not represented as realized historical controls, and the seed
offset alone does not prove historical seed independence. The selected
all-age window archives record their sampling controls.

Complete-posterior acceptance requires zero divergences and tree-depth hits,
maximum R-hat ≤1.01, minimum bulk and tail ESS ≥400 and BFMI ≥0.3, including
all child effects. Each new fit must also pass the fixed conditional target
covariance screen: 2,000 replicas using actual pre/no-TB child/repeat membership,
with both observed ICC and within-child mean square inside inclusive central
99% predictive bounds. At least 20 repeat children and 30 within-child degrees
of freedom are required; unevaluable samples block completion. This is a
model-adequacy screen, not a calibrated classical multiple-testing procedure.

The frozen corrective protocol permits one common extension of all five
Table S1 fits to 25,600 retained draws per chain only if all five initial fits
pass sampler and covariance-PPC gates but at least one attenuation MCSE bound
exceeds one percentage point. The model, priors, seed, tuning, initialization
and thresholds remain fixed. Every original posterior draw and non-timing
sampler statistic must match the longer sequence's prefix exactly. Initial
results and failed precision checks are preserved. Continued failure blocks
acceptance; no further extension, selected seed or relaxed gate is authorized
by this rule. The other fits are not extended under this precision provision.

The previous source-reproduction validation passed 267 mean/quantile
comparisons across 89 estimands and diagnostics over 31,910 posterior scalars.
Its ten fits used 3,200 retained draws per chain, needed no extension, and its
M12 synthetic suite had 729 passing tests. These are historical checks of
the previous model. The revised specification is not required to reproduce
those estimates. Independent review instead checks the revised 64 effect
rows, 25 hazard rows, 300 support rows, all 46 table-input rows, exact prior-fit
reuse, eight attenuation bounds and frozen adequacy criteria. Old-versus-new
effects, interval widths and null inclusion are descriptive comparisons.
Archived manuscript rounding remains a provenance check, not a new fit gate.

The cumulative command is `run.py --through M12` with the private input and
analysis configurations; `--sensitivity-sampling FILE` supplies explicit
controls, while earlier flags keep their module-specific meanings. Final
original-source and copied-source cumulative replays passed for the historical
release. The corrective revision requires its own complete reconstruction and
independent acceptance receipts bound to its reviewed source. It preserves the
separately approved [Figure S3 score-membership correction](PRETB_MODELS.md)
and its remaining tie/boundary conventions. Four antenatal flow counts remain
unavailable; neither historical nor revised model checks reconstruct them.
