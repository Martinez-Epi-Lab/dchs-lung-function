# Visit-specific FeNO models

M11 implements eleven selected regressions from the verified M03 FeNO
contracts: five infancy-TB comparisons for Figure 3B and six distinct
measurement-timing comparisons for Figure S4. Both families passed independent
source/sample and full-precision numerical validation in the original release.
M14 renders their figures.

## Selected samples and comparisons

| Family | Nominal assessments | Selected exposure | Reference |
|---|---|---|---|
| Main Figure 3B | 12 months, 24 months, 8 years, 9 years, 10 years | First TB before age one, with measurement on or after diagnosis | Pre/no-prior-TB state at measurement |
| Supplement Figure S4, earliest visit | 6–10 weeks | First TB after measurement | No recorded TB |
| Supplement Figure S4, later visits | 12 months, 24 months, 8 years, 9 years, 10 years | First TB on or before measurement, at any diagnosis age | No recorded TB |

The source visit codes `96 months` and `FeNO` display as 8 years and 9 years,
respectively. These are nominal assessment labels, not elapsed time since TB.
The main and supplementary analyses have separate sample rules and reference
groups; their observations and effect estimates are not interchangeable.

All samples require positive FeNO and a dated test. Duplicate child/test-date
records are resolved before fitting, favoring an explicitly labelled FeNO
visit. The selected duplicate decision is retained in private preparation
evidence. Natural-log FeNO is the response; nonpositive values do not enter
the logarithm. Complete-case selection uses the outcome, selected exposure
terms, sex, exact age and current height.

The main analysis joins eligible measurements to the fixed respiratory cohort.
First TB before age one defines infancy TB. A measurement on the diagnosis
date is post-TB. Later-onset post-TB observations receive a separate indicator
when present; measurements before later TB remain in the pre/no-prior-TB
reference state. At the four later selected visits, exact age is at least
one and below 10.5 years, and measurement precedes the child's censor date.

The selected 12-month result uses `nominal12_sex_age_height`. It retains
nominal 12-month attendance before the first birthday, provided it precedes
censoring, and excludes measurements taken before diagnosis in children who
had infancy TB. It does not apply the generic exact-age-at-least-one mask.
The other four main results use `sex_age_height`. The other adjustment
scenarios produced by the source script are outside this selected figure.

The supplementary analysis uses its own validated PFT visit samples without
adding the main respiratory-cohort, exact-age or censor restrictions. It
classifies each measurement as on/after first TB, before a later first TB,
or from a child with no recorded TB. Both observed TB categories enter the
model as separate terms when present. At 6–10 weeks, the selected comparison
is future TB because no recorded TB preceded measurement in the selected
sample. Later selected estimates use the post-TB coefficient; future-TB
measurements remain a separate adjustment category. This analysis is not
restricted to diagnoses during infancy.

## Regression and uncertainty

Each visit uses a separate Gaussian generalized estimating equation with an
identity link for natural-log FeNO, an independence working correlation and
child-level clustering. The mean model contains an intercept, the selected
exposure indicators, sex, exact assessment age and current height. Main
age/height columns are centered and divided by their sample standard
deviations after complete-case selection. Supplementary age/height scaling
occurs within the nominal-visit frame before complete-case selection. M11
uses the validated M03 design matrices and preserves this distinction.

For design matrix X, response y and fitted coefficients beta, let
A = X'X and let s_g = X_g'(y_g − X_g beta) be the score summed within child g.
The source robust covariance is

`inverse(A) × sum_g(s_g s_g') × inverse(A)`.

Its finite-sample multiplier is one. There is no HC1, HC2, HC3, cluster-count
or residual-degrees-of-freedom correction to that sandwich. Residual scale
is `sum((y − X beta)^2) / (n − p)`, where p is the number of design columns;
the separately retained naive covariance is that scale times `inverse(A)`.
The Gaussian-independence coefficient solution also agrees with ordinary
least squares, which provides an independent numerical check.

The main source allows 60 fitting iterations and the supplementary source
250; both use convergence tolerance 1e-6 and robust covariance. Confidence
limits are `exp(beta ± 1.96 × robust SE)`, and two-sided p-values use the
standard-normal Wald reference. No t reference or multiplicity adjustment is
applied. These are frequentist confidence intervals, not Bayesian credible
intervals. Exponentiated exposure coefficients are adjusted geometric mean
ratios (GMRs); percentage differences are `100 × (GMR − 1)`. They are not
wheeze risk ratios, odds ratios or incidence rate ratios.

The selected source uses uncorrected sandwich covariance and normal Wald
inference even for small exposure categories. That method is preserved
explicitly. Reproducing it does not establish general small-sample coverage
or a causal effect. Exact aggregate counts, including zero and counts below
ten, remain in the private aggregate outputs.

## Outputs, checks and execution

The selected output contains eleven estimate rows, 63 coefficient rows,
726 full robust/naive covariance cells and 33 exposure-support rows. Three
exposure categories are retained for every fit. Labels retain the family, visit, nominal age,
selected term and comparison. A missing exposure contrast receives an
explicit non-estimable status with its support retained; numerical failures
block successful completion.

Independent validation reconstructs coefficients using QR, accumulates the
child score covariance directly, and checks both complete covariance matrices,
standard errors, transforms, intervals, p-values, model counts and source
selections. Floating-point comparisons use relative and absolute tolerances
of 1e-10 fixed before results were inspected. Rank, sample membership, term
order and counts are checked exactly. Unscaled estimating-equation score
norms must be below 1e-6. Archived outputs contain selected contrasts;
complete fresh matrices are checked against independent equations, not
claimed to have archived full-matrix references. Manuscript rounding is a
separate check and cannot substitute for full-precision agreement.

`--through M11` is the cumulative execution target. It verifies M01–M10
before fitting the eleven FeNO regressions. M11 performs no posterior
sampling; earlier sampling flags retain their meanings for preceding modules.
Private aggregate outputs under `feno_models/<fingerprint>/` contain
`estimates.csv`, `coefficients.csv`, `covariance.csv`, `counts.csv`, diagnostics,
summary and receipt files. Every resume recomputes the deterministic fits and
verifies the complete publication inventory. Source, prepared contracts and
runtime identities bind the result. Successful cumulative execution writes
`feno_current.json` only after all required preceding checks pass.

All eleven selected fits passed independent numerical validation: 2,368
checks across 65,293 comparison cells, including complete coefficients and
covariance matrices. The prior source/reference review passed 365 checks,
and all 15 rounded main-manuscript estimate/interval values agree with the
selected calculations. The M11 baseline synthetic suite had 621 passing tests.
Successful cumulative execution from the original source tree and an
identical source copy remains a delivery acceptance requirement; numerical
fit validation alone does not establish that replay has completed.

The retained pre-TB covariance limitations and approved Figure S3 membership
correction are documented in [PRETB_MODELS.md](PRETB_MODELS.md). Four upstream
antenatal flow counts remain unavailable. Infant and school-age FeNO were measured in
different testing contexts, with no selected assessments between approximately
ages two and eight. These visit-specific associations do not describe a
continuous FeNO trajectory or establish a shared mechanism with wheeze.
