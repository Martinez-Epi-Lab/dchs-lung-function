# Corrective scientific and release revision

This revision addresses model-adequacy, follow-up, numerical-precision and
publication defects identified after the original source-reproduction release.
The original release and its private results remain an immutable comparison.
Faithful reproduction of those results did not establish that their repeated
measurement covariance or ascertainment assumptions were adequate.

## Scientific specification

The 29 continuous target/shared-mechanics joint models add an independent
target-specific child intercept. The standardized target mean includes
`tau_target * target_raw[child]`, with `tau_target ~ HalfNormal(1)` and
independent `target_raw ~ Normal(0,1)`. The shared mechanics factor retains its
unit variance and orientation. Existing direct/shared TB contrast definitions
are unchanged. The additional intercept permits target-specific repeated
measurement covariance without forcing a larger cross-outcome loading.

Threshold-family first-TB risk time ends at the earliest eligible diagnosis,
study exit, administrative endpoint or threshold birthday. Diagnosis must
remain strictly before the first or fifth birthday; diagnosis on the observed
exit or administrative endpoint is included. There is no assumption of complete
post-exit surveillance. The lung-function observation window is a separate
definition: joint-model measurements respect study exit, while only designated
restricted-window analyses truncate lung measurements at the TB administrative
endpoint. These distinctions preserve the intended age-specific estimands.

The approved Figure S3 actual-risk-membership uncertainty correction remains.
The retained Cox tie and risk-boundary conventions are documented separately;
this revision does not restore legacy uncertainty. Existing source-specific
oscillometry unit conversions remain unchanged.

## Prospective validation

The candidate is first evaluated on pre-bronchodilator native FEV1/FVC, its GLI
z-score and FEV1 z-score: the three targets with independently demonstrated
covariance underfit. The protocol was frozen before revised-fit inspection.
All continuous joint models also receive a reproducible posterior predictive
screen on their actual pre/no-TB children and measurement multiplicities.

For 2,000 deterministically selected posterior draws, replicate conditional
Gaussian target observations using saved child effects. Check the unequal-size
ANOVA intraclass correlation and within-child mean square against their central
99% predictive intervals. Eligibility requires at least 20 children with repeat
measurements and 30 within-child degrees of freedom. Insufficient information
is explicitly `not_evaluable`, never an unqualified adequacy pass. This screen
is model criticism, not a calibrated classical multiple-testing procedure.

Sampler gates remain four chains, rank-normalized R-hat at most 1.01, bulk and
tail ESS at least 400, every-chain BFMI at least 0.3, no divergences and no
maximum-tree-depth hits. All child effects are included. Revised estimates must
be compared with the original release, but agreement with a superseded model
is not an acceptance requirement. No fixed percentage inflation is applied to
intervals, and no assumed direction of loading bias selects the result.

### Primary diagnostic-remediation amendment

Two primary fits in the initial 1,200-retained-draw reconstruction failed the
unchanged R-hat limit while passing the other sampler and covariance gates.
Independent chain reviews found low-frequency location imbalance and no
demonstrated separate mode in the inspected global projections. These fits
remain failures; passing predictive checks does not override poor convergence.

A new amendment, specified after those initial results and before further
sampling, fixes 4,800 retained draws per chain for all twelve primary models.
The model, 1,600 tuning iterations, seeds, initialization, target acceptance and
all gates remain unchanged. Preserve the initial failed run, verify every
1,200-draw posterior and non-timing sampler-statistic prefix exactly, and examine
later-block stability alongside complete diagnostics and predictive checks.
There is no interim stopping, early-draw removal, seed selection or further
automatic primary extension. If acceptance still fails, retain the evidence
and reassess the cause. This is distinct from the initially prespecified Table
S1 precision extension below.

## Attenuation and Monte Carlo precision

Attenuation remains `100 * (1 - abs(adjusted mean) / abs(core mean))`. It is a
comparison of posterior mean magnitudes, not a posterior distribution of a
draw-wise ratio, a mediation estimand or proof of confounding. Sign reversals
are flagged; a zero core mean is undefined. The computational uncertainty of
both posterior means must accompany the approximate whole-percentage display.

A first-order, covariance-agnostic upper bound for its Monte Carlo standard
error is

```text
100 * MCSE_adjusted / abs(core_mean)
  + 100 * abs(adjusted_mean) * MCSE_core / abs(core_mean)^2
```

This Cauchy bound does not presume independent errors merely because models
have different specifications. It relies on estimated marginal MCSEs and the
delta approximation; it is not a confidence or credible interval and does not
guarantee identical rounded digits in independent runs. The prospective target
is at most one percentage point for every selected attenuation.

The five Table S1 fits initially retain 12,800 draws per chain. Other sensitivity
fits retain 3,200. If all five S1 fits pass the sampler and predictive gates but
any attenuation exceeds the precision bound, one common extension to 25,600
draws is permitted. Seeds, tuning and initialization stay fixed; retained draw
and non-timing sampler prefixes must reproduce exactly. Further failure remains
explicit and cannot be resolved by changing seeds or relaxing the gate.

Existing per-target seed assignments are retained, including assignments shared
by related specifications. Unique seeds are not required for valid marginal
fits and would not by themselves establish independent Monte Carlo errors.
The conservative precision bound above allows arbitrary cross-fit correlation;
same-seed stability across specifications is not used as evidence of precision.

## Interpretation and release scope

Exact small aggregate counts, estimates and intervals remain available.
Sparse-event binary/Cox inference is qualified rather than suppressed.
Trajectory bands are pointwise population-mean intervals. First-stage nuisance
estimates and scales remain plug-in quantities: unconditional two-stage
uncertainty has not been established by this correction. Common height and sex
terms cancel from the additive trajectory differences, though uncertainty in
those standardization choices can affect absolute curves.

Final release acceptance requires independent calculation and sample checks,
actual exported figure-content bounds, a clean extracted-source test run,
verified resume behavior and hosted checks. Raw data, participant effects,
manuscripts, licensed fonts and private audit artifacts are excluded from the
source release. Public visibility and a reuse license remain separate choices.
