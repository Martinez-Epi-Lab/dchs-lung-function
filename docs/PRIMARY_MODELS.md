# Primary spirometry joint models

M04 implements twelve continuous fits: pre/post-bronchodilator FEV1/FVC,
FEV1 and FVC, each on its native and GLI z-score scales. It consumes the
M03 primary contracts. It does not substitute a pooled sample for the
outcome-specific available-case populations or refit the nuisance trends
using post-TB observations.

All twelve primary fits in the preserved V3 corrective reconstruction passed
sampler diagnostics and the target-outcome covariance screen. Exploratory
checks identified covariance problems in auxiliary mechanics processes, so
these passes do not establish adequacy of the complete joint model. The
[corrective revision](CORRECTIVE_REVISION.md#current-status) remains under review.

## Statistical specification

Each fit contains six Gaussian observation processes and a piecewise
exponential first-TB process. Four mechanics indicators share a child factor
with standard Normal prior. Their loadings and residual standard deviations
have HalfNormal(1) priors. The source orientations represent better mechanics
with higher factor values: resistance and ReE are reversed; compliance and
XeE retain their signs.

LCI has a separate child process: a standard Normal child effect multiplied
by a HalfNormal(1) scale. Its observation residual SD also has a
HalfNormal(1) prior and its loading is fixed at one. This
process does not enter either the spirometry target or the TB hazard.

Mechanics and LCI each have post-TB shifts for diagnosis before five years,
an additional infancy shift, and a separate shift for diagnosis at five years
or later. The spirometry target has a Normal(0,1) mechanics loading and direct
post-TB terms with the same age coding. Each shift, direct term and hazard
covariate coefficient has a Normal(0,0.75) prior. Target residual SD has a
HalfNormal(1) prior. The target also has a separate child intercept
`target_process = tau_target * target_raw`, with `target_raw ~ Normal(0,1)`
and `tau_target ~ HalfNormal(1)`. This effect and the mechanics factor are
independent in their priors; they need not be independent in the posterior.
It enters only the target mean, not the mechanics indicators, LCI or TB hazard.
The added intercept allows repeated target measurements to share variation
beyond the mechanics factor. It does not change the algebra of the reported
TB contrasts.

The log hazard contains the mechanics factor, male sex and an interval
intercept. Its age boundaries are 0, 1, 3, 5, 8, 10 and 20 years. The six
intercepts have Normal priors with SD 1.25 and means equal to the logarithms
of 0.045, 0.018, 0.006, 0.006, 0.006 and 0.006. M03 supplies exact exposure
durations and event indicators. The likelihood sums event log hazards minus
integrated hazards; it does not treat observed TB ages as Gaussian outcomes.
Follow-up ends at first TB or the earlier of study exit and the configured
administrative endpoint. Lung measurements are retained through study exit;
an additional administrative measurement cutoff applies only to the designated
restricted-follow-up analyses.

The mechanics factor is estimated jointly using the available mechanics,
target and first-TB processes, including later measurements. Its hazard
association is therefore a joint-model association, not a purely prospective
prediction from a factor measured only before TB. A higher factor represents
better mechanics. Adding a separate target intercept does not determine the
direction or size of any change in the loading or TB effects.

Every target total association combines its direct term with the product of
the target loading and mechanics shift, calculated for each posterior draw.
Pooled contrasts use counts of distinct children with post-TB target
observations. Multiplication by the pre-TB nuisance-model residual SD saved by M03 restores
reported units exactly once; the fitted observation SD is not used for this
conversion. Native FEV1/FVC is already in percentage points at that
stage. Eight contrast arrays are retained, including three age groups, two
pooled contrasts and three age-group differences.

Nuisance regressions and their residual scales are estimated in preparation
and then treated as fixed in the joint fit. Posterior intervals are conditional
on those first-stage estimates; they do not propagate their estimation
uncertainty. The direction and magnitude of any omitted uncertainty depend
on cross-stage covariance and the estimand, so no universal interval-inflation
factor is applied.

## Sampling and acceptance

`config/primary_sampling.json` fixes four chains, 1,600 tuning iterations and
4,800 retained draws per chain, target acceptance 0.95, maximum tree depth
10, and `adapt_diag` initialization. The four positive initial loading
values are 0.25, 0.35, 0.45 and 0.55; initial target SD is 0.9. The selected
target seeds are versioned in `fit_spec`; the new-run configuration adds a
fixed 104729 offset. Reusing a seed does not itself establish biased inference,
and different seed labels do not establish independence between estimates.
Sampling controls and realized counts are retained with the private draws.
Thread controls request one
BLAS thread before numerical libraries load; observed library capabilities are
recorded separately, and Apple Accelerate's actual thread count is not observable
through threadpoolctl. Two chains run simultaneously by default.

The 4,800-draw setting is a separately recorded diagnostic-remediation amendment
after two initial 1,200-draw fits exceeded the unchanged R-hat limit. It applies
to all twelve primary models with the original tuning, seeds and initialization.
Exact original posterior and non-timing sampler-statistic prefixes must be
verified; early draws are not discarded. This amendment does not imply that
longer sampling guarantees convergence, and no further automatic primary
extension is authorized by this specification.

Acceptance requires zero divergences and zero retained maximum-tree-depth
hits, rank-normalized R-hat at most 1.01, bulk and tail ESS at least 400, and
BFMI at least 0.3 in every chain. R-hat and ESS checks cover every posterior
scalar, including every child effect. Missing variables, incomplete latent
coordinates and nonfinite diagnostics fail validation. These are this
repository's prospective acceptance rules; they extend the recorded
reference-run checks.

The [corrective protocol](CORRECTIVE_REVISION.md) adds a separate covariance
adequacy screen. It selects target measurements in the pre/no-TB state and
verifies that every modeled post-TB term is zero. Using the actual child and
repeat pattern, it generates 2,000 conditional Gaussian replicas from evenly
spaced saved posterior draws and a target-specific fixed seed. Both the
untruncated unequal-repeat ANOVA intraclass correlation and within-child mean
square must fall inside their inclusive central 99% predictive intervals.
At least 20 children with repeated measurements and 30 within-child degrees
of freedom are required. A sample that cannot be evaluated blocks completion;
it is not a pass. These checks are an adequacy screen, not a calibrated
multiple-testing procedure or a replacement for convergence diagnostics.

The prior source-reproduction validation checked all 96 target/contrast
combinations. For each posterior mean and each 95% interval endpoint, the
allowed absolute difference is four times the square root of the sum of
the two runs' squared Monte Carlo standard errors. Endpoint checks use
quantile-specific MCSE, as supported by [ArviZ 0.22.0](https://python.arviz.org/en/v0.22.0/api/generated/arviz.mcse.html).
This gave 288 prespecified comparisons for the previous model. Matching rounded manuscript values
alone is insufficient. Reference posterior arrays contain global variables;
original latent draws and sampler statistics are unavailable, so historical
all-parameter diagnostics are supported by their saved audits. All new
latent draws and sampler statistics are retained privately.

Those historical checks establish source reproduction, not adequacy of the
previous covariance structure. Revised fits use a changed target likelihood
and must pass the frozen diagnostics and covariance screen. Old-versus-new
effect, interval and null-inclusion comparisons are descriptive audits; old
estimates are not targets that the revised model must match.

If a fit fails, retain its evidence and use a separately fingerprinted,
reviewed sampling configuration for any longer run. Do not change the model,
drop difficult parameters, choose seeds by their results or relax an acceptance
threshold to obtain a pass.

## Runtime and saved runs

`requirements-models.lock.txt` includes preparation dependencies and the
selected PyMC 5.25.1, PyTensor 2.31.7 and ArviZ 0.22.0 versions. The validated
environment is Python 3.12 on macOS ARM64 with the system Clang toolchain.
Other platforms require their own clean-install and compiler validation.
The small runtime adapter removes PyTensor 2.31.7's unsupported `-ld64`
argument on macOS. It changes compilation configuration, not model math.
Compiler caches remain in private work directories. Start the command in
a fresh process so the private runtime is configured before importing PyMC.

The sampling fingerprint includes the actual prepared contract, model and
sampling code, environment lock, runtime and resolved sampling controls.
The validation fingerprint separately includes summary/diagnostic code and
acceptance criteria. A changed input, model or sampler cannot reuse the old
raw fit. A changed diagnostic or reporting implementation can revalidate
verified raw draws without new sampling.

Raw files have `sampled` status. Aggregate results retain separate sampler
diagnostic and covariance-PPC acceptance fields. Either failure remains explicit
and prevents a successful batch pointer. A convergence pass does not establish
model adequacy. Complete reconstruction and independent scientific review are
additional acceptance requirements recorded in private evidence.

`--resume` verifies posterior and sampler archives, metadata, complete model
variable inventory and result files. It does not continue an interrupted
chain from its last draw. An interrupted unsaved fit is sampled again.
Directory-descriptor locks remain held by the live owning process and release
on its exit, including abnormal termination. Lock files from the prior protocol
require explicit verification before migration; they are not silently removed.
Incomplete staging output is never accepted as a completed fit.

## Results

`estimates.csv` contains full-precision means, medians, equal-tailed 95%
credible intervals, probabilities, MCSE and sample/weight metadata. Selection
fields identify 24 Table 2 estimates, 60 Table S6 estimates and eight
continuous Figure 2 estimates across the twelve fits. `shared_factor.csv`
also preserves the loading summaries used by M07. M13/M14 own
final manuscript rendering; the result interface does not round or edit the
manuscript. Exact small aggregate counts are retained.
