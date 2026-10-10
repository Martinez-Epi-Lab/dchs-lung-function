# Mechanics models and shared-factor summaries

M07 fits six individual mechanics/first-TB joint models for Table S5 and the retained mechanics prose. It also derives the six Figure S5 shared-factor associations from accepted M04 posterior draws. Tables and figures are rendered by M13 and M14.

## Selected outcomes and samples

| Target | Measurement | Orientation for fitting | Native unit | Reporting unit |
|---|---|---:|---|---|
| R | Model-derived resistance | +1 | cmH2O*s/L | cmH2O*s/L |
| C | Model-derived compliance | -1 | L/cmH2O | mL/cmH2O |
| ReE | End-expiratory resistance | +1 | hPa*s/L | hPa*s/L |
| ReI | End-inspiratory resistance | +1 | hPa*s/L | hPa*s/L |
| XeE | End-expiratory reactance | -1 | hPa*s/L | hPa*s/L |
| XeI | End-inspiratory reactance | -1 | hPa*s/L | hPa*s/L |

M03 constructs a separate available-case sample for each outcome using its mechanics measurement date, height and verified sex encoding, retaining observations through study exit. The linked child cohort contributes to the first-TB process even when a child has no valid measurement of a particular outcome. Each contract retains its observation indices, nuisance design, residual scale, event arrays and outcome-specific exposed-child weights.

For each outcome, ordinary least squares on pre/no-TB measurements removes the effects of visit, sample-standardized height and male sex. The adverse-oriented residuals are divided by the pre/no-TB residual standard deviation, using `ddof` equal to the number of nuisance-design columns. Its variance denominator is the training sample size minus that number. Same-day measurements are post-TB. First-diagnosis age below one is nested within below five; diagnoses at or above five form a separate category. Source date and cohort rules remain in the verified preparation module.

The joint fit treats those nuisance coefficients and residual scales as fixed. Its posterior intervals are conditional on their first-stage estimates; their estimation uncertainty is not propagated. No universal direction or inflation factor for that omission is assumed.

## Joint model

For child `i`, the shared outcome effect is `b_i = tau_child * child_z_i`, where `child_z_i ~ Normal(0, 1)` and `tau_child ~ HalfNormal(1)`. Standardized mechanics measurements follow a normal distribution with residual scale `sigma_outcome ~ HalfNormal(1)` and mean

```text
b_i + shift_1to5 * post_before5
    + shift_infancy_additional * post_infancy
    + shift_late * post_at_after5.
```

The piecewise-exponential first-TB process uses age intervals `[0,1)`, `[1,3)`, `[3,5)`, `[5,8)`, `[8,10)` and `[10,20)`. Its log hazard is

```text
log_baseline_hazard[interval] + assoc_outcome * b_i + hazard_male * male_i.
```

The three shifts, association and sex coefficient have independent `Normal(0, 0.75)` priors. Log baseline hazards have standard deviation 1.25 and prior means equal to the logarithms of `0.045, 0.018, 0.006, 0.006, 0.006, 0.006`. Event likelihood contributions use the complete child-specific exposure intervals and first-event indicator. Each of the six models has its own outcome-specific child effect.

These six graphs already have a target child intercept, so they do not receive a second one in the corrective revision. Follow-up is bounded by study exit and the administrative event endpoint. The child effect is estimated jointly from the complete available measurements and first-TB process, including later measurements. Its hazard association is not a purely prospective prediction from measurements taken only before TB.

## Contrasts and units

The infancy effect is `shift_1to5 + shift_infancy_additional`; the one-to-under-five effect is `shift_1to5`; the later effect is `shift_late`. Pooled before-five and any-TB contrasts use the numbers of distinct children with post-TB measurements of that outcome. All weighted effects and differences are constructed draw by draw.

The aggregate output retains eight contrasts per outcome: the three age categories, both pooled effects, infancy minus one-to-under-five, later minus infancy, and later minus one-to-under-five. Table S5 selects the first five, giving 30 estimate cells (six measures by five contrasts). Native draws are standardized draws times the outcome residual scale divided by its fitting orientation. Compliance reporting then multiplies native draws by 1,000 once. Means and equal-tailed 95% posterior intervals are calculated after transformation, so negative orientations reverse interval order correctly. Probability of worse mechanics is evaluated on the adverse-oriented draws.

Six additional aggregate hazard summaries retain the source transformation `exp(assoc_outcome * tau_child)`, the first-TB hazard ratio per between-child SD poorer mechanics. Posterior probabilities for a hazard ratio use one as their null, with separate below, equal and above-one fields; the probability correction does not change the point or interval transformation. Exact observation and child counts, including small or zero groups, are retained.

## Combined shared-factor associations

Figure S5 selects the pre/post GLI z-score models for FEV1/FVC, FEV1 and FVC. Their M04 mechanics factor has unit variance and is oriented toward better combined respiratory mechanics. For each selected posterior draw, the association with one model-defined SD poorer mechanics is

```text
-loading_ratio_mechanics * target_residual_scale.
```

The six summaries retain their own outcome-specific samples and scales. They describe jointly informed model associations; the factor is informed by the fitted respiratory mechanics, school-age spirometry and first-TB process. They do not establish a pre-TB-only prediction or a within-visit bronchodilator response. The derived-result runner requires existing complete accepted primary fits and has no sampling path. It verifies full posterior inventories, realized controls, all-variable diagnostics and covariance-PPC acceptance before reading the loading draws. The revised primary fits include their separate target child intercept; Figure S5 therefore uses the newly estimated loading rather than substituting a historical value. The direction or magnitude of that change is assessed from the results, not assumed.

## Sampling, diagnostics and verification

The mechanics defaults in `config/mechanics_sampling.json` retain four chains, 1,600 tuning iterations and 1,200 retained draws per chain, `jitter+adapt_diag` initialization, target acceptance 0.95 and maximum tree depth 10. Source seeds and the configured offset are recorded; seed labels alone do not establish cross-fit independence. All 16 posterior variables are retained privately, including both child-effect arrays. The seed, initialization, controls, actual chain/draw/tuning counts, runtime and sampler statistics are recorded for every fit.

Acceptance requires zero divergences and tree-depth hits, maximum rank-normalized R-hat at most 1.01, minimum bulk and tail ESS of 400, and minimum BFMI of 0.3. These gates cover every posterior scalar, including child effects. Global parameter traces are inspected privately. Failed fits retain their raw evidence and cannot publish cumulative success.

The corrective acceptance also applies the fixed conditional covariance screen to these six target processes. Using actual pre/no-TB child membership and repeats, 2,000 saved-draw replicas must place both observed ANOVA ICC and within-child mean square inside central 99% predictive intervals. Samples need at least 20 repeat children and 30 within-child degrees of freedom; unevaluable results do not pass. The screen tests repeated-outcome adequacy separately from sampler convergence and is not a calibrated multiple-testing procedure.

Independent source checks reconstruct the samples and compare complete model log densities, gradients and deterministics. The previous source-reproduction review used four combined Monte Carlo standard errors, with quantile-specific errors for interval limits. Its 180 comparisons covered 144 mechanics contrast statistics, 18 hazard statistics and 18 shared-factor statistics. Those historical counts do not establish acceptance of the corrective revision or require changed primary-factor summaries to match old values. Current review separately verifies frozen covariance criteria, sampler diagnostics, transformations, display selections and exact outcome-specific counts, then reports old-versus-new changes descriptively.

## Execution and artifacts

```sh
.venv/bin/python run.py --through M07 --config config/paths.local.json --analysis config/analysis.local.json --resume
```

`--sampling FILE` controls preceding M04 fits; `--mechanics-sampling FILE` controls the six M07 fits. `--jobs 2` or `--jobs 3` enables parallel model fits. Both configurations enforce the minimum accepted sampling controls.

The cumulative command verifies M01-M06 before completing M07. Raw fits are stored privately under `mechanics_fits/<target>/<sampling fingerprint>/`. Aggregate diagnostics, estimates, hazard summaries, counts and metadata are stored under `mechanics_models/<target>/<validation fingerprint>/`. Six shared-factor rows and their primary-fit bindings are stored under `shared_factor/<fingerprint>/` in the aggregate directory. The repository contains only code, documentation, example configuration and synthetic tests.

Sampling and validation have separate fingerprints, allowing reporting or validation corrections to recheck unchanged raw draws. Publication uses locks, staging directories and atomic publication. Resume checks complete inventories, file hashes, realized sampling metadata and recomputes aggregates, diagnostics and covariance checks from the verified draws. The cumulative `mechanics_current.json` pointer is written only after all required models and shared-factor summaries pass. Full reconstruction and source delivery have separate acceptance receipts.
