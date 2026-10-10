# Spirometry trajectories

M06 fits native FEV1/FVC (percentage points) and GLI FEV1/FVC z-score trajectories for first TB before age one and before age five. These four analyses each have a full model and a separately fitted parallel-trajectory model. Figure 1 uses the full-model predictions. Parallel differences used in numerical prose come from the reduced models. M14 renders the figures; M08 fits the X6 trajectories.

## Sample and design

The verified M03 contracts use the common sample with native and GLI ratios available, exact measurement age from 5 through 10 years inclusive, and complete date, height and sex covariates. Native ratios are converted to percentage points in preparation. Exposure starts on the diagnosis date, with strict diagnosis-age thresholds. Post-TB measurements after a first diagnosis at or above the cutoff, and prediagnosis measurements from children with future TB, have separate nuisance terms. No-recorded-TB measurements define the reference group.

The full design contains the intercept, target TB, later TB, future TB, standardized height, male sex, three centered natural-cubic age-spline columns, and three target-TB-by-age-spline interactions. Patsy constructs `cr(age, df=3, constraints="center", lower_bound=5, upper_bound=10) - 1` on the training sample. Prediction uses that training basis and its stored knot/centering state. The reduced design removes only the three interaction terms. Both fits retain the identical sample.

## Estimation and uncertainty

A Gaussian random-intercept model estimates the within-child covariance. Let `lambda = tau2 / sigma2`, where `tau2` is the random-intercept variance and `sigma2` is the residual variance. For each candidate lambda, generalized least squares estimates the fixed effects. The profiled REML objective is

```text
(n - p) * log(SSE / (n - p))
  + sum_i log(1 + n_i * lambda)
  + log(det(X' V_lambda^-1 X)).
```

The selected search evaluates zero and 90 positive candidates on an equally spaced log grid from -10 to 12. A positive grid minimum is refined using the neighboring positive grid bracket and 60 golden-section iterations; a zero minimum remains zero. Fit metadata retains the search diagnostics, variance components, sample size, conditioning and acceptance status.

The model covariance is `sigma2 * inverse(X' V_lambda^-1 X)`. Reported uncertainty uses the child-score sandwich with correction `G/(G-1) * (n-1)/(n-p)`. No additional residual-variance multiplier is applied to this sandwich. Confidence intervals use estimate plus or minus 1.96 robust standard errors.

The full fit supplies a three-degree-of-freedom interaction Wald test and a four-degree-of-freedom test of the target main effect plus its interactions. The reduced fit supplies the constant target difference, interval and two-sided normal p-value. These are raw p-values. No multiple-testing family is redefined for this selected subset, and no likelihood-ratio comparison is made between REML fits with different fixed-effect designs.

The normal and chi-square reference distributions are asymptotic. The finite-cluster sandwich multiplier does not itself establish small-sample calibration, particularly for sparsely supported exposure groups or influential children. It is not a cluster-degrees-of-freedom t or F reference calculation.

## Standardized predictions

Predictions use 101 equally spaced ages from 5 to 10, with two groups at each age. Later and future TB indicators are zero in both groups. Sex is standardized to the measurement-weighted male proportion. Expected height is estimated by ordinary least squares using cubic age centered at 7.5 and male sex, then transformed with the training measurement-level height mean and sample standard deviation. The same expected height and sex values are used for both groups.

Prediction bands use the full robust fixed-effect covariance and quantify uncertainty in the population mean. They are pointwise 95% intervals, not simultaneous bands over the age grid or individual prediction intervals. They condition on the fitted height-standardization relationship, its mean/SD and the observed sex proportion; uncertainty in that first stage is not propagated into the absolute curves. Six exposed-minus-reference contrasts at integer ages 5 through 10 use the covariance of the difference vector, preserving the covariance between the two predictions. Common height and sex terms cancel exactly in that difference. The full-fit interaction Wald test does not use the predicted height relationship. These contrasts are distinct from the reduced model's constant parallel difference.

## Execution and artifacts

```sh
.venv/bin/python run.py --through M06 --config config/paths.local.json --analysis config/analysis.local.json --resume
```

The cumulative command verifies M01-M05 before fitting M06. It writes aggregate artifacts outside the repository in `trajectory_models/<fingerprint>/`: four global-test rows, 808 prediction rows, 24 age-contrast rows, 84 coefficients, 1,800 robust/model covariance cells, diagnostics, summary and completion receipt. Coefficients and covariance retain both full and reduced fits. Participant records and identifiers are never included in these aggregate artifacts.

Fingerprints bind the selected M03 contracts, model and execution source, dependency lock and runtime. Resume verifies all artifact hashes, complete row inventories, fit statuses, sample-count consistency, covariance validity and inferential relationships. Changed dependencies require a different run identity. Numerical failures retain explicit failure evidence and do not publish the cumulative success pointer. `trajectory_current.json` is written only after all four analyses and the cumulative prerequisites pass.

## Verification

Synthetic tests cover sample and exposure boundaries, training-basis predictions, unequal visit counts, height/sex standardization, correlated contrast uncertainty, variance-component boundaries, covariance calculations, numerical failures and artifact invalidation. Private scientific validation independently reconstructs the selected sample, fits, full/reduced coefficients, both covariance matrices, tests, predictions and age contrasts. Its comparison protocol is fixed before inspecting the new outputs. Private manuscript checks distinguish full-model curves from reduced-model prose differences. The corrective revision does not substitute the continuous joint model for these separately fitted trajectories or apply a universal interval-width adjustment. Current model and publication acceptance is bound to its own reconstruction receipts.
