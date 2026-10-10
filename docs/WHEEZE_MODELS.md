# Age-specific wheeze models

M10 uses the verified M03 respiratory contracts to fit the selected primary
sex-adjusted analysis and the analysis additionally adjusted for prenatal
maternal smoking and birthweight z-score. The two scenarios use the same model implementation. M12 reuses the
adjusted result after verifying its fit. The selected outputs feed Figure 3A, Table S4 and Figure S6;
manuscript layout and figure rendering remain in M13–M14.

## Observations and risk time

Follow-up begins at birth and ends at the earliest formal exit, age ten years
six months or configured administrative endpoint. Scheduled current-wheeze
responses are observed binary outcomes from scheduled contacts; missing
responses are excluded. Participant-initiated episodes arise from qualifying
positive sick/pneumonia contacts or dedicated positive wheeze contacts.
Consecutive qualifying contacts separated by no more than 30 days form one
episode, dated at the first contact. Scheduled responses are not episodes.

Separate scenario ledgers retain children, scheduled assessments and positive
responses, episode counts, recurrent person-time and first-TB events. Exact
aggregate cells, including counts below ten and observed cells with zero
events, are retained in the external analysis outputs.

Post-TB exposure begins on the first TB date, without an exclusion lag.
First TB before age one defines infancy TB; later first TB has separate
nuisance effects. Attained-age effect bands are 0–<1, 1–<3, 3–<5 and 5–10.5
years. All four are fitted; the latter three are reported. Recurrence risk
continues after TB, while the TB survival contribution stops at first TB.
Risk intervals split at baseline age boundaries and the exact TB age.
Events and observations at censoring are excluded.

The source uses a calendar ten-year-six-month censor date alongside a decimal
10.5-year observation bound. Calendar follow-up can extend slightly beyond
10.5 decimal years. Recurrence intervals retain that calendar follow-up; the
TB baseline exposure matrix ends at decimal age 10.5. The difference is
quantified in private validation evidence. Scheduled responses and episode
contacts must still be below decimal age 10.5.

## Joint probability model

A child-level standard-normal latent propensity links three processes:
scheduled Bernoulli current wheeze with a logit link, recurrent Poisson
episodes with a log person-time offset, and piecewise-exponential first-TB
survival. The same child covariates enter each process. Scheduled age uses
the five-column centered natural cubic spline prepared from the scheduled
training ages, with the original column means and sample standard deviations.
Recurrence and TB baseline rates use intervals bounded by 0, 0.25, 0.5, 1,
2, 3, 5, 8 and 10.5 years.

The prior distributions are:

| Parameter | Prior |
|---|---|
| Child latent propensity | Normal(0, 1) |
| Scheduled and episode latent scales | HalfNormal(1) |
| Scheduled intercept | Normal(−3, 1.5) |
| Scheduled age coefficients | Normal(0, 1) |
| Process covariate coefficients | Normal(0, 0.75) |
| Infancy-TB and other-TB age-band log effects | Normal(0, 1) |
| Episode baseline log rates | Normal(log(0.05), 1.5) |
| TB baseline log hazards | Normal(log(0.012), 1.5) |
| TB latent association | Normal(0, 0.75) |

Normal prior arguments are mean and standard deviation. The source coding
defines male as sex code 2 in this respiratory cohort. Birthweight
standardization is fixed before complete-case selection. Complete posterior
arrays include every child propensity and all model deterministics.

## Estimands

Conditional odds ratios describe scheduled current wheeze; episode incidence
rate ratios describe participant-initiated recurrent episodes. Age-pair
contrasts exponentiate differences of the corresponding log effects within
each posterior draw. Global age summaries use the posterior mean vector and
sample covariance of two log-effect differences, with a chi-square reference
on two degrees of freedom. These are posterior-Wald reference summaries.

Figure 3A uses a different transformation of the primary scheduled model.
For every posterior draw, probabilities under pre/no TB and post-infancy TB
are integrated over the same standard-normal latent distribution and the
same primary-cohort sex distribution. Each probability is averaged uniformly
over its attained-age band, and their ratio is then calculated. This is a
population-standardized ratio of current-wheeze assessment probabilities,
not cumulative incidence. It does not use the fitted child effects or the
latent distribution selected by TB status, and it is not a causal estimate.

Integration uses 80 Gauss–Hermite nodes for the latent propensity and 48
Gauss–Legendre nodes for age. Repeating with 120 latent nodes checks numerical
convergence. Each draw must preserve the direction of its conditional odds
ratio, with the probability ratio between that odds ratio and one.

## Sampling, diagnostics and validation

The selected fresh controls were fixed before inspecting fresh results:
four chains, 3,200 retained draws and 2,200 tuning steps per chain, target
acceptance 0.95, maximum tree depth 10 and `jitter+adapt_diag` initialization.
The primary and adjusted seeds are 2026195862 and 2026196862, respectively;
each adds 104729 to its documented source seed. The archived runs used four
chains with 800 retained draws, 1,000 tuning steps and target acceptance 0.93.

All posterior variables must have R-hat at most 1.01, bulk and tail effective
sample sizes at least 400, BFMI at least 0.3, and no divergences or maximum
tree-depth hits. Diagnostic acceptance and independent scientific reference
agreement are separate checks. The frozen scientific comparison uses 81
median and interval-endpoint comparisons across conditional effects, age
contrasts and primary standardized ratios. Each tolerance is four times the
combined, endpoint-specific Monte Carlo standard error of the archived and
fresh estimate; rounded manuscript values are not acceptance thresholds.

Posterior predictive checks use 200 evenly spaced retained draws and the
fitted child effects. A separate deterministic random seed generates binary
responses and Poisson episodes. Exports retain only cell-level replicate
aggregates, draw indices, observed support and predictive summaries. Checks
recompute these summaries and compare simulated means with analytic
conditional moments. Coverage findings describe model fit and are not a
requirement that every observed cell lie inside its predictive interval.

Raw fits and sampler metadata stay under private `wheeze_fits/` directories.
Separate `wheeze_models/` validation directories contain aggregate effects,
age contrasts, global summaries, support, predictive checks and standardized
ratios. Source, input, environment and sampling hashes bind raw fit receipts;
reporting hashes bind validation receipts. Resume verifies both and recomputes
aggregates. Cumulative success writes `wheeze_current.json` only after both
selected models and all preceding modules pass their execution checks.

The retained M09 covariance limitations and approved Figure S3 membership
correction are documented in [PRETB_MODELS.md](PRETB_MODELS.md). Four antenatal
flow counts remain unavailable. Full-release acceptance is specified separately
in [RELEASE.md](RELEASE.md).
