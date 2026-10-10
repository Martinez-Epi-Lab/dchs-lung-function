# Binary spirometry models

M05 fits the selected binary spirometry analyses using the M03 outcome-specific
contracts. Pre- and post-bronchodilator analyses use their own available
measurements. No common pre/post sample or additional exact-age restriction is
imposed. Test dates are retained through study exit, as in the selected
preparation specification.

## Outcomes and exposure

Below the lower limit of normal (LLN) means a GLI z score strictly less than
−1.645. A value equal to −1.645 is not below LLN. The ratio, FEV1 and FVC binary
outcomes each require their own observed score. Patterns require both the
ratio and FVC scores from the same bronchodilator stage:

| Pattern | FEV1/FVC | FVC |
|---|---|---|
| Normal | At or above LLN | At or above LLN |
| Obstructive | Below LLN | At or above LLN |
| Possible restriction | At or above LLN | Below LLN |
| Mixed | Below LLN | Below LLN |

Any abnormal pattern is the union of obstructive, possible restriction and
mixed. Missing measurements remain missing; they do not become normal or
negative outcomes. The mixed category is retained in the count ledger, but
has no separately selected regression model.

The regression reference is children with no recorded TB. Measurements from
future cases before their diagnosis enter a separate indicator. Measurements
on the diagnosis date are post-TB. Diagnosis-age boundaries are less than one,
one to less than five, and five years or older. The pooled-before-five model
contains a distinct later-TB term; the age-group model contains disjoint
infancy, one-to-less-than-five and later-TB terms. The all-age model contains
one post-TB term. Each model also retains the future-TB indicator.

## Estimation and selected fits

The model is modified-Poisson GEE with a log link, independence working
correlation and child-cluster robust sandwich covariance. There are no
observation weights or offsets and no finite-cluster covariance multiplier.
The maximum iteration count is 250. Covariates are standardized height
(sample SD, calculated within the outcome-specific sample), male sex and
planned-visit indicators in chronological order. Constant or redundant
nuisance columns are explicitly recorded. Required exposure terms are never
silently assigned another interpretation.

Risk ratios are `exp(beta)` with two-sided normal intervals
`exp(beta ± 1.96 × robust SE)`. The model does not use logistic odds ratios or
standardized logistic risks for these selected estimands.

Covariance is evaluated by solving the Poisson information system for each
child's score and summing the outer products of the resulting influence
vectors. Diagonal rescaling improves numerical conditioning before the solve
and is reversed afterwards. This computes the same robust sandwich without
changing coefficients, weighting or covariance scale. It avoids numerical
asymmetry from multiplying ill-conditioned inverse matrices in zero-event
categories. The original library covariance and the numerical differences
are retained as aggregate diagnostics.

The default selection has 28 fits:

- Pre/post ratio below LLN, FEV1 below LLN, FVC below LLN and obstruction:
  pooled-before-five, disjoint age-group and all-age specifications (24 fits).
- Pre-bronchodilator any abnormal pattern and possible restriction:
  pooled-before-five and disjoint age-group specifications (4 fits).

The latter two outcomes retain the separate pattern-source population,
independently verified to match the relevant prepared contracts. Their
post-bronchodilator sample definitions remain available in M03, but are not
additional selected M05 regressions.

The contrast output contains 46 selected rows. Full coefficient and covariance
files also retain the later-TB nuisance terms in the supplementary pattern
models. `selected_for_figure2` identifies the eight binary figure rows.

## Non-estimability and validation

Every contrast retains its exact observation, child and event counts,
including zero and counts below ten. Zero or all-positive exposed outcomes,
absent exposure terms and unsuccessful numerical fits receive explicit
statuses. A missing estimate is not replaced with a risk ratio of one or
zero. Expected boundary non-estimability is distinguished from a numerical
failure; numerical failures prevent a successful module completion marker.

An estimable contrast with very few exposed events is still reported with its
exact counts, risk ratio and interval. Sparse-event normal-Wald inference can
be unstable even when coefficients and sandwich calculations converge. The
selected covariance has no finite-cluster adjustment, and numerical agreement
with an independent calculation does not establish small-sample coverage or
calibration of a significance threshold. These limitations are not addressed
by the independent target intercept added to the separate continuous joint
models; that covariance correction does not change this binary GEE model.

Validation checks the outcome classification, exposure state, model matrix,
cluster membership and counts before comparing results. Independent
Poisson score equations and a separately calculated cluster sandwich check
the coefficients, full covariance matrices, standard errors and intervals.
Boundary coefficients that approach an infinite limit cannot be treated as
stable finite maximum-likelihood estimates. Their event support and limiting
behavior are checked separately. Selected archived estimates and the eight
Figure 2 binary rows are additional scientific checks.

## Execution and outputs

```sh
.venv/bin/python run.py --through M05 --config config/paths.local.json --analysis config/analysis.local.json --resume
```

This command verifies or runs M01–M04 first, requiring diagnostic acceptance
of all twelve primary models. M04 sampling options still apply to that
prerequisite; selecting only a subset with `--targets` is not permitted for
cumulative M05. The binary fits themselves are deterministic and run in the
pinned preparation environment included in the full model dependency lock.

The aggregate directory contains `binary_models/<fingerprint>/` with
`estimates.csv`, `coefficients.csv`, `covariance.csv`, `pattern_counts.csv`,
`diagnostics.json`, `summary.json` and a completion receipt. The full
coefficient and covariance arrays support later review; the selected contrast
table records whether each estimate is reportable. Pattern counts retain all
four categories in each exposure state, including zero cells.

Identity checks bind the selected prepared contracts, implementation,
dependency lock and actual runtime. Work/output locks and atomic publication
protect completed artifacts. `--resume` verifies the entire saved file
inventory and its diagnostic status. Changed or missing artifacts fail
validation. Only an accepted cumulative run writes `binary_current.json` in
private work. Command success establishes numerical acceptance; agreement
with the selected scientific reference is independently documented in private
validation evidence. Manuscript rendering and source delivery have separate
acceptance receipts; current reconstruction also requires the preceding
primary models' covariance-PPC acceptance.
