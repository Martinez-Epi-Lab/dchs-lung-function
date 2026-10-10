# Preparation methods and boundaries

## Source inputs and linkage

M01 retains source rows and numeric Stata category codes, recording value labels separately. Identifier normalization rejects missing, ambiguous or colliding keys. Declared date formats are explicit; malformed nonempty values cannot silently become missing. CSV width, encoding, required fields and declared child-level uniqueness are checked before linkage. The optional wheeze CSV companion is aligned by identifier and compared across every substantive field, including dates and event names. Numeric tolerance accommodates export precision; categorical values, missingness and dates must agree.

The TB episode role requires `tbcat` to contain exactly `Confirmed tb` or `Unconfirmed tb`. Missing values, unknown strings and numeric encodings, including labelled Stata codes, are rejected. No unreviewed category mapping can silently change confirmed-TB histories. This vocabulary is specific to the episode input; it does not recode baseline categories.

Shared preparation constructs first/recurrent TB histories and measurement-specific before/after states, including same-day diagnosis in post-TB observations. It preserves original source codes and adds canonical visit labels. Pre/post-bronchodilator GLI scores, selected South African oscillometry reference scores and paired columns are recalculated from source measurements. Reference scores are left missing for unsupported reference ranges and nonpositive measurements.

The selected GLI Global lookup is `GLIgl_lookup` from rspiro version 0.5; its supplied DESCRIPTION records GPL (>= 2). The lookup is supplied separately and identified by checksum. Python `rdata` reads it; no R fitting is required. Lookup redistribution terms remain distinct from the repository's eventual reuse license.

Baseline and PFT sex have different explicit encodings. Baseline-wide covariate standardization and respiratory-cohort birthweight standardization retain their separate original scaling samples. M02 calls the shared deterministic linkage/scoring utilities; M03 then exports final model-specific contracts without depending on archived derived files.

## Cohorts and descriptive rules

The joint population comprises children represented in the PFT source. The descriptive population is the union of selected eligible outcome samples with the selected timing restrictions. Each later analytical sample remains separately defined. Source PFT representation does not mean contribution to every modeled outcome.

Table 1 uses explicit fields, denominators and summary rules; parity uses median/quartiles. Verified inherited binary recodes for sex, site, ethnicity, prematurity, HIV exposure, child HIV and low birthweight use equality to the specified positive code. A missing source value therefore enters the reference category for those fields only. Original values and missing-to-reference counts remain available in the private audit. Smoking/alcohol direct indicators retain missing denominators. Small aggregate counts are not suppressed.

Child flow and maternal livebirth/plurality counts are derived from supplied child records. Antenatal enrollment, antenatal loss to follow-up, pregnancy loss and total antenatal exclusions require an earlier recruitment/disposition source. Those four nodes remain missing with `not_reconstructable_from_supplied_inputs` status; values from the manuscript are not substituted. The modality schedule records observed valid measurements by modality and nominal visit.

## Model preparation

There are 75 PFT contracts covering primary continuous spirometry, binary patterns, trajectories, individual mechanics, covariate/window sensitivities, unpaired/paired/change MMEF, X6 and 17 pre-TB Cox outcomes. M07 derives shared-factor summaries from primary posterior draws. Every contract retains its selected dates, age rules, pairing, nuisance design, scale, follow-up arrays and child weights. Nominal visits and exact-age windows are separate rules.

Primary nuisance trends are estimated on pre/no-TB observations using the selected visit, height and sex specification. These deterministic preparation calculations do not refit later joint outcome models. Cox measurement carry-forward ends at the earliest next eligible measurement, two-year ceiling, first TB or selected censor. Cox estimation and robust covariance are checked in M09.

Every PFT joint-model process requires measurements on or before recorded study exit. The early-TB sensitivity, MMEF and X6 joint families retain their strict diagnosis-age cutoffs: first TB before age one or before age five, as specified. Their event-risk time ends at the earliest eligible first TB, cutoff age, study exit or configured administrative endpoint. A diagnosis exactly on exit or the administrative endpoint is observed; a diagnosis exactly at the age cutoff is outside the early-TB estimand. A known diagnosis after observed follow-up does not count as a threshold event. Missing exit uses the administrative endpoint. Missing or malformed chronology and nonpositive follow-up fail validation rather than creating zero-time children.

The TB administrative endpoint limits hazard ascertainment, not every lung-function process. Full-follow-up families may retain measurements after that endpoint, provided they precede or equal study exit; designated window sensitivities additionally restrict measurements to the administrative endpoint. Threshold contracts record that endpoint so model validators can independently reproduce risk arrays. This observed-follow-up rule corrects the earlier threshold convention that credited every noncase through the cutoff age.

Required exposure terms cannot be silently dropped when constant or collinear. Repeated models explicitly allow a zero-support exposure category to have no coefficient and an ensuing nonestimable result. Supported constant exposures fail because the contrast is not identified.

Respiratory preparation uses the fixed descriptive cohort. Calendar age ten years six months, formal exit and the configured administrative endpoint define censoring, with measurements strictly before censor and exact age below 10.5. Scheduled wheeze and participant-initiated episodes are separate processes. Consecutive qualifying contacts at most 30 days apart form one episode; there is no post-diagnosis lag. Risk time is split at age boundaries and exact TB timing. Primary and adjusted scenarios have separate complete-case samples.

Main FeNO retains the nominal 12-month exception: early attendance is eligible, while eventual infancy-TB cases cannot contribute their prediagnosis measurements to that comparison. Other main visits use the selected exact-age and censoring rules. Supplementary FeNO retains its distinct future/post/never-TB comparator and does not inherit the main cohort restriction. Supplementary age/height scaling precedes complete-case filtering; main scaling follows it. Dedicated FeNO labels take priority on duplicate dates. No exposure contrast yields an explicit nonestimable status and no target coefficient.

## Validation scope

Preparation comparisons check membership, original values, scores, matrices, transformations and saved aggregate evidence. Synthetic tests cover malformed imports, joins, dates, reference limits, pairing, episode chains, exposure contrasts and execution failures. The preparation environment was independently installed from the pinned dependency file. Preparation covers all 17 statistical families. Later modules perform outcome fitting, uncertainty comparisons, diagnostics, rendering and manuscript reproduction; preparation checks alone do not validate those results.
