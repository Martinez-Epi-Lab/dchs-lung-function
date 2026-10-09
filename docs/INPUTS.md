# Input configuration

The source contract is `config/input_contract.json`. It defines input roles without embedding personal paths or study records. Actual source identities remain in the completed private inventory and local configuration.

| Role | Purpose | Required for full reconstruction |
|---|---|---|
| `baseline` | Baseline characteristics, covariates and censoring | Yes |
| `tb_events` | Dated TB events used in longitudinal linkage | Yes |
| `pulmonary_function` | Original longitudinal pulmonary-function measurements | Yes |
| `respiratory_tb_dates` | Diagnosis-date linkage for wheeze and aligned FeNO | Yes |
| `wheeze_assessments` | Scheduled assessments and illness-triggered contacts | Yes; selected local binding verified |
| `gli_lookup` | Selected reference-score coefficients and lookup | Yes; separately supplied rspiro 0.5 lookup |
| `pft_dictionary` | Source variable definitions | Optional file-level check |
| `wheeze_dictionary` | Source variable definitions | Optional file-level check |
| `wheeze_csv_check` | Companion representation for import comparison | Optional; selected local pair verified |

These are neutral roles within the same study dataset. The selected analysis definitions and model-specific samples remain explicit.

## Local file format

Copy the example configuration to a local file and supply real paths. Relative paths resolve against the configuration file's directory, independently of the shell's working directory. Configuration files named `*.local.json` are ignored by Git. Input files should remain outside the repository.

Each configured input uses this structure:

```json
{
  "path": "../authorized-inputs/source.dta",
  "sha256": "optional 64-character hexadecimal file checksum"
}
```

Omit `sha256` when a checksum is not yet recorded; the explanatory string above is not a valid checksum. The reviewed local bindings include checksums. Execution hashes every supplied input regardless of whether an expected checksum was provided, includes it in the run identity and checks it again before accepting the run or resuming it.

The complete file has `schema_version`, `inputs`, `directories` and `fonts` objects, as shown in the example. All required roles must be present. Unknown input roles and unexpected fields are rejected to catch spelling mistakes. Optional input roles can be omitted; if provided, their files must pass the same checks.

`directories.work` and `directories.outputs` identify private generated-file locations outside the source repository. They cannot contain source code, inputs, fonts or configuration, or overlap each other. They need not already exist; file preflight does not create them. `fonts.arial` and `fonts.arial_bold` point to separately supplied font files. The current check verifies readability, not font identity or embedding.

## Reconstruction boundary

Historical linked, reference-scored, paired and covariate-enriched PFT datasets are private comparison evidence. M02-M03 regenerate their required values from the supplied source inputs; archived derived files are not execution prerequisites.

The wheeze binding identified during M01 passes identity and schema checks. Its Stata and CSV representations agree across all 255 fields for 1,082 children under the declared import comparison rules. If a CSV companion is configured, every preparation run repeats the substantive-field comparison. The file-level preflight alone does not establish equivalence.

The GLI lookup is a computational reference dependency. Supply the selected rspiro 0.5 `R/sysdata.rda` containing `GLIgl_lookup`; its package DESCRIPTION records GPL (>= 2). Its contents, sex tables and supported coefficient structure are validated during preparation. The source repository contains no copied lookup or proprietary fonts. A future release may add a separately reviewed acquisition route.

Four antenatal flow counts require an upstream recruitment/disposition source beyond the supplied child records. The flow output retains missing values and an explicit unresolved status for those nodes. They are not filled with manuscript constants.

## Analysis configuration and checks

Copy `config/analysis.example.json` and set `event_end` and `respiratory_end` to the selected administrative dates in ISO `YYYY-MM-DD` form. These settings are supplied privately and included in the preparation fingerprint. The respiratory observation rules retain their selected strict-before-censor boundary; an endpoint setting does not override that rule.

`--check-config` validates file readability, optional checksums and safe directory separation. `--through M01` additionally checks source format, encoding, required fields, dates, identifier uniqueness, missingness and row preservation. `--through M02` adds linkage, reference scoring and descriptive outputs. `--through M03` adds model-specific samples, designs and follow-up arrays. Participant-level comparisons and intermediate artifacts stay local; console output uses aggregate summaries and neutral role names.
