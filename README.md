# Cognition and neuropsychiatric tests: Elastic Net models

> **Practice code.** This repository contains exploratory practice code written while preparing the analyses for a paper on chronic pain and adverse childhood experiences. It is not the final analysis pipeline, and its purpose is to understand how machine learning works.

## What the script does

`ML trainingpy.py` fits Elastic Net models to find which neuropsychological and neuropsychiatric tests keep predictive weight for **chronic pain** and for **childhood adversity** (threat and deprivation), in cognitively normal (CN) participants and participants with Alzheimer's disease (AD). The goal is to decide which tests enter the later structural equation models (SEM).

The same 11 tests are used as penalized predictors in four models:

| Model | Outcome | Sample | Method |
|---|---|---|---|
| M1 | Pain severity | Participants with chronic pain only (CP_1 = 1) | `ElasticNetCV` |
| M2 | Chronic pain yes/no (CP_1) | Whole sample | `LogisticRegressionCV` (elastic net, saga) |
| M3 | Threat | Whole sample | `ElasticNetCV` |
| M4 | Deprivation | Whole sample | `ElasticNetCV` |

M1 and M2 together form a two-part (hurdle) model: is there pain, and if so, how much?

## Header comments from the script

These are the comments at the beginning of the script, translated from Spanish:

```text
Elastic Net on the neuropsychological tests (BrainLat, CN and AD).

Goal: identify which tests (_total) keep predictive weight for chronic pain and for
childhood adversity (threat, deprivation), to decide which ones enter the SEM models.

Models (same 11 tests as penalized predictors; covariates NOT penalized):
  M1  severity (only CP_1 = 1): mean of z(intensity), z(frequency), z(number of regions),
                                standardized within participants with pain              -> ElasticNetCV
  M2  CP_1 (yes/no)           : presence of chronic pain (whole sample)                 -> LogisticRegressionCV (elasticnet, saga)
      M1 + M2 = two-part (hurdle) model: is there pain? and, if so, how much?
  M3  threat          : mean of binary items on physical/psychological violence         -> ElasticNetCV
  M4  deprivation     : mean of binary items on material/care deprivation               -> ElasticNetCV

Unpenalized covariates (residualization, Frisch-Waugh-Lovell theorem):
  - M1, M3, M4: tests and outcome are residualized on the covariates (OLS) and the Elastic Net is fit
    on the residuals. This is exactly equivalent to an Elastic Net with unpenalized covariates.
  - M2 (logistic): the tests are residualized on the covariates and the covariates enter as
    predictors. An approximation (in logistic regression the equivalence is not exact), but it prevents
    a test from absorbing the effect of a covariate.
  - The residualization is refit within each CV fold and each bootstrap sample (no leakage).
  - Baseline = unpenalized OLS / logistic regression with covariates only; incremental R²/AUC is reported.

Data and conventions:
  - CN and AD only (FTD excluded). -99 -> NaN.
  - Tests oriented "higher = worse" (mmse, mint, digits, minisea are reversed); log1p on faq, gad, gds, npi.
  - Out-of-range values or imputed placeholders -> NaN (see clean_tests).
  - Standardization (z) done once, on the final sample of complete cases for exposures, covariates
    and the three pain indicators (same sample as the SEM models). Missing tests are dropped
    per model (final N is printed).
  - Covariates: age, sex (female = demo_sex == 1), pack-years, years of education (cog_ed) and
    AD diagnosis (by default; --sin_dx removes it, only as a sensitivity analysis).
  - Exposures: binarized items from the "Variables binarizadas" sheet; mean per domain with >= 50% of items.
  - Pain indicators: no pain (CP_1 == 0) is coded 0 (66% zeros: zero-inflation).

Self-contained: requires only pandas, numpy, scikit-learn, matplotlib, openpyxl. Does not import other scripts.

Usage:
  python "ML trainingpy.py" [sin_fp] [--sin_dx] [--data=PATH.xlsx] [--out=FOLDER]
    sin_fp  : excludes minisea_fp_total (large loss of cases)
    --sin_dx: does not adjust for diagnosis (sensitivity)
    --data  : ACEs_julio_final.xlsx (sheets "Variables binarizadas" and "data raw")
    --out   : output folder (default: next to this script)
  If --data is not given, the ACES_DATA environment variable is used and, if it does not exist,
  ACEs_julio_final.xlsx in the same folder as this script.

Outputs (folder resultados_enet_<variant>[_sin_dx]):
  enet_coeficientes.csv   coefficient, 95% bootstrap CI, selection frequency and CV performance per variable and model
  enet_pesos_tests.png    test coefficients per model (dark = stable)
```

## Covariates

All models adjust for the same covariates. They are **not penalized**, so the Elastic Net can't shrink them away and a test can't take credit for a covariate's effect.

| Covariate | Source column(s) | Coding |
|---|---|---|
| Age | `demo_age` | Years |
| Sex | `demo_sex` | 1 = female (`demo_sex == 1`), 0 = otherwise |
| Education | `cog_ed` | Years of education |
| Smoking (pack-years) | `demo_smoke`, `demo_smoke_cpd`, `demo_smoke_years` | (cigarettes per day / 20) × years smoked; never-smokers (`demo_smoke == 0`) = 0 |
| AD diagnosis | `clinical_diagnosis` | 1 = AD, 0 = CN. Included by default; `--sin_dx` removes it as a sensitivity analysis |

All covariates are z-standardized on the analysis sample.

**How they are left unpenalized:**

- **M1, M3, M4 (linear):** the tests and the outcome are residualized on the covariates with ordinary least squares, and the Elastic Net is fit on the residuals. By the Frisch–Waugh–Lovell theorem, this is exactly equivalent to an Elastic Net with unpenalized covariates.
- **M2 (logistic):** the tests are residualized on the covariates, and the covariates are entered as additional predictors. In logistic regression this is an approximation rather than an exact equivalence.
- The residualization is refit inside every cross-validation fold and every bootstrap sample, so no information leaks from test data into training.
- A baseline model with only the covariates (unpenalized OLS or logistic regression) is fit for comparison, and the script reports how much R² or AUC the tests add on top of it.

## Missing data and imputation

**The script does not impute missing values.** It uses complete cases. It also removes imputed placeholder values that were found in the source data:

- **Missing codes:** `-99` is treated as missing.
- **Imputed placeholders in the source file are removed:** repeated values that came from an earlier imputation are set to missing rather than used. These are `gds_total = 2.900925`, `npi_total = 4.285323`, and test values stored as text (for example `'2,900,925'`).
- **Impossible values are removed:** `gds_total > 15` (GDS-15 maximum), `npi_total > 144` (NPI-12 maximum) and `udsfaq_total > 30` (FAQ maximum) are set to missing. These are assumed to be data-entry errors such as a lost decimal point.
- **Analysis sample:** participants must have complete data on both exposures, all covariates and the three pain indicators. After that, participants missing any of the included tests are dropped. `minisea_fp_total` loses many cases, so the `sin_fp` option excludes it to keep a larger sample. The final N for each model is printed when the script runs.
- **Adversity domains:** a domain score is computed only if at least 50% of its items are answered; otherwise it is missing. When at least half are answered, the score is the mean of the answered items.
- **Pain regions:** a region not recorded is counted as "not marked".

## How the other variables were scored

### Childhood adversity (exposures)

Items come from the already-binarized sheet *Variables binarizadas* (1 = adversity present). Items with the suffixes `period___1` and `period___2` are the same question asked for two different childhood periods, and each is counted as a separate item.

| Domain | Items | Score |
|---|---|---|
| **Threat** | Physical violence (hitting, attacks, property: 6 items) + psychological violence (humiliation, prohibitions, monitoring, demands: 8 items) | Mean of the 14 binary items (proportion endorsed) |
| **Deprivation** | Lack of household assets and services (20 items, e.g. electricity, running water, refrigerator, TV, car) + basic care needs (7 items, e.g. finances, food, balanced diet, care) | Mean of the 27 binary items (proportion endorsed) |

Both scores are z-standardized on the analysis sample.

### Chronic pain

| Indicator | Source | Scoring |
|---|---|---|
| Chronic pain (yes/no) | `CP_1` | 1 = chronic pain, 0 = none. Outcome of M2 |
| Intensity | `CP_Intensity` | As recorded |
| Frequency | `CP_Frequency` | Recoded to increasing order: original 2 → 1, 3 → 2, 4 → 3, 1 → 4 |
| Number of regions | `CP_*_1` region columns | Number of anatomical groups (0–5) with at least one painful region: axial (neck, upper back, lower back, chest, abdomen), upper limbs (shoulders, arms, elbows, wrists, hands), lower limbs (legs, knees, ankles, feet), head (head, face) and other |

Participants without chronic pain (`CP_1 = 0`) get 0 on intensity, frequency and number of regions. About two-thirds of the sample has no pain, which is why severity is modeled separately (M1).

**Pain severity (M1 outcome):** among participants with pain only, intensity, frequency and number of regions are each z-standardized within that group and then averaged. Standardizing within the pain group keeps the composite from mostly measuring "pain vs. no pain".

### Neuropsychological and neuropsychiatric tests (predictors)

All tests are oriented so that **higher = worse**. Tests where a high score means better performance are multiplied by −1. Skewed symptom scales are log-transformed with log(1 + x). All tests are then z-standardized on the analysis sample.

| Test (column) | Measures | Direction | Transformation |
|---|---|---|---|
| `mmse_total` | Global cognition (MMSE) | Reversed | None |
| `cog_mint_total` | Naming (MINT) | Reversed | None |
| `cog_digits_forward_total` | Digit span forward | Reversed | None |
| `cog_digits_backward_total` | Digit span backward | Reversed | None |
| `minisea_em_total` | Emotion recognition (Mini-SEA) | Reversed | None |
| `minisea_fp_total` | Faux pas / theory of mind (Mini-SEA) | Reversed | None |
| `udsfaq_total` | Functional activities (FAQ) | As is | log(1 + x) |
| `uls_total` | Loneliness (ULS) | As is | None |
| `gad_total` | Anxiety (GAD) | As is | log(1 + x) |
| `gds_total` | Depression (GDS-15) | As is | log(1 + x) |
| `npi_total` | Neuropsychiatric symptoms (NPI) | As is | log(1 + x) |

`udsfaq_total` is read from the *data raw* sheet and merged by `ID`, because it is not in the binarized sheet.

## Model fitting and stability

- **Hyperparameters:** L1 ratio from {0.2, 0.5, 0.8, 0.95}. 60 penalty values (linear models) or 12 values of C (logistic model). Chosen by 5-fold cross-validation, scored by AUC for M2.
- **Performance:** nested cross-validation with 5 outer folds. The residualization and the hyperparameter choice are redone inside each outer fold, and the reported R² or AUC comes from out-of-sample predictions.
- **Stability:** 200 bootstrap resamples with the hyperparameters fixed, giving 95% percentile intervals for the coefficients and each test's selection frequency (how often its coefficient is non-zero). Tests selected in at least 80% of resamples are considered stable.
- **Random seed:** 2026.

## Requirements

- Python 3.11 or later (required by pandas 3). Tested with Python 3.14.
- Packages: `pip install numpy pandas scikit-learn matplotlib openpyxl`

## Data

**The data are not included in this repository** and must not be uploaded; `.gitignore` blocks Excel files. The script expects an Excel file (by default `ACEs_julio_final.xlsx`) with two sheets:

- **Variables binarizadas**, with the column headers in row 2: demographics, pain variables, tests and binarized adversity items.
- **data raw**: used only for `udsfaq_total`, matched by `ID`.

## How to run

The script looks for the data file in this order:

1. The `--data` argument: `python "ML trainingpy.py" --data="C:\path\to\ACEs_julio_final.xlsx"`
2. The `ACES_DATA` environment variable
3. `ACEs_julio_final.xlsx` in the same folder as the script

Options:

| Option | Effect |
|---|---|
| `sin_fp` | Excludes `minisea_fp_total` to keep more cases |
| `--sin_dx` | Does not adjust for AD diagnosis (sensitivity analysis) |
| `--out=FOLDER` | Output folder (default: next to the script) |

Example: `python "ML trainingpy.py" sin_fp --sin_dx`

## Outputs

Results are saved to `resultados_enet_<variant>[_sin_dx]/`:

- `enet_coeficientes.csv`: for each variable and model, the standardized coefficient, 95% bootstrap interval, selection frequency, N and cross-validated performance (with and without the tests).
- `enet_pesos_tests.png`: bar chart of the test coefficients for each model. Dark bars are tests selected in at least 80% of bootstrap resamples.
