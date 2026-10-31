## Updated plots and source tables

Run `python run_all.py` to prepare all 781 rows, train the models and produce the updated figures. Install `requirements.txt` first. Real TabPFN/XGB training has not been run in this delivery environment; no new ML scores are supplied or fabricated.

- Figure 5 legends report train/test R² and RMSE, using the same rows as their scatter markers. RMSE is in mmol/g.
- Figure 7 main panels use per-isotherm Langmuir. Its test metrics are N/A for held-out isotherm, composition and source protocols. The available-support plot shows full-test ML results and eligible-test Langmuir results; use the common-support plot for direct comparisons. With zero common test rows all three matched scores are N/A. The pooled pressure-only reference remains explicitly named in the raw export for supplementary inspection, not the principal plots.
- Figure 8 now supplies a separate 1×3 figure for each protocol (R², RMSE, MAE), with two boxes per panel (TabPFN/XGB) and five held-out-fold scores per box. No folds from different protocols are pooled. Dots show each fold, lines the median, diamonds the mean; whiskers extend to 1.5 IQR. The score CSV is also exported.
- `tables/Dataset_summary.csv` and `.md` have Coal rank, P range, T range, Authors. `tables/Table_4_Source_inventory.csv` and `.md` complete every rank entry and retain source links, counts and ranges for all 781 rows. Regenerate using `python summarize_sources.py`. These are replacement tables for the revised draft, not old 777-row counts.
- Four display ranks: Lignite, Bituminous, Semi-bituminous, Anthracite. Semi-bituminous is the user-requested display label for supplied Sub-Bi; original labels remain unchanged. This is not a verified scientific reclassification. Hu/Li provisional composition-inferred ranks are starred; they are not independently reported ranks.
- `plot_depth_inventory.py` plots Section 3.6 from `run5_depth_capacity.csv`: inventory components, Q0.10–Q0.90 bands when available, and pressure/temperature/porosity profiles. It runs automatically after modeling, or independently with `python plot_depth_inventory.py --results results_revised`. Protocol: full-data TabPFN refit using the first two training composition IDs (edit in RUN 5b). This is illustrative, not independent field validation. Existing gradient/porosity assumptions are preserved and must not be described as verified basin inputs. Model quantiles do not include geological-input uncertainty. No numeric storage predictions are generated without a real fitted model.

To replot after an existing local model run: `python plot_manuscript.py --results results_revised`.

The prior full-data workflow is retained below; the updates above supersede its earlier figure descriptions.

# Full-dataset H₂ coal evaluation

## Current analysis population

**All 781 observations are retained for ML training/evaluation.** The corrected input is preserved unchanged in `data/revised_input.xlsx`. Composition totals pass the 0.02 wt.% rounding tolerance; no additional dry-basis conversion is applied. Moisture remains unchanged. Any small closure residual is recorded as a fixed-carbon rounding adjustment, with original values preserved.

The two monotonicity flags are diagnostic only. Langmuir availability is also diagnostic only: no observation is deleted because its isotherm lacks fitting support. The old 717/734 subsets are no longer the default population.

```python
MONO_MODE = 'flag'
USE_COMMON_SUPPORT = False
```

The dataset contains 34 composition groups, 85 source-aware isotherms and 9 source DOI groups. The separate five-fold Langmuir availability audit covers 719 observations; this number is NOT a filtered ML dataset.

## Primary and matched comparisons

Each protocol uses the same train/test partition for all models, with seed 42:

| Protocol | Training rows | Test rows | Per-isotherm Langmuir test coverage |
|---|---:|---:|---:|
| A: ordinary random rows | 624 | 157 | 142 |
| B: held-out isotherm | 624 | 157 | 0 |
| C: held-out composition | 624 | 157 | 0 |
| D: held-out source DOI | 625 | 156 | 0 |

Protocol A now uses ordinary `train_test_split(test_size=0.2, random_state=42)`, without a constraint favoring Langmuir coverage. B/C/D preserve whole groups and choose a seeded combination of group sizes nearest to 20% of rows, without looking at targets or errors. D cannot reach 157 without splitting a source, so it uses 156. This group-size balancing is not uniform group sampling.

**Primary ML result:** score TabPFN and XGB on every test observation (157 for A/B/C, 156 for D). All training observations are retained. Both models use the six numeric inputs only.

**Matched Langmuir result:** restrict metric calculation, not training, to rows where training-only per-isotherm Langmuir predictions exist. In A, TabPFN, XGB and per-isotherm Langmuir are all scored on exactly the same 142 test observations. The other 15 remain in the full ML test evaluation. For training comparisons, the matching mask covers 605 of A's 624 rows. The mask is recomputed for each split and every CV fold.

Do not compare a full-test ML score with a partial-test Langmuir score. `support=available` contains each model's coverage, including full ML coverage; `support=common_per_isotherm` contains the exact common row mask. The export records n and n_total and includes the Boolean `Langmuir_eligible` on each prediction row.

Per-isotherm Langmuir parameters cannot be estimated for completely withheld isotherms in B/C/D without an additional parameter-transfer model. Their per-isotherm test values remain blank and matched metrics N/A. A separate `Langmuir_pooled` pressure-only curve predicts all test rows, but it ignores composition and temperature and is explicitly labelled as a different reference. It remains in the raw exports only; the main bar plots use per-isotherm Langmuir.

## Fitting and grouping

Per-isotherm Langmuir uses training rows only, at least three unique pressures, bounds nL≥0 and PL≥1e-6 MPa, and maxfev=20000. Fit coverage/failures are logged. No test-target fitting or residual-based deletion is performed.

Isotherm keys contain source DOI + rounded proximate composition + temperature. Composition groups are global rounded composition proxies, not verified specimen identities. Source groups use the supplied DOI/Link, with Author as fallback. Rank labels are not grouping keys. Source metadata and reporting bases remain subject to verification against original publications.

TabPFN and XGB use MinMax scalers fitted to training rows only. XGB retains 50 Optuna trials, with tuning folds inside outer training and grouping appropriate to the outer protocol. The optional five-fold supplement uses shuffled KFold for A and GroupKFold for B/C/D. Its metrics also include full-coverage and exactly matched-support rows. Grouped folds need not contain exactly 20% of observations.

## Local execution

Extract the entire ZIP. Use your working TabPFN/PyTorch environment and checkpoint where possible.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs preparation/Langmuir/split audits, regenerates the Excel audit and data figure, and then runs the full modeling script. To regenerate model figures without fitting:

```bash
python plot_manuscript.py --results results_revised
```

For an initial single-split run, set `RUN_1_CV=False` and RUN 2–6 flags to False in `tabpfn_h2_coal_revised.py`. Full analyses and CV are enabled by default. `build_main.py` regenerates the main script and would overwrite manual flag changes. The supplied TabPFN configuration uses average_before_softmax=False; actual versions/settings are recorded at runtime. The requirements are not a historical lockfile; retain checkpoint identity separately.

## Output files

- `Revised_Data_Audit.xlsx`: all 781 rows, metadata, rank estimates, availability audits, split details and actual Langmuir predictions.
- `data/model_dataset.csv`: **781 rows** with source identifiers, inputs, target and metadata.
- `data/model_input.csv`: six numerical predictors, Author and target. Coal rank is excluded.
- `data/langmuir_support_audit.csv`: five-fold fitting availability; `Langmuir_available=False` does not mean an invalid or excluded observation.
- `results_revised/train_test_predictions.xlsx`: generated locally, including all train/test rows, ML and Langmuir predictions, `Langmuir_eligible`, metrics, and `Langmuir_matched_rows` for easy plotting.
- `CV_test_predictions` and `CV_metrics` sheets: optional five-fold predictions and metrics with both scoring supports.
- CSV equivalents provide the same plotting data. `original_excel_row` connects results to the attachment; CV row_idx indexes the full modeling table.
- `results_revised/manuscript_figures`: full-population ML parity with train/test markers, residuals, full-coverage and matched-Langmuir metric bars, five-fold metric boxplots, input correlations/distributions, and 2×2 TabPFN/XGB leverage and relevancy panels.
- Original RUN 2–6 outputs remain in `results_revised` and `paper_tables.xlsx`.

Leverage uses the training design with intercept and pseudoinverse, h*=3 rank/n_train, and residual/training RMSE; it is descriptive, not a calibrated interval. Relevancy factors are test-only Pearson correlations, not causal effects or model feature importance. CV boxplots summarize five test-fold metric scores: circles are individual folds, lines are medians and diamonds are means.

RUN 3 uses condition-blocked P/T holdouts (12/5 MPa and 333.2/303.2 K), separate from A–D. Temperature monotonicity uses low-temperature-trained models. RUN 4 and Monte Carlo use full-data fits. Monte Carlo retains 3000 LHS samples and exploratory expanded ranges, with pressure-window CDFs including 20 MPa. Ordos capacity remains illustrative and not field-validated. All these analyses now use the full 781-row modeling population.

## Coal-rank metadata

The Hu and Li missing-label groups remain provisionally estimated as Bituminous. The method uses three nearest unique reference compositions after standardization and inverse-distance interpolation of ordinal rank codes. Rank labels, neighbors, weights and reference-label conflicts are exported. These are exploratory estimates, not formal ASTM classifications or calibrated probabilities. Reported and estimated ranks remain separate and neither enters model fitting or evaluation splits. Formal classification limitations: https://store.astm.org/d0388-19.html and https://pubs.usgs.gov/circ/c891/table1.htm.

## Execution status

Preparation, full-data split design and actual Langmuir predictions were rerun. All 781 rows are retained. Tests exercise exports, CV, plots and exactly matched metrics with stand-in estimators; those test outputs are not delivered as scientific ML results. Actual TabPFN/XGB and RUN 2–6 fitting require local execution because their libraries/checkpoint are unavailable here. No previous subset-based ML predictions are reused.
