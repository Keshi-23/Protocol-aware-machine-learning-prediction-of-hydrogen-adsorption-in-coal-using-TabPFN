# Full-dataset H₂ coal evaluation

**All 781 observations are retained for ML training/evaluation.** The corrected input is preserved unchanged in `data/revised_input.xlsx`. Composition totals pass the 0.02 wt.% rounding tolerance; no additional dry-basis conversion is applied. Moisture remains unchanged. Any small closure residual is recorded as a fixed-carbon rounding adjustment, with original values preserved.

## Primary and matched comparisons

Each protocol uses the same train/test partition for all models, with seed 42:

| Protocol | Training rows | Test rows | Per-isotherm Langmuir test coverage |
|---|---:|---:|---:|
| A: ordinary random rows | 624 | 157 | 142 |
| B: held-out isotherm | 624 | 157 | 0 |
| C: held-out composition | 624 | 157 | 0 |
| D: held-out source DOI | 625 | 156 | 0 |

Protocol A now uses ordinary `train_test_split(test_size=0.2, random_state=42)`, without a constraint favoring Langmuir coverage. B/C/D preserve whole groups and choose a seeded combination of group sizes nearest to 20% of rows, without looking at targets or errors. D cannot reach 157 without splitting a source, so it uses 156. This group-size balancing is not uniform group sampling.

**Primary ML result:** score TabPFN and XGB on every test observation (157 for A/B/C, 156 for D). All training observations are retained. Both models use the six numeric inputs only
