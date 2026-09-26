# Life Expectancy Prediction (Regression)

Predicting life expectancy across countries and years from socioeconomic and
health indicators (income, education, healthcare access, immunization
coverage, mortality, and more), using a Random Forest implemented from
scratch as a bagging ensemble of decision trees, compared against Ridge
Regression and AdaBoost.

## Highlights

- **Custom Random Forest**, implemented from scratch on top of
  `DecisionTreeRegressor` (bootstrap sampling + prediction averaging,
  `get_params`/`set_params` wired up for `GridSearchCV` compatibility), 
  validated against scikit-learn's own `RandomForestRegressor` under
  matching hyperparameters, landing within ~1% RMSE of it (1.98 vs 2.00).
- **Test RMSE of 1.88 years / MAE of 1.13 years** on a held-out test set,
  ahead of Ridge Regression (RMSE 4.14) and AdaBoost (RMSE 3.08) under the
  same evaluation.
- **Feature importance analysis**: averaged variance-reduction importances
  across the ensemble's trees to identify which indicators actually drive
  the model's predictions (`HIV/AIDS` and `Adult Mortality` dominate),
  cross-checked against training-set correlations for consistency.
- **Residual analysis**: checked predictions vs. actual and the residual
  distribution for systematic bias or heteroscedasticity - none found.
- **Leak-free preprocessing**: every statistic used for a modeling decision
  (which features to drop, imputation values, hyperparameters) is computed
  from the training split only, then applied unchanged to validation, test,
  and the final evaluation set.
- **Per-country median imputation** for missing values, with a global
  training-median fallback for countries with insufficient training data -
  a more informative choice than a single dataset-wide median given how much
  countries vary on most indicators.

## Approach

1. **Data cleaning** - corrected inconsistent development-status labels,
   fixed a malformed column name, and validated every rate/percentage
   feature against its physically valid range, treating out-of-range values
   as missing rather than trusting them.
2. **Train / validation / test split** (60/20/20), with all downstream
   preprocessing decisions fit on the training split only.
3. **Feature selection** - dropped `Population` after confirming (on
   training data) both a data-quality issue — implausible year-to-year
   jumps for the same country and negligible correlation with the target.
4. **Missing-value imputation** - per-country median from the training set,
   falling back to the global training median where a country has too few
   training rows.
5. **Modeling**  a custom Random Forest (from scratch), Ridge Regression
   (with and without feature standardization), and AdaBoost, each tuned via
   5-fold cross-validation on the training set and compared on a shared
   validation set.
6. **Evaluation** - final model selected by validation RMSE, scored on the
   untouched test set (RMSE/MAE), then examined via feature importance and
   residual analysis before generating predictions on a separate evaluation
   set.

## Results

| Model | Validation RMSE |
|---|---|
| Custom Random Forest | 1.98 |
| AdaBoost | 3.08 |
| Ridge Regression | 4.14 |

**Final model (Custom Random Forest) on the test set:** RMSE 1.88, MAE 1.13.

## Stack

Python, pandas, NumPy, scikit-learn (`DecisionTreeRegressor`, `Ridge`,
`AdaBoostRegressor`, `GridSearchCV`), Matplotlib.
