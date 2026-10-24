# Semester 3 report outline

Title: Zero-Shot Time-Series Foundation Models for Retail Demand Forecasting: A Comparative Evaluation of Chronos-2, LightGBM, and Seasonal Naive

Use this outline after a real M5 run. Do not fill the results chapter with the
synthetic demo.

Chronos-2 is an AI model developed by Amazon that is designed to forecast future values from historical time-series data.

## 1. Introduction

- State the research question from the experiment summary.
- Explain why the output is a 7- or 28-day sales forecast with an interval, not
  only a single number.
- Say clearly that inventory replenishment is the Semester 4 extension and is
  not evaluated here.

## 2. Data

- Walmart M5: daily sales, calendar, and weekly price.
- Why the full 30,490 series are not used. Report the seed, thresholds, and
  the number of fast, slow, and intermittent series actually selected.
- Class labels are computed before the first evaluation origin.
- Limitations to state in this chapter, not only at the end:
  - recorded sales are not unconstrained demand
  - a zero is not a stockout label
  - the files do not contain inventory, lead time, or cost

## 3. Methods

- Seasonal naive with weekly seasonality, plus residual quantiles fit on the
  history available at the origin.
- LightGBM direct multi-horizon quantile models. One global model per quantile,
  refit at each origin on past rows only.
- Chronos-2 zero-shot forecasts from the recorded checkpoint. No neural network
  is trained in this project.
- Known future covariates: calendar fields. Price is last-known-at-origin unless
  the report justifies the other assumption.
- Say that LightGBM is trained on the study series and Chronos-2 is not. That
  difference is part of the comparison, not a hidden advantage to ignore.

## 4. Evaluation protocol

- Chronological origins. No random row split.
- Horizon and number of non-overlapping windows.
- MAE and MASE for point forecasts. Do not use MAPE as the primary metric.
- Pinball loss at 0.1, 0.5, and 0.9.
- Interval coverage and width for the 0.1 to 0.9 interval.
- Report macro scores by demand class, not only one pooled score.
- Runtime and memory.

## 5. Results

- Insert `reports/experiment_summary.md` tables.
- Insert the EDA figures and `reports/figures/mae_by_class.png`.
- Answer the research question with the observed direction and magnitude.
  A result in which LightGBM or seasonal naive wins on some classes is valid.
- Show one fast series and one intermittent series from the dashboard.

## 6. Discussion

- Where the interval is too narrow or too wide.
- Whether intermittent series drive the difference between models.
- Checkpoint, context length, and the possibility that public retail series
  overlap Chronos-2 pretraining. Check the model card and say what cannot be
  ruled out.
- Hyperparameters were fixed before looking at the test windows.

## 7. Conclusion and Semester 4 handoff

- What was built.
- What the evidence does and does not support.
- Semester 4 plan, in one page: inventory simulation, lead-time demand
  uncertainty, and cost-service trade-offs. Do not claim those results now.

## Demonstration checklist

- `python -m demand_forecast.cli demo` runs.
- The M5 study has been run with seasonal naive, LightGBM, and Chronos-2.
- The dashboard opens on the M5 forecasts.
- The report states the data limitations above.
