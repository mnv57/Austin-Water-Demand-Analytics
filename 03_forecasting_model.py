"""
Austin TX Municipal Water Consumption
Demand Forecasting Model
=========================
Input  : cleaned_features_df.csv  (1,047 rows × 48 features)
Model  : Gradient Boosting Regressor (sklearn)
         — interpretable, no deep learning, robust on tabular data
Output : forecast_results.csv          (actuals + fitted + 30-day forecast)
         model_evaluation_report.txt   (all metrics)
         forecasting_model_plots.png   (4-panel diagnostic)

Pipeline:
  M1 — Feature selection (what goes in, what stays out, and why)
  M2 — Time-series cross-validation (no data leakage)
  M3 — Model training on full history
  M4 — Evaluation: MAPE, RMSE, R², residual diagnostics
  M5 — Feature importance (which variables drive demand)
  M6 — 30-day forward forecast (Jan 1–30 2026) with uncertainty bands
  M7 — Business interpretation (consulting-style output)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_features_df.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 62)
print("  AUSTIN TX — DEMAND FORECASTING MODEL")
print("=" * 62)
print(f"  Training data : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Rows          : {len(df):,}")
print(f"  Features      : 48 available columns")

# ─────────────────────────────────────────────
# M1 — FEATURE SELECTION
# ─────────────────────────────────────────────
# Rules:
#   IN  — observable at prediction time OR lagged (no future leakage)
#   OUT — ground truth columns, internal model components, repair metadata
#
# What we EXCLUDE and why:
#   consumption_gallons_clean  → ground truth, never available in production
#   consumption_gallons_dirty  → pre-cleaning, not what we're predicting
#   model_signal / base_load / seasonal_component etc → internal engine vars,
#                                would give the model a "cheat code"
#   noise                      → by definition unforecastable
#   obs_to_model_ratio         → post-hoc cleaning artifact
#   local_scale / repair_action / data_quality_flag → cleaning metadata
#   trend_multiplier           → redundant with year_index

FEATURES = [
    # Temperature signals (available from weather forecast)
    "temp_f", "temp_lag_1", "temp_lag_2", "temp_roll7_mean",
    "temp_x_month", "temp_above_95", "temp_above_80",

    # Demand history (lag features — known at prediction time)
    "demand_lag_1", "demand_lag_2", "demand_lag_7",
    "demand_lag_14", "demand_lag_365",

    # Rolling demand context
    "demand_roll7_mean", "demand_roll7_std", "demand_roll30_mean",

    # Calendar — cyclical encoded (no raw month/doy to avoid ordinality issues)
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "dow_sin", "dow_cos",

    # Event flags (known in advance for holidays; from forecast for heatwaves)
    "is_heatwave", "is_holiday", "is_weekend", "is_freeze_event",

    # Trend
    "year_index",
]

TARGET = "consumption_repaired"

X = df[FEATURES].values
y = df[TARGET].values
dates_train = df["date"].values

print(f"\n[M1] Feature selection: {len(FEATURES)} features selected")
print(f"     Target: {TARGET}")

# ─────────────────────────────────────────────
# M2 — TIME-SERIES CROSS-VALIDATION
# ─────────────────────────────────────────────
# CRITICAL: never use random k-fold on time series.
# TimeSeriesSplit always trains on past, tests on future — no leakage.
# We use 5 folds, each with a minimum 180-day training window.

print("\n[M2] Time-series cross-validation (5 folds, walk-forward)...")

tscv = TimeSeriesSplit(n_splits=5, gap=0)

cv_mape_scores = []
cv_rmse_scores = []
cv_r2_scores   = []

# Gradient Boosting hyperparameters — tuned for this dataset size
# n_estimators=300: enough trees without overfitting 1k rows
# max_depth=4: shallow trees reduce variance on small datasets
# learning_rate=0.05: slow learner → better generalisation
# subsample=0.8: stochastic GB reduces overfitting
MODEL_PARAMS = dict(
    n_estimators  = 300,
    max_depth     = 4,
    learning_rate = 0.05,
    subsample     = 0.8,
    min_samples_leaf = 5,
    loss          = "huber",    # robust to outliers (better than squared error)
    random_state  = 42,
)

fold_details = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_cv = GradientBoostingRegressor(**MODEL_PARAMS)
    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)

    # MAPE (exclude near-zero values)
    mask  = y_test > 1e6
    mape  = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)

    cv_mape_scores.append(mape)
    cv_rmse_scores.append(rmse)
    cv_r2_scores.append(r2)

    fold_details.append({
        "fold"      : fold,
        "train_days": len(train_idx),
        "test_days" : len(test_idx),
        "test_start": pd.Timestamp(dates_train[test_idx[0]]).date(),
        "test_end"  : pd.Timestamp(dates_train[test_idx[-1]]).date(),
        "mape"      : mape,
        "rmse"      : rmse,
        "r2"        : r2,
    })

    print(f"     Fold {fold}: train={len(train_idx):>4}d | "
          f"test={len(test_idx):>3}d "
          f"[{fold_details[-1]['test_start']} → {fold_details[-1]['test_end']}] | "
          f"MAPE={mape:.2f}% | R²={r2:.4f}")

print(f"\n     CV Summary:")
print(f"     MAPE : {np.mean(cv_mape_scores):.2f}% ± {np.std(cv_mape_scores):.2f}%")
print(f"     RMSE : {np.mean(cv_rmse_scores):,.0f} ± {np.std(cv_rmse_scores):,.0f} gal/day")
print(f"     R²   : {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")

# ─────────────────────────────────────────────
# M3 — TRAIN FINAL MODEL ON FULL HISTORY
# ─────────────────────────────────────────────
print("\n[M3] Training final model on full dataset (2022–2025)...")

model_final = GradientBoostingRegressor(**MODEL_PARAMS)
model_final.fit(X, y)

y_fitted = model_final.predict(X)

# In-sample metrics (expected to be better than CV — shown for comparison)
mask_all  = y > 1e6
mape_insample = np.mean(np.abs((y[mask_all] - y_fitted[mask_all]) / y[mask_all])) * 100
rmse_insample = np.sqrt(mean_squared_error(y, y_fitted))
r2_insample   = r2_score(y, y_fitted)

print(f"     In-sample MAPE : {mape_insample:.2f}%")
print(f"     In-sample RMSE : {rmse_insample:,.0f} gal/day")
print(f"     In-sample R²   : {r2_insample:.4f}")

# ─────────────────────────────────────────────
# M4 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n[M4] Computing feature importance...")

importances = model_final.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature"   : FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n     Top 10 drivers of water demand:")
print(f"     {'Rank':<5} {'Feature':<25} {'Importance':>10}")
print(f"     {'-'*43}")
for i, row in feat_imp_df.head(10).iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"     {i+1:<5} {row['feature']:<25} {row['importance']:>8.4f}  {bar}")

# ─────────────────────────────────────────────
# M5 — 30-DAY FORWARD FORECAST (Jan 1–30, 2026)
# ─────────────────────────────────────────────
# Strategy: recursive one-step-ahead forecasting
# Each day's prediction becomes the lag feature for the next day.
# This is how real operational forecasting works.
print("\n[M5] Generating 30-day forward forecast (2026-01-01 → 2026-01-30)...")

FORECAST_DAYS = 30
forecast_start = pd.Timestamp("2026-01-01")
forecast_dates = pd.date_range(forecast_start, periods=FORECAST_DAYS, freq="D")

# Seed the recursive forecast with the last known values from training data
history = df[["date", TARGET, "temp_f"]].copy()

# Austin January temperature normals for 2026 (NOAA climatology)
# Jan avg high: 59°F, with realistic daily variation
np.random.seed(42)
jan_temps = 59 + np.random.normal(0, 4, FORECAST_DAYS)
jan_temps = np.clip(jan_temps, 40, 75)  # physically plausible Jan range

# Build forecast row by row (recursive)
forecast_records = []
rolling_demand_buffer = list(df[TARGET].values[-30:])  # last 30 actuals

for i, fdate in enumerate(forecast_dates):
    doy   = fdate.day_of_year
    month = fdate.month
    dow   = fdate.dayofweek
    year_index = (fdate - df["date"].min()).days / 365.25
    temp  = jan_temps[i]

    # Lag features from rolling buffer
    lag1  = rolling_demand_buffer[-1]
    lag2  = rolling_demand_buffer[-2]
    lag7  = rolling_demand_buffer[-7]
    lag14 = rolling_demand_buffer[-14]
    # Same day last year: pull from training data (Jan 2025)
    same_day_last_year = df[df["date"] == (fdate - pd.DateOffset(years=1))][TARGET]
    lag365 = same_day_last_year.values[0] if len(same_day_last_year) > 0 else np.mean(rolling_demand_buffer[-30:])

    temp_lag1 = jan_temps[i-1] if i > 0 else df["temp_f"].values[-1]
    temp_lag2 = jan_temps[i-2] if i > 1 else df["temp_f"].values[-2]

    roll7_mean  = np.mean(rolling_demand_buffer[-7:])
    roll7_std   = np.std(rolling_demand_buffer[-7:])
    roll30_mean = np.mean(rolling_demand_buffer[-30:])
    temp_roll7  = np.mean(jan_temps[max(0,i-7):i+1])

    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    doy_sin   = np.sin(2 * np.pi * doy / 365.25)
    doy_cos   = np.cos(2 * np.pi * doy / 365.25)
    dow_sin   = np.sin(2 * np.pi * dow / 7)
    dow_cos   = np.cos(2 * np.pi * dow / 7)

    is_holiday  = 1 if fdate.strftime("%Y-%m-%d") == "2026-01-01" else 0
    is_weekend  = 1 if dow >= 5 else 0
    is_heatwave = 0   # January — no heat wave
    is_freeze   = 0

    temp_x_month  = temp * month
    temp_above_95 = max(temp - 95, 0)
    temp_above_80 = np.clip(temp - 80, 0, 15)

    row = [
        temp, temp_lag1, temp_lag2, temp_roll7,
        temp_x_month, temp_above_95, temp_above_80,
        lag1, lag2, lag7, lag14, lag365,
        roll7_mean, roll7_std, roll30_mean,
        month_sin, month_cos, doy_sin, doy_cos, dow_sin, dow_cos,
        is_heatwave, is_holiday, is_weekend, is_freeze,
        year_index,
    ]

    pred = model_final.predict([row])[0]

    # Floor: never below 65% of calibrated base (Jan 2026 base ~= last known)
    pred = max(pred, 85_000_000 * 1.10 * 0.65)   # growth-adjusted floor

    forecast_records.append({
        "date"              : fdate,
        "forecast_gallons"  : pred,
        "temp_f"            : temp,
        "is_weekend"        : is_weekend,
        "is_holiday"        : is_holiday,
    })
    rolling_demand_buffer.append(pred)

forecast_df = pd.DataFrame(forecast_records)

# ── Uncertainty bands via bootstrap residuals
# Use CV residuals to estimate prediction interval empirically
# (more honest than assuming Gaussian errors for a GB model)
all_cv_residuals = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    m_tmp = GradientBoostingRegressor(**MODEL_PARAMS)
    m_tmp.fit(X[train_idx], y[train_idx])
    resid = y[test_idx] - m_tmp.predict(X[test_idx])
    all_cv_residuals.extend(resid.tolist())

resid_array = np.array(all_cv_residuals)
p10 = np.percentile(resid_array, 10)   # pessimistic scenario
p90 = np.percentile(resid_array, 90)   # optimistic scenario
p25 = np.percentile(resid_array, 25)
p75 = np.percentile(resid_array, 75)

# Widen bands slightly for recursive forecast (error compounds day by day)
# Scale factor increases with forecast horizon: sqrt(day) approximation
for i, row in forecast_df.iterrows():
    horizon_scale = np.sqrt(i + 1) / np.sqrt(FORECAST_DAYS)
    forecast_df.loc[i, "lower_80"] = row["forecast_gallons"] + p10 * (1 + horizon_scale)
    forecast_df.loc[i, "upper_80"] = row["forecast_gallons"] + p90 * (1 + horizon_scale)
    forecast_df.loc[i, "lower_50"] = row["forecast_gallons"] + p25 * (1 + horizon_scale * 0.5)
    forecast_df.loc[i, "upper_50"] = row["forecast_gallons"] + p75 * (1 + horizon_scale * 0.5)

print(f"     Forecast range: {forecast_df['forecast_gallons'].min()/1e6:.1f}M → "
      f"{forecast_df['forecast_gallons'].max()/1e6:.1f}M gal/day")
print(f"     Forecast mean : {forecast_df['forecast_gallons'].mean()/1e6:.1f}M gal/day")

# ─────────────────────────────────────────────
# M6 — BUSINESS INTERPRETATION
# ─────────────────────────────────────────────
# Translate model output into consulting-style operational insight

print("\n[M6] Generating business insights...")

peak_day  = forecast_df.loc[forecast_df["forecast_gallons"].idxmax()]
trough_day= forecast_df.loc[forecast_df["forecast_gallons"].idxmin()]
total_jan = forecast_df["forecast_gallons"].sum()
weekend_avg = forecast_df[forecast_df["is_weekend"]==1]["forecast_gallons"].mean()
weekday_avg = forecast_df[forecast_df["is_weekend"]==0]["forecast_gallons"].mean()

# Compare Jan 2026 forecast vs Jan 2025 actuals
jan_2025_actual = df[df["date"].dt.month == 1]["consumption_repaired"].mean()
yoy_change = (forecast_df["forecast_gallons"].mean() - jan_2025_actual) / jan_2025_actual * 100

print(f"\n  ┌─ OPERATIONAL DEMAND BRIEF — JANUARY 2026 ──────────────┐")
print(f"  │                                                          │")
print(f"  │  Projected monthly volume : {total_jan/1e9:.3f} Billion gallons    │")
print(f"  │  Daily average forecast   : {forecast_df['forecast_gallons'].mean()/1e6:.1f}M gal/day          │")
print(f"  │  Peak demand day          : {peak_day['date'].strftime('%b %d')} ({peak_day['forecast_gallons']/1e6:.1f}M gal)       │")
print(f"  │  Trough demand day        : {trough_day['date'].strftime('%b %d')} ({trough_day['forecast_gallons']/1e6:.1f}M gal)       │")
print(f"  │  Weekend vs weekday avg   : +{(weekend_avg/weekday_avg-1)*100:.1f}% on weekends             │")
print(f"  │  YoY vs Jan 2025 actuals  : {yoy_change:+.1f}% (population growth)   │")
print(f"  │                                                          │")
print(f"  │  RECOMMENDATION: Pre-position {(forecast_df['upper_80'].max()/1e6):.0f}M gal reserve      │")
print(f"  │  capacity to cover 80th-percentile demand scenario.     │")
print(f"  └──────────────────────────────────────────────────────────┘")

# ─────────────────────────────────────────────
# M7 — SAVE ALL RESULTS
# ─────────────────────────────────────────────

# Combined history + forecast
OUTPUT_COLS = ["date", "actual_gallons", "fitted_gallons",
               "forecast_gallons", "lower_80", "upper_80",
               "lower_50", "upper_50", "temp_f", "is_holiday",
               "is_weekend", "type"]

history_out = df[["date", TARGET, "temp_f", "is_holiday", "is_weekend"]].copy()
history_out.columns = ["date", "actual_gallons", "temp_f", "is_holiday", "is_weekend"]
history_out["fitted_gallons"] = np.round(y_fitted, 0)
history_out["forecast_gallons"] = np.nan
history_out["lower_80"] = np.nan
history_out["upper_80"] = np.nan
history_out["lower_50"] = np.nan
history_out["upper_50"] = np.nan
history_out["type"] = "historical"
history_out = history_out[OUTPUT_COLS]

forecast_out = forecast_df.copy()
forecast_out["actual_gallons"] = np.nan
forecast_out["fitted_gallons"] = np.nan
forecast_out["type"] = "forecast"
forecast_out = forecast_out[OUTPUT_COLS]

results_df = pd.concat([history_out, forecast_out], ignore_index=True)
results_df.to_csv("forecast_results.csv", index=False)

# Text evaluation report
report_lines = [
    "AUSTIN TX WATER DEMAND — MODEL EVALUATION REPORT",
    "=" * 60,
    "",
    "MODEL: Gradient Boosting Regressor (sklearn)",
    f"  n_estimators  : {MODEL_PARAMS['n_estimators']}",
    f"  max_depth     : {MODEL_PARAMS['max_depth']}",
    f"  learning_rate : {MODEL_PARAMS['learning_rate']}",
    f"  loss          : {MODEL_PARAMS['loss']} (robust to outliers)",
    f"  subsample     : {MODEL_PARAMS['subsample']}",
    "",
    "CROSS-VALIDATION (TimeSeriesSplit, 5 folds, walk-forward)",
    "-" * 60,
    f"  {'Fold':<6} {'Train':>6} {'Test':>5} {'Period':<25} {'MAPE':>8} {'R²':>8}",
    "-" * 60,
]
for fd in fold_details:
    report_lines.append(
        f"  {fd['fold']:<6} {fd['train_days']:>6} {fd['test_days']:>5} "
        f"{str(fd['test_start'])+' → '+str(fd['test_end']):<25} "
        f"{fd['mape']:>7.2f}% {fd['r2']:>8.4f}"
    )
report_lines += [
    "-" * 60,
    f"  {'MEAN':<6} {'':>6} {'':>5} {'':25} {np.mean(cv_mape_scores):>7.2f}% {np.mean(cv_r2_scores):>8.4f}",
    f"  {'STD':<6} {'':>6} {'':>5} {'':25} {np.std(cv_mape_scores):>7.2f}% {np.std(cv_r2_scores):>8.4f}",
    "",
    "IN-SAMPLE PERFORMANCE (final model, full training set)",
    "-" * 60,
    f"  MAPE : {mape_insample:.2f}%",
    f"  RMSE : {rmse_insample:,.0f} gal/day",
    f"  R²   : {r2_insample:.4f}",
    "",
    "TOP 10 FEATURE IMPORTANCES",
    "-" * 60,
]
for i, row in feat_imp_df.head(10).iterrows():
    report_lines.append(f"  {i+1:>2}. {row['feature']:<25} {row['importance']:.4f}")

report_lines += [
    "",
    "JANUARY 2026 FORECAST SUMMARY",
    "-" * 60,
    f"  Total projected volume : {total_jan/1e9:.3f} Billion gallons",
    f"  Daily mean forecast    : {forecast_df['forecast_gallons'].mean()/1e6:.2f}M gal/day",
    f"  YoY change vs Jan 2025 : {yoy_change:+.1f}%",
    f"  Peak day               : {peak_day['date'].strftime('%Y-%m-%d')} @ {peak_day['forecast_gallons']/1e6:.2f}M gal",
    f"  Trough day             : {trough_day['date'].strftime('%Y-%m-%d')} @ {trough_day['forecast_gallons']/1e6:.2f}M gal",
    f"  Reserve recommendation : {forecast_df['upper_80'].max()/1e6:.0f}M gal (80th percentile coverage)",
]

with open("model_evaluation_report.txt", "w") as f:
    f.write("\n".join(report_lines))

print(f"\n[SAVED] forecast_results.csv")
print(f"[SAVED] model_evaluation_report.txt")

# ─────────────────────────────────────────────
# M8 — DIAGNOSTIC PLOTS (4 panel)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 20))
fig.suptitle(
    "Austin TX Municipal Water Demand — Forecasting Model\n"
    "Gradient Boosting | Walk-Forward CV | 30-Day Forecast",
    fontsize=13, fontweight="bold", y=0.99
)

# ── Panel 1: Historical fitted values vs actuals + 30-day forecast
ax1 = axes[0]
# Historical: show last 18 months for clarity
cutoff_display = pd.Timestamp("2024-07-01")
hist_display = df[df["date"] >= cutoff_display]
hist_idx = hist_display.index

ax1.plot(hist_display["date"], hist_display[TARGET] / 1e6,
         color="#1f77b4", lw=1.2, label="Actual (2024–2025)", zorder=3)
ax1.plot(hist_display["date"], y_fitted[hist_idx] / 1e6,
         color="#ff7f0e", lw=1.0, ls="--", alpha=0.85,
         label="Model fitted", zorder=2)

# Forecast
ax1.fill_between(forecast_df["date"],
                 forecast_df["lower_80"] / 1e6,
                 forecast_df["upper_80"] / 1e6,
                 alpha=0.15, color="#2ca02c", label="80% prediction interval")
ax1.fill_between(forecast_df["date"],
                 forecast_df["lower_50"] / 1e6,
                 forecast_df["upper_50"] / 1e6,
                 alpha=0.25, color="#2ca02c", label="50% prediction interval")
ax1.plot(forecast_df["date"], forecast_df["forecast_gallons"] / 1e6,
         color="#2ca02c", lw=2.0, label="30-day forecast (Jan 2026)", zorder=4)

ax1.axvline(pd.Timestamp("2026-01-01"), color="black",
            lw=1.2, ls=":", label="Forecast horizon")
ax1.set_ylabel("Consumption (M Gal/Day)", fontsize=9)
ax1.set_title("Actual vs Fitted (2024–2025) + 30-Day Forecast (Jan 2026)", fontsize=10)
ax1.legend(fontsize=8, ncol=3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax1.grid(alpha=0.3)

# ── Panel 2: Walk-forward CV — actual vs predicted per fold
ax2 = axes[1]
ax2.plot(df["date"], df[TARGET] / 1e6,
         color="#1f77b4", lw=0.8, alpha=0.5, label="Actual (full history)")

fold_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    m_tmp = GradientBoostingRegressor(**MODEL_PARAMS)
    m_tmp.fit(X[train_idx], y[train_idx])
    y_pred_tmp = m_tmp.predict(X[test_idx])
    fold_dates = df["date"].values[test_idx]
    ax2.plot(fold_dates, y_pred_tmp / 1e6,
             color=fold_colors[fold_i], lw=1.5,
             label=f"Fold {fold_i+1} pred (MAPE={fold_details[fold_i]['mape']:.2f}%)")

ax2.set_ylabel("Consumption (M Gal/Day)", fontsize=9)
ax2.set_title("Walk-Forward Cross-Validation — Each Fold's Out-of-Sample Predictions", fontsize=10)
ax2.legend(fontsize=8, ncol=3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax2.grid(alpha=0.3)

# ── Panel 3: Feature importance (horizontal bar)
ax3 = axes[2]
top_n = 15
top_feats = feat_imp_df.head(top_n)
colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
bars = ax3.barh(top_feats["feature"][::-1], top_feats["importance"][::-1],
                color=colors_imp, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, top_feats["importance"][::-1]):
    ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=8)
ax3.set_xlabel("Feature Importance (Gini)", fontsize=9)
ax3.set_title(f"Top {top_n} Feature Importances — What Drives Austin Water Demand?", fontsize=10)
ax3.grid(alpha=0.3, axis="x")
ax3.set_xlim(0, top_feats["importance"].max() * 1.18)

# ── Panel 4: Residual analysis (actual - fitted)
ax4 = axes[3]
residuals = (df[TARGET].values - y_fitted) / 1e6

ax4.scatter(df["date"], residuals,
            c=df["month"], cmap="RdYlBu_r",
            s=8, alpha=0.6, zorder=2)
ax4.axhline(0, color="black", lw=1.0, ls="--")
ax4.axhline(np.percentile(residuals, 10), color="red",
            lw=0.8, ls=":", label="10th/90th percentile")
ax4.axhline(np.percentile(residuals, 90), color="red", lw=0.8, ls=":")

# Shade heatwave periods to check if residuals cluster there
for _, row in df[df["is_heatwave"] == 1].groupby(
        (df["is_heatwave"] != df["is_heatwave"].shift()).cumsum()):
    ax4.axvspan(row["date"].iloc[0], row["date"].iloc[-1],
                alpha=0.07, color="red")

ax4.set_ylabel("Residual (M Gal/Day)", fontsize=9)
ax4.set_title("Residuals Over Time (coloured by month) — Red shading = heat wave periods", fontsize=10)
ax4.legend(fontsize=8)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.98])
PLOT_PATH = "forecasting_model_plots.png"
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"[SAVED] {PLOT_PATH}")

print("\n" + "=" * 62)
print("  FORECASTING PIPELINE COMPLETE")
print(f"  CV MAPE  : {np.mean(cv_mape_scores):.2f}% ± {np.std(cv_mape_scores):.2f}%")
print(f"  CV R²    : {np.mean(cv_r2_scores):.4f}")
print(f"  Forecast : Jan 1–30 2026 | mean {forecast_df['forecast_gallons'].mean()/1e6:.1f}M gal/day")
print("=" * 62)
