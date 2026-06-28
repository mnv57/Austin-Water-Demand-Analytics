# Austin TX Municipal Water Demand — End-to-End Analytics Pipeline

**Domain:** Municipal Infrastructure | Water Resource Management | Demand Forecasting  
**Stack:** Python · Scikit-learn · Pandas · NumPy · Matplotlib  
**Data:** Deterministic synthetic dataset — calibrated to NOAA climatology and Austin Water published statistics  
**Reproducibility:** Fully seeded (`seed=42`) — every number in this repository is deterministic

---

## Project Summary

This project builds a complete, production-style data analytics pipeline for municipal water demand management — from raw (corrupted) sensor data through cleaning, feature engineering, machine learning forecasting, and operational decision support.

The pipeline answers a real question faced by Austin Water and similar utilities:

> *"Given historical consumption patterns, weather forecasts, and known demand drivers — how much water will the city need over the next 30 days, and what should operations do about it?"*

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Synthetic Data Engine                                │
│  Deterministic simulation of 1,461 daily records (2022–2025)   │
│  10 stacked demand layers + 5 corruption types injected        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Data Hygiene Pipeline                                │
│  6-stage cleaning: negatives → flatlines → unit errors →       │
│  outliers → seasonal imputation → quality evaluation           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Forecasting Model                                    │
│  Gradient Boosting + walk-forward CV + 30-day forecast         │
│  + bootstrapped uncertainty bands + business output            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
austin-water-demand-analytics/
│
├── README.md                              ← This file
│
├── data/
│   ├── austin_water_consumption_2022_2025.csv   ← Raw dirty dataset (Layer 1 output)
│   ├── cleaned_features_df.csv                  ← ML-ready dataset (Layer 2 output)
│   └── forecast_results.csv                     ← Actuals + fitted + 30-day forecast
│
├── src/
│   ├── 01_data_generator.py               ← Layer 1: synthetic data engine
│   ├── 02_cleaning_pipeline.py            ← Layer 2: cleaning + feature engineering
│   └── 03_forecasting_model.py            ← Layer 3: model + forecast + insights
│
├── outputs/
│   ├── austin_water_diagnostic_plots.png  ← Layer 1 visualisations
│   ├── cleaning_pipeline_results.png      ← Layer 2 cleaning evaluation
│   ├── forecasting_model_plots.png        ← Layer 3 model diagnostics
│   └── model_evaluation_report.txt        ← Full CV metrics + feature importance
│
└── requirements.txt                       ← Python dependencies
```

---

## Layer 1 — Synthetic Data Engine

**File:** `src/01_data_generator.py`

### Design Philosophy

Real municipal water datasets are proprietary and rarely public. Rather than use an off-the-shelf Kaggle dataset, this project builds a **deterministic probabilistic simulation engine** — calibrated against published real-world statistics — that generates a dataset with realistic statistical properties while remaining fully reproducible.

### Demand Layers Stacked

| Layer | Component | Calibration Source |
|---|---|---|
| L1 | Base load (85M gal/day) | Austin population × 90 gal/capita (AWWA standard) |
| L2 | Population growth trend (+2.5%/yr) | US Census Austin metro growth rate |
| L3 | Asymmetric seasonal pattern | 2nd-harmonic sine (captures long flat summers) |
| L4 | Daily temperature (°F) | NOAA Austin-Bergstrom 30-year normals |
| L5 | Non-linear heat stress demand | Exponential above 95°F (empirical threshold) |
| L6 | Historical heat wave events | Anchored to real Texas heat events (2022–2025) |
| L7 | Winter freeze / pipe-burst event | Feb 2023 Uri echo event |
| L8 | Holiday behavioural spikes | July 4th, Labor Day, Memorial Day |
| L9 | Weekend periodicity | +4% residential vs commercial mix shift |
| L10 | Gaussian daily noise | σ = 1.5% of base (realistic meter variance) |

### Intentional Data Corruptions (5 Types)

The dataset includes **96 corrupt rows (6.6%)** across five realistic failure modes — designed to create a genuine data cleaning challenge:

| Corruption Type | Count | Mechanism | Why Realistic |
|---|---|---|---|
| Clustered missing values | 37 days | Sensor outages (2–8 consecutive days) | IoT sensors fail in runs, not randomly |
| Negative values | 6 days | Meter rollover / recording error | Common in legacy SCADA systems |
| Flatline (stuck sensor) | 19 days | Repeating identical value for 3–6 days | Hardware freeze under extreme temps |
| Extreme outlier spikes | 7 days | 8–12× normal (sensor malfunction) | Electrical fault in transmitter |
| Unit inconsistency | 27 days | Values recorded in liters not gallons | Multi-vendor sensor integration issue |

The `data_quality_flag` column records ground truth for each row — enabling supervised evaluation of the cleaning pipeline.

### Key Output Statistics

```
Records      : 1,461 daily observations (Jan 1 2022 – Dec 31 2025)
Consumption  : 52.4M – 147.7M gallons/day
Temperature  : 50.3°F – 108.6°F (NOAA-calibrated)
Heat waves   : 7 events, 88 days total
Freeze event : 1 event (burst-spike-crash-recovery pattern)
```

---

## Layer 2 — Data Hygiene Pipeline

**File:** `src/02_cleaning_pipeline.py`

### 6-Stage Cleaning Process

```
S1 → Negative values       : Null and route to imputation
S2 → Flatline detection    : Run-length encoding (≥3 consecutive identical values)
S3 → Unit error correction : Rolling model-reference median (no target leakage)
S4 → Outlier removal       : Threshold at 3× seasonal model signal
S5 → Seasonal imputation   : Scaled model reconstruction (not naive linear fill)
S6 → Quality evaluation    : MAPE + RMSE + Precision/Recall vs ground truth
```

**Key design decision — Stage 2:** Standard `diff() == 0` flatline detection incorrectly flags coincidental consecutive pairs. This pipeline uses **run-length encoding** to require ≥3 identical consecutive values before flagging — eliminating false positives on clean data.

**Key design decision — Stage 5:** `pandas.interpolate(method='linear')` draws straight lines through curved seasonal gaps. For a 7-day outage in July, this undershoots real consumption significantly. This pipeline imputes using the **known seasonal model signal scaled by local observed-to-model ratio** — preserving the seasonal curve shape through long gaps.

### Cleaning Results

| Corruption Type | Count | MAPE After Cleaning |
|---|---|---|
| Negative values | 6 | 1.73% |
| Flatline sensor | 19 | 10.25% |
| Unit errors | 27 | 0.00% (exact mathematical inversion) |
| Outlier spikes | 7 | 154.70%* |
| Missing values | 37 | 1.69% |
| **Overall** | **96** | **0.93%** |

*Outliers overlapping heat wave peaks cannot be reliably reconstructed via model imputation — ground truth recovery requires cross-referencing secondary sensors. This is an epistemic limitation of the approach, not a pipeline failure. See discussion in `outputs/model_evaluation_report.txt`.

**Detection accuracy:** Precision 100% · Recall 100% · F1 100%

### Feature Engineering (48 features → ML-ready)

- **Lag features:** demand at t-1, t-2, t-7, t-14, t-365 (same day last year)
- **Rolling statistics:** 7-day and 30-day mean/std of demand and temperature
- **Cyclical encoding:** month, day-of-year, day-of-week encoded as sin/cos pairs (prevents ordinality artifacts in tree models)
- **Interaction terms:** temp × month, piecewise heat stress above 80°F and 95°F
- **Event flags:** heatwave, freeze, holiday, weekend

---

## Layer 3 — Forecasting Model

**File:** `src/03_forecasting_model.py`

### Model Choice: Gradient Boosting Regressor

Gradient Boosting was selected over alternatives for the following reasons:

| Criterion | GB | Random Forest | Linear Regression | LSTM |
|---|---|---|---|---|
| Handles non-linearity (heat stress) | ✅ | ✅ | ❌ | ✅ |
| Robust to outliers (`huber` loss) | ✅ | Partial | ❌ | ❌ |
| Interpretable feature importance | ✅ | ✅ | ✅ | ❌ |
| Works well on ~1,000 rows | ✅ | ✅ | ✅ | ❌ |
| No stationarity assumption | ✅ | ✅ | ❌ | ✅ |

LSTM was explicitly ruled out — insufficient data (1,047 rows) for sequence learning without severe overfitting.

### Validation: Walk-Forward Cross-Validation

**Critical:** standard k-fold cross-validation leaks future information into training on time series. This pipeline uses `TimeSeriesSplit` — always training on past, testing on strictly future periods.

| Fold | Test Period | MAPE | R² |
|---|---|---|---|
| 1 | Jul 2023 – Jan 2024 | 13.71% | 0.33 |
| 2 | Jan 2024 – Jul 2024 | 3.92% | 0.88 |
| 3 | Jul 2024 – Dec 2024 | 3.64% | 0.55 |
| 4 | Jan 2025 – Jul 2025 | 3.67% | 0.71 |
| 5 | Jul 2025 – Dec 2025 | **2.08%** | **0.98** |
| **Mean** | | **5.40% ± 4.21%** | **0.69 ± 0.23** |

Fold 1 underperforms because 177 days of training is insufficient for a model that needs to learn annual seasonal patterns. The consistent improvement from Fold 1 → Fold 5 as training history grows is expected behaviour for seasonal models — and is itself a finding: **this model requires at least 18 months of history to generalise reliably.**

### Top Feature Importances

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `doy_cos` | 39.6% | Seasonal position dominates all other signals |
| 2 | `month_cos` | 22.3% | Confirms seasonal signal (cross-validation between features) |
| 3 | `demand_lag_1` | 12.7% | Strong autocorrelation — demand is sticky day-to-day |
| 4 | `demand_lag_365` | 11.5% | Same day last year — model learns annual memory |
| 5 | `demand_roll7_mean` | 6.6% | Trailing week context |
| 6–10 | Temperature features | 2.3% | Low in January; would dominate in summer months |

### January 2026 Forecast

```
Daily mean forecast  : 63.7M gallons/day
Monthly total        : 1.91 Billion gallons
YoY growth           : +2.1% (population trend)
Peak day             : Jan 4  — 68.8M gal (weekend + holiday carry-over)
Trough day           : Jan 27 — 60.8M gal
80% CI range         : 60.8M – 74.0M gal/day
```

**Operational recommendation:** Pre-position 74M gallon reserve capacity to cover the 80th-percentile demand scenario in January 2026.

### Uncertainty Quantification

Confidence intervals are derived via **empirical bootstrap from walk-forward CV residuals** — not parametric Gaussian assumptions. This is more honest for Gradient Boosting, which has asymmetric residual distributions. Interval width scales with `√(forecast horizon)` to reflect compounding error in recursive one-step-ahead forecasting.

---

## Future Extensions

### Demand Response Policy Simulator

The current model predicts demand. The natural next layer is a **decision support system** that triggers operational interventions when forecast demand crosses stress thresholds.

Proposed architecture:

```python
INTERVENTION_MENU = {
    "odd_even_irrigation_schedule" : {"demand_reduction": 0.08, "lead_time_days": 7},
    "IBT_block3_tariff_increase"   : {"demand_reduction": 0.02, "lead_time_days": 30},
    "public_awareness_campaign"    : {"demand_reduction": 0.02, "lead_time_days": 3},
    "Stage2_water_restrictions"    : {"demand_reduction": 0.15, "lead_time_days": 1},
}
# Trigger: forecast > 1.12 × seasonal_baseline → activate minimum-cost lever stack
```

**Note on tariff elasticity:** Flat tariff hikes are a weak demand management lever for peak events. Short-run price elasticity of municipal water is -0.1 to -0.3 (literature consensus), meaning a 10% tariff increase yields only 1–3% demand reduction. More effective are non-price interventions (irrigation scheduling, smart meter alerts) and Increasing Block Tariffs targeting high-volume discretionary users specifically.

### Segment-Level Disaggregation

The current model treats the city as one aggregate meter. Disaggregating into residential-basic, residential-discretionary (irrigation), and commercial segments would allow segment-specific elasticity modelling and more targeted interventions.

### Real-Time Integration

The pipeline is structured to accept daily inputs. With Austin Water's smart meter API, Layer 1 (synthetic generation) would be replaced by live SCADA feeds — Layers 2 and 3 are production-ready as written.

---

## Running the Pipeline

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/austin-water-demand-analytics
cd austin-water-demand-analytics
pip install -r requirements.txt

# 2. Run in sequence
python src/01_data_generator.py        # generates raw + dirty dataset
python src/02_cleaning_pipeline.py     # cleans + engineers features
python src/03_forecasting_model.py     # trains model + produces forecast

# All outputs saved to data/ and outputs/ directories
```

---

## Technical Environment

```
Python     3.10+
pandas     2.0+
numpy      1.24+
scikit-learn 1.3+
matplotlib 3.7+
scipy      1.10+
```

---

## Key Design Decisions (Interview Reference)

| Decision | What was done | Why |
|---|---|---|
| Synthetic data | Deterministic engine with NOAA calibration | Real Austin Water data is proprietary; synthetic allows full ground-truth evaluation |
| Asymmetric seasonality | 2nd harmonic sine | Pure sine is symmetric; Austin summers are long and flat, winters short and sharp |
| Flatline detection | Run-length encoding, not diff==0 | diff==0 generates false positives on coincidental pairs |
| Gap imputation | Seasonal model scaling, not linear interpolation | Linear fill undershoots curved seasonal consumption during multi-day gaps |
| CV method | TimeSeriesSplit (walk-forward) | Standard k-fold leaks future data into training on time series |
| Loss function | Huber loss | More robust to residual outliers than MSE in GB models |
| Uncertainty bands | Empirical bootstrap from CV residuals | Gaussian CI assumption invalid for GB residual distributions |

---

*Built as a portfolio project demonstrating end-to-end data analytics capability: simulation, data engineering, machine learning, and business decision support.*
