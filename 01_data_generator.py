"""
Austin, Texas Municipal Smart Meter Water Consumption
Deterministic Synthetic Data Generation Engine
=================================================
Date Range : 2022-01-01 to 2025-12-31 (Daily frequency)
Unit       : Gallons/day (city-wide aggregate)
Base Load  : 85,000,000 gallons/day (calibrated to ~978k population @ 90 gal/capita)
Seed       : 42 (fully reproducible)

Layers built:
  L1 - Base load with population growth trend
  L2 - Asymmetric seasonal pattern (long hot summer, short mild winter)
  L3 - Real Austin weather (monthly climatology + daily noise)
  L4 - Temperature-responsive demand (non-linear above 95°F)
  L5 - Heat wave events (historically anchored)
  L6 - Freeze/pipe-burst event (Uri echo)
  L7 - Holiday behavioural spikes
  L8 - Weekly periodicity (residential vs commercial mix)
  L9 - Gaussian daily noise
  L10- Five categories of intentional data corruption
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. GLOBAL SEED — deterministic reproducibility
# ─────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)          # modern NumPy Generator (better than np.random)

# ─────────────────────────────────────────────
# 1. DATE SPINE
# ─────────────────────────────────────────────
START = pd.Timestamp("2022-01-01")
END   = pd.Timestamp("2025-12-31")
dates = pd.date_range(START, END, freq="D")
n     = len(dates)                         # 1,461 days

print(f"[INFO] Generating {n} daily records from {START.date()} to {END.date()}")

# ─────────────────────────────────────────────
# 2. HELPER ARRAYS
# ─────────────────────────────────────────────
doy   = dates.day_of_year.to_numpy()       # day of year 1–366
month = dates.month.to_numpy()
dow   = dates.dayofweek.to_numpy()         # 0=Mon, 6=Sun
year  = dates.year.to_numpy()
days_elapsed = (dates - START).days.to_numpy().astype(float)

# ─────────────────────────────────────────────
# 3. LAYER 1 — BASE LOAD + POPULATION GROWTH TREND
# ─────────────────────────────────────────────
# Austin is one of the fastest-growing US cities: ~2.5% YoY population growth
BASE_LOAD       = 85_000_000.0             # gallons/day at Jan 1 2022
ANNUAL_GROWTH   = 0.025

trend_multiplier = 1 + ANNUAL_GROWTH * (days_elapsed / 365.25)
base = BASE_LOAD * trend_multiplier

print(f"[L1]  Base load range: {base.min():,.0f} → {base.max():,.0f} gal/day")

# ─────────────────────────────────────────────
# 4. LAYER 2 — ASYMMETRIC SEASONAL PATTERN
# ─────────────────────────────────────────────
# Pure sine = symmetric (wrong for Austin).
# Adding 2nd harmonic makes summer broad/flat and winter sharp — realistic.
# Phase shift 80 days ≈ peak around June 20 (Austin summer peak)
# Amplitude: 30% swing on base (real Austin seasonal variation is 25–40%)
phi   = (doy - 80) * 2 * np.pi / 365.25
seasonal = (
    base * 0.30 * np.sin(phi)              # primary harmonic: 30% amplitude
    + base * 0.08 * np.sin(2 * phi)       # 2nd harmonic: asymmetry (flat summer, sharp winter)
)

print(f"[L2]  Seasonal swing: {seasonal.min():,.0f} → {seasonal.max():,.0f} gal/day")

# ─────────────────────────────────────────────
# 5. LAYER 3 — REAL AUSTIN WEATHER PATTERN
# ─────────────────────────────────────────────
# Source: NOAA Austin-Bergstrom 30-year normals (monthly avg high °F)
MONTHLY_TEMP_NORMALS = {
    1: 59, 2: 63, 3: 70, 4: 78, 5: 85,
    6: 92, 7: 96, 8: 97, 9: 90, 10: 80,
    11: 68, 12: 60
}

# Interpolate monthly normals to smooth daily temperature baseline
month_centers = np.array(list(MONTHLY_TEMP_NORMALS.keys()))
month_temps   = np.array(list(MONTHLY_TEMP_NORMALS.values()))

# Use midpoint of each month as knot, extend cyclically for smooth year wrap
extended_months = np.concatenate([month_centers - 12, month_centers, month_centers + 12])
extended_temps  = np.tile(month_temps, 3)
temp_interp     = interp1d(extended_months, extended_temps, kind="cubic")

# Map each date's fractional month position
frac_month = month + (dates.day.to_numpy() - 15) / 30.0
temp_baseline = temp_interp(frac_month)

# Add day-to-day variation: ±4°F (realistic for Texas)
temp_noise    = rng.normal(0, 4.0, n)

# Add inter-annual variation: some years run hotter/cooler
year_bias = {2022: +1.5, 2023: +2.0, 2024: +0.5, 2025: +1.0}
annual_bias = np.array([year_bias[y] for y in year])

temp_daily = temp_baseline + temp_noise + annual_bias

print(f"[L3]  Temperature range: {temp_daily.min():.1f}°F → {temp_daily.max():.1f}°F")

# ─────────────────────────────────────────────
# 6. LAYER 4 — TEMPERATURE-RESPONSIVE DEMAND
# ─────────────────────────────────────────────
# Three regimes:
#   < 80°F : no heat effect
#   80–95°F: linear increase (mild irrigation, AC cooling towers)
#   > 95°F : exponential (lawn panic-watering, evaporative AC, behavioural surge)

heat_effect = np.zeros(n)

moderate_mask = (temp_daily >= 80) & (temp_daily <= 95)
extreme_mask  = temp_daily > 95

heat_effect[moderate_mask] = (
    base[moderate_mask] * 0.0008 * (temp_daily[moderate_mask] - 80)
)
heat_effect[extreme_mask] = (
    base[extreme_mask] * 0.001 * (temp_daily[extreme_mask] - 95) ** 1.8
)

print(f"[L4]  Heat effect range: {heat_effect.min():,.0f} → {heat_effect.max():,.0f} gal/day")

# ─────────────────────────────────────────────
# 7. LAYER 5 — HISTORICAL HEAT WAVE EVENTS
# ─────────────────────────────────────────────
# Anchored to real Texas heat events (ERCOT/NWS records)
# Format: (start_date, end_date, surge_fraction)
HEATWAVE_EVENTS = [
    ("2022-07-15", "2022-07-28", 0.18),   # Extended July 2022 heat dome
    ("2022-08-22", "2022-08-30", 0.12),   # Late August 2022 secondary wave
    ("2023-06-20", "2023-07-04", 0.22),   # June/July 2023 record heat
    ("2023-08-18", "2023-08-27", 0.16),   # August 2023 continuation
    ("2024-07-08", "2024-07-19", 0.15),   # July 2024 heat event
    ("2025-06-25", "2025-07-08", 0.19),   # June 2025 early onset heat
    ("2025-08-05", "2025-08-18", 0.14),   # August 2025 late summer
]

heatwave_effect = np.zeros(n)
is_heatwave     = np.zeros(n, dtype=bool)

for start_str, end_str, surge in HEATWAVE_EVENTS:
    hw_start = pd.Timestamp(start_str)
    hw_end   = pd.Timestamp(end_str)
    mask     = (dates >= hw_start) & (dates <= hw_end)
    # Smooth the surge: ramp up over 2 days, plateau, ramp down over 2 days
    idx = np.where(mask)[0]
    if len(idx) == 0:
        continue
    ramp = np.ones(len(idx))
    ramp[0]  = 0.4
    ramp[1]  = 0.75
    if len(idx) > 2:
        ramp[-1] = 0.4
    if len(idx) > 3:
        ramp[-2] = 0.75
    heatwave_effect[idx] += base[idx] * surge * ramp
    is_heatwave[idx] = True

print(f"[L5]  Heat wave days: {is_heatwave.sum()}")

# ─────────────────────────────────────────────
# 8. LAYER 6 — WINTER FREEZE / PIPE BURST EVENT
# ─────────────────────────────────────────────
# Uri 2021 was catastrophic. Model a smaller echo-event in Feb 2023.
# Pattern: pre-freeze normal → burst spike (+35%) → supply cut crash (-40%)
#          → recovery ramp over ~5 days
FREEZE_EVENTS = [
    {
        "label"       : "Feb2023_freeze",
        "burst_start" : "2023-02-02",
        "burst_end"   : "2023-02-03",
        "crash_start" : "2023-02-04",
        "crash_end"   : "2023-02-05",
        "recovery_end": "2023-02-09",
        "burst_surge" : 0.35,
        "crash_drop"  : -0.40,
    }
]

freeze_effect   = np.zeros(n)
is_freeze       = np.zeros(n, dtype=bool)

for ev in FREEZE_EVENTS:
    for i, d in enumerate(dates):
        ds = d.strftime("%Y-%m-%d")
        if ev["burst_start"] <= ds <= ev["burst_end"]:
            freeze_effect[i] = base[i] * ev["burst_surge"]
            is_freeze[i] = True
        elif ev["crash_start"] <= ds <= ev["crash_end"]:
            freeze_effect[i] = base[i] * ev["crash_drop"]
            is_freeze[i] = True
        elif ev["recovery_end"] >= ds > ev["crash_end"]:
            # Linear recovery from crash to normal over recovery window
            recovery_days = (
                pd.Timestamp(ev["recovery_end"]) - pd.Timestamp(ev["crash_end"])
            ).days
            day_in_recovery = (d - pd.Timestamp(ev["crash_end"])).days
            frac = day_in_recovery / recovery_days
            freeze_effect[i] = base[i] * ev["crash_drop"] * (1 - frac)
            is_freeze[i] = True

print(f"[L6]  Freeze event days: {is_freeze.sum()}")

# ─────────────────────────────────────────────
# 9. LAYER 7 — HOLIDAY BEHAVIOURAL SPIKES
# ─────────────────────────────────────────────
# Major outdoor activity days → elevated residential use
HOLIDAY_SPIKES = {
    # July 4th — cookouts, lawn parties
    "2022-07-04": 0.12, "2023-07-04": 0.11,
    "2024-07-04": 0.13, "2025-07-04": 0.12,
    # Labor Day weekend Sundays
    "2022-09-05": 0.09, "2023-09-04": 0.08,
    "2024-09-02": 0.09, "2025-09-01": 0.08,
    # Memorial Day weekends
    "2022-05-30": 0.07, "2023-05-29": 0.07,
    "2024-05-27": 0.08, "2025-05-26": 0.07,
}

holiday_effect = np.zeros(n)
is_holiday     = np.zeros(n, dtype=bool)

for d_str, surge in HOLIDAY_SPIKES.items():
    mask = dates == pd.Timestamp(d_str)
    holiday_effect[mask] = base[mask] * surge
    is_holiday[mask] = True

print(f"[L7]  Holiday spike days: {is_holiday.sum()}")

# ─────────────────────────────────────────────
# 10. LAYER 8 — WEEKLY PERIODICITY
# ─────────────────────────────────────────────
# Weekends: higher residential irrigation, lower commercial
# Net effect: +4% on Saturday/Sunday
weekend_effect = np.where(dow >= 5, base * 0.04, 0.0)

# ─────────────────────────────────────────────
# 11. LAYER 9 — GAUSSIAN DAILY NOISE
# ─────────────────────────────────────────────
# Real meter data has day-to-day stochastic variation (~1.5% of base)
noise = rng.normal(0, base * 0.015)

# ─────────────────────────────────────────────
# 12. CLEAN SIGNAL ASSEMBLY
# ─────────────────────────────────────────────
consumption_clean = (
    base
    + seasonal
    + heat_effect
    + heatwave_effect
    + freeze_effect
    + holiday_effect
    + weekend_effect
    + noise
)

# Floor at 60% of base (system minimum — never goes to zero)
consumption_clean = np.maximum(consumption_clean, base * 0.60)

print(f"\n[SIGNAL] Clean consumption range:")
print(f"         Min : {consumption_clean.min():>15,.0f} gal/day")
print(f"         Max : {consumption_clean.max():>15,.0f} gal/day")
print(f"         Mean: {consumption_clean.mean():>15,.0f} gal/day")

# ─────────────────────────────────────────────
# 13. LAYER 10 — INTENTIONAL DATA CORRUPTION
# ─────────────────────────────────────────────
# Five realistic corruption types for data cleaning challenge

consumption_dirty = consumption_clean.copy().astype(float)
quality_flag      = np.full(n, "CLEAN", dtype=object)

print("\n[CORRUPT] Injecting data quality issues...")

# ── Corruption Type 1: CLUSTERED MISSING VALUES (sensor outages)
# Real sensors fail for consecutive days, not randomly
# Inject ~8 outage clusters of 2–7 days each
np.random.seed(SEED + 1)                  # separate seed for corruption layer
outage_starts = rng.choice(
    np.arange(30, n - 10), size=8, replace=False
)
missing_count = 0
for start_idx in outage_starts:
    length = int(rng.integers(2, 8))
    end_idx = min(start_idx + length, n)
    consumption_dirty[start_idx:end_idx] = np.nan
    quality_flag[start_idx:end_idx] = "MISSING"
    missing_count += (end_idx - start_idx)

print(f"  Type 1 — Missing values (clustered): {missing_count} days")

# ── Corruption Type 2: NEGATIVE VALUES (meter rollover / recording error)
neg_idx = rng.choice(
    np.where(quality_flag == "CLEAN")[0], size=6, replace=False
)
for idx in neg_idx:
    consumption_dirty[idx] = -abs(consumption_clean[idx]) * rng.uniform(0.1, 0.5)
    quality_flag[idx] = "NEGATIVE"

print(f"  Type 2 — Negative values (meter error): {len(neg_idx)} days")

# ── Corruption Type 3: EXTREME OUTLIER SPIKES (sensor malfunction)
# 8–12x normal — clearly impossible, distinct from real heat spikes
spike_idx = rng.choice(
    np.where(quality_flag == "CLEAN")[0], size=7, replace=False
)
for idx in spike_idx:
    consumption_dirty[idx] = consumption_clean[idx] * rng.uniform(8.0, 12.0)
    quality_flag[idx] = "OUTLIER_HIGH"

print(f"  Type 3 — Extreme outlier spikes: {len(spike_idx)} days")

# ── Corruption Type 4: FLAT-LINE PERIODS (stuck sensor / frozen value)
# Sensor gets stuck and repeats the same value for 3–6 consecutive days
flatline_starts = rng.choice(
    np.where(quality_flag == "CLEAN")[0][10:-10], size=4, replace=False
)
flatline_count = 0
for start_idx in flatline_starts:
    length = int(rng.integers(3, 7))
    end_idx = min(start_idx + length, n)
    stuck_value = consumption_clean[start_idx]
    consumption_dirty[start_idx:end_idx] = stuck_value
    quality_flag[start_idx:end_idx] = "FLATLINE"
    flatline_count += (end_idx - start_idx)

print(f"  Type 4 — Flat-line (stuck sensor): {flatline_count} days")

# ── Corruption Type 5: UNIT INCONSISTENCY (liters vs gallons mix-up)
# Subtle: ~2% of rows recorded in liters instead of gallons
# A value in liters = value_gallons × 3.785 — easy to miss, hard to fix
clean_indices = np.where(quality_flag == "CLEAN")[0]
n_unit_errors = max(1, int(len(clean_indices) * 0.02))
unit_idx = rng.choice(clean_indices, size=n_unit_errors, replace=False)
for idx in unit_idx:
    consumption_dirty[idx] = consumption_clean[idx] * 3.785
    quality_flag[idx] = "UNIT_ERROR"

print(f"  Type 5 — Unit inconsistency (liters/gallons): {n_unit_errors} days")

# Summary
unique, counts = np.unique(quality_flag, return_counts=True)
print(f"\n[CORRUPT] Quality flag distribution:")
for flag, count in zip(unique, counts):
    pct = count / n * 100
    print(f"  {flag:<15} : {count:>4} days ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 14. BUILD FINAL DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "date"                    : dates,
    "consumption_gallons_dirty": np.round(consumption_dirty, 0),
    "consumption_gallons_clean": np.round(consumption_clean, 0),
    "temp_f"                  : np.round(temp_daily, 1),
    "base_load"               : np.round(base, 0),
    "seasonal_component"      : np.round(seasonal, 0),
    "heat_effect"             : np.round(heat_effect, 0),
    "heatwave_effect"         : np.round(heatwave_effect, 0),
    "freeze_effect"           : np.round(freeze_effect, 0),
    "holiday_effect"          : np.round(holiday_effect, 0),
    "weekend_effect"          : np.round(weekend_effect, 0),
    "noise"                   : np.round(noise, 0),
    "day_of_year"             : doy,
    "day_of_week"             : dow,
    "month"                   : month,
    "year"                    : year,
    "is_heatwave"             : is_heatwave.astype(int),
    "is_freeze_event"         : is_freeze.astype(int),
    "is_holiday"              : is_holiday.astype(int),
    "is_weekend"              : (dow >= 5).astype(int),
    "data_quality_flag"       : quality_flag,
    "trend_multiplier"        : np.round(trend_multiplier, 6),
})

# ─────────────────────────────────────────────
# 15. SAVE OUTPUT
# ─────────────────────────────────────────────
OUTPUT_PATH = "austin_water_consumption_2022_2025.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[SAVED] Dataset → {OUTPUT_PATH}")
print(f"        Shape   : {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 16. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 18))
fig.suptitle(
    "Austin TX Municipal Water Consumption — Synthetic Data Engine\n"
    "Deterministic (seed=42) | 2022-01-01 to 2025-12-31 | Daily",
    fontsize=13, fontweight="bold", y=0.98
)

# ── Plot 1: Full clean signal with event annotations
ax1 = axes[0]
ax1.plot(dates, consumption_clean / 1e6, color="#1f77b4", lw=0.9, label="Clean Signal")
ax1.fill_between(
    dates,
    (consumption_clean - noise) / 1e6,
    consumption_clean / 1e6,
    alpha=0.15, color="#1f77b4"
)
# Shade heatwave periods
in_hw = False
for i, d in enumerate(dates):
    if is_heatwave[i] and not in_hw:
        hw_start_plot = d
        in_hw = True
    elif not is_heatwave[i] and in_hw:
        ax1.axvspan(hw_start_plot, d, alpha=0.15, color="red", label="_nolegend_")
        in_hw = False

# Freeze annotation
for ev in FREEZE_EVENTS:
    ax1.axvspan(
        pd.Timestamp(ev["burst_start"]),
        pd.Timestamp(ev["recovery_end"]),
        alpha=0.2, color="cyan", label="Freeze event"
    )

ax1.set_ylabel("Consumption (Million Gallons/Day)", fontsize=9)
ax1.set_title("Clean Signal — All Layers Stacked", fontsize=10)
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax1.grid(alpha=0.3)

# ── Plot 2: Dirty signal with corruption highlighted
ax2 = axes[1]
clean_series = pd.Series(consumption_clean / 1e6, index=dates)
dirty_series = pd.Series(consumption_dirty / 1e6, index=dates)

ax2.plot(dates, clean_series, color="#1f77b4", lw=0.8, alpha=0.6, label="Clean")
ax2.plot(dates, dirty_series, color="#ff7f0e", lw=0.6, alpha=0.8, label="Dirty")

# Highlight each corruption type
corruption_colors = {
    "MISSING"     : ("purple",  0.4),
    "NEGATIVE"    : ("red",     0.8),
    "OUTLIER_HIGH": ("crimson", 0.8),
    "FLATLINE"    : ("green",   0.4),
    "UNIT_ERROR"  : ("orange",  0.8),
}
for flag, (color, alpha) in corruption_colors.items():
    mask = quality_flag == flag
    if mask.any():
        ax2.scatter(
            dates[mask],
            np.where(mask, dirty_series.values, np.nan)[mask],
            color=color, s=25, zorder=5,
            label=flag.replace("_", " ").title()
        )

ax2.set_ylabel("Consumption (Million Gallons/Day)", fontsize=9)
ax2.set_title("Dirty Signal — Corruptions Highlighted", fontsize=10)
ax2.legend(fontsize=7, ncol=3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax2.grid(alpha=0.3)

# ── Plot 3: Signal decomposition stacked area
ax3 = axes[2]
components = {
    "Base Load"   : base / 1e6,
    "Seasonal"    : seasonal / 1e6,
    "Heat Effect" : (heat_effect + heatwave_effect) / 1e6,
    "Freeze"      : freeze_effect / 1e6,
    "Holiday/WE"  : (holiday_effect + weekend_effect) / 1e6,
    "Noise"       : noise / 1e6,
}
colors_comp = ["#4878d0", "#ee854a", "#d65f5f", "#59a14f", "#b07aa1", "#bab0ac"]
bottom = np.zeros(n)
for (label, values), color in zip(components.items(), colors_comp):
    pos_vals = np.maximum(values, 0)
    ax3.bar(dates, pos_vals, bottom=bottom, label=label,
            color=color, alpha=0.85, width=1.0)
    bottom += pos_vals

ax3.set_ylabel("Consumption (Million Gallons/Day)", fontsize=9)
ax3.set_title("Signal Decomposition by Layer", fontsize=10)
ax3.legend(fontsize=7, ncol=3, loc="upper left")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=7)
ax3.grid(alpha=0.2, axis="y")

# ── Plot 4: Temperature vs Consumption scatter (coloured by month)
ax4 = axes[3]
clean_mask = quality_flag == "CLEAN"
scatter = ax4.scatter(
    temp_daily[clean_mask],
    consumption_clean[clean_mask] / 1e6,
    c=month[clean_mask], cmap="RdYlBu_r",
    s=8, alpha=0.6
)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label("Month", fontsize=8)
cbar.set_ticks([1, 3, 6, 9, 12])
cbar.set_ticklabels(["Jan", "Mar", "Jun", "Sep", "Dec"])
ax4.axvline(95, color="red", lw=1.2, ls="--", label="95°F threshold")
ax4.axvline(80, color="orange", lw=1.0, ls="--", label="80°F threshold")
ax4.set_xlabel("Daily Temperature (°F)", fontsize=9)
ax4.set_ylabel("Consumption (Million Gallons/Day)", fontsize=9)
ax4.set_title("Temperature vs Consumption (clean days only)", fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
PLOT_PATH = "austin_water_diagnostic_plots.png"
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"[SAVED] Diagnostic plots → {PLOT_PATH}")

# ─────────────────────────────────────────────
# 17. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET SUMMARY STATISTICS")
print("=" * 60)
print(f"\nDate range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Total records  : {len(df):,}")
print(f"Columns        : {list(df.columns)}")
print(f"\nClean consumption (gallons/day):")
print(f"  Min    : {df['consumption_gallons_clean'].min():>15,.0f}")
print(f"  Max    : {df['consumption_gallons_clean'].max():>15,.0f}")
print(f"  Mean   : {df['consumption_gallons_clean'].mean():>15,.0f}")
print(f"  Std Dev: {df['consumption_gallons_clean'].std():>15,.0f}")
print(f"\nTemperature (°F):")
print(f"  Min    : {df['temp_f'].min():>8.1f}")
print(f"  Max    : {df['temp_f'].max():>8.1f}")
print(f"  Mean   : {df['temp_f'].mean():>8.1f}")
print(f"\nEvent days:")
print(f"  Heat wave days : {df['is_heatwave'].sum():>4}")
print(f"  Freeze days    : {df['is_freeze_event'].sum():>4}")
print(f"  Holiday days   : {df['is_holiday'].sum():>4}")
print(f"  Weekend days   : {df['is_weekend'].sum():>4}")
print(f"\nData quality:")
print(df['data_quality_flag'].value_counts().to_string())
print("\n" + "=" * 60)
print("Generation complete. Fully reproducible with seed=42.")
print("=" * 60)
