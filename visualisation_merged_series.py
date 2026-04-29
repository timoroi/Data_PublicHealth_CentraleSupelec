import matplotlib.pyplot as plt
import math

import pandas as pd
import matplotlib.dates as mdates


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column_name in df.columns:
        if str(column_name).strip().lower() == "date":
            renamed[column_name] = "date"
        else:
            renamed[column_name] = str(column_name).strip().lower()
    return df.rename(columns=renamed)


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


df_monthly = pd.read_csv("data/google_trends/merged_trends.csv")
df_monthly_2 = pd.read_csv("data/google_trends/monthly_campaigns_2005_2026.csv")

df_monthly = normalize_columns(df_monthly)
df_monthly_2 = normalize_columns(df_monthly_2)

df_monthly = prepare_timeseries(df_monthly)
df_monthly_2 = prepare_timeseries(df_monthly_2)

# Compare only common numeric columns and shared dates.
common_cols = sorted(
    set(df_monthly.select_dtypes(include="number").columns)
    & set(df_monthly_2.select_dtypes(include="number").columns)
)
common_dates = df_monthly.index.intersection(df_monthly_2.index)

if not common_cols:
    raise ValueError("No common numeric columns found between df_monthly and df_monthly_2.")

if common_dates.empty:
    raise ValueError("No common dates found between df_monthly and df_monthly_2.")

df1 = df_monthly.loc[common_dates, common_cols]
df2 = df_monthly_2.loc[common_dates, common_cols]

print(f"Common dates: {common_dates.min().date()} -> {common_dates.max().date()} ({len(common_dates)} rows)")
print(f"Common columns ({len(common_cols)}): {common_cols}")

# =========================
# 1) Setup plot
# =========================
n_cols = 2
n_rows = math.ceil(len(common_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

# =========================
# 2) Plot both series + differences
# =========================
for i, col in enumerate(common_cols):
    ax = axes[i]
    series_a = df1[col]
    series_b = df2[col]
    diff = series_a - series_b

    ax.plot(df1.index, series_a, color="#1f77b4", linewidth=2, label="df_monthly")
    ax.plot(df2.index, series_b, color="#ff7f0e", linewidth=2, linestyle="--", label="df_monthly_2")

    mae = diff.abs().mean()
    max_abs_diff = diff.abs().max()
    ax.set_title(f"{col} | MAE={mae:.2f} | Max|Delta|={max_abs_diff:.2f}", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)

    if i == 0:
        ax.legend(loc="upper left")

# Hide unused axes if the number of columns is odd.
for j in range(len(common_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
