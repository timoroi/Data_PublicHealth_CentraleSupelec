from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.tsa.stattools import grangercausalitytests


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
MEDIA_CLOUD_PATH = BASE_DIR / "octobre_rose_media_cloud_daily.csv"
GOOGLE_TRENDS_PATH = BASE_DIR / "octobre_rose_all_seasons.csv"
MAX_LAG = 3
ALPHA = 0.05


def safe_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def load_media_cloud() -> pd.DataFrame:
    df = pd.read_csv(MEDIA_CLOUD_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df = df.dropna(subset=["year", "date", "ratio"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["year", "date", "ratio"]].rename(columns={"ratio": "media_cloud"})


def load_google_trends() -> pd.DataFrame:
    df = pd.read_csv(GOOGLE_TRENDS_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["window_start"] = pd.to_datetime(df["window_start"], dayfirst=True, errors="coerce")
    df["campaign_index"] = pd.to_numeric(df["campaign_index"], errors="coerce")
    df = df.dropna(subset=["date", "window_start", "campaign_index"]).copy()
    df["year"] = df["window_start"].dt.year.astype(int)
    return df[["year", "date", "campaign_index"]].rename(columns={"campaign_index": "google_trends"})


def zscore_within_year(series: pd.Series) -> pd.Series:
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def prepare_merged_panel() -> pd.DataFrame:
    media = load_media_cloud()
    trends = load_google_trends()

    merged = (
        media.merge(trends, on=["year", "date"], how="inner", validate="one_to_one")
        .dropna(subset=["media_cloud", "google_trends"])
        .sort_values(["year", "date"])
        .reset_index(drop=True)
    )

    quality = (
        merged.groupby("year")
        .agg(
            n_days=("date", "nunique"),
            media_std=("media_cloud", "std"),
            trends_std=("google_trends", "std"),
        )
        .reset_index()
    )
    print("Data quality by year:")
    print(quality.to_string(index=False))

    expected_days = int(quality["n_days"].median())
    valid_years = quality.loc[
        quality["n_days"].between(expected_days - 2, expected_days + 2)
        & quality["media_std"].fillna(0).gt(0)
        & quality["trends_std"].fillna(0).gt(0),
        "year",
    ].tolist()

    removed_years = sorted(set(merged["year"].unique()) - set(valid_years))
    if removed_years:
        print(f"Removed years failing quality checks: {removed_years}")

    merged = merged.loc[merged["year"].isin(valid_years)].copy()
    merged["media_cloud"] = merged.groupby("year")["media_cloud"].transform(zscore_within_year)
    merged["google_trends"] = merged.groupby("year")["google_trends"].transform(zscore_within_year)
    merged = merged.dropna(subset=["media_cloud", "google_trends"]).copy()
    merged = merged.sort_values(["year", "date"]).reset_index(drop=True)
    merged["day_index"] = merged.groupby("year").cumcount() + 1
    return merged[["year", "date", "media_cloud", "google_trends", "day_index"]]


def run_granger_per_year(
    df: pd.DataFrame,
    target_col: str,
    source_col: str,
    max_lag: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for year, g in df.groupby("year", sort=True):
        g = g.sort_values("day_index")[[target_col, source_col]].dropna()
        if len(g) <= max_lag + 1:
            continue
        if g[target_col].nunique() < 2 or g[source_col].nunique() < 2:
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message="verbose is deprecated since functions should not print results",
                )
                test_out = grangercausalitytests(
                    g[[target_col, source_col]], maxlag=max_lag, verbose=False
                )
        except InfeasibleTestError:
            continue

        for lag in range(1, max_lag + 1):
            rows.append(
                {
                    "year": int(year),
                    "lag": lag,
                    "direction": f"{source_col} -> {target_col}",
                    "p_value": float(test_out[lag][0]["ssr_ftest"][1]),
                }
            )

    return pd.DataFrame(rows)


def pooled_granger_fe(
    df: pd.DataFrame,
    target_col: str,
    source_col: str,
    max_lag: int,
) -> Tuple[float, float, str]:
    tmp = df[["year", "day_index", target_col, source_col]].copy()

    target_lags = []
    source_lags = []
    for lag in range(1, max_lag + 1):
        target_lag = f"{target_col}_lag{lag}"
        source_lag = f"{source_col}_lag{lag}"
        tmp[target_lag] = tmp.groupby("year")[target_col].shift(lag)
        tmp[source_lag] = tmp.groupby("year")[source_col].shift(lag)
        target_lags.append(target_lag)
        source_lags.append(source_lag)

    tmp = tmp.dropna().copy()

    rhs = target_lags + source_lags + ["C(year)"]
    model = smf.ols(formula=f"{target_col} ~ " + " + ".join(rhs), data=tmp).fit(cov_type="HC3")
    hypothesis = " , ".join([f"{lag_name} = 0" for lag_name in source_lags])
    test = model.wald_test(hypothesis, scalar=True)
    return float(np.asarray(test.statistic).squeeze()), float(test.pvalue), hypothesis


def run_leave_one_year_out(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for excluded_year in sorted(df["year"].unique()):
        subset = df.loc[df["year"] != excluded_year].copy()

        stat_fw, pval_fw, _ = pooled_granger_fe(subset, "media_cloud", "google_trends", MAX_LAG)
        rows.append(
            {
                "excluded_year": int(excluded_year),
                "direction": "google_trends -> media_cloud",
                "p_value": float(pval_fw),
                "wald_stat": float(stat_fw),
            }
        )

        stat_rv, pval_rv, _ = pooled_granger_fe(subset, "google_trends", "media_cloud", MAX_LAG)
        rows.append(
            {
                "excluded_year": int(excluded_year),
                "direction": "media_cloud -> google_trends",
                "p_value": float(pval_rv),
                "wald_stat": float(stat_rv),
            }
        )

    return pd.DataFrame(rows, columns=["excluded_year", "direction", "p_value", "wald_stat"])


def plot_loo_pvalues(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for direction, group in results.groupby("direction"):
        group = group.sort_values("excluded_year")
        ax.plot(group["excluded_year"], group["p_value"], marker="o", linewidth=2, label=direction)
    ax.axhline(ALPHA, color="red", linestyle="--", linewidth=1.5, label="0.05 threshold")
    ax.set_title("Leave-One-Year-Out Granger p-values")
    ax.set_xlabel("Excluded year")
    ax.set_ylabel("p-value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loo_boxplot(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    directions = list(results["direction"].drop_duplicates())
    box_data = [results.loc[results["direction"] == d, "p_value"] for d in directions]
    ax.boxplot(box_data, tick_labels=directions)
    ax.axhline(ALPHA, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Leave-One-Year-Out Granger p-values")
    ax.set_ylabel("p-value")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cleaned_time_series(merged: pd.DataFrame, path: Path) -> None:
    plot_df = merged.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df["date"], plot_df["media_cloud"], color="#d81b60", linewidth=1.6, label="media_cloud (z-score)")
    ax.plot(plot_df["date"], plot_df["google_trends"], color="#1e88e5", linewidth=1.3, label="google_trends (z-score)")
    ax.set_title("Octobre Rose Cleaned Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(merged: pd.DataFrame, path: Path) -> None:
    plot_df = merged.copy()
    x = plot_df["google_trends"].to_numpy()
    y = plot_df["media_cloud"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=18, alpha=0.55, color="#6a1b9a")
    ax.plot(x_line, y_line, color="black", linewidth=1.5)
    ax.set_title(f"Octobre Rose Correlation Plot (r = {corr:.3f})")
    ax.set_xlabel("google_trends (z-score)")
    ax.set_ylabel("media_cloud (z-score)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def deseason_and_standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    out = df.copy()
    corr_before = float(out[["media_cloud", "google_trends"]].corr().iloc[0, 1])

    out["media_cloud_deseason"] = out["media_cloud"] - out.groupby("day_index")["media_cloud"].transform("mean")
    out["google_trends_deseason"] = out["google_trends"] - out.groupby("day_index")["google_trends"].transform("mean")

    out["media_cloud_deseason"] = out.groupby("year")["media_cloud_deseason"].transform(zscore_within_year)
    out["google_trends_deseason"] = out.groupby("year")["google_trends_deseason"].transform(zscore_within_year)
    out = out.dropna(subset=["media_cloud_deseason", "google_trends_deseason"]).copy()

    corr_after = float(out[["media_cloud_deseason", "google_trends_deseason"]].corr().iloc[0, 1])
    out = out[["year", "date", "day_index", "media_cloud_deseason", "google_trends_deseason"]].copy()
    return out, corr_before, corr_after


def main() -> None:
    merged_output = safe_output_path(BASE_DIR / "merged_event_panel_clean.csv")
    per_year_output = safe_output_path(BASE_DIR / "granger_per_year_octobre_rose_clean.csv")
    pooled_output = safe_output_path(BASE_DIR / "granger_pooled_octobre_rose_clean.csv")
    loo_output = safe_output_path(BASE_DIR / "granger_leave_one_year_out_octobre_rose_clean.xlsx")
    loo_plot_output = safe_output_path(BASE_DIR / "granger_loo_pvalues_octobre_rose_clean.png")
    loo_boxplot_output = safe_output_path(BASE_DIR / "granger_loo_boxplot_octobre_rose_clean.png")
    cleaned_ts_output = safe_output_path(BASE_DIR / "octobre_rose_cleaned_time_series.png")
    corr_output = safe_output_path(BASE_DIR / "octobre_rose_correlation_plot.png")
    deseason_panel_output = safe_output_path(BASE_DIR / "merged_event_panel_deseason.csv")
    deseason_pooled_output = safe_output_path(BASE_DIR / "granger_pooled_octobre_rose_deseason.csv")
    deseason_loo_plot_output = safe_output_path(BASE_DIR / "granger_loo_pvalues_octobre_rose_deseason.png")
    deseason_loo_boxplot_output = safe_output_path(BASE_DIR / "granger_loo_boxplot_octobre_rose_deseason.png")

    merged = prepare_merged_panel()
    year_counts = merged.groupby("year")["date"].size()
    if year_counts.min() < (MAX_LAG + 2):
        raise ValueError(
            f"Some years have too few rows for max_lag={MAX_LAG}. Min rows/year={year_counts.min()}"
        )

    merged_to_save = merged.copy()
    merged_to_save["date"] = pd.to_datetime(merged_to_save["date"]).dt.strftime("%Y-%m-%d")
    merged_to_save.to_csv(merged_output, index=False)

    per_year_forward = run_granger_per_year(merged, "media_cloud", "google_trends", MAX_LAG)
    per_year_reverse = run_granger_per_year(merged, "google_trends", "media_cloud", MAX_LAG)
    per_year_results = pd.concat([per_year_forward, per_year_reverse], ignore_index=True)
    per_year_results.to_csv(per_year_output, index=False)

    stat_fw, pval_fw, hyp_fw = pooled_granger_fe(merged, "media_cloud", "google_trends", MAX_LAG)
    stat_rv, pval_rv, hyp_rv = pooled_granger_fe(merged, "google_trends", "media_cloud", MAX_LAG)
    pooled_results = pd.DataFrame(
        [
            {
                "direction": "google_trends -> media_cloud",
                "max_lag": MAX_LAG,
                "wald_stat": stat_fw,
                "p_value": pval_fw,
                "null_hypothesis": hyp_fw,
            },
            {
                "direction": "media_cloud -> google_trends",
                "max_lag": MAX_LAG,
                "wald_stat": stat_rv,
                "p_value": pval_rv,
                "null_hypothesis": hyp_rv,
            },
        ]
    )
    pooled_results.to_csv(pooled_output, index=False)

    loo_results = run_leave_one_year_out(merged)
    loo_results.to_excel(loo_output, index=False)

    plot_loo_pvalues(loo_results, loo_plot_output)
    plot_loo_boxplot(loo_results, loo_boxplot_output)
    plot_cleaned_time_series(merged, cleaned_ts_output)
    plot_correlation(merged, corr_output)

    deseason_df, corr_before, corr_after = deseason_and_standardize(merged)
    deseason_to_save = deseason_df.copy()
    deseason_to_save["date"] = pd.to_datetime(deseason_to_save["date"]).dt.strftime("%Y-%m-%d")
    deseason_to_save.to_csv(deseason_panel_output, index=False)

    stat_fw_ds, pval_fw_ds, hyp_fw_ds = pooled_granger_fe(
        deseason_df.rename(
            columns={
                "media_cloud_deseason": "media_cloud",
                "google_trends_deseason": "google_trends",
            }
        ),
        "media_cloud",
        "google_trends",
        MAX_LAG,
    )
    stat_rv_ds, pval_rv_ds, hyp_rv_ds = pooled_granger_fe(
        deseason_df.rename(
            columns={
                "media_cloud_deseason": "media_cloud",
                "google_trends_deseason": "google_trends",
            }
        ),
        "google_trends",
        "media_cloud",
        MAX_LAG,
    )
    deseason_pooled = pd.DataFrame(
        [
            {
                "direction": "google_trends -> media_cloud_deseason",
                "max_lag": MAX_LAG,
                "wald_stat": stat_fw_ds,
                "p_value": pval_fw_ds,
                "null_hypothesis": hyp_fw_ds,
            },
            {
                "direction": "media_cloud_deseason -> google_trends",
                "max_lag": MAX_LAG,
                "wald_stat": stat_rv_ds,
                "p_value": pval_rv_ds,
                "null_hypothesis": hyp_rv_ds,
            },
        ]
    )
    deseason_pooled.to_csv(deseason_pooled_output, index=False)

    deseason_loo = run_leave_one_year_out(
        deseason_df.rename(
            columns={
                "media_cloud_deseason": "media_cloud",
                "google_trends_deseason": "google_trends",
            }
        )
    )
    plot_loo_pvalues(deseason_loo, deseason_loo_plot_output)
    plot_loo_boxplot(deseason_loo, deseason_loo_boxplot_output)

    print("\nKey results:")
    print(pooled_results.to_string(index=False))
    print("\nDeseason correlation check:")
    print(f"Correlation before deseasonalization: {corr_before:.6f}")
    print(f"Correlation after deseasonalization:  {corr_after:.6f}")
    print("\nDeseasonalized pooled results:")
    print(deseason_pooled.to_string(index=False))
    print(f"Saved merged panel: {merged_output}")
    print(f"Saved per-year Granger: {per_year_output}")
    print(f"Saved pooled Granger: {pooled_output}")
    print(f"Saved leave-one-year-out results: {loo_output}")
    print(f"Saved LOO p-values plot: {loo_plot_output}")
    print(f"Saved LOO boxplot: {loo_boxplot_output}")
    print(f"Saved cleaned time series plot: {cleaned_ts_output}")
    print(f"Saved correlation plot: {corr_output}")
    print(f"Saved deseasonalized panel: {deseason_panel_output}")
    print(f"Saved deseasonalized pooled results: {deseason_pooled_output}")
    print(f"Saved deseasonalized LOO p-values plot: {deseason_loo_plot_output}")
    print(f"Saved deseasonalized LOO boxplot: {deseason_loo_boxplot_output}")


if __name__ == "__main__":
    main()
