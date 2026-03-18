from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.tsa.stattools import grangercausalitytests


# Event window: Dec 26 (year-1) to Feb 6 (year)
WINDOW_START = (12, 26)
WINDOW_END = (2, 6)


def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") >= header.count(",") else ","
    return pd.read_csv(path, sep=sep)


def extract_year_from_filename(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.stem)
    if not match:
        raise ValueError(f"Could not extract year from filename: {path.name}")
    return int(match.group(1))


def clean_columns(columns: List[str]) -> List[str]:
    return [str(c).strip().strip('"').lower() for c in columns]


def enforce_event_window(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    start = pd.to_datetime(
        {
            "year": df[year_col] - 1,
            "month": WINDOW_START[0],
            "day": WINDOW_START[1],
        }
    )
    end = pd.to_datetime(
        {"year": df[year_col], "month": WINDOW_END[0], "day": WINDOW_END[1]}
    )
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "date"]).copy()
    df["day_index"] = df.groupby("year").cumcount() + 1
    return df


def load_media_cloud(media_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for file in sorted(media_dir.glob("*.csv")):
        try:
            year = extract_year_from_filename(file)
        except ValueError:
            continue
        df = read_csv_auto_sep(file)
        df.columns = clean_columns(df.columns.tolist())

        required = {"date", "count", "total_count", "ratio"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{file.name} is missing columns: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df["total_count"] = pd.to_numeric(df["total_count"], errors="coerce")
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
        df["year"] = year

        df = enforce_event_window(df)
        df = df.dropna(subset=["date"]).sort_values("date")
        frames.append(df[["year", "date", "count", "total_count", "ratio"]])

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {media_dir}")

    out = pd.concat(frames, ignore_index=True)
    return add_day_index(out)


def pick_trends_column(df: pd.DataFrame, keyword: str) -> str:
    keyword = keyword.lower().strip()
    if keyword in df.columns:
        return keyword

    numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    if numeric_cols:
        # Fallback: choose the first numeric series if exact keyword is not found.
        return numeric_cols[0]

    raise ValueError("No numeric Google Trends column found.")


def load_google_trends_from_directory(trends_dir: Path, keyword: str = "dry january") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for file in sorted(trends_dir.glob("*.csv")):
        try:
            year = extract_year_from_filename(file)
        except ValueError:
            continue
        df = read_csv_auto_sep(file)
        df.columns = clean_columns(df.columns.tolist())

        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})
        if "date" not in df.columns:
            raise ValueError(f"{file.name} must contain 'Time' or 'date' column")

        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        trends_col = pick_trends_column(df, keyword=keyword)

        tmp = df[["date", trends_col]].rename(columns={trends_col: "google_trends"})
        tmp["year"] = year
        tmp = enforce_event_window(tmp)
        tmp = tmp.dropna(subset=["date"]).sort_values("date")

        frames.append(tmp[["year", "date", "google_trends"]])

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {trends_dir}")

    out = pd.concat(frames, ignore_index=True)
    return add_day_index(out)


def load_google_trends_from_file(trends_file: Path, keyword: str = "dry january") -> pd.DataFrame:
    df = read_csv_auto_sep(trends_file)
    df.columns = clean_columns(df.columns.tolist())

    if "date" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    if "date" not in df.columns:
        raise ValueError(f"{trends_file.name} must contain 'date' or 'time' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if "campaign_index" in df.columns:
        trends_col = "campaign_index"
    else:
        trends_col = pick_trends_column(df, keyword=keyword)

    out = df[["date", trends_col]].rename(columns={trends_col: "google_trends"}).copy()

    if "season" in df.columns:
        season_year = (
            df["season"]
            .astype(str)
            .str.extract(r"(\d{4})\s*-\s*(\d{4})", expand=True)[1]
        )
        out["year"] = pd.to_numeric(season_year, errors="coerce")
    elif "window_end" in df.columns:
        out["year"] = pd.to_datetime(df["window_end"], errors="coerce").dt.year
    else:
        out["year"] = np.where(out["date"].dt.month == 12, out["date"].dt.year + 1, out["date"].dt.year)

    fallback_year = pd.Series(
        np.where(out["date"].dt.month == 12, out["date"].dt.year + 1, out["date"].dt.year),
        index=out.index,
    )
    out["year"] = out["year"].fillna(fallback_year)

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["google_trends"] = pd.to_numeric(out["google_trends"], errors="coerce")
    out = out.dropna(subset=["date", "year", "google_trends"]).copy()
    out["year"] = out["year"].astype(int)

    out = enforce_event_window(out)
    out = out.sort_values(["year", "date"]).reset_index(drop=True)
    return add_day_index(out[["year", "date", "google_trends"]])


def load_google_trends(trends_path: Path, keyword: str = "dry january") -> pd.DataFrame:
    if trends_path.is_file():
        return load_google_trends_from_file(trends_path, keyword=keyword)
    if trends_path.is_dir():
        return load_google_trends_from_directory(trends_path, keyword=keyword)
    raise FileNotFoundError(f"Google Trends path not found: {trends_path}")


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
            pval = test_out[lag][0]["ssr_ftest"][1]
            rows.append(
                {
                    "year": int(year),
                    "lag": lag,
                    "direction": f"{source_col} -> {target_col}",
                    "p_value": float(pval),
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
        t_col = f"{target_col}_lag{lag}"
        s_col = f"{source_col}_lag{lag}"
        tmp[t_col] = tmp.groupby("year")[target_col].shift(lag)
        tmp[s_col] = tmp.groupby("year")[source_col].shift(lag)
        target_lags.append(t_col)
        source_lags.append(s_col)

    tmp = tmp.dropna().copy()

    rhs = target_lags + source_lags + ["C(year)"]
    formula = f"{target_col} ~ " + " + ".join(rhs)
    model = smf.ols(formula=formula, data=tmp).fit(cov_type="HC3")

    hypothesis = " , ".join([f"{name} = 0" for name in source_lags])
    test = model.wald_test(hypothesis, scalar=True)

    stat = float(np.asarray(test.statistic).squeeze())
    pval = float(test.pvalue)
    return stat, pval, hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Granger causality on repeated Dry January windows (no cross-year continuity)."
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=Path(r"C:\Users\abdel\Downloads\Media cloud jour"),
        help="Directory containing Media Cloud yearly CSV files.",
    )
    parser.add_argument(
        "--trends-path",
        "--trends-dir",
        dest="trends_path",
        type=Path,
        default=Path(r"C:\Users\abdel\Downloads\Media cloud jour\dry_january_all_seasons.csv"),
        help="Google Trends source: either a directory of yearly CSVs or one all-seasons CSV file.",
    )
    parser.add_argument(
        "--trends-keyword",
        type=str,
        default="dry january",
        help="Google Trends keyword column to use (after quote stripping).",
    )
    parser.add_argument(
        "--media-var",
        type=str,
        default="ratio",
        choices=["count", "total_count", "ratio"],
        help="Media Cloud variable used in Granger tests.",
    )
    parser.add_argument("--max-lag", type=int, default=3, help="Maximum lag for Granger tests.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save prepared datasets and test results.",
    )
    args = parser.parse_args()

    media_df = load_media_cloud(args.media_dir)
    trends_df = load_google_trends(args.trends_path, keyword=args.trends_keyword)

    merged = (
        media_df.merge(
            trends_df[["year", "date", "google_trends"]],
            on=["year", "date"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["year", "date"])
        .reset_index(drop=True)
    )
    merged = add_day_index(merged.drop(columns=["day_index"]))

    year_counts = merged.groupby("year")["date"].size()
    if year_counts.min() < (args.max_lag + 2):
        raise ValueError(
            f"Some years have too few rows for max_lag={args.max_lag}. Min rows/year={year_counts.min()}"
        )

    # Direction 1: Google Trends -> Media Cloud
    per_year_forward = run_granger_per_year(
        merged,
        target_col=args.media_var,
        source_col="google_trends",
        max_lag=args.max_lag,
    )
    stat_fw, pval_fw, hyp_fw = pooled_granger_fe(
        merged,
        target_col=args.media_var,
        source_col="google_trends",
        max_lag=args.max_lag,
    )

    # Direction 2: Media Cloud -> Google Trends
    per_year_reverse = run_granger_per_year(
        merged,
        target_col="google_trends",
        source_col=args.media_var,
        max_lag=args.max_lag,
    )
    stat_rv, pval_rv, hyp_rv = pooled_granger_fe(
        merged,
        target_col="google_trends",
        source_col=args.media_var,
        max_lag=args.max_lag,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    media_df.to_csv(args.out_dir / "media_cloud_all_years.csv", index=False)
    trends_df.to_csv(args.out_dir / "google_trends_all_years.csv", index=False)
    merged.to_csv(args.out_dir / "merged_event_panel.csv", index=False)

    per_year_results = pd.concat([per_year_forward, per_year_reverse], ignore_index=True)
    per_year_results.to_csv(args.out_dir / "granger_per_year.csv", index=False)

    pooled_summary = pd.DataFrame(
        [
            {
                "direction": f"google_trends -> {args.media_var}",
                "max_lag": args.max_lag,
                "wald_stat": stat_fw,
                "p_value": pval_fw,
                "null_hypothesis": hyp_fw,
            },
            {
                "direction": f"{args.media_var} -> google_trends",
                "max_lag": args.max_lag,
                "wald_stat": stat_rv,
                "p_value": pval_rv,
                "null_hypothesis": hyp_rv,
            },
        ]
    )
    pooled_summary.to_csv(args.out_dir / "granger_pooled_fe.csv", index=False)

    print("Data prepared and Granger analysis completed.")
    print(f"Rows in merged panel: {len(merged)}")
    print("\nPooled FE Granger-style test (HC3 robust):")
    print(pooled_summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
