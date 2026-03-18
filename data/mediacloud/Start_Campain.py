import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/mediacloud/mediacloud_monthly_1103.csv"
OUTPUT_DIR = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/mediacloud"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_YEARLY_CSV = os.path.join(OUTPUT_DIR, "campaigns_mediacloud_yearly_from_monthly.csv")
OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "campaigns_mediacloud_summary_from_monthly.csv")


# ============================================================
# Helpers
# ============================================================

def compute_yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly MediaCloud counts into yearly counts.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    yearly = (
        df.groupby(["campaign", "year"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "annual_count"})
    )

    yearly = yearly.sort_values(["campaign", "year"]).reset_index(drop=True)

    yearly["prev_annual_count"] = yearly.groupby("campaign")["annual_count"].shift(1)

    yearly["growth_ratio_vs_prev"] = yearly.apply(
        lambda row: (
            row["annual_count"] / row["prev_annual_count"]
            if pd.notna(row["prev_annual_count"]) and row["prev_annual_count"] > 0
            else pd.NA
        ),
        axis=1
    )

    yearly["absolute_increase_vs_prev"] = yearly.apply(
        lambda row: (
            row["annual_count"] - row["prev_annual_count"]
            if pd.notna(row["prev_annual_count"])
            else pd.NA
        ),
        axis=1
    )

    return yearly


def detect_first_mention_year(df_campaign: pd.DataFrame):
    """
    First year with annual_count > 0.
    """
    nonzero = df_campaign[df_campaign["annual_count"] > 0]
    if nonzero.empty:
        return None
    return int(nonzero["year"].min())


def detect_popularization_year(
    df_campaign: pd.DataFrame,
    min_annual_count: int = 500,
    min_growth_ratio: float = 1.5
):
    """
    Heuristic for 'popularization year':
    first year where:
    - annual_count >= min_annual_count
    - growth_ratio_vs_prev >= min_growth_ratio
    """
    candidates = df_campaign[
        (df_campaign["annual_count"] >= min_annual_count)
        & (df_campaign["growth_ratio_vs_prev"].notna())
        & (df_campaign["growth_ratio_vs_prev"] >= min_growth_ratio)
    ]

    if candidates.empty:
        return None

    return int(candidates.iloc[0]["year"])


def build_summary(yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Build one summary row per campaign.
    """
    rows = []

    for campaign, df_campaign in yearly.groupby("campaign"):
        df_campaign = df_campaign.sort_values("year").reset_index(drop=True)

        first_year = detect_first_mention_year(df_campaign)
        popularization_year = detect_popularization_year(df_campaign)

        peak_row = df_campaign.loc[df_campaign["annual_count"].idxmax()]

        rows.append({
            "campaign": campaign,
            "first_nonzero_year": first_year,
            "popularization_year": popularization_year,
            "max_annual_count_year": int(peak_row["year"]),
            "max_annual_count": int(peak_row["annual_count"])
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"campaign", "date", "count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    yearly = compute_yearly_counts(df)
    summary = build_summary(yearly)

    yearly.to_csv(OUTPUT_YEARLY_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\nYearly counts:")
    print(yearly.head(20))

    print("\nSummary:")
    print(summary)

    print("\nSaved files:")
    print(OUTPUT_YEARLY_CSV)
    print(OUTPUT_SUMMARY_CSV)


if __name__ == "__main__":
    main()