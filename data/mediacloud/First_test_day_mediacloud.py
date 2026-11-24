import os
import time
import random
import requests
import pandas as pd

# ----------------------------
# Campaign configuration
# ----------------------------
campaign_name = "Dry January"
campaign_query = "dry january OR mois sans alcool OR arreter de boire OR janvier sans alcool OR arreter alcool"
MC_API_KEY = "c7632b98c833655afc3269e67027f5a900f369ee"

# ----------------------------
# Test window
# ----------------------------
test_start = "2024-12-15"
test_end   = "2025-01-31"

days_before_peak = 15
days_after_peak  = 30

# ----------------------------
# Output directory
# ----------------------------
output_dir = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/mediacloud"
os.makedirs(output_dir, exist_ok=True)


def pause():
    t = random.uniform(3, 6)
    print(f"  Sleeping {t:.1f}s...")
    time.sleep(t)


def fetch_daily_counts(query, start_date, end_date):
    params = {
        "q":          query,
        "start_date": start_date,
        "end_date":   end_date,
        "api_key":    MC_API_KEY
    }
    response = requests.get(
        "https://search.mediacloud.org/api/v1/story-count-over-time",
        params=params
    )
    
    # DEBUG
    print(f"  Status code: {response.status_code}")
    print(f"  URL called: {response.url}")
    print(f"  Response text: {response.text[:500]}")
    
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["counts"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ----------------------------
# Step 1: fetch test window
# ----------------------------
print("Step 1 — fetching test window...")
pause()

df_broad = fetch_daily_counts(campaign_query, test_start, test_end)

if df_broad.empty:
    raise ValueError("No data returned for the test window.")

# Aggregate by week to find peak week
df_broad["week"] = df_broad["date"].dt.to_period("W").apply(lambda r: r.start_time)
df_weekly = df_broad.groupby("week")["count"].sum().reset_index()
df_weekly.columns = ["week_start", "campaign_index"]

# ----------------------------
# Step 2: detect peak week
# ----------------------------
peak_idx   = df_weekly["campaign_index"].idxmax()
peak_week  = df_weekly.loc[peak_idx, "week_start"]
peak_value = df_weekly.loc[peak_idx, "campaign_index"]

print(f"\nDetected peak week: {peak_week.date()}  (count: {peak_value})")

peak_date = peak_week + pd.Timedelta(days=3)

# ----------------------------
# Step 3: build local window
# ----------------------------
local_start = (peak_date - pd.Timedelta(days=days_before_peak)).strftime("%Y-%m-%d")
local_end   = (peak_date + pd.Timedelta(days=days_after_peak)).strftime("%Y-%m-%d")

print(f"\nLocal window: {local_start} → {local_end}")

# ----------------------------
# Step 4: fetch local window
# ----------------------------
print("\nStep 4 — fetching local window...")
pause()

df_local = fetch_daily_counts(campaign_query, local_start, local_end)

if df_local.empty:
    raise ValueError("No data returned for the local window.")

# ----------------------------
# Add metadata
# ----------------------------
df_local = df_local.rename(columns={"count": "campaign_index"})
df_local["campaign"]            = campaign_name
df_local["query"]               = campaign_query
df_local["detected_peak_date"]  = peak_date.strftime("%Y-%m-%d")
df_local["detected_peak_value"] = peak_value
df_local["window_start"]        = local_start
df_local["window_end"]          = local_end

df_local = df_local[[
    "date", "campaign_index", "campaign", "query",
    "detected_peak_date", "detected_peak_value",
    "window_start", "window_end"
]]

# ----------------------------
# Export CSV
# ----------------------------
output_path = os.path.join(output_dir, "dry_january_mediacloud_peak_window.csv")
df_local.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nSaved to: {output_path}")
print("\nPreview:")
print(df_local.head())