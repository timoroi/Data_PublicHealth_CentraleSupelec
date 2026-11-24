"""
Visualisation des résultats Granger : Google Trends vs Mediacloud — Dry January

Fichiers attendus dans le même dossier que ce script :
- granger_summary.csv
- panel_trends_mediacloud.csv

Sorties :
- figures/01_series_by_season.png
- figures/02_pvalues_by_season.png
- figures/03_lags_by_season.png
- figures/04_media_volume_by_season.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "granger_summary.csv"
PANEL_PATH = BASE_DIR / "panel_trends_mediacloud.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------

summary = pd.read_csv(SUMMARY_PATH)
panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])

season_summary = summary[summary["scope"].str.startswith("saison")].copy()
season_summary["season"] = season_summary["scope"].str.replace("saison ", "", regex=False)

ALPHA = 0.05


# ---------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return s * 0
    return 100 * (s - s.min()) / (s.max() - s.min())


def add_significance_line(ax):
    ax.axhline(ALPHA, linestyle="--", linewidth=1)
    ax.text(
        0.01,
        ALPHA + 0.005,
        "5% threshold",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
    )


# ---------------------------------------------------------------------
# 4. Plot daily series by season
# ---------------------------------------------------------------------

seasons = sorted(panel["season"].unique())
n_cols = 2
n_rows = (len(seasons) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), sharex=False)
axes = axes.flatten()

for ax, season in zip(axes, seasons):
    sub = panel[panel["season"] == season].sort_values("date").copy()

    ax.plot(sub["date"], normalize_series(sub["trends"]), label="Google Trends, normalized")
    ax.plot(sub["date"], normalize_series(sub["mc_count"]), label="Mediacloud, normalized")

    ax.set_title(season)
    ax.set_ylabel("Index 0-100")
    ax.tick_params(axis="x", rotation=30)

for ax in axes[len(seasons):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("Dry January — Google Trends vs Mediacloud by season", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG_DIR / "01_series_by_season.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------
# 5. Plot p-values by direction
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))

x = range(len(season_summary))
width = 0.38

ax.bar(
    [i - width / 2 for i in x],
    season_summary["p_mc->trends"],
    width=width,
    label="Mediacloud -> Google Trends",
)
ax.bar(
    [i + width / 2 for i in x],
    season_summary["p_trends->mc"],
    width=width,
    label="Google Trends -> Mediacloud",
)

add_significance_line(ax)
ax.set_xticks(list(x))
ax.set_xticklabels(season_summary["season"], rotation=45, ha="right")
ax.set_ylabel("Best p-value across lags")
ax.set_title("Dry January — Granger p-values by season")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "02_pvalues_by_season.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------
# 6. Plot best lags by direction
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    season_summary["season"],
    season_summary["lag_mc->trends"],
    marker="o",
    label="Best lag: Mediacloud -> Google Trends",
)
ax.plot(
    season_summary["season"],
    season_summary["lag_trends->mc"],
    marker="o",
    label="Best lag: Google Trends -> Mediacloud",
)

ax.set_xticks(range(len(season_summary)))
ax.set_xticklabels(season_summary["season"], rotation=45, ha="right")
ax.set_ylabel("Lag in days")
ax.set_title("Dry January — Best Granger lag by season")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "03_lags_by_season.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------
# 7. Plot Mediacloud total volume by season
# ---------------------------------------------------------------------

volume = (
    panel.groupby("season", as_index=False)
    .agg(
        total_articles=("mc_count", "sum"),
        max_daily_articles=("mc_count", "max"),
        mean_daily_articles=("mc_count", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(volume["season"], volume["total_articles"])
ax.set_xticks(range(len(volume)))
ax.set_xticklabels(volume["season"], rotation=45, ha="right")
ax.set_ylabel("Total Mediacloud articles")
ax.set_title("Dry January — Mediacloud volume by season")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_media_volume_by_season.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------
# 8. Console summary
# ---------------------------------------------------------------------

print("\n=== Dry January — Season-level Granger summary ===")
cols = [
    "scope",
    "n_obs",
    "p_mc->trends",
    "lag_mc->trends",
    "p_trends->mc",
    "lag_trends->mc",
    "interpretation",
]
print(summary[cols].to_string(index=False))

print("\nFigures saved in:", FIG_DIR)
