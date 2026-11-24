"""
Visualisation des résultats Granger LOYO : Google Trends vs Mediacloud — Mars Bleu

Fichiers attendus dans le même dossier que ce script :
- loyo_train_results.csv
- loyo_test_results.csv
- loyo_stability.csv
- panel_trends_mediacloud.csv

Sorties :
- figures/loyo_01_series_by_season.png       — séries normalisées GT vs MC par saison
- figures/loyo_02_train_pvalues.png          — p-values des folds d'entraînement
- figures/loyo_03_test_pvalues.png           — p-values des saisons de test
- figures/loyo_04_train_vs_test.png          — comparaison train vs test côte à côte
- figures/loyo_05_lags.png                   — meilleurs lags train vs test
- figures/loyo_06_stability.png              — carte de stabilité LOYO
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# 1. Chemins
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
TRAIN_PATH  = BASE_DIR / "loyo_train_results.csv"
TEST_PATH   = BASE_DIR / "loyo_test_results.csv"
STAB_PATH   = BASE_DIR / "loyo_stability.csv"
PANEL_PATH  = BASE_DIR / "panel_trends_mediacloud.csv"
FIG_DIR     = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Chargement
# ---------------------------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
stab_df  = pd.read_csv(STAB_PATH)
panel    = pd.read_csv(PANEL_PATH, parse_dates=["date"])

train_df["left_out"] = (
    train_df["scope"]
    .str.replace("train (sans ", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.strip()
)
test_df["left_out"] = (
    test_df["scope"]
    .str.replace("test (", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.strip()
)

seasons = stab_df["left_out_season"].tolist()
ALPHA   = 0.05

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return s * 0.0
    return 100.0 * (s - s.min()) / (s.max() - s.min())


def add_significance_line(ax, label: bool = True):
    ax.axhline(ALPHA, color="red", linestyle="--", linewidth=1, alpha=0.8)
    if label:
        ax.text(
            0.01, ALPHA + 0.008,
            f"α = {ALPHA}",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color="red",
        )


# ---------------------------------------------------------------------------
# 4. Figure 01 — Séries normalisées par saison
# ---------------------------------------------------------------------------

all_seasons = sorted(panel["season"].unique())
n_cols = 2
n_rows = (len(all_seasons) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), sharex=False)
axes = axes.flatten()

for ax, season in zip(axes, all_seasons):
    sub = panel[panel["season"] == season].sort_values("date").copy()
    ax.plot(sub["date"], normalize_series(sub["trends"]),   label="Google Trends (norm.)")
    ax.plot(sub["date"], normalize_series(sub["mc_count"]), label="Mediacloud (norm.)")
    ax.set_title(season)
    ax.set_ylabel("Index 0–100")
    ax.tick_params(axis="x", rotation=30)

for ax in axes[len(all_seasons):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)
fig.suptitle("Mars Bleu LOYO — Google Trends vs Mediacloud par saison", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG_DIR / "loyo_01_series_by_season.png", dpi=200)
plt.close(fig)
print("Saved: loyo_01_series_by_season.png")

# ---------------------------------------------------------------------------
# 5. Figure 02 — p-values des folds d'entraînement
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 5))
x     = np.arange(len(train_df))
width = 0.38

ax.bar(x - width / 2, train_df["p_mc->trends"], width=width, label="MC → GT")
ax.bar(x + width / 2, train_df["p_trends->mc"], width=width, label="GT → MC")

add_significance_line(ax)
ax.set_xticks(x)
ax.set_xticklabels(train_df["left_out"], rotation=45, ha="right")
ax.set_ylabel("Meilleure p-value (sur les lags)")
ax.set_title("Mars Bleu LOYO — Folds d'entraînement : p-values Granger\n"
             "(saison en abscisse = saison laissée de côté)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "loyo_02_train_pvalues.png", dpi=200)
plt.close(fig)
print("Saved: loyo_02_train_pvalues.png")

# ---------------------------------------------------------------------------
# 6. Figure 03 — p-values des saisons de test
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(test_df))

ax.bar(x - width / 2, test_df["p_mc->trends"], width=width, label="MC → GT")
ax.bar(x + width / 2, test_df["p_trends->mc"], width=width, label="GT → MC")

add_significance_line(ax)
ax.set_xticks(x)
ax.set_xticklabels(test_df["left_out"], rotation=45, ha="right")
ax.set_ylabel("Meilleure p-value (sur les lags)")
ax.set_title("Mars Bleu LOYO — Saisons de test : p-values Granger\n"
             "(chaque barre = Granger sur la saison seule)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "loyo_03_test_pvalues.png", dpi=200)
plt.close(fig)
print("Saved: loyo_03_test_pvalues.png")

# ---------------------------------------------------------------------------
# 7. Figure 04 — Comparaison train vs test (4 sous-graphes)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

directions = [
    ("MC → GT", "train_p_mc->gt", "test_p_mc->gt",  axes[0, 0], axes[0, 1]),
    ("GT → MC", "train_p_gt->mc", "test_p_gt->mc",  axes[1, 0], axes[1, 1]),
]

for label, train_col, test_col, ax_train, ax_test in directions:
    x = np.arange(len(stab_df))

    colors_train = [
        "#2ecc71" if p < ALPHA else "#e74c3c"
        for p in stab_df[train_col].fillna(1)
    ]
    ax_train.bar(x, stab_df[train_col].fillna(1), color=colors_train, edgecolor="white")
    add_significance_line(ax_train)
    ax_train.set_xticks(x)
    ax_train.set_xticklabels(stab_df["left_out_season"], rotation=45, ha="right", fontsize=8)
    ax_train.set_title(f"{label} — Entraînement (sans la saison)")
    ax_train.set_ylabel("p-value")

    colors_test = [
        "#2ecc71" if p < ALPHA else "#e74c3c"
        for p in stab_df[test_col].fillna(1)
    ]
    ax_test.bar(x, stab_df[test_col].fillna(1), color=colors_test, edgecolor="white")
    add_significance_line(ax_test)
    ax_test.set_xticks(x)
    ax_test.set_xticklabels(stab_df["left_out_season"], rotation=45, ha="right", fontsize=8)
    ax_test.set_title(f"{label} — Test (saison seule)")
    ax_test.set_ylabel("p-value")

sig_patch   = mpatches.Patch(color="#2ecc71", label=f"p < {ALPHA} (significatif)")
insig_patch = mpatches.Patch(color="#e74c3c", label=f"p ≥ {ALPHA} (non significatif)")
fig.legend(handles=[sig_patch, insig_patch], loc="upper center", ncol=2, fontsize=10)
fig.suptitle("Mars Bleu LOYO — Comparaison train vs test (p-values)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "loyo_04_train_vs_test.png", dpi=200)
plt.close(fig)
print("Saved: loyo_04_train_vs_test.png")

# ---------------------------------------------------------------------------
# 8. Figure 05 — Meilleurs lags train vs test
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (direction_label, train_lag_col, test_lag_col) in zip(
    axes,
    [
        ("MC → GT", "train_lag_mc->gt", "test_lag_mc->gt"),
        ("GT → MC", "train_lag_gt->mc", "test_lag_gt->mc"),
    ],
):
    ax.plot(
        stab_df["left_out_season"],
        pd.to_numeric(stab_df[train_lag_col], errors="coerce"),
        marker="o", linestyle="-", label="Train (sans la saison)",
    )
    ax.plot(
        stab_df["left_out_season"],
        pd.to_numeric(stab_df[test_lag_col], errors="coerce"),
        marker="s", linestyle="--", label="Test (saison seule)",
    )
    ax.set_xticks(range(len(stab_df)))
    ax.set_xticklabels(stab_df["left_out_season"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Lag optimal (jours)")
    ax.set_title(f"Meilleur lag — {direction_label}")
    ax.legend(fontsize=9)

fig.suptitle("Mars Bleu LOYO — Meilleurs lags Granger : train vs test")
fig.tight_layout()
fig.savefig(FIG_DIR / "loyo_05_lags.png", dpi=200)
plt.close(fig)
print("Saved: loyo_05_lags.png")

# ---------------------------------------------------------------------------
# 9. Figure 06 — Carte de stabilité LOYO
# ---------------------------------------------------------------------------

dir_order  = ["MC->GT", "GT->MC", "BIDIR", "NONE"]
dir_to_int = {d: i for i, d in enumerate(dir_order)}
colors_map = ["#2196F3", "#FF9800", "#9C27B0", "#9E9E9E"]

cols_to_show = {
    "direction_all":   "Toutes\nsaisons",
    "direction_train": "Train\n(sans Y)",
    "direction_test":  "Test\n(Y seule)",
}

matrix = np.array([
    [dir_to_int.get(str(stab_df[col].iloc[i]), 3) for col in cols_to_show]
    for i in range(len(stab_df))
])

fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(stab_df) + 2)))

cmap = ListedColormap(colors_map)
ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto")

ax.set_xticks(range(len(cols_to_show)))
ax.set_xticklabels(list(cols_to_show.values()), fontsize=10)
ax.set_yticks(range(len(stab_df)))
ax.set_yticklabels(stab_df["left_out_season"], fontsize=9)

for i in range(len(stab_df)):
    for j, col in enumerate(cols_to_show):
        val = str(stab_df[col].iloc[i])
        ax.text(j, i, val, ha="center", va="center", fontsize=8,
                color="white" if val != "NONE" else "#333333", fontweight="bold")

for i, inf in enumerate(stab_df["influential"]):
    if inf:
        ax.add_patch(plt.Rectangle(
            (-0.5, i - 0.5), len(cols_to_show), 1,
            fill=False, edgecolor="red", linewidth=2.5, clip_on=False,
        ))

legend_patches = [
    mpatches.Patch(color=colors_map[i], label=dir_order[i]) for i in range(4)
]
inf_patch = mpatches.Patch(fill=False, edgecolor="red", linewidth=2,
                            label="Saison influente")
legend_patches.append(inf_patch)
ax.legend(
    handles=legend_patches, loc="upper left",
    bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9,
)

ax.set_title("Mars Bleu LOYO — Stabilité de la direction causale\n"
             "(rouge = retirer cette saison change la conclusion)")
fig.tight_layout()
fig.savefig(FIG_DIR / "loyo_06_stability.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: loyo_06_stability.png")

# ---------------------------------------------------------------------------
# 10. Résumé console
# ---------------------------------------------------------------------------

print("\n=== Mars Bleu LOYO — Résultats folds d'entraînement ===")
print(train_df[["scope", "n_obs", "p_mc->trends", "lag_mc->trends",
                "p_trends->mc", "lag_trends->mc", "direction"]].to_string(index=False))

print("\n=== Mars Bleu LOYO — Résultats saisons de test ===")
print(test_df[["scope", "n_obs", "p_mc->trends", "lag_mc->trends",
               "p_trends->mc", "lag_trends->mc", "direction"]].to_string(index=False))

print("\n=== Mars Bleu LOYO — Stabilité ===")
print(stab_df[[
    "left_out_season", "direction_all", "direction_train", "direction_test",
    "train_test_agree", "influential",
]].to_string(index=False))

n_inf    = stab_df["influential"].sum()
n_agree  = stab_df["train_test_agree"].sum()
n_total  = len(stab_df)
majority = stab_df["direction_train"].mode().iloc[0]
print(f"\nSaisons influentes           : {n_inf}/{n_total}")
print(f"Accord train/test            : {n_agree}/{n_total}")
print(f"Direction majoritaire (train): {majority}")
print(f"\nFigures sauvegardées dans : {FIG_DIR}")
