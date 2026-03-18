# 11/03 Données monthly, csv séparés, 5 termes pour chaque campagne (apres snowball manuel)

import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pytrends.request import TrendReq

pytrends = TrendReq(hl="fr-FR", tz=120, timeout=(60, 120))

OUTPUT_DIR = Path(r"C:\Users\tedvm\Centrale\PoleProjetS8\Data_PublicHealth_CentraleSupelec\data\google_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT = OUTPUT_DIR / "trends_plot.png"

CAMPAIGNS = {
    "Dry January":  "dry january + mois sans alcool + arreter de boire + janvier sans alcool + arreter alcool",
    "Octobre Rose": "Octobre rose + cancer du sein + mammographie + dépistage cancer du sein + cancer du sein symptôme",
    "Mars Bleu":    "Mars bleu + cancer colorectal + dépistage colon + coloscopie + cancer colon",
    "Movember":     "Movember + cancer prostate + movember prostate + cancer prostate psa + cancer prostate traitement",
}

monthly = {}

for campaign, query in CAMPAIGNS.items():
    print(f"\n── {campaign} ──")
    slug = campaign.lower().replace(" ", "_")
    csv_path = OUTPUT_DIR / f"{slug}.csv"

    for attempt in range(3):
        try:
            pytrends.build_payload([query], timeframe="2005-01-01 2016-01-01", geo="FR")
            df = pytrends.interest_over_time().drop(columns=["isPartial"], errors="ignore")
            time.sleep(10)

            if df.empty:
                raise ValueError("Aucune donnée renvoyée par Google Trends.")

            df.index = pd.to_datetime(df.index)
            df = df.resample("MS").mean().round(1)

            df_out = df.reset_index()
            df_out.columns = ["date", "interest"]
            df_out["campaign"] = campaign
            df_out[["campaign", "date", "interest"]].to_csv(csv_path, index=False)

            monthly[campaign] = df
            print(f"✓ {len(df)} mois récupérés → {csv_path}")
            break

        except Exception as e:
            print(f"Erreur lors de la requête : {e}")
            wait = 30 * (attempt + 1)
            print(f"⚠ tentative {attempt+1}/3 — attente {wait}s")
            time.sleep(wait)

    time.sleep(80)  # petite pause entre campagnes

if monthly:
    fig, ax = plt.subplots(figsize=(16, 6))
    for campaign, df in monthly.items():
        col = df.columns[0]
        ax.plot(df.index, df[col], linewidth=2, label=campaign)

    ax.set_title("Google Trends — Campagnes santé publique (FR) 2005–2016",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Intérêt agrégé (0–100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Plot → {OUTPUT_PLOT}")