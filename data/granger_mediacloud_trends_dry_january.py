"""
Test de causalité de Granger entre la couverture médiatique (Mediacloud)
et l'attention du public (Google Trends) sur Dry January.

Workflow
--------
1) Charger les Google Trends journaliers (dry_january_all_seasons.csv).
2) Pour chaque saison (déc-fév de chaque année), interroger l'API Mediacloud v4
   pour obtenir le nombre d'articles francophones par jour qui mentionnent
   les mêmes termes que la requête Google Trends.
3) Aligner les deux séries par date.
4) Tester la stationnarité (ADF) et différencier si nécessaire.
5) Appliquer le test de causalité de Granger dans les deux sens :
     - Mediacloud -> Google Trends
     - Google Trends -> Mediacloud
   Saison par saison ET sur toutes saisons concaténées, puis comparer.
6) Exporter les séries combinées et un récapitulatif des p-values.

Pré-requis
----------
    pip install mediacloud pandas numpy statsmodels python-dotenv
"""

from __future__ import annotations

import os
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import mediacloud.api
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

MC_API_KEY = os.getenv("MC_API_KEY", "")

# Collection Mediacloud "French language online news"
MC_COLLECTION_ID = 34412146

# Mots-clés Google Trends (même requête que dans les scripts de collecte)
TRENDS_QUERY = (
    "dry january + mois sans alcool + arreter de boire "
    "+ janvier sans alcool + arreter alcool"
)

INPUT_CSV  = Path.home() / "Desktop" / "Tronc commun 2A" / "Projet S8" / "Data_PublicHealth_CentraleSupelec" / "data" / "google_trends" / "dry_january_all_seasons.csv"
OUTPUT_DIR = Path.home() / "Desktop" / "Tronc commun 2A" / "Projet S8" / "Data_PublicHealth_CentraleSupelec" / "data" / "outputs_granger_dry_january"
OUTPUT_DIR.mkdir(exist_ok=True)

FRENCH_STOPWORDS = {"de", "du", "des", "le", "la", "les", "un", "une",
                    "et", "ou", "à", "en", "au", "aux"}

# Lags à tester pour Granger (en jours).
GRANGER_MAX_LAG = 7

# Politesse API : pause entre saisons.
SLEEP_BETWEEN_SEASONS = 1.0


# ---------------------------------------------------------------------------
# 1. CHARGEMENT GOOGLE TRENDS
# ---------------------------------------------------------------------------

def load_trends(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"]   = pd.to_datetime(df["window_end"])
    df = df.rename(columns={"campaign_index": "trends"})
    df = df[["date", "season", "trends", "query", "window_start", "window_end"]].sort_values("date").reset_index(drop=True)
    return df


def build_mc_query_from_trends_query(trends_query: str) -> str:
    """
    Convertit la requête Google Trends 'A + B + C' en requête Mediacloud
    Elasticsearch '(termeA) OR (termeB) OR (termeC)' où chaque terme est
    représenté par un AND de ses mots significatifs (stopwords retirés).

    Ex : 'dry january + mois sans alcool'
       -> '((dry AND january) OR (mois AND sans AND alcool)) AND language:fr'
    """
    raw_terms = [t.strip() for t in trends_query.split("+") if t.strip()]
    term_clauses = []
    for term in raw_terms:
        words = [w.lower() for w in term.split() if w.lower() not in FRENCH_STOPWORDS]
        if not words:
            continue
        if len(words) == 1:
            term_clauses.append(words[0])
        else:
            term_clauses.append("(" + " AND ".join(words) + ")")

    or_block = " OR ".join(term_clauses)
    return f"({or_block}) AND language:fr"


# ---------------------------------------------------------------------------
# 2. MEDIACLOUD : COMPTAGE D'ARTICLES PAR JOUR
# ---------------------------------------------------------------------------

def fetch_mediacloud_daily(
    mc_search: mediacloud.api.SearchApi,
    query: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Retourne un DataFrame [date, mc_count] jour par jour sur la fenêtre.
    Utilise story_count_over_time (une seule requête par saison).
    Les jours sans article sont imputés à 0.
    """
    results = mc_search.story_count_over_time(
        query,
        start_date=start_date,
        end_date=end_date,
        collection_ids=[MC_COLLECTION_ID],
    )

    if not results:
        return pd.DataFrame(columns=["date", "mc_count"])

    rows = []
    for r in results:
        date_str = r.get("date")
        count    = r.get("count", r.get("total_count", 0))
        if date_str is None:
            continue
        rows.append({"date": pd.to_datetime(date_str).normalize(), "mc_count": int(count)})

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    out = (
        out.set_index("date")
           .reindex(full_idx, fill_value=0)
           .rename_axis("date")
           .reset_index()
    )
    return out


def build_combined_panel(trends_df: pd.DataFrame, mc_api_key: str, mc_query: str) -> pd.DataFrame:
    mc_search = mediacloud.api.SearchApi(mc_api_key)

    pieces = []
    for season, sub in trends_df.groupby("season", sort=True):
        start_date = sub["window_start"].iloc[0].date()
        end_date   = sub["window_end"].iloc[0].date()
        print(f"[Mediacloud] {season} : {start_date} -> {end_date}  ({(end_date - start_date).days + 1} jours)")

        mc_df = fetch_mediacloud_daily(mc_search, mc_query, start_date, end_date)
        merged = sub.merge(mc_df, on="date", how="left").fillna({"mc_count": 0})
        merged["mc_count"] = merged["mc_count"].astype(int)
        pieces.append(merged)

        time.sleep(SLEEP_BETWEEN_SEASONS)

    panel = pd.concat(pieces, ignore_index=True).sort_values(["season", "date"])
    return panel


# ---------------------------------------------------------------------------
# 3. STATIONNARITE + GRANGER
# ---------------------------------------------------------------------------

def adf_pvalue(series: pd.Series) -> float:
    series = series.dropna()
    if series.nunique() < 2 or len(series) < 10:
        return np.nan
    try:
        return adfuller(series, autolag="AIC")[1]
    except Exception:
        return np.nan


def make_stationary(series: pd.Series, alpha: float = 0.05) -> tuple[pd.Series, int]:
    s = series.copy()
    for d in range(3):
        p = adf_pvalue(s)
        if np.isnan(p) or p < alpha:
            return s, d
        s = s.diff().dropna()
    return s, 2


def run_granger(
    y: pd.Series,
    x: pd.Series,
    max_lag: int = GRANGER_MAX_LAG,
) -> dict:
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < max_lag + 5:
        return {"best_lag": None, "best_pvalue": np.nan, "pvalues_by_lag": {}}

    try:
        res = grangercausalitytests(df[["y", "x"]], maxlag=max_lag)
    except Exception as e:
        return {"best_lag": None, "best_pvalue": np.nan, "pvalues_by_lag": {}, "error": str(e)}

    pvals = {lag: round(res[lag][0]["ssr_ftest"][1], 5) for lag in res}
    best_lag = min(pvals, key=pvals.get)
    return {
        "best_lag": best_lag,
        "best_pvalue": pvals[best_lag],
        "pvalues_by_lag": pvals,
    }


def granger_both_directions(
    label: str,
    trends: pd.Series,
    mc: pd.Series,
    max_lag: int = GRANGER_MAX_LAG,
) -> dict:
    trends_s, d_t = make_stationary(trends)
    mc_s,     d_m = make_stationary(mc)

    common = pd.concat([trends_s, mc_s], axis=1).dropna()
    common.columns = ["trends", "mc"]

    mc_to_trends = run_granger(common["trends"], common["mc"], max_lag)
    trends_to_mc = run_granger(common["mc"], common["trends"], max_lag)

    return {
        "label": label,
        "n_obs": len(common),
        "diff_order_trends": d_t,
        "diff_order_mc": d_m,
        "mc_to_trends": mc_to_trends,
        "trends_to_mc": trends_to_mc,
    }


# ---------------------------------------------------------------------------
# 4. INTERPRETATION
# ---------------------------------------------------------------------------

def interpret(result: dict, alpha: float = 0.05) -> str:
    p_mc_to_t = result["mc_to_trends"]["best_pvalue"]
    p_t_to_mc = result["trends_to_mc"]["best_pvalue"]

    def sig(p):
        return (not np.isnan(p)) and (p < alpha)

    mc_causes_t = sig(p_mc_to_t)
    t_causes_mc = sig(p_t_to_mc)

    if mc_causes_t and not t_causes_mc:
        return "Mediacloud Granger-cause Google Trends (les médias précèdent l'attention)."
    if t_causes_mc and not mc_causes_t:
        return "Google Trends Granger-cause Mediacloud (l'attention précède la couverture)."
    if mc_causes_t and t_causes_mc:
        return "Causalité bidirectionnelle (feedback loop)."
    return "Aucune causalité de Granger significative dans un sens ou dans l'autre."


def summarize(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "scope": r["label"],
            "n_obs": r["n_obs"],
            "diff_trends": r["diff_order_trends"],
            "diff_mc": r["diff_order_mc"],
            "p_mc->trends": r["mc_to_trends"]["best_pvalue"],
            "lag_mc->trends": r["mc_to_trends"]["best_lag"],
            "p_trends->mc": r["trends_to_mc"]["best_pvalue"],
            "lag_trends->mc": r["trends_to_mc"]["best_lag"],
            "interpretation": interpret(r),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    print("== Chargement Google Trends ==")
    trends_df = load_trends(INPUT_CSV)
    print(f"  {len(trends_df)} lignes, {trends_df['season'].nunique()} saisons.")

    mc_query = build_mc_query_from_trends_query(TRENDS_QUERY)
    print("\n== Alignement des requêtes ==")
    print(f"  Google Trends : {TRENDS_QUERY}")
    print(f"  Mediacloud    : {mc_query}")

    print("\n== Récupération Mediacloud (peut prendre un peu de temps) ==")
    panel = build_combined_panel(trends_df, MC_API_KEY, mc_query)
    panel_path = OUTPUT_DIR / "panel_trends_mediacloud.csv"
    panel.to_csv(panel_path, index=False)
    print(f"  Panel exporté -> {panel_path}")

    print("\n== Test de Granger ==")
    results = []

    for season, sub in panel.groupby("season"):
        sub = sub.sort_values("date").reset_index(drop=True)
        if sub["mc_count"].nunique() < 3 or sub["trends"].nunique() < 3:
            print(f"  [{season}] série trop plate -> skip")
            continue
        r = granger_both_directions(
            label=f"saison {season}",
            trends=sub["trends"].astype(float).reset_index(drop=True),
            mc=sub["mc_count"].astype(float).reset_index(drop=True),
        )
        results.append(r)
        print(f"  [{season}] {interpret(r)}")

    print("\n== Toutes saisons concaténées (après différenciation par saison) ==")
    pooled_pieces = []
    for season, sub in panel.groupby("season"):
        sub = sub.sort_values("date").reset_index(drop=True)
        t_s, _ = make_stationary(sub["trends"].astype(float))
        m_s, _ = make_stationary(sub["mc_count"].astype(float))
        block = pd.concat([t_s.reset_index(drop=True),
                           m_s.reset_index(drop=True)], axis=1).dropna()
        block.columns = ["trends", "mc"]
        pooled_pieces.append(block)

    pooled = pd.concat(pooled_pieces, ignore_index=True)
    mc_to_trends = run_granger(pooled["trends"], pooled["mc"], GRANGER_MAX_LAG)
    trends_to_mc = run_granger(pooled["mc"], pooled["trends"], GRANGER_MAX_LAG)
    pooled_result = {
        "label": "toutes saisons (pooled)",
        "n_obs": len(pooled),
        "diff_order_trends": "par saison",
        "diff_order_mc": "par saison",
        "mc_to_trends": mc_to_trends,
        "trends_to_mc": trends_to_mc,
    }
    results.append(pooled_result)
    print(f"  {interpret(pooled_result)}")

    summary = summarize(results)
    summary_path = OUTPUT_DIR / "granger_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n== Résumé exporté -> {summary_path} ==")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
