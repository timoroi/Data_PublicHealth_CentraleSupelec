"""
Test de causalité de Granger – Leave-One-Year-Out (LOYO)
entre la couverture médiatique (Mediacloud) et l'attention du public
(Google Trends) sur Dry January.

Workflow LOYO
-------------
Pour chaque saison Y :
  - Ensemble d'entraînement  : toutes les saisons SAUF Y (données poolées,
    différenciées par saison pour la stationnarité).
  - Ensemble de test         : uniquement la saison Y.
  → Le test de Granger est appliqué sur l'entraînement ET sur le test.
  → On compare : le résultat de l'entraînement est-il cohérent avec le test ?
    La saison Y est-elle "influente" (son retrait change-t-il la conclusion) ?

Sorties
-------
  loyo_train_results.csv  – résultats Granger sur chaque fold d'entraînement
  loyo_test_results.csv   – résultats Granger sur chaque saison laissée de côté
  loyo_stability.csv      – tableau de stabilité (accord train/test, influence)
  panel_trends_mediacloud.csv – panel brut (réutilisé si déjà présent)

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

MC_COLLECTION_ID = 34412146

TRENDS_QUERY = (
    "dry january + mois sans alcool + arreter de boire "
    "+ janvier sans alcool + arreter alcool"
)

INPUT_CSV  = (
    Path.home() / "Desktop" / "Tronc commun 2A" / "Projet S8"
    / "Data_PublicHealth_CentraleSupelec" / "data" / "google_trends"
    / "dry_january_all_seasons.csv"
)
OUTPUT_DIR = (
    Path.home() / "Desktop" / "Tronc commun 2A" / "Projet S8"
    / "Data_PublicHealth_CentraleSupelec" / "data"
    / "outputs_granger_dry_january"
)
OUTPUT_DIR.mkdir(exist_ok=True)

FRENCH_STOPWORDS = {"de", "du", "des", "le", "la", "les", "un", "une",
                    "et", "ou", "à", "en", "au", "aux"}

GRANGER_MAX_LAG = 7
SLEEP_BETWEEN_SEASONS = 1.0
ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. CHARGEMENT GOOGLE TRENDS
# ---------------------------------------------------------------------------

def load_trends(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"]         = pd.to_datetime(df["date"])
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"]   = pd.to_datetime(df["window_end"])
    df = df.rename(columns={"campaign_index": "trends"})
    return (
        df[["date", "season", "trends", "query", "window_start", "window_end"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_mc_query_from_trends_query(trends_query: str) -> str:
    raw_terms = [t.strip() for t in trends_query.split("+") if t.strip()]
    clauses = []
    for term in raw_terms:
        words = [w.lower() for w in term.split() if w.lower() not in FRENCH_STOPWORDS]
        if not words:
            continue
        clauses.append(words[0] if len(words) == 1 else "(" + " AND ".join(words) + ")")
    return f"({' OR '.join(clauses)}) AND language:fr"


# ---------------------------------------------------------------------------
# 2. MEDIACLOUD : COMPTAGE D'ARTICLES PAR JOUR
# ---------------------------------------------------------------------------

def fetch_mediacloud_daily(
    mc_search: mediacloud.api.SearchApi,
    query: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    results = mc_search.story_count_over_time(
        query,
        start_date=start_date,
        end_date=end_date,
        collection_ids=[MC_COLLECTION_ID],
    )
    if not results:
        return pd.DataFrame(columns=["date", "mc_count"])

    rows = [
        {"date": pd.to_datetime(r["date"]).normalize(),
         "mc_count": int(r.get("count", r.get("total_count", 0)))}
        for r in results if r.get("date") is not None
    ]
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    return (
        out.set_index("date")
           .reindex(full_idx, fill_value=0)
           .rename_axis("date")
           .reset_index()
    )


def build_combined_panel(
    trends_df: pd.DataFrame,
    mc_api_key: str,
    mc_query: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Construit le panel brut. Si cache_path existe déjà, le charge directement."""
    if cache_path is not None and cache_path.exists():
        print(f"  Panel déjà présent, chargement depuis {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["date", "window_start", "window_end"])

    mc_search = mediacloud.api.SearchApi(mc_api_key)
    pieces = []
    for season, sub in trends_df.groupby("season", sort=True):
        start_date = sub["window_start"].iloc[0].date()
        end_date   = sub["window_end"].iloc[0].date()
        print(f"  [Mediacloud] {season} : {start_date} -> {end_date}")
        mc_df  = fetch_mediacloud_daily(mc_search, mc_query, start_date, end_date)
        merged = sub.merge(mc_df, on="date", how="left").fillna({"mc_count": 0})
        merged["mc_count"] = merged["mc_count"].astype(int)
        pieces.append(merged)
        time.sleep(SLEEP_BETWEEN_SEASONS)

    panel = pd.concat(pieces, ignore_index=True).sort_values(["season", "date"])
    if cache_path is not None:
        panel.to_csv(cache_path, index=False)
        print(f"  Panel exporté -> {cache_path}")
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


def make_stationary(series: pd.Series, alpha: float = ALPHA) -> tuple[pd.Series, int]:
    s = series.copy()
    for d in range(3):
        p = adf_pvalue(s)
        if np.isnan(p) or p < alpha:
            return s, d
        s = s.diff().dropna()
    return s, 2


def run_granger(y: pd.Series, x: pd.Series, max_lag: int = GRANGER_MAX_LAG) -> dict:
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < max_lag + 5:
        return {"best_lag": None, "best_pvalue": np.nan, "pvalues_by_lag": {}}
    try:
        res = grangercausalitytests(df[["y", "x"]], maxlag=max_lag, verbose=False)
    except Exception as e:
        return {"best_lag": None, "best_pvalue": np.nan, "pvalues_by_lag": {}, "error": str(e)}

    pvals = {lag: round(res[lag][0]["ssr_ftest"][1], 5) for lag in res}
    best_lag = min(pvals, key=pvals.get)
    return {"best_lag": best_lag, "best_pvalue": pvals[best_lag], "pvalues_by_lag": pvals}


def pool_seasons(panel: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """
    Concatène les séries différenciées (par saison) pour un ensemble de saisons.
    Retourne un DataFrame [trends, mc] prêt pour run_granger.
    """
    pieces = []
    for season in seasons:
        sub = panel[panel["season"] == season].sort_values("date").reset_index(drop=True)
        if sub["mc_count"].nunique() < 3 or sub["trends"].nunique() < 3:
            continue
        t_s, _ = make_stationary(sub["trends"].astype(float))
        m_s, _ = make_stationary(sub["mc_count"].astype(float))
        block = pd.concat(
            [t_s.reset_index(drop=True), m_s.reset_index(drop=True)], axis=1
        ).dropna()
        block.columns = ["trends", "mc"]
        pieces.append(block)
    if not pieces:
        return pd.DataFrame(columns=["trends", "mc"])
    return pd.concat(pieces, ignore_index=True)


def granger_pooled(label: str, pooled: pd.DataFrame) -> dict:
    """Lance Granger dans les deux sens sur un pool déjà différencié."""
    if pooled.empty:
        empty_r = {"best_lag": None, "best_pvalue": np.nan, "pvalues_by_lag": {}}
        return {
            "label": label, "n_obs": 0,
            "diff_order_trends": "n/a", "diff_order_mc": "n/a",
            "mc_to_trends": empty_r, "trends_to_mc": empty_r,
        }
    mc_to_trends = run_granger(pooled["trends"], pooled["mc"])
    trends_to_mc = run_granger(pooled["mc"], pooled["trends"])
    return {
        "label": label,
        "n_obs": len(pooled),
        "diff_order_trends": "par saison",
        "diff_order_mc": "par saison",
        "mc_to_trends": mc_to_trends,
        "trends_to_mc": trends_to_mc,
    }


def granger_single_season(label: str, sub: pd.DataFrame) -> dict:
    """Lance Granger sur une seule saison (différenciée si besoin)."""
    t_s, d_t = make_stationary(sub["trends"].astype(float).reset_index(drop=True))
    m_s, d_m = make_stationary(sub["mc_count"].astype(float).reset_index(drop=True))
    common = pd.concat([t_s.reset_index(drop=True), m_s.reset_index(drop=True)], axis=1).dropna()
    common.columns = ["trends", "mc"]
    mc_to_trends = run_granger(common["trends"], common["mc"])
    trends_to_mc = run_granger(common["mc"], common["trends"])
    return {
        "label": label,
        "n_obs": len(common),
        "diff_order_trends": d_t,
        "diff_order_mc": d_m,
        "mc_to_trends": mc_to_trends,
        "trends_to_mc": trends_to_mc,
    }


# ---------------------------------------------------------------------------
# 4. INTERPRETATION & RESUME
# ---------------------------------------------------------------------------

def sig(p: float, alpha: float = ALPHA) -> bool:
    return (not np.isnan(p)) and (p < alpha)


def interpret(result: dict, alpha: float = ALPHA) -> str:
    p_mc  = result["mc_to_trends"]["best_pvalue"]
    p_gt  = result["trends_to_mc"]["best_pvalue"]
    mc_ok = sig(p_mc, alpha)
    gt_ok = sig(p_gt, alpha)
    if mc_ok and not gt_ok:
        return "MC->GT  (médias précèdent l'attention)"
    if gt_ok and not mc_ok:
        return "GT->MC  (attention précède la couverture)"
    if mc_ok and gt_ok:
        return "BIDIR   (feedback loop)"
    return "NONE    (aucune causalité significative)"


def direction(result: dict, alpha: float = ALPHA) -> str:
    p_mc  = result["mc_to_trends"]["best_pvalue"]
    p_gt  = result["trends_to_mc"]["best_pvalue"]
    mc_ok = sig(p_mc, alpha)
    gt_ok = sig(p_gt, alpha)
    if mc_ok and not gt_ok:
        return "MC->GT"
    if gt_ok and not mc_ok:
        return "GT->MC"
    if mc_ok and gt_ok:
        return "BIDIR"
    return "NONE"


def result_to_row(r: dict) -> dict:
    return {
        "scope":          r["label"],
        "n_obs":          r["n_obs"],
        "diff_trends":    r["diff_order_trends"],
        "diff_mc":        r["diff_order_mc"],
        "p_mc->trends":   r["mc_to_trends"]["best_pvalue"],
        "lag_mc->trends": r["mc_to_trends"]["best_lag"],
        "p_trends->mc":   r["trends_to_mc"]["best_pvalue"],
        "lag_trends->mc": r["trends_to_mc"]["best_lag"],
        "direction":      direction(r),
        "interpretation": interpret(r),
    }


# ---------------------------------------------------------------------------
# 5. LOYO
# ---------------------------------------------------------------------------

def run_loyo(panel: pd.DataFrame) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """
    Effectue la validation croisée Leave-One-Year-Out.

    Retourne :
      train_results  – liste de résultats Granger sur les folds d'entraînement
      test_results   – liste de résultats Granger sur chaque saison laissée de côté
      stability_df   – DataFrame récapitulatif de stabilité
    """
    seasons_all = sorted(panel["season"].unique())
    n = len(seasons_all)
    print(f"\n  {n} saisons disponibles : {seasons_all}")

    # Référence : toutes saisons poolées
    pooled_all = pool_seasons(panel, seasons_all)
    all_r      = granger_pooled(label="all seasons", pooled=pooled_all)
    dir_all    = direction(all_r)

    train_results  = []
    test_results   = []
    stability_rows = []

    for left_out in seasons_all:
        train_seasons = [s for s in seasons_all if s != left_out]
        print(f"\n  [LOYO] Saison laissée de côté : {left_out}")
        print(f"         Entraînement sur         : {train_seasons}")

        # --- Fold d'entraînement ---
        pooled_train = pool_seasons(panel, train_seasons)
        train_r = granger_pooled(label=f"train (sans {left_out})", pooled=pooled_train)
        train_results.append(train_r)
        print(f"         Entraînement             : {interpret(train_r)}")

        # --- Saison de test seule ---
        sub_test = (
            panel[panel["season"] == left_out]
            .sort_values("date")
            .reset_index(drop=True)
        )
        too_flat = sub_test["mc_count"].nunique() < 3 or sub_test["trends"].nunique() < 3
        if too_flat:
            print(f"         Test [{left_out}]             : série trop plate, skip.")
            empty_r = {"best_lag": None, "best_pvalue": np.nan, "pvalues_by_lag": {}}
            test_r = {
                "label": f"test ({left_out})",
                "n_obs": len(sub_test),
                "diff_order_trends": "n/a",
                "diff_order_mc": "n/a",
                "mc_to_trends": empty_r,
                "trends_to_mc": empty_r,
            }
        else:
            test_r = granger_single_season(label=f"test ({left_out})", sub=sub_test)
            print(f"         Test [{left_out}]             : {interpret(test_r)}")
        test_results.append(test_r)

        dir_train    = direction(train_r)
        dir_test     = direction(test_r)
        influential  = (dir_train != dir_all)

        stability_rows.append({
            "left_out_season":       left_out,
            "n_train_seasons":       len(train_seasons),
            "n_train_obs":           train_r["n_obs"],
            "n_test_obs":            test_r["n_obs"],
            "direction_all":         dir_all,
            "direction_train":       dir_train,
            "direction_test":        dir_test,
            "train_p_mc->gt":        train_r["mc_to_trends"]["best_pvalue"],
            "train_lag_mc->gt":      train_r["mc_to_trends"]["best_lag"],
            "train_p_gt->mc":        train_r["trends_to_mc"]["best_pvalue"],
            "train_lag_gt->mc":      train_r["trends_to_mc"]["best_lag"],
            "test_p_mc->gt":         test_r["mc_to_trends"]["best_pvalue"],
            "test_lag_mc->gt":       test_r["mc_to_trends"]["best_lag"],
            "test_p_gt->mc":         test_r["trends_to_mc"]["best_pvalue"],
            "test_lag_gt->mc":       test_r["trends_to_mc"]["best_lag"],
            "train_test_agree":      dir_train == dir_test,
            "influential":           influential,
            "train_interpretation":  interpret(train_r),
            "test_interpretation":   interpret(test_r),
        })

    stability_df = pd.DataFrame(stability_rows)
    return train_results, test_results, stability_df


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("== Chargement Google Trends ==")
    trends_df = load_trends(INPUT_CSV)
    print(f"  {len(trends_df)} lignes, {trends_df['season'].nunique()} saisons.")

    mc_query = build_mc_query_from_trends_query(TRENDS_QUERY)
    print("\n== Requêtes ==")
    print(f"  Google Trends : {TRENDS_QUERY}")
    print(f"  Mediacloud    : {mc_query}")

    cache_path = OUTPUT_DIR / "panel_trends_mediacloud.csv"
    print("\n== Récupération / chargement du panel Mediacloud ==")
    panel = build_combined_panel(trends_df, MC_API_KEY, mc_query, cache_path=cache_path)

    print("\n== Leave-One-Year-Out Granger ==")
    train_results, test_results, stability_df = run_loyo(panel)

    # --- Export CSV ---
    train_df = pd.DataFrame([result_to_row(r) for r in train_results])
    train_path = OUTPUT_DIR / "loyo_train_results.csv"
    train_df.to_csv(train_path, index=False)

    test_df = pd.DataFrame([result_to_row(r) for r in test_results])
    test_path = OUTPUT_DIR / "loyo_test_results.csv"
    test_df.to_csv(test_path, index=False)

    stab_path = OUTPUT_DIR / "loyo_stability.csv"
    stability_df.to_csv(stab_path, index=False)

    # -----------------------------------------------------------------------
    # AFFICHAGE DES RESULTATS
    # -----------------------------------------------------------------------

    sep = "=" * 65

    print(f"\n{sep}")
    print("RESULTATS GRANGER – FOLDS D'ENTRAINEMENT (toutes saisons sauf Y)")
    print(sep)
    print(train_df[[
        "scope", "n_obs",
        "p_mc->trends", "lag_mc->trends",
        "p_trends->mc", "lag_trends->mc",
        "direction",
    ]].to_string(index=False))

    print(f"\n{sep}")
    print("RESULTATS GRANGER – SAISONS DE TEST (saison Y seule)")
    print(sep)
    print(test_df[[
        "scope", "n_obs",
        "p_mc->trends", "lag_mc->trends",
        "p_trends->mc", "lag_trends->mc",
        "direction",
    ]].to_string(index=False))

    print(f"\n{sep}")
    print("STABILITE LOYO – TABLEAU COMPLET")
    print(sep)
    print(stability_df[[
        "left_out_season",
        "direction_all", "direction_train", "direction_test",
        "train_p_mc->gt", "train_p_gt->mc",
        "test_p_mc->gt",  "test_p_gt->mc",
        "train_test_agree", "influential",
    ]].to_string(index=False))

    print(f"\n{sep}")
    print("SYNTHESE")
    print(sep)

    n_total       = len(stability_df)
    n_influential = stability_df["influential"].sum()
    n_agree       = stability_df["train_test_agree"].sum()
    majority      = stability_df["direction_train"].mode().iloc[0]

    print(f"  Saisons analysées                                   : {n_total}")
    print(f"  Saisons 'influentes' (retrait change la conclusion) : {n_influential}/{n_total}")
    print(f"  Folds où train et test s'accordent                  : {n_agree}/{n_total}")
    print(f"  Direction majoritaire (folds d'entraînement)        : {majority}")

    if n_influential == 0:
        print("\n  => Résultat ROBUSTE : aucune saison unique ne fait basculer la conclusion.")
    else:
        seasons_inf = stability_df.loc[stability_df["influential"], "left_out_season"].tolist()
        print(f"\n  => ATTENTION : retirer {seasons_inf} change la conclusion globale.")

    print(f"\n  Fichiers exportés dans : {OUTPUT_DIR}")
    print(f"    - {train_path.name}")
    print(f"    - {test_path.name}")
    print(f"    - {stab_path.name}")
    print(sep)


if __name__ == "__main__":
    main()
