"""Chronological value-betting backtest.

Selection rule (fixed before looking at test returns):

    For each match and side k, the model's estimated expected net return of a
    one-unit stake at decimal odds o_k is   EV_k = p_poisson_k * o_k - 1.
    Bet one unit on the single side with the largest EV_k if EV_k > tau.

At most one bet per match is placed. Bets are settled at the raw Bet365
pre-closing decimal odds: net return o_k - 1 on a win and -1 on a loss.

The threshold tau is chosen on the validation seasons only, from a grid
declared in ``TAU_GRID``, then frozen and applied once to the test seasons.

Margin hurdle: if the de-vigged market probabilities q_k were the true
probabilities, every bet would have expected return q_k * o_k - 1 =
1 / overround - 1 < 0. This is reported for the selected bets as the
"market-null expected ROI".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SIDES, odds_matrix, market_probs
from .evaluate import PROB_COLS, cluster_bootstrap_mean

TAU_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
MIN_VALIDATION_BETS = 100

SELECTION_COLUMNS = ["match_id", "selected_side", "model_probability", "odds",
                     "estimated_ev", "realized_return"]


def select_bets(df: pd.DataFrame, tau: float, prob_cols=PROB_COLS["poisson"]) -> pd.DataFrame:
    """Apply the EV rule; returns one row per bet in the shared selection schema plus context columns."""
    p = df[list(prob_cols)].to_numpy(float)
    o = odds_matrix(df)
    ev = p * o - 1
    best = ev.argmax(axis=1)
    rows = np.arange(len(df))
    best_ev = ev[rows, best]
    keep = best_ev > tau
    k, r = best[keep], rows[keep]
    won = df["y"].to_numpy()[r] == k
    odds = o[r, k]
    q = market_probs(df)[r, k]
    bets = pd.DataFrame({
        "match_id": df["match_id"].to_numpy()[r],
        "selected_side": np.array(SIDES)[k],
        "model_probability": p[r, k],
        "odds": odds,
        "estimated_ev": best_ev[keep],
        "realized_return": np.where(won, odds - 1.0, -1.0),
        "market_probability": q,
        "market_null_ev": q * odds - 1.0,
        "date": df["date"].to_numpy()[r],
        "season": df["season"].to_numpy()[r],
        "matchday": df["matchday"].to_numpy()[r],
    })
    return bets


def summarise(bets: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    if bets.empty:
        return {"n_bets": 0}
    ci = cluster_bootstrap_mean(bets["realized_return"].to_numpy(), bets["matchday"].to_numpy(),
                                n_boot, seed)
    return {
        "n_bets": int(len(bets)),
        "n_matchdays": ci["n_clusters"],
        "profit_units": float(bets["realized_return"].sum()),
        "roi": ci["mean"],
        "roi_ci95": [ci["ci_low"], ci["ci_high"]],
        "roi_boot_se": ci["boot_se"],
        "hit_rate": float((bets["realized_return"] > 0).mean()),
        "mean_odds": float(bets["odds"].mean()),
        "mean_model_ev": float(bets["estimated_ev"].mean()),
        "market_null_expected_roi": float(bets["market_null_ev"].mean()),
        "per_bet_return_sd": float(bets["realized_return"].std(ddof=1)),
        "side_counts": {s: int((bets["selected_side"] == s).sum()) for s in SIDES},
    }


def choose_threshold(validation: pd.DataFrame, grid=TAU_GRID, min_bets=MIN_VALIDATION_BETS,
                     n_boot: int = 2000, seed: int = 0) -> tuple[float, pd.DataFrame]:
    """Pick tau maximising validation ROI among thresholds with at least ``min_bets`` bets.

    Ties go to the smaller threshold (more bets). Only validation rows are ever passed in.
    """
    candidates = [(tau, select_bets(validation, tau)) for tau in grid]
    if not any(len(bets) >= min_bets for _, bets in candidates):
        raise ValueError("No threshold meets the minimum validation bet count")
    rows = []
    for tau, bets in candidates:
        s = summarise(bets, n_boot, seed)
        rows.append({"tau": tau, **{k: s.get(k) for k in ("n_bets", "roi", "roi_ci95", "profit_units",
                                                          "market_null_expected_roi")}})
    table = pd.DataFrame(rows)
    eligible = table[table["n_bets"] >= min_bets]
    best = eligible.sort_values(["roi", "tau"], ascending=[False, True]).iloc[0]
    return float(best["tau"]), table


def selection_export(bets: pd.DataFrame) -> pd.DataFrame:
    """Shared return-inference import columns, plus model_probability."""
    return bets[SELECTION_COLUMNS].copy()
