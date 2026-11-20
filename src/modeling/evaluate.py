"""Experiment A2: probabilistic forecast comparison.

Forecasters (all available before kick-off):
* ``naive``   - empirical H/D/A frequencies of all earlier matches in the dataset
* ``poisson`` - walk-forward Poisson model (A1)
* ``market``  - Bet365 pre-closing odds with the margin removed by normalisation

Scores are proper scoring rules, i.e. expected losses minimised by the true
outcome distribution:
* multiclass Brier  B = sum_k (p_k - 1{Y = k})^2         (0 is perfect, 2 is worst)
* log loss          L = -log p_Y

Differences between forecasters are paired per match. Their uncertainty is
estimated by resampling whole matchdays (ISO weeks) with replacement, which
keeps fixtures that share a round together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SIDES

PROB_COLS = {
    "naive": ["p_naive_home", "p_naive_draw", "p_naive_away"],
    "poisson": ["p_poisson_home", "p_poisson_draw", "p_poisson_away"],
    "market": ["p_market_home", "p_market_draw", "p_market_away"],
}


def naive_frequencies(matches: pd.DataFrame) -> pd.DataFrame:
    """H/D/A frequencies of all matches on strictly earlier dates (expanding window)."""
    m = matches.sort_values("date")
    onehot = pd.get_dummies(m["outcome"]).reindex(columns=list(SIDES), fill_value=0).astype(float)
    by_date = onehot.groupby(m["date"]).sum()
    prior = by_date.cumsum().shift(1).fillna(0.0)
    prior = prior.div(prior.sum(axis=1).replace(0, np.nan), axis=0).fillna(1 / 3)
    out = prior.reindex(m["date"]).to_numpy()
    return pd.DataFrame(out, columns=PROB_COLS["naive"]).assign(match_id=m["match_id"].to_numpy())


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.eye(p.shape[1])[y]
    return ((p - onehot) ** 2).sum(axis=1)


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))


def per_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"].to_numpy()
    out = {}
    for name, cols in PROB_COLS.items():
        p = df[cols].to_numpy(float)
        out[f"brier_{name}"] = brier(p, y)
        out[f"logloss_{name}"] = log_loss(p, y)
    return pd.DataFrame(out, index=df.index)


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int = 5000,
                           seed: int = 0, level: float = 0.95) -> dict:
    """Mean with a percentile CI from resampling clusters (matchdays) with replacement.

    The bootstrap mean is the ratio sum(values in sampled clusters) / count, so
    unequal cluster sizes are handled correctly.
    """
    values = np.asarray(values, float)
    codes, uniq = pd.factorize(np.asarray(clusters))
    sums = np.bincount(codes, values, len(uniq))
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    a = (1 - level) / 2
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, a)),
        "ci_high": float(np.quantile(boot, 1 - a)),
        "boot_se": float(boot.std(ddof=1)),
        # Percentile-bootstrap two-sided p-value for H0: mean = 0, floored at 1/n_boot.
        "p_value_two_sided": float(min(1.0, max(2 * min((boot <= 0).mean(), (boot >= 0).mean()),
                                                1.0 / n_boot))),
        "n": int(len(values)),
        "n_clusters": int(len(uniq)),
    }


def compare_forecasters(df: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean scores and paired differences (Poisson - market, Poisson - naive, market - naive)."""
    scores = per_match_scores(df)
    clusters = df["matchday"].to_numpy()
    result = {"n_matches": int(len(df)), "scores": {}, "differences": {}}
    for metric in ("brier", "logloss"):
        result["scores"][metric] = {n: float(scores[f"{metric}_{n}"].mean()) for n in PROB_COLS}
        for a, b in (("poisson", "market"), ("poisson", "naive"), ("market", "naive")):
            diff = scores[f"{metric}_{a}"] - scores[f"{metric}_{b}"]
            result["differences"][f"{metric}:{a}-{b}"] = cluster_bootstrap_mean(
                diff.to_numpy(), clusters, n_boot, seed)
    return result


def calibration_table(df: pd.DataFrame, forecaster: str, n_bins: int = 10) -> pd.DataFrame:
    """One-vs-rest reliability bins per outcome with Wilson 95% intervals."""
    rows = []
    for k, side in enumerate(SIDES):
        p = df[PROB_COLS[forecaster][k]].to_numpy(float)
        hit = (df["y"].to_numpy() == k).astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        for i in range(len(edges) - 1):
            mask = b == i
            n = int(mask.sum())
            if n == 0:
                continue
            obs = hit[mask].mean()
            z = 1.959964
            centre = (obs + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append({"forecaster": forecaster, "side": side, "bin": i, "n": n,
                         "mean_pred": float(p[mask].mean()), "observed": float(obs),
                         "ci_low": float(centre - half), "ci_high": float(centre + half)})
    return pd.DataFrame(rows)
