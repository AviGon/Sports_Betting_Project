"""Shared numerical checks, cluster inference and figure conventions."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.data.prepare import PROBS

SIDE_CODES = np.array(["H", "D", "A"])


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Expected positive n by 3 probabilities summing to 1")
    return p


def one_hot(outcomes):
    y = np.asarray(outcomes)
    if not np.isin(y, SIDE_CODES).all():
        raise ValueError("Outcomes must be H/D/A")
    return (y[:, None] == SIDE_CODES).astype(float)


def cluster_mean(values, clusters, alpha=.05):
    """CR1 sandwich for a mean; Student reference with G-1 cluster df."""
    x = np.asarray(values, dtype=float)
    labels = np.asarray(clusters)
    if len(x) == 0 or len(x) != len(labels) or not np.isfinite(x).all() or pd.isna(labels).any():
        raise ValueError("Need finite values and matching nonmissing clusters")
    mean = x.mean()
    codes, unique = pd.factorize(labels)
    g = len(unique)
    if g < 2:
        return {"n": len(x), "clusters": g, "effect": float(mean), "se": None, "ci_low": None, "ci_high": None, "p_value": None}
    scores = np.bincount(codes, weights=x-mean, minlength=g)
    se = np.sqrt(g/(g-1) * np.square(scores).sum()) / len(x)
    critical = t.ppf(1-alpha/2, g-1)
    p = float(2*t.sf(abs(mean/se), g-1)) if se > 0 else (1. if abs(mean) < 1e-14 else 0.)
    return {"n": len(x), "clusters": g, "effect": float(mean), "se": float(se),
            "ci_low": float(mean-critical*se), "ci_high": float(mean+critical*se), "p_value": p}


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Finite p-values in [0,1] required")
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def long_outcomes(df):
    p = probabilities(df[PROBS])
    y = one_hot(df.outcome)
    return pd.DataFrame({"match_id": np.repeat(df.match_id.to_numpy(), 3),
                         "date": np.repeat(df.date.to_numpy(), 3),
                         "season": np.repeat(df.season.to_numpy(), 3),
                         "side": np.tile(SIDE_CODES, len(df)), "p": p.ravel(),
                         "y": y.ravel(), "residual": (y-p).ravel()})


def plot_setup():
    # Local cache prevents writes outside the project in restricted environments.
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
    return plt
