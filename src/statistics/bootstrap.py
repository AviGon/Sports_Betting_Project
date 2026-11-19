"""Historical return bootstrap, with selectable resampling units."""
import numpy as np
import pandas as pd

from src.statistics.monte_carlo import validate_bets


def confidence_interval(matches, bets, method="match", n_paths=5000, seed=226, alpha=.05):
    """Percentile interval for ROI; resample complete clusters with replacement.

    Repeated cluster draws carry both cluster returns and stakes. With unequal
    cluster sizes each path is a ratio, not a mean of cluster means.
    """
    joined = validate_bets(matches, bets)
    if method not in ("ordinary", "match", "date", "season"):
        raise ValueError("method must be ordinary/match/date/season")
    if n_paths < 2000 or not 0 < alpha < 1:
        raise ValueError("Need >=2,000 paths and alpha in (0,1)")
    if method == "ordinary":
        sums = joined.realized_return.to_numpy(float)
        sizes = np.ones(len(joined))
    else:
        key = {"match": "match_id", "date": "date", "season": "season"}[method]
        if joined[key].isna().any():
            raise ValueError(f"Missing {key} clusters")
        groups = joined.groupby(key, sort=True).realized_return.agg(["sum", "size"])
        sums, sizes = groups["sum"].to_numpy(), groups["size"].to_numpy()
    n = len(sums)
    if n < 2:
        raise ValueError("Need at least two resampling units")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_paths)
    for start in range(0, n_paths, 256):
        sample = rng.integers(0, n, size=(min(256, n_paths-start), n))
        draws[start:start+len(sample)] = sums[sample].sum(axis=1)/sizes[sample].sum(axis=1)
    result = {"method": method, "n_units": n, "n_bets": len(joined), "n_paths": n_paths, "seed": seed,
              "observed_roi": float(joined.realized_return.mean()),
              "confidence_level": 1-alpha, "ci": np.quantile(draws, [alpha/2, 1-alpha/2]).tolist(),
              "few_clusters": bool(n < 20)}
    return result, draws
