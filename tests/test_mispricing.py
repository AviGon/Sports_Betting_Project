import numpy as np
import pandas as pd

from src.statistics.common import cluster_mean, fdr_bh, long_outcomes
from src.statistics.mispricing import analyze


def fixture():
    n = 120
    return pd.DataFrame({"match_id": [str(i) for i in range(n)], "date": np.repeat(np.arange(60), 2),
                         "season": np.repeat(np.arange(6), 20), "home_team": "a", "away_team": "b",
                         "outcome": np.tile(["H", "D", "A"], 40), "p_market_home": .5,
                         "p_market_draw": .25, "p_market_away": .25})


def test_match_multiclass_residuals_sum_to_zero():
    long = long_outcomes(fixture())
    np.testing.assert_allclose(long.groupby("match_id").residual.sum(), 0, atol=1e-15)


def test_single_global_fdr_and_team_thresholds():
    result = analyze(fixture())
    rows = result["tests"]
    assert {r["group"] for r in rows} == {"team_win", "probability_band", "outcome"}
    np.testing.assert_allclose([r["q_value_bh"] for r in rows], fdr_bh([r["p_value"] for r in rows]))
    assert all(r["clusters"] == 6 for r in rows)
    assert not analyze(fixture(), min_matches=1000)["tests"]


def test_cluster_se_retains_common_shocks():
    residual = np.repeat([-.2, .2, -.2, .2], 20)
    season = np.repeat(np.arange(4), 20)
    clustered = cluster_mean(residual, season)
    independent = cluster_mean(residual, np.arange(80))
    assert clustered["se"] > independent["se"]*3
