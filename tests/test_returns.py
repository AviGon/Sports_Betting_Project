import numpy as np
import pandas as pd
import pytest

from src.statistics.bootstrap import confidence_interval
from src.statistics.monte_carlo import home_baseline, settle, simulate, simulation_inputs, validate_bets


def matches():
    return pd.DataFrame({"match_id": ["a", "b", "c"], "date": ["2020-01-01", "2020-01-01", "2021-01-02"],
                         "season": ["2019-20", "2019-20", "2020-21"], "outcome": ["H", "D", "A"],
                         "odds_home": [2., 2., 2.], "p_market_home": [.5]*3,
                         "p_market_draw": [.25]*3, "p_market_away": [.25]*3})


def test_settlement_and_roi():
    np.testing.assert_array_equal(settle([2, 3, 4], ["H", "D", "H"], ["H", "D", "A"]), [1, 2, -1])
    bets = home_baseline(matches())
    assert bets.realized_return.mean() == pytest.approx(-1/3)


def test_joint_categorical_settlement_and_theoretical_expectation():
    m = matches().iloc[:1]
    bets = pd.DataFrame({"match_id": ["a"]*3, "selected_side": ["H", "D", "A"], "odds": [3.]*3,
                         "estimated_ev": np.nan, "realized_return": [2., -1., -1.]})
    r, paths = simulate(m, bets, n_paths=4000, seed=5)
    # Exactly one of three mutually exclusive bets wins in EVERY path.
    np.testing.assert_array_equal(paths, 0)
    assert r["expected_roi"] == 0
    assert r["theoretical_sd_roi"] == 0


def test_seed_determinism_and_mc_average():
    m = matches()
    bets = home_baseline(m)
    r, a = simulate(m, bets, n_paths=20000, seed=42)
    _, b = simulate(m, bets, n_paths=20000, seed=42)
    np.testing.assert_array_equal(a, b)
    assert abs(r["simulation_mean_roi"]-r["expected_roi"]) < 5*r["mc_se_of_mean"]
    assert r["simulation_sd_roi"] == pytest.approx(r["theoretical_sd_roi"], rel=.03)


def test_duplicates_unknown_ids_missing_odds_and_inconsistent_returns():
    m = matches()
    bets = home_baseline(m)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_bets(m, pd.concat([bets, bets.iloc[:1]]))
    for col, value in (("odds", np.nan), ("match_id", "unknown"), ("realized_return", 100), ("selected_side", "X")):
        bad = bets.copy()
        bad.loc[0, col] = value
        with pytest.raises(ValueError):
            validate_bets(m, bad)


def test_bootstrap_match_clustering_keeps_joint_bets_together():
    m = matches()
    bets = pd.DataFrame({"match_id": np.repeat(m.match_id, 3), "selected_side": np.tile(["H", "D", "A"], 3),
                         "odds": 3., "estimated_ev": np.nan, "realized_return": np.tile([-1., -1., -1.], 3)})
    bets["realized_return"] = settle(bets.odds, bets.selected_side, np.repeat(m.outcome, 3))
    r, paths = confidence_interval(m, bets, method="match", n_paths=2000)
    np.testing.assert_array_equal(paths, 0)
    assert r["n_units"] == 3 and r["ci"] == [0., 0.]
    _, iid_paths = confidence_interval(m, bets, method="ordinary", n_paths=2000)
    assert iid_paths.std() > .1


def test_block_bootstrap_counts_reproducibility_and_unequal_stakes():
    m = matches()
    bets = home_baseline(m)
    for method, n in (("ordinary", 3), ("match", 3), ("date", 2), ("season", 2)):
        r, a = confidence_interval(m, bets, method=method, n_paths=2000, seed=123)
        _, b = confidence_interval(m, bets, method=method, n_paths=2000, seed=123)
        assert r["n_units"] == n
        np.testing.assert_array_equal(a, b)
    r, draws = confidence_interval(m, bets, method="date", n_paths=2000)
    # Date clusters hold 2 bets with total return 0, and 1 losing bet.
    assert set(np.round(np.unique(draws), 8)) <= {0., -1., round(-1/3, 8)}
