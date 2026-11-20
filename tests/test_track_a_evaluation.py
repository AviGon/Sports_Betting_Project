import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest import choose_threshold, select_bets, summarise
from src.modeling.data import add_derived
from src.modeling.evaluate import brier, calibration_table, cluster_bootstrap_mean, log_loss, naive_frequencies


def test_scores_known_values():
    p = np.array([[1., 0, 0], [.5, .25, .25]])
    y = np.array([0, 1])
    np.testing.assert_allclose(brier(p, y), [0, .25 + .5625 + .0625])
    np.testing.assert_allclose(log_loss(p, y), [0, np.log(4)])


def test_naive_frequencies_use_only_earlier_dates():
    m = pd.DataFrame({"match_id": ["a", "b", "c", "d"],
                      "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08", "2020-01-15"]),
                      "outcome": ["H", "A", "D", "H"]})
    out = naive_frequencies(m).set_index("match_id")
    np.testing.assert_allclose(out.loc["a"].to_numpy(float), [1 / 3] * 3)
    np.testing.assert_allclose(out.loc["b"].to_numpy(float), [1 / 3] * 3)   # same day as a: no leak
    np.testing.assert_allclose(out.loc["c"].to_numpy(float), [.5, 0, .5])
    np.testing.assert_allclose(out.loc["d"].to_numpy(float), [1 / 3] * 3)


def test_cluster_bootstrap_is_reproducible_and_degenerate_for_constants():
    v = np.full(50, .3)
    c = np.repeat(np.arange(10), 5)
    r = cluster_bootstrap_mean(v, c, 500, seed=1)
    assert r["ci_low"] == pytest.approx(.3) and r["ci_high"] == pytest.approx(.3)
    x = np.random.default_rng(0).normal(size=50)
    assert cluster_bootstrap_mean(x, c, 500, 4) == cluster_bootstrap_mean(x, c, 500, 4)


def market_frame():
    df = pd.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "date": pd.to_datetime(["2023-01-07", "2023-01-07", "2023-01-14"]),
        "season": "2022-23",
        "outcome": ["H", "A", "D"],
        "odds_home": [2.0, 1.5, 3.0], "odds_draw": [3.5, 4.0, 3.2], "odds_away": [4.0, 6.0, 2.4],
        "p_poisson_home": [.60, .60, .30], "p_poisson_draw": [.25, .20, .30], "p_poisson_away": [.15, .20, .40],
    })
    inv = 1 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    q = inv / inv.sum(axis=1, keepdims=True)
    df[["p_market_home", "p_market_draw", "p_market_away"]] = q
    return add_derived(df)


def test_selection_rule_and_settlement():
    df = market_frame()
    bets = select_bets(df, tau=0.05).set_index("match_id")
    # m1: EVs .20, -.125, -.40 -> H, wins at 2.0
    assert bets.loc["m1", "selected_side"] == "H"
    assert bets.loc["m1", "realized_return"] == pytest.approx(1.0)
    # m2: EVs -.10, -.20, +.20 -> A, wins at 6.0
    assert bets.loc["m2", "selected_side"] == "A"
    assert bets.loc["m2", "realized_return"] == pytest.approx(5.0)
    # m3: best EV is H at .30*3 - 1 = -.10 -> no bet
    assert "m3" not in bets.index
    assert (bets.estimated_ev > .05).all()
    assert not bets.reset_index().duplicated(["match_id"]).any()


def test_market_null_ev_is_the_negative_margin():
    df = market_frame()
    bets = select_bets(df, tau=-1)
    overround = (1 / df[["odds_home", "odds_draw", "odds_away"]]).sum(axis=1).to_numpy()
    np.testing.assert_allclose(bets.market_null_ev.to_numpy(), 1 / overround - 1)


def test_threshold_choice_prefers_smaller_tau_on_ties():
    df = market_frame()
    tau, table = choose_threshold(df, grid=(0.0, 0.05), min_bets=1, n_boot=50)
    assert tau == 0.0 and len(table) == 2
    assert summarise(select_bets(df, 0.0), 50)["n_bets"] == 2


def test_calibration_bins_cover_all_matches():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([3, 2, 2], 500)
    df = pd.DataFrame(p, columns=["p_market_home", "p_market_draw", "p_market_away"])
    df["y"] = [rng.choice(3, p=row) for row in p]
    cal = calibration_table(df, "market")
    assert cal.groupby("side").n.sum().eq(500).all()
