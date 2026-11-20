import numpy as np
import pandas as pd
import pytest

from src.modeling.data import add_derived
from src.modeling.poisson import (PoissonConfig, fit_poisson, outcome_probs, promoted_teams,
                                  walk_forward)


def test_outcome_probs_are_a_distribution_and_symmetric():
    p = outcome_probs(1.3, 1.3)
    assert p.sum() == pytest.approx(1)
    assert p[0] == pytest.approx(p[2])
    stronger_home = outcome_probs(2.0, 0.8)
    assert stronger_home[0] > stronger_home[2]


def test_outcome_probs_match_monte_carlo():
    rng = np.random.default_rng(1)
    x, y = rng.poisson(1.6, 400_000), rng.poisson(1.1, 400_000)
    empirical = np.array([(x > y).mean(), (x == y).mean(), (x < y).mean()])
    np.testing.assert_allclose(outcome_probs(1.6, 1.1), empirical, atol=3e-3)


def test_fit_recovers_simulated_strengths():
    rng = np.random.default_rng(0)
    n = 8
    att = rng.normal(0, .3, n); att -= att.mean()
    dfn = rng.normal(0, .3, n); dfn -= dfn.mean()
    mu, h = 0.2, 0.3
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j] * 40
    hi, ai = np.array(pairs).T
    xg = rng.poisson(np.exp(mu + h + att[hi] - dfn[ai]))
    yg = rng.poisson(np.exp(mu + att[ai] - dfn[hi]))
    m, hh, a, d, ok = fit_poisson(hi, ai, xg, yg, np.ones(len(hi)), n, ridge=1e-6)
    assert ok
    assert hh == pytest.approx(h, abs=.05)
    # Attack/defence are identified up to a common shift; compare centred values.
    np.testing.assert_allclose(a - a.mean(), att, atol=.08)
    np.testing.assert_allclose(d - d.mean(), dfn, atol=.08)


def toy_league(seed=0, seasons=("2015-16", "2016-17")):
    rng = np.random.default_rng(seed)
    teams = list("ABCDEF")
    rows, k = [], 0
    for s_i, season in enumerate(seasons):
        start = pd.Timestamp(f"{2015 + s_i}-08-01")
        for r in range(10):
            for i in range(0, len(teams), 2):
                h, a = rng.permutation(teams)[:2]
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                rows.append({"match_id": f"m{k}", "date": start + pd.Timedelta(days=7 * r),
                             "season": season, "home_team": h, "away_team": a,
                             "home_goals": hg, "away_goals": ag,
                             "outcome": "H" if hg > ag else "D" if hg == ag else "A"})
                k += 1
    return add_derived(pd.DataFrame(rows))


def test_walk_forward_never_uses_same_day_or_future_results():
    m = toy_league()
    cfg = PoissonConfig(xi=.002, ridge=1.0)
    base = walk_forward(m, cfg, predict_seasons=("2016-17",)).set_index("match_id")
    cutoff = m.loc[m.season == "2016-17", "date"].sort_values().unique()[4]
    tampered = m.copy()
    late = tampered.date >= cutoff
    tampered.loc[late, "home_goals"] = 9
    tampered.loc[late, "away_goals"] = 0
    tampered.loc[late, "outcome"] = "H"
    other = walk_forward(tampered, cfg, predict_seasons=("2016-17",)).set_index("match_id")
    unaffected = m.loc[(m.season == "2016-17") & (m.date <= cutoff), "match_id"]
    affected = m.loc[m.date > cutoff, "match_id"]
    cols = ["p_poisson_home", "p_poisson_draw", "p_poisson_away"]
    pd.testing.assert_frame_equal(base.loc[unaffected, cols], other.loc[unaffected, cols])
    assert not np.allclose(base.loc[affected, cols], other.loc[affected, cols])


def test_promoted_teams_are_new_relative_to_previous_season():
    m = pd.DataFrame({"season": ["2015-16", "2015-16", "2016-17", "2016-17"],
                      "home_team": ["A", "B", "A", "C"]})
    assert promoted_teams(m) == {"2015-16": set(), "2016-17": {"C"}}
