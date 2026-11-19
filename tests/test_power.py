import numpy as np
import pytest

from src.statistics.power import detection_power, edge_model, monte_carlo_power, required_sample_size


def test_edge_tilt_is_bounded_and_has_requested_roi():
    odds = np.array([1.01, 1.03, 1.1, 2., 4., 10.])
    for edge in (0., .01, .02, .03, .05):
        q, m = edge_model(odds, edge)
        assert ((q > 0) & (q < 1)).all()
        assert np.mean(odds*q-1) == pytest.approx(edge, abs=1e-11)
        assert m["alternative_variance"] > 0


def test_zero_edge_type_one_error_and_positive_edge_sample_sizes():
    _, m = edge_model([2.]*100, 0)
    assert detection_power(1000, 0, m["null_variance"], m["alternative_variance"]) == pytest.approx(.05)
    assert required_sample_size(0, 1, 1) is None
    n = required_sample_size(.05, 1, 1-.05**2)
    assert detection_power(n, .05, 1, 1-.05**2) >= .8
    assert detection_power(n-1, .05, 1, 1-.05**2) < .8
    assert required_sample_size(.01, 1, 1) > required_sample_size(.05, 1, 1)*20


def test_mc_power_agrees_with_clt_and_seed_is_deterministic():
    odds = np.array([2.]*100)
    r = monte_carlo_power(odds, .05, 2500, n_paths=6000, seed=31)
    repeat = monte_carlo_power(odds, .05, 2500, n_paths=6000, seed=31)
    assert r == repeat
    theory = detection_power(2500, .05, 1, 1-.05**2)
    assert abs(r["power"]-theory) < .03
    null = monte_carlo_power(odds, 0., 2500, n_paths=6000, seed=32)
    assert abs(null["power"]-.05) < .015


def test_invalid_edge_and_prices_are_rejected():
    with pytest.raises(ValueError):
        edge_model([1.01], .05)
    with pytest.raises(ValueError):
        edge_model([np.nan], .01)
