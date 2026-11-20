import numpy as np
import pytest

from src.statistics.calibration import brier_score, log_loss, logistic_calibration
from src.statistics.common import cluster_mean, fdr_bh


def test_proper_scores_known_values():
    p = np.array([[.5, .25, .25], [.2, .3, .5]])
    assert brier_score(p, ["H", "A"]) == pytest.approx((.375+.38)/2)
    assert log_loss(p, ["H", "A"]) == pytest.approx(-np.log(.5))
    with pytest.raises(ValueError):
        brier_score([[.2, .2, .2]], ["D"])
    with pytest.raises(ValueError):
        log_loss(p, ["H"])


def test_bh_preserves_original_order_and_monotonicity():
    np.testing.assert_allclose(fdr_bh([.04, .001, .02, .8]), [.053333333, .004, .04, .8])


def test_cluster_mean_matches_manual_calculation():
    result = cluster_mean([0, 1, 0, 1], ["a", "a", "b", "b"])
    assert result["effect"] == .5 and result["se"] == 0
    assert cluster_mean([1, -1], ["a", "a"])["p_value"] is None


def test_logistic_calibration_recovers_known_relationship():
    rng = np.random.default_rng(42)
    p = rng.uniform(.1, .9, 30000)
    y = rng.binomial(1, p)
    result = logistic_calibration(p, y, np.arange(len(p)) % 30)
    assert result["intercept"] == pytest.approx(0, abs=.06)
    assert result["slope"] == pytest.approx(1, abs=.06)
