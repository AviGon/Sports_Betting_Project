"""Cross-pipeline checks against the versioned data and frozen betting outputs."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import MATCHES_PATH, dataset_sha256
from src.modeling.backtest import select_bets
from src.modeling.data import DATA_PATH, TEST, TRAIN, VALIDATION, load_matches
from src.modeling.evaluate import PROB_COLS, naive_frequencies, per_match_scores
from src.statistics.bootstrap import confidence_interval
from src.statistics.monte_carlo import validate_bets

RESULTS = Path("reports/modeling/results")


def test_pipelines_share_identical_data_and_exclude_closing_features():
    shared = pd.read_csv(MATCHES_PATH)
    modeling = load_matches()
    assert DATA_PATH == MATCHES_PATH
    for col in ("match_id", "season", "home_team", "away_team", "home_goals", "away_goals", "outcome",
                "odds_home", "odds_draw", "odds_away", *PROB_COLS["market"]):
        pd.testing.assert_series_equal(shared[col], modeling[col])
    assert not any("closing" in col or col.startswith("raw_") for col in modeling)


def test_dataset_fingerprint_ignores_line_endings(tmp_path):
    unix = tmp_path / "unix.csv"
    windows = tmp_path / "windows.csv"
    unix.write_bytes(b"id,result\n1,H\n")
    windows.write_bytes(b"id,result\r\n1,H\r\n")
    assert dataset_sha256(unix) == dataset_sha256(windows)


def test_frozen_heldout_bets_match_predictions_odds_and_settlement():
    matches = load_matches()
    predictions = pd.read_csv(RESULTS / "predictions.csv")
    lock = json.loads((RESULTS / "threshold_lock.json").read_text(encoding="utf-8"))
    config = json.loads((RESULTS / "poisson_config.json").read_text(encoding="utf-8"))
    assert lock["selected_on"] == list(VALIDATION)
    assert config["selected_on"] == list(TRAIN)
    test = matches.merge(predictions, on="match_id", validate="one_to_one")
    test = test.loc[test.season.isin(TEST)]
    expected = select_bets(test, lock["tau"]).set_index("match_id").sort_index()
    saved = pd.read_csv(RESULTS / "selections_test.csv").set_index("match_id").sort_index()
    pd.testing.assert_index_equal(expected.index, saved.index)
    pd.testing.assert_series_equal(expected.selected_side, saved.selected_side)
    np.testing.assert_allclose(expected[["model_probability", "odds", "estimated_ev", "realized_return"]],
                               saved[["model_probability", "odds", "estimated_ev", "realized_return"]], atol=1e-12, rtol=0)
    checked = validate_bets(matches, saved.reset_index())
    assert set(checked.season) <= set(TEST) and checked.match_id.is_unique


def test_saved_predictions_reproduce_test_scores_at_full_precision():
    matches = load_matches()
    predictions = pd.read_csv(RESULTS / "predictions.csv")
    np.testing.assert_allclose(predictions[PROB_COLS["poisson"]].sum(axis=1), 1, rtol=0, atol=1e-12)
    frame = matches.merge(predictions, on="match_id").merge(naive_frequencies(matches), on="match_id")
    test = frame.loc[frame.season.isin(TEST)]
    scores = per_match_scores(test)
    recorded = json.loads((RESULTS / "model_evaluation.json").read_text(encoding="utf-8"))["test"]
    for metric in ("brier", "logloss"):
        for forecaster in PROB_COLS:
            np.testing.assert_allclose(scores[f"{metric}_{forecaster}"].mean(),
                                       recorded["scores"][metric][forecaster], rtol=0, atol=1e-12)


def test_weekly_roi_ci_agrees_between_bootstrap_implementations():
    matches = load_matches()
    bets = pd.read_csv(RESULTS / "selections_test.csv")
    # Use the SAME resampling units/seed in both APIs; date and week CIs need
    # not agree when they deliberately use different dependence assumptions.
    weekly = matches.copy()
    weekly["date"] = weekly.matchday
    result, _ = confidence_interval(weekly, bets, method="date", n_paths=5000, seed=2025)
    backtest = json.loads((RESULTS / "backtest.json").read_text(encoding="utf-8"))["test"]
    np.testing.assert_allclose(result["ci"], backtest["roi_ci95"], rtol=0, atol=1e-12)
    assert result["n_units"] == backtest["n_matchdays"]


def test_pipeline_provenance_matches_shared_input():
    for path in (RESULTS / "run_metadata.json", Path("reports/results/run_metadata.json"),
                 Path("reports/results/run_metadata_poisson_test.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["input_lf_sha256"] == dataset_sha256()
        if metadata.get("bets_sha256"):
            assert metadata["bets_sha256"] == hashlib.sha256((RESULTS / "selections_test.csv").read_bytes()).hexdigest()
