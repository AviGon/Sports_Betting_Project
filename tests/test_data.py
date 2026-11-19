import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare import ODDS, PROBS, clean_season, devig, validate


def fixture():
    return pd.DataFrame({"Date": ["01/08/2015", "02/08/15"], "HomeTeam": ["Man City", "Liverpool"],
                         "AwayTeam": ["Arsenal", "Chelsea"], "FTHG": [2, 0], "FTAG": [1, 0], "FTR": ["H", "D"],
                         "B365H": [2., 3.], "B365D": [4., 3.], "B365A": [4., 3.]})


def test_devig():
    np.testing.assert_allclose(devig([2, 4, 4]), [.5, .25, .25])
    np.testing.assert_allclose(devig([3, 3, 3]), [1/3]*3)
    for bad in ([1, 2, 3], [np.nan, 2, 3], [np.inf, 2, 3]):
        with pytest.raises(ValueError):
            devig(bad)


def test_clean_dates_aliases_missing_time_and_stable_ids():
    a, excluded, _ = clean_season(fixture(), "2015-16")
    assert not excluded and a.home_team.iloc[0] == "Manchester City"
    assert a.kickoff_time.isna().all()
    validate(a)
    raw = fixture()
    raw.loc[0, "Date"] = "03/08/2015"
    b, _, _ = clean_season(raw, "2015-16")
    assert a.match_id.tolist() == b.match_id.tolist()


def test_exclusions_duplicates_conflicts():
    raw = fixture()
    raw.loc[0, "B365H"] = np.nan
    out, excluded, _ = clean_season(raw, "2015-16")
    assert len(out) == 1 and excluded[0]["reason"] == "invalid_or_missing_b365_odds"
    out, excluded, _ = clean_season(pd.concat([fixture(), fixture().iloc[:1]]), "2015-16")
    assert len(out) == 2 and excluded[0]["reason"] == "exact_duplicate_fixture"
    raw = pd.concat([fixture(), fixture().iloc[:1]], ignore_index=True)
    raw.loc[2, "B365H"] = 5
    with pytest.raises(ValueError, match="Conflicting"):
        clean_season(raw, "2015-16")


def test_no_outcome_leakage_into_market_probabilities():
    raw = fixture()
    a, _, _ = clean_season(raw, "2015-16")
    raw.loc[0, ["FTHG", "FTAG", "FTR"]] = [0, 3, "A"]
    b, _, _ = clean_season(raw, "2015-16")
    np.testing.assert_array_equal(a[PROBS], b[PROBS])


def test_goal_mismatch_is_documented():
    raw = fixture()
    raw.loc[0, "FTR"] = "A"
    out, excluded, _ = clean_season(raw, "2015-16")
    assert len(out) == 1 and excluded[0]["reason"] == "outcome_goal_mismatch"


def test_covid_delayed_season_and_invalid_dates():
    raw = fixture()
    raw["Date"] = ["26/07/2020", "31/02/2020"]
    out, excluded, _ = clean_season(raw, "2019-20")
    assert len(out) == 1 and excluded[0]["reason"] == "invalid_date"


def test_consensus_schema_standardization_and_missing_closing():
    raw = fixture()
    raw[["BbAvH", "BbAvD", "BbAvA"]] = [[2., 4., 4.], [3., 3., 3.]]
    a, _, _ = clean_season(raw, "2015-16")
    assert a.consensus_opening_source.eq("BbAv").all()
    assert a.odds_consensus_closing_home.isna().all()
    assert a.p_consensus_closing_home.isna().all()
    np.testing.assert_allclose(a[["p_consensus_opening_home", "p_consensus_opening_draw", "p_consensus_opening_away"]].sum(axis=1), 1)
    raw = raw.rename(columns={"BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA"})
    b, _, _ = clean_season(raw, "2015-16")
    np.testing.assert_allclose(a.odds_consensus_opening_home, b.odds_consensus_opening_home)


def test_incomplete_optional_triplet_is_not_imputed():
    raw = fixture()
    raw[["AvgCH", "AvgCD", "AvgCA"]] = [[2., np.nan, 4.], [3., 3., 3.]]
    out, excluded, _ = clean_season(raw, "2015-16")
    assert len(out) == 2 and not excluded
    assert out.loc[0, ["p_consensus_closing_home", "p_consensus_closing_draw", "p_consensus_closing_away"]].isna().all()
    assert out.loc[1, ["p_consensus_closing_home", "p_consensus_closing_draw", "p_consensus_closing_away"]].sum() == pytest.approx(1)


def test_empirical_dataset_contract():
    path = Path("data/processed/epl_matches.csv")
    if not path.exists():
        pytest.skip("Build shared dataset first")
    df = pd.read_csv(path)
    validate(df)
    manifest = json.loads(Path("data/processed/dataset_manifest.json").read_text())
    assert len(manifest["seasons"]) == 11
    assert sum(s["included_rows"] for s in manifest["seasons"]) == len(df)
    assert all(s["source_rows"] == s["included_rows"] + s["excluded_rows"] for s in manifest["seasons"])
    assert len(manifest["exclusions"]) == sum(s["excluded_rows"] for s in manifest["seasons"])
