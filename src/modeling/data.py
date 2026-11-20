"""Loading the shared dataset and the chronological split used by Track A.

The split is fixed in code before any test-season result is inspected:

* ``BURN_IN``     2015-16           model history only, no predictions scored
* ``TRAIN``       2016-17..2019-20  Poisson hyperparameters (decay, shrinkage) are chosen here
* ``VALIDATION``  2020-21..2021-22  the value-betting EV threshold is chosen here
* ``TEST``        2022-23..2025-26  scored once with everything frozen
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/epl_matches.csv")
SIDES = ("H", "D", "A")
SIDE_NAMES = {"H": "home", "D": "draw", "A": "away"}

BURN_IN = ("2015-16",)
TRAIN = ("2016-17", "2017-18", "2018-19", "2019-20")
VALIDATION = ("2020-21", "2021-22")
TEST = ("2022-23", "2023-24", "2024-25", "2025-26")
SPLITS = {"train": TRAIN, "validation": VALIDATION, "test": TEST}


def split_of(season: str) -> str:
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    return "burn_in"


def load_matches(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Core columns of the shared dataset, sorted chronologically.

    Only pre-match information (teams, date, Bet365 pre-closing odds) and the
    target labels are kept; closing quotes are dropped so they cannot leak into
    features or settlement.
    """
    cols = ["match_id", "date", "kickoff_time", "season", "home_team", "away_team",
            "home_goals", "away_goals", "outcome",
            "odds_home", "odds_draw", "odds_away",
            "p_market_home", "p_market_draw", "p_market_away"]
    df = pd.read_csv(path, usecols=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "kickoff_time", "match_id"], na_position="first").reset_index(drop=True)
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    iso = df["date"].dt.isocalendar()
    # A "matchday" cluster: all fixtures in the same ISO week (a weekend round
    # plus any midweek games). Used for dependence-aware bootstrap resampling.
    df["matchday"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["split"] = df["season"].map(split_of)
    df["y"] = df["outcome"].map({s: i for i, s in enumerate(SIDES)}).astype(int)
    return df


def market_probs(df: pd.DataFrame) -> np.ndarray:
    return df[["p_market_home", "p_market_draw", "p_market_away"]].to_numpy(float)


def odds_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
