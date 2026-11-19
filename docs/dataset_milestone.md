# Shared dataset milestone — Ksheer

The completed-season dataset contains **4,180 matches** (380 per season, eleven seasons), from **2015-08-08 through 2026-05-24**. All requested fixtures have valid results and Bet365 pre-closing 1X2 odds; **zero rows were excluded**. The exclusion ledger is retained with headers. Partial 2026–27 is excluded. The COVID-delayed July 2020 fixtures belong to 2019–20 and are retained. Kickoff time is absent for all 1,520 matches before 2019–20.

Created: `src/data/prepare.py`, `tests/test_data.py`, `data/processed/epl_matches.csv`, `data_dictionary.md`, `dataset_manifest.json`, `exclusions.csv`, `docs/dataset_audit.md`, `docs/repository_structure.md`, dependency files and setup instructions. Actual schemas range from 62 to 132 columns; provenance and optional-quote missingness are recorded per season. Raw downloads remain ignored.

Person A reads `data/processed/epl_matches.csv`, joins on `match_id` and uses H/D/A outcome order. Core fields are `match_id, date, home_team, away_team, home_goals, away_goals, outcome, odds_home, odds_draw, odds_away, p_market_home, p_market_draw, p_market_away`. Optional `season`, `kickoff_time` and `raw_<source header>` columns retain provenance. Raw odds settle profits; normalized reciprocal probabilities define market expectations. Do not use target labels or closing quotes as opening-price features.

Reproduce with `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements-lock.txt`, `.venv/bin/python -m src.data.prepare`, `.venv/bin/python -m pytest -q`. First build downloads missing files; repeat builds use local caches. Refresh with `--download`; compare manifest hashes.

Milestone validation: **7 tests passed**, including actual-data contract validation, probability construction, invalid data, duplicate/conflicting fixtures, alias/date normalization, COVID postponements and outcome-independence of market probabilities. Dataset work is committed before implementing experiments.
