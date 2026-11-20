# Repository structure and ownership

`main` holds the shared dataset (prepared by Ksheer, imported from `ksheer/statistical-analysis` at `b36a91b`) and Track A. Track B lives on `ksheer/statistical-analysis`.

- `src/data/`: shared preparation contract (Ksheer).
- `data/raw/`: ignored cached football-data downloads and retrieval metadata.
- `data/processed/`: versioned shared CSV, dictionary, provenance manifest and exclusion ledger.
- `src/modeling/`: Track A (Avinash) — `poisson.py` (A1), `evaluate.py` (A2), `backtest.py` (A3), `run.py` (pipeline).
- `tests/`: dataset validation (`test_data.py`) and Track A tests (`test_poisson.py`, `test_track_a_evaluation.py`).
- `docs/track_a.md`: Track A protocol, mathematics, results and limitations.
- `reports/track_a/`: Track A results and figures; `reports/track_a_report.md`: report contribution draft.

Track B paths (`src/statistics/`, `reports/results/`, `reports/figures/`, `docs/experiments/`) are not used by Track A. When the branches are merged, only `README.md` and this file need manual conflict resolution.
