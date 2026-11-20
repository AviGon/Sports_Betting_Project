# Repository structure and ownership

`main` contains the shared dataset and both research tracks. Track B was merged from `ksheer/statistical-analysis` at `b36a91b`.

- `src/data/`: shared preparation contract (Ksheer).
- `data/raw/`: ignored cached football-data downloads and retrieval metadata.
- `data/processed/`: versioned shared CSV, dictionary, provenance manifest and exclusion ledger.
- `src/modeling/`: Track A (Avinash) — `poisson.py` (A1), `evaluate.py` (A2), `backtest.py` (A3), `run.py` (pipeline).
- `src/statistics/`: Track B (Ksheer) — calibration, mispricing, Monte Carlo, bootstrap and power modules.
- `tests/`: dataset validation (`test_data.py`), Track A tests (`test_poisson.py`, `test_track_a_evaluation.py`) and Track B tests (`test_calibration.py`, `test_mispricing.py`, `test_power.py`, `test_returns.py`).
- `docs/track_a.md`: Track A protocol, mathematics, results and limitations.
- `docs/experiments/`: Track B assumptions, methods, computed results and limitations.
- `reports/track_a/`: Track A results and figures.
- `reports/figures/`, `reports/results/`: Track B figures and machine-readable summaries; `reports/ksheer_report.md`: Track B contribution draft.

Each track writes only to its own paths, so rerunning one never overwrites the other's outputs.
