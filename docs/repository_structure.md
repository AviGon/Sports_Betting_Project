# Repository structure and ownership

The inspected main branch contained only README.md (initial commit 2b61499). Work proceeds on `ksheer/statistical-analysis`.

- `src/data/`: Ksheer maintains the shared preparation contract for both teammates.
- `data/raw/`: ignored cached football-data downloads and retrieval metadata.
- `data/processed/`: versioned shared CSV, dictionary, provenance manifest and exclusion ledger.
- `src/statistics/`: Ksheer's calibration, mispricing, Monte Carlo, bootstrap and power modules.
- `tests/`: synthetic unit fixtures and validation of actual processed data.
- `docs/experiments/`: mathematical assumptions, methods, computed results and limitations.
- `reports/figures/`: generated statistical figures; `reports/ksheer_report.md`: concise contribution draft.
- `reports/results/`: machine-readable computed summaries.

Person A independently owns Poisson modeling, prediction comparisons and chronological betting backtests. No implementation of that work belongs in this branch. Future Person A files must not be overwritten. The first milestone is the dataset contract and tests, committed before statistical experiments.
