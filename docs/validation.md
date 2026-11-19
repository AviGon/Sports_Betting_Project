# Validation and reproduction record

Ksheer's implementation is on `ksheer/statistical-analysis`, based on the original main commit `2b61499`. The shared dataset milestone was committed as `832e874` before experiments. No Person A modeling or chronological betting backtest was implemented.

Local environment: Python 3.14.4; exact dependency versions in `requirements-lock.txt`. Result provenance and the processed input hash are in `reports/results/run_metadata.json`. Raw file provenance, schemas and coverage are in `data/processed/dataset_manifest.json`.

Completed checks:

- `.venv/bin/python -m src.data.prepare`: 4,180 matches, eleven completed seasons, zero excluded rows. A second cached rebuild reproduced the same processed data contract.
- `.venv/bin/python -m src.statistics.run`: all four experiments and four bootstrap methods ran against the actual dataset. Outputs include four inspected 300-dpi figures, result JSON, simulation paths, fixed baseline selections, documentation and a 420-word report contribution.
- `.venv/bin/python -m pytest -q`: **27 tests passed**. Tests cover actual-data invariants and missing-data accounting, synthetic cleaning edge cases, duplicate conflicts, consensus schema standardization, incomplete optional quote groups, de-vigging, proper scores, BH correction, cluster uncertainty, joint categorical settlement, known ROI, deterministic seeds, theoretical versus simulated moments, no outcome/estimated-EV leakage into null simulations, clustered return resampling and power validation.
- `.venv/bin/python -m compileall -q src`: passed.
- `git diff --check`: passed.
- Imported-selection CLI smoke checks: Monte Carlo and bootstrap accepted the generated baseline CSV through the external interface with 2,000 draws; observed/expected ROI matched the internal baseline. Temporary imported-result copies were removed afterwards.
- Calibration, mispricing, Monte Carlo and power PNGs were opened and visually checked for labels, legibility, legends and layout. The final Monte Carlo caption uses the selected fixture date range so future subset imports are labeled correctly.

GitHub Actions is configured to run the data/statistics tests on Python 3.14 using pinned dependencies. Local verification does not by itself establish a successful hosted CI run.

Scientific limits are recorded per experiment: provider snapshot timing, multiplicative margin assumptions, few season clusters, overlapping FDR hypotheses, independent-market null simulations, conditional strategy selection, historical bootstrap exchangeability and the prospective iid odds-mixture power design. Unit fixtures are synthetic; every reported empirical result uses the downloaded historical matches.
