# Sports Betting Project — ECE 225A

**Can we beat the bookmakers?** Can statistical models identify profitable inefficiencies in EPL betting markets, and can observed effects be distinguished from randomness?

- **Track A (Person A, Avinash Gondela) — this branch (`main`):** Poisson goals model, forecast comparison against bookmaker probabilities, chronological value-betting backtest. See [docs/track_a.md](docs/track_a.md).
- **Track B (Person B, Ksheer):** shared data preparation, market calibration, mispricing tests, Monte Carlo and power analysis on `ksheer/statistical-analysis`.

See [repository structure](docs/repository_structure.md).

## Setup

Tested with Python 3.10.4 (Track A) and 3.14 (dataset build, Track B).

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q
```

## Reproduce Track A

```sh
.venv/bin/python -m src.modeling.run                 # tune on 2016–20, lock τ on 2020–22, test on 2022–26
.venv/bin/python -m src.modeling.run --skip-tuning   # reuse saved Poisson hyperparameters
```

Outputs go to `reports/track_a/`:

| File | Content |
|---|---|
| `results/predictions.csv` | `match_id, lambda_home, lambda_away, p_poisson_home, p_poisson_draw, p_poisson_away` (walk-forward, 2016–17 onward) |
| `results/selections_test.csv` | Test-season bets in the Track B import schema: `match_id, selected_side, model_probability, odds, estimated_ev, realized_return` |
| `results/selections_validation.csv` | Same format, validation seasons |
| `results/model_evaluation.json` | Brier / log-loss scores and paired bootstrap CIs per split and season |
| `results/backtest.json` | Locked τ, ROI, bootstrap CI, market-null expected ROI, threshold sensitivity |
| `results/poisson_config.json`, `threshold_lock.json` | Frozen choices made before the test evaluation |
| `figures/a2_forecast_scores.png`, `a2_calibration_test.png`, `a3_backtest.png` | Report figures |

Headline (test seasons 2022–26): the bookmaker beats the Poisson model (Brier 0.574 vs 0.586, difference 95% CI [0.007, 0.017]). The value-betting rule returns −8.3% per unit, 95% CI [−18.2%, +2.3%], which is consistent with no edge after the ≈5% margin.

To analyse Track A bets with Track B's simulations (on Ksheer's branch):
`python -m src.statistics.run --only monte-carlo --bets reports/track_a/results/selections_test.csv`.

## Reproduce the shared dataset

Requires internet access on the first data build.

```sh
.venv/bin/python -m src.data.prepare
```

Use `--download` to refresh sources; otherwise preparation reuses ignored `data/raw/` caches. Source hashes and schemas are in `data/processed/dataset_manifest.json`. The default covers only completed seasons 2015–16 through 2025–26 and fails if a requested season is incomplete. Partial 2026–27 is excluded.

The input is [epl_matches.csv](data/processed/epl_matches.csv); join by `match_id`. See the [dictionary](data/processed/data_dictionary.md) and [audit](docs/dataset_audit.md). Unit stakes and net returns are used throughout. Odds and normalized market probabilities are separate; raw decimal odds are used for profits.
