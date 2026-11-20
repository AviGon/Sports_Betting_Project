# Sports Betting Project — ECE 225A

**Can we beat the bookmakers?** Can statistical models identify profitable inefficiencies in EPL betting markets, and can observed effects be distinguished from randomness?

- **Track A (Person A, Avinash Gondela):** Poisson goals model, forecast comparison against bookmaker probabilities, chronological value-betting backtest. Code in `src/modeling/`; see [docs/track_a.md](docs/track_a.md).
- **Track B (Person B, Ksheer):** shared data preparation, market calibration, mispricing tests, Monte Carlo and power analysis. Code in `src/statistics/`; see [docs/experiments/](docs/experiments/).

See [repository structure](docs/repository_structure.md).

## Setup

Dataset build and Track B were tested with Python 3.14.4; Track A was tested with Python 3.10.4. `requirements-lock.txt` pins the exact environment used for Track B's committed outputs; use that file instead of `requirements.txt` to reproduce those dependency versions on a compatible Python/platform.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q
```

## Reproduce the shared dataset

Requires internet access on the first data build.

```sh
.venv/bin/python -m src.data.prepare
```

Use `--download` to refresh sources; otherwise preparation reuses ignored `data/raw/` caches. Source hashes and schemas are in `data/processed/dataset_manifest.json`. The default covers only completed seasons 2015–16 through 2025–26 and fails if a requested season is incomplete. Partial 2026–27 is excluded.

The input is [epl_matches.csv](data/processed/epl_matches.csv); join by `match_id`. See the [dictionary](data/processed/data_dictionary.md), [audit](docs/dataset_audit.md) and downstream selection interface in the dictionary. Unit stakes and net returns are used throughout. Odds and normalized market probabilities are separate; use raw decimal odds for profits.

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

## Reproduce Track B (Ksheer's experiments)

```sh
.venv/bin/python -m src.statistics.run
.venv/bin/python -m pytest -q
```

The full run validates the versioned dataset, computes calibration and cluster-based mispricing tests, runs 10,000 categorical outcome simulations (seed 225), 5,000 historical bootstrap draws per resampling method (seed 226), and 4,000 power experiments per point (base seed 227). Outputs are deterministic for the pinned environment. No network is needed when using the committed processed CSV. Per-point seeds, dependency versions, input hash and run arguments are recorded under `reports/results/`.

Select a phase with `--only calibration`, `mispricing`, `monte-carlo`, `bootstrap` or `power`. Precision controls are `--paths`, `--bootstrap-paths` and `--power-paths` (all at least 2,000). `--seed` changes the base seed. The full run regenerates the [concise report contribution](reports/ksheer_report.md), [figures](reports/figures/) and [experiment methods/results](docs/experiments/).

The always-home baseline is a fixed independent benchmark. To analyze Track A's selections:

```sh
.venv/bin/python -m src.statistics.run --only monte-carlo --bets reports/track_a/results/selections_test.csv
.venv/bin/python -m src.statistics.run --only bootstrap --bets reports/track_a/results/selections_test.csv
```

Required CSV headers: `match_id, selected_side, odds, estimated_ev, realized_return`. H/D/A sides, finite decimal odds >1, one unit stake per row, net realized returns; `estimated_ev` may be blank. Unknown matches, missing odds, inconsistent settlement and duplicate match/side rows are rejected. Different sides of one fixture settle against one categorical match outcome in every simulation. Imported outputs have `_imported` suffixes and do not replace baseline outputs; a later import replaces earlier `_imported` results. Imported odds are supplied by Person A; their execution timing and out-of-sample selection validity remain their responsibility.

## Interpretation

Track B's analyses are market diagnostics. De-vigged probabilities are an investigated null model. Mispricing tests use a single BH family, season-cluster uncertainty and date/BY sensitivities. Historical bootstrap intervals and market-null outcome intervals answer different questions. Power calculations construct hypothetical advantages under a break-even null; simulated positive edges are not evidence of profitable strategies. Details, actual numbers and limitations are in each experiment document.
