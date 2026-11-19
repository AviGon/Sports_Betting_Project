# Sports Betting Project — ECE 225A

Can statistical models identify profitable inefficiencies in EPL betting markets, and can observed effects be distinguished from randomness?

Ksheer maintains shared data preparation and market statistics on `ksheer/statistical-analysis`. Person A independently implements Poisson modeling, prediction comparisons and chronological backtests. See [repository structure](docs/repository_structure.md).

## Reproduce the shared dataset

Requires Python 3.11+ and internet access on the first data build.

Tested with Python 3.14.4. `requirements-lock.txt` pins the exact environment used for the committed outputs; use that file instead of `requirements.txt` to reproduce those dependency versions on a compatible Python/platform.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.data.prepare
.venv/bin/python -m pytest -q
```

Use `--download` to refresh sources; otherwise preparation reuses ignored `data/raw/` caches. Source hashes and schemas are in `data/processed/dataset_manifest.json`. The default covers only completed seasons 2015–16 through 2025–26 and fails if a requested season is incomplete. Partial 2026–27 is excluded.

Person A's input is [epl_matches.csv](data/processed/epl_matches.csv); join by `match_id`. See the [dictionary](data/processed/data_dictionary.md), [audit](docs/dataset_audit.md) and downstream selection interface in the dictionary. Unit stakes and net returns are used throughout. Odds and normalized market probabilities are separate; use raw decimal odds for profits.

## Reproduce Ksheer's experiments

```sh
.venv/bin/python -m src.statistics.run
.venv/bin/python -m pytest -q
```

The full run validates the versioned dataset, computes calibration and cluster-based mispricing tests, runs 10,000 categorical outcome simulations (seed 225), 5,000 historical bootstrap draws per resampling method (seed 226), and 4,000 power experiments per point (base seed 227). Outputs are deterministic for the pinned environment. No network is needed when using the committed processed CSV. Per-point seeds, dependency versions, input hash and run arguments are recorded under `reports/results/`.

Select a phase with `--only calibration`, `mispricing`, `monte-carlo`, `bootstrap` or `power`. Precision controls are `--paths`, `--bootstrap-paths` and `--power-paths` (all at least 2,000). `--seed` changes the base seed. The full run regenerates the [concise report contribution](reports/ksheer_report.md), [figures](reports/figures/) and [experiment methods/results](docs/experiments/).

The always-home baseline is a fixed independent benchmark. To analyze Person A's selections:

```sh
.venv/bin/python -m src.statistics.run --only monte-carlo --bets path/to/selections.csv
.venv/bin/python -m src.statistics.run --only bootstrap --bets path/to/selections.csv
```

Required CSV headers: `match_id, selected_side, odds, estimated_ev, realized_return`. H/D/A sides, finite decimal odds >1, one unit stake per row, net realized returns; `estimated_ev` may be blank. Unknown matches, missing odds, inconsistent settlement and duplicate match/side rows are rejected. Different sides of one fixture settle against one categorical match outcome in every simulation. Imported outputs have `_imported` suffixes and do not replace baseline outputs; a later import replaces earlier `_imported` results. Imported odds are supplied by Person A; their execution timing and out-of-sample selection validity remain their responsibility.

## Interpretation

The completed analyses are market diagnostics, not Person A's prediction or betting-backtest implementation. De-vigged probabilities are an investigated null model. Mispricing tests use a single BH family, season-cluster uncertainty and date/BY sensitivities. Historical bootstrap intervals and market-null outcome intervals answer different questions. Power calculations construct hypothetical advantages under a break-even null; simulated positive edges are not evidence of profitable strategies. Details, actual numbers and limitations are in each experiment document.
