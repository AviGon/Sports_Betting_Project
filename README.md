# Sports Betting Project — ECE 225A

Can statistical models identify profitable inefficiencies in EPL betting markets, and can observed effects be distinguished from randomness?

Ksheer maintains shared data preparation and market statistics on `ksheer/statistical-analysis`. Person A independently implements Poisson modeling, prediction comparisons and chronological backtests. See [repository structure](docs/repository_structure.md).

## Reproduce the shared dataset

Requires Python 3.11+ and internet access on the first data build.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.data.prepare
.venv/bin/python -m pytest -q
```

Use `--download` to refresh sources; otherwise preparation reuses ignored `data/raw/` caches. Source hashes and schemas are in `data/processed/dataset_manifest.json`. The default covers only completed seasons 2015–16 through 2025–26 and fails if a requested season is incomplete. Partial 2026–27 is excluded.

Person A's input is [epl_matches.csv](data/processed/epl_matches.csv); join by `match_id`. See the [dictionary](data/processed/data_dictionary.md), [audit](docs/dataset_audit.md) and downstream selection interface in the dictionary. Unit stakes and net returns are used throughout. Odds and normalized market probabilities are separate; use raw decimal odds for profits.
