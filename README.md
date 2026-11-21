# EPL Betting Market Analysis

**Research question:** Can statistical models identify profitable inefficiencies in English Premier League betting markets, and can the observed effects be distinguished from randomness?

## Data and methodology

Both pipelines read the same versioned [EPL dataset](data/processed/epl_matches.csv): **4,180 completed matches**, 380 per season from 2015–16 through 2025–26. Partial 2026–27 is excluded. The [dataset audit](docs/dataset_audit.md) records schema differences, exclusions and source provenance; the [dictionary](data/processed/data_dictionary.md) specifies the shared contract.

Bet365 pre-closing decimal odds are retained for settlement. Market probabilities are multiplicatively de-vigged: `p_k = (1/o_k) / sum_j(1/o_j)`. The independent Poisson goals model uses time decay, ridge shrinkage and a promoted-team prior. It refits using only strictly earlier match dates; same-day results never enter predictions. Closing quotes are excluded from the modeling loader.

The chronological protocol uses 2015–16 as burn-in, 2016–17 through 2019–20 for choosing Poisson hyperparameters, 2020–21 through 2021–22 for choosing the betting threshold, and 2022–23 through 2025–26 for testing. Past test-period matches may update later forecasts, but hyperparameters and the threshold remain fixed. The selected values are decay **0.002/day**, ridge **2**, and threshold **0.025**. At most one unit bet is placed per fixture, on the side with largest `p_poisson * odds - 1` above the threshold.

Forecasts are compared with bookmaker probabilities and expanding historical outcome frequencies using unscaled multiclass Brier score and natural-log loss. Paired score differences and betting ROI use whole-ISO-week percentile bootstrap intervals. Paired p-values use a null-centered bootstrap, with BH correction across the six score comparisons per split; training and validation comparisons are descriptive after model/threshold selection. Forecast calibration intervals also resample whole weeks.

Market calibration and mispricing diagnostics use all eleven seasons, with season-cluster inference and one BH family covering 28 probability-band, outcome and team tests. Fixed-selection Monte Carlo draws **one categorical outcome per match**, settling all bets on that fixture jointly. Historical bootstrap intervals resample observed returns and answer a different question from market-null outcome simulations. The existing power analysis considers hypothetical advantages under a break-even null, not discovered profitable strategies.

## Reproduced experimental results

- **Held-out forecasts, 1,520 matches:** Brier scores are naive **0.645617**, Poisson **0.586231**, market **0.574199**; log losses are **1.067452**, **0.983642**, **0.966169** respectively. Poisson minus market Brier difference is **0.012032**, with a pointwise weekly-bootstrap 95% CI of **[0.006854, 0.017406]**.
- **Held-out value betting:** **1,154 bets**, profit **−96.10 units**, ROI **−8.3276%**, weekly-bootstrap 95% CI **[−18.1548%, 2.2586%]**. Market-null expected ROI is **−5.1507%**. Validation ROI was **9.4236%** on 609 bets and was used to select the threshold; it is not held-out evidence of profitability.
- **Held-out return uncertainty:** 10,000 market-null paths give a central 95% ROI interval **[−15.3565%, 5.5598%]**, observed-return percentile **28.05**, and upper-tail Monte Carlo p **0.7195**. The date-block historical bootstrap 95% CI is **[−18.5245%, 2.4468%]**. The different intervals use different null models/resampling units and are not contradictory. The season bootstrap has only **four** test-season blocks and is too fragile for a strong significance conclusion.
- **Full-period market diagnostics:** Brier **0.568975**, log loss **0.960206**; **0 of 28** mispricing tests survive BH at 5%. The always-home benchmark returns **−3.8813%**, versus a market-null expectation of **−4.2945%**.
- **Hypothetical power:** approximate independent-bet counts for 80% power at one-sided 5% significance are **119,546** for 1% expected ROI, **29,969** for 2%, **13,356** for 3%, and **4,834** for 5%. Monte Carlo validates these designs. No finite sample reaches 80% power at a true 0% edge.

These results do not demonstrate a profitable strategy or prove market efficiency. The model assigns positive estimated EV to selections that lose historically. Conditional market-null simulations do not establish the true outcome probabilities.

## Project structure

- `src/data/`: shared data preparation, validation and portable fingerprints.
- `src/modeling/`: Poisson fitting, chronological forecasts, forecast comparison and betting backtest.
- `src/statistics/`: calibration, mispricing, categorical simulations, historical bootstrap and power.
- `data/processed/`: shared CSV, dictionary, provenance manifest and exclusions. `data/raw/` is an ignored download cache.
- `reports/modeling/results/` and `reports/modeling/figures/`: frozen modeling choices, predictions, betting selections, comparisons and figures.
- `reports/results/` and `reports/figures/`: full-period statistics and benchmarks; `_poisson_test` identifies held-out selection inference.
- `docs/experiments/`: [modeling protocol](docs/experiments/modeling.md), [market calibration](docs/experiments/calibration.md), [mispricing tests](docs/experiments/mispricing.md), [Monte Carlo](docs/experiments/monte_carlo.md), [bootstrap](docs/experiments/bootstrap.md) and [power](docs/experiments/statistical_power.md). Held-out [Monte Carlo](docs/experiments/monte_carlo_poisson_test.md) and [bootstrap](docs/experiments/bootstrap_poisson_test.md) details are recorded separately from the baseline.
- `tests/`: numerical, leakage, dataset and cross-pipeline consistency tests. See [verification notes](docs/validation.md).

## Reproducibility

Run from the repository root. Requires Python **3.11+**; the combined pipelines were verified with Python **3.14.4**. The lock file pins the dependency versions used for the reproduced outputs on a compatible platform.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.modeling.run
.venv/bin/python -m src.statistics.run
.venv/bin/python -m src.statistics.run --only returns --bets reports/modeling/results/selections_test.csv --label poisson_test
```

On Windows use `.venv\Scripts\python` in place of `.venv/bin/python`. To use compatible unpinned dependencies, install `requirements.txt`. The committed processed data supports all experiments offline. To rebuild it, run `.venv/bin/python -m src.data.prepare`; first use downloads missing CSVs from [football-data.co.uk](https://www.football-data.co.uk/englandm.php). `--download` refreshes cached sources and may change them, so compare manifest hashes before interpreting differences.

The modeling command reruns the declared training grid and validation rule; `--skip-tuning` reuses the saved Poisson configuration. Bootstrap defaults are 5,000 draws, seed 2025. Statistics defaults are 10,000 outcome paths (seed 225), 5,000 bootstrap draws (seed 226) and 4,000 power simulations per point (base seed 227). Select statistics with `--only calibration`, `mispricing`, `monte-carlo`, `bootstrap`, `returns` or `power`; precision controls are `--paths`, `--bootstrap-paths` and `--power-paths` (at least 2,000).

Run metadata records dependencies, seeds, source hashes and arguments. `input_sha256` is the exact file-byte hash; `input_lf_sha256` normalizes CRLF to LF so Windows and Unix copies can be compared. CSV predictions retain full precision to reproduce recorded scores. Numerical optimizer/library differences can cause tiny floating-point changes without changing selections or conclusions.

Selection imports require `match_id, selected_side, odds, estimated_ev, realized_return`. Sides are H/D/A; stakes are one unit; odds must be finite decimal values >1; net return is `odds - 1` for a win and `-1` otherwise. The estimated-EV header is required but values may be blank. Unknown IDs, duplicate match/side rows, invalid quotes and inconsistent settlement are rejected. `--label poisson_test` keeps these outputs separate from the always-home benchmark. Reusing a label replaces its previous outputs. The pipelines regenerate scientific results, figures and experiment documentation; they do not generate a final report.

## Assumptions and limits

Pre-closing snapshots are not synchronized first-ever opening prices; execution, commission, stake limits and transaction costs are not modeled. Independent Poisson misses goal dependence and unobserved team information. Hyperparameter and threshold choices are fixed by the historical protocol, not an externally registered prospective study. ISO weeks approximate dependence blocks rather than official match rounds. Resampling assumes blocks are independent/exchangeable; only eleven seasons exist overall and four in the test sample. Pointwise confidence intervals are not simultaneous, BH relies on dependence assumptions, and failure to reject is not evidence of no effect. Strategy fitting/reselection uncertainty is not included in conditional bootstrap or Monte Carlo inference.
