# Scientific verification and reproducibility

The existing modeling, market-statistics and betting experiments were reproduced on the combined codebase. No predictive model, strategy search, data source or research experiment was added. Generated outputs contain scientific results and experiment notes, not a final report.

## Shared data and chronological separation

- Both loaders use `src.data.MATCHES_PATH` and the same 4,180 completed fixtures. Core identifiers, teams, goals, results, seasons, Bet365 prices and de-vigged probabilities agree column-for-column. The modeling loader excludes closing quotes and validates the shared contract.
- The previously different input hashes are exactly explained by line endings: LF `20694f2dfc0c5c3c52128e1709e03f90ed0b888a360bfb6bf5b5bb22ded8dd52`; CRLF `0d409fbf6b31703608ea93455cc82e6037ea2b319359359cee667ad183f1f8ca`. Both pipelines now record exact and normalized hashes. The dataset itself was not changed.
- Poisson hyperparameters were reproduced on 2016–17 through 2019–20, the threshold on 2020–21 through 2021–22, and evaluation on 2022–23 through 2025–26. The choices remain decay 0.002/day, ridge 2 and threshold 0.025.
- Walk-forward and expanding-frequency tests tamper with same-day/future outcomes and check that earlier forecasts do not change. The model deliberately updates later test forecasts using earlier test matches; this is chronological updating, not future leakage. No externally registered prospective protocol is claimed.
- All 1,154 exported test selections belong to the held-out seasons. Integration tests reconstruct the choices from saved predictions and verify the selected side, original decimal odds, estimated EV and realized settlement. The input selections hash is recorded for return inference.

## Verified corrections

- Forecast-comparison two-sided p-values now use absolute tails of a null-centered cluster-bootstrap distribution. The previous calculation used signs of the uncentered distribution. BH adjustment covers the six paired comparisons within each split; training/validation comparisons remain descriptive after selection. Score means and percentile confidence intervals are preserved.
- Forecast calibration frequency intervals now resample whole ISO weeks rather than assuming independent Bernoulli trials. Existing bin counts and frequency estimates are preserved; uncertainty bars and their labels are corrected. Constant forecasts now produce a valid single bin instead of silently dropping every row. Fewer than two clusters cannot produce a confidence interval.
- Threshold selection now rejects a candidate grid with no threshold meeting the declared minimum validation count, instead of silently weakening the rule. This does not change the eligible historical choice.
- CSV predictions and selections retain full precision. Earlier six-decimal exports lost up to about 5e-7 in model probabilities/EV and could not reconstruct full-precision scores exactly. Current saved predictions reproduce the recorded scores to 1e-12.
- The return bootstrap documentation now states the actual number of represented seasons, including four for the held-out selections. Market-null outcome intervals are consistently distinguished from historical mean-return confidence intervals.

## Preserved and reproduced results

The five existing full-period result JSONs (`calibration`, `mispricing`, `monte_carlo`, `bootstrap`, `power`) are numerically identical to the pre-cleanup versions. The shared data CSV is unchanged. Validation/test bet counts, selected IDs and sides, odds, total profit, ROI, side counts and weekly ROI intervals are unchanged. Full-precision return serialization differs from prior six-decimal CSVs only at floating-point roundoff (maximum 4.45e-16).

A full refit and training-grid search on the pinned environment produced forecast-score changes of at most 1.31e-10 versus the prior Windows environment. These are optimizer/library roundoff, not changed scientific conclusions. Poisson test Brier remains 0.586231 versus market 0.574199. Test profit remains −96.10 units and ROI −8.3276%, with weekly-bootstrap 95% CI [−18.1548%, +2.2586%].

Held-out return inference uses 10,000 categorical outcome paths (seed 225), with one outcome per fixture shared by all its bets. Expected market-null ROI is −5.1507%, 95% simulation interval [−15.3565%, +5.5598%], observed percentile 28.05 and upper-tail p 0.7195. Four historical bootstrap methods use 5,000 draws (seed 226); the date-block CI is [−18.5245%, +2.4468%]. Both bootstrap implementations agree when given the same ISO-week units and seed; different date/week intervals reflect different resampling assumptions.

## Checks

Verified on Python 3.14.4 with `requirements-lock.txt`:

```sh
.venv/bin/python -m src.modeling.run
.venv/bin/python -m src.modeling.run --skip-tuning
.venv/bin/python -m src.statistics.run
.venv/bin/python -m src.statistics.run --only returns --bets reports/modeling/results/selections_test.csv --label poisson_test
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
git diff --check
```

**49 tests pass.** They cover data invariants/exclusions, mathematical scores, de-vigging, multiple comparisons, cluster resampling, null centering, deterministic seeds, Poisson fits, same-day/future leakage, joint categorical settlement, ROI, theoretical simulation moments, hypothetical power, full-precision reconstruction and cross-pipeline provenance. All source modules and three CLI help entry points were checked. Local Markdown links, file references and removal of internal labels were audited. The corrected calibration and held-out Monte Carlo figures were opened and visually inspected. Existing figure categories and research results are retained under scientific names.

GitHub Actions is configured for the full test suite. These checks establish local reproducibility, not the status of an unrun hosted CI job.

## Remaining limitations

No implementation failure remains from these checks. Four season blocks give an unstable held-out season-bootstrap interval even though its endpoints exclude zero; it is not strong evidence of a negative true edge. Only eleven season clusters support full-period inference. Independent-match market-null simulation, approximate ISO-week blocks, conditional strategy selection and exchangeability assumptions remain explicit. Bookmaker probabilities are not known truths, and price execution timing is unverified. Historical training/threshold selection and exploratory sensitivity results are not independent prospective confirmation.
