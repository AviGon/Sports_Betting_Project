# Historical bootstrap confidence intervals

Strategy: `poisson_test`. These are percentile CIs for a historical mean return, distinct from the Monte Carlo experiment's simulation of new match outcomes under a specified market model. Reuse `src.statistics.bootstrap.confidence_interval(matches, bets, method=..., n_paths=5000, seed=226)` with the shared selection interface. Each row is one unit stake.

Resample with replacement the same number of units as observed:

- `ordinary`: individual bet rows; ignores shared-match dependence, useful only as an iid comparison.
- `match`: complete fixtures; different sides of one fixture stay together. Assumes match clusters independent.
- `date`: complete calendar match dates; retains all bets on the same date, allows contemporaneous dependence and assumes dates independent. This is a date-block bootstrap, not a moving multi-day bootstrap.
- `season`: whole seasons; retains serial dependence within each season. With only 4 seasons this interval is highly approximate; between-season independence and exchangeability can fail under structural changes (including COVID).

For each draw compute total sampled net returns / total sampled unit stakes. With unequal cluster sizes this is a ratio, not an unweighted average of cluster means. Keep repeated clusters with their full multiplicity. No model refitting or strategy reselection is performed. These CIs condition on historical strategy choices and do not correct data mining, market regime shifts or systematic missingness. Larger clusters need not always produce wider intervals: limited clusters can produce unstable variance estimates.

## Computed 95% percentile intervals

- ordinary: [-18.005%, 1.658%], 1,154 resampling units, 5,000 draws, seed 226.
- match: [-17.970%, 1.548%], 1,154 resampling units, 5,000 draws, seed 226.
- date: [-18.524%, 2.447%], 428 resampling units, 5,000 draws, seed 226.
- season: [-12.148%, -1.581%], 4 resampling units, 5,000 draws, seed 226.

Observed ROI: -8.328%. See the result JSON for exact values.
