# Track A report contribution (draft) — Avinash Gondela

## Predictive methodology

We model home and away goals as conditionally independent Poisson variables with
log-rates μ + h + att_home − def_away and μ + att_away − def_home. Team strengths are re-estimated before
every match date by time-weighted (half-life ≈ 1 year), ridge-penalised maximum likelihood on the previous
three years of results, and outcome probabilities follow by summing the joint score distribution.
Hyperparameters were chosen on 2016–20, the betting threshold on 2020–22, and all reported test results
use 2022–26 (1,520 matches) once.

## Can we predict better than the bookmaker?

No. On the test seasons the Poisson model's Brier score (0.586) beats naive historical frequencies
(0.646) by 0.059 [0.045, 0.074] but is worse than the de-vigged Bet365 probabilities (0.574) by
0.012 [0.007, 0.017] (95% matchday-cluster bootstrap); log loss gives the same ordering. Both forecasts are
approximately calibrated (Figure A2), so the market's advantage is sharper discrimination, not better
calibration: it prices information that goals alone do not contain.

## Does disagreeing with the market pay?

Betting one unit whenever the model's estimated expected value exceeded τ = 0.025 (chosen on
validation) produced 1,154 test bets and a return of −8.3% per unit staked, 95% CI [−18.2%, +2.3%].
The model estimated +20% average EV on these bets. The bookmaker's margin alone implies −5.2%, which lies
inside the interval. Ksheer's market-null Monte Carlo on the same bets gives p = 0.72 for doing at least
this well by chance. The apparent +9.4% validation profit did not persist. Any edge the Poisson
model has is too small to survive the bookmaker margin, and the losses are consistent with having no edge at all.

Figures: `reports/track_a/figures/a2_forecast_scores.png`, `a2_calibration_test.png`, `a3_backtest.png`.
Full methods and numbers: `docs/track_a.md`.
