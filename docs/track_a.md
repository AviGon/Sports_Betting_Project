# Track A — Poisson modelling, forecast comparison and value betting

Owner: Avinash Gondela (Person A). Code: `src/modeling/`. Tests: `tests/test_poisson.py`,
`tests/test_track_a_evaluation.py`. Outputs: `reports/track_a/`.

Reproduce (about one minute; add `--skip-tuning` to reuse the saved hyperparameters):

```sh
python -m src.modeling.run
python -m pytest -q
```

## Pre-registered protocol

| Split | Seasons | Used for |
|---|---|---|
| burn-in | 2015–16 | model history only; no forecasts scored |
| train | 2016–17 … 2019–20 | choosing Poisson decay ξ and shrinkage (log loss) |
| validation | 2020–21, 2021–22 | choosing the EV threshold τ (written to `threshold_lock.json`) |
| test | 2022–23 … 2025–26 | one evaluation with everything frozen |

Rules applied in code:

* Every forecast for a match on date *d* is made by a model fitted only on matches dated strictly
  before *d* (same-day fixtures share one fit). `test_walk_forward_never_uses_same_day_or_future_results`
  tampers with later results and checks that earlier forecasts do not change.
* Only Bet365 pre-closing odds (`odds_*`) are used, both to compute EV and to settle bets. Closing quotes are never loaded.
* τ is chosen on validation from the fixed grid {0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20}
  (maximum validation ROI among thresholds with ≥ 100 bets, ties → smaller τ). The test-season ROI for
  other τ values is reported only as an *exploratory* sensitivity, not as evidence.
* Uncertainty: percentile bootstrap that resamples whole **matchdays** (ISO weeks), keeping
  fixtures from the same round together. 5,000 resamples, seed 2025.

## A1 — Poisson model

For home team *i* and away team *j*,

X ~ Poisson(λ), log λ = μ + h + att_i − def_j  Y ~ Poisson(ν), log ν = μ + att_j − def_i,

with X ⟂ Y given the parameters. Hence P(X = x, Y = y) = Pois(x; λ)·Pois(y; ν) and
P(H) = Σ_{x>y}, P(D) = Σ_{x=y}, P(A) = Σ_{x<y} (truncated at 10 goals, renormalised).

The parameters maximise the time-weighted, ridge-penalised log-likelihood

Σ_m w_m [x_m log λ_m − λ_m + y_m log ν_m − ν_m] − r Σ_t [(att_t − c^att_t)² + (def_t − c^def_t)²],
w_m = exp(−ξ · age_m),

over a 3-year look-back window (L-BFGS-B with analytic gradient). Promoted teams (not in the
previous season, which the published fixture list reveals before kick-off) are shrunk toward the
average strength of the three weakest established teams of the previous fit instead of toward 0.

Hyperparameters chosen on train-season log loss (`hyperparameter_search.csv`):
**ξ = 0.002/day (half-life ≈ 347 days), r = 2**. Train log loss ranged from 0.9424 (selected) to 0.9503 across the 12 grid points.

## A2 — Forecast comparison

Multiclass Brier score B = Σ_k (p_k − 1{Y=k})² and log loss −log p_Y; both are proper scoring
rules, so the true outcome distribution minimises their expected value.

| Split (matches) | Naive Brier | Poisson Brier | Market Brier | Naive LL | Poisson LL | Market LL |
|---|---|---|---|---|---|---|
| train (1,520) | 0.638 | 0.557 | **0.547** | 1.057 | 0.942 | **0.930** |
| validation (760) | 0.654 | 0.585 | **0.576** | 1.078 | 0.985 | **0.972** |
| test (1,520) | 0.646 | 0.586 | **0.574** | 1.068 | 0.984 | **0.966** |

Paired differences on the test seasons (95% matchday-bootstrap CI, 143 matchdays):

| Difference | Mean | 95% CI |
|---|---|---|
| Brier: Poisson − market | +0.0120 | [+0.0069, +0.0174] |
| Log loss: Poisson − market | +0.0175 | [+0.0097, +0.0254] |
| Brier: Poisson − naive | −0.0594 | [−0.0737, −0.0448] |
| Brier: market − naive | −0.0714 | [−0.0850, −0.0579] |

The Poisson model is far better than naive frequencies, but the bookmaker is significantly better
than the Poisson model in every split and in 9 of 10 seasons (`a2_forecast_scores.png`).
Both are close to calibrated on the test seasons (`a2_calibration_test.png`); the main visible Poisson
miscalibration is the top away-win decile (forecast ≈ 0.66, observed ≈ 0.62).

## A3 — Value-betting backtest

Rule: bet one unit on the side with the largest EV = p_poisson · odds − 1 if EV > τ; at most one bet
per match. Settlement: +(odds − 1) on a win, −1 on a loss.

Margin hurdle: if the de-vigged market probabilities q were true, each bet's expected return is
q·o − 1 = 1/overround − 1 (≈ −5% for Bet365), whatever is selected. Reported as "market-null ROI".

Validation chose **τ = 0.025** (validation: 609 bets, ROI +9.4%, 95% CI [−6.4%, +26.8%]).

Frozen rule on the test seasons:

| Quantity | Value |
|---|---|
| Bets | 1,154 (H 439, D 52, A 663) over 142 matchdays |
| Profit | −96.1 units |
| ROI | **−8.3%**, 95% CI [−18.2%, +2.3%] |
| Market-null expected ROI | −5.2% |
| Mean model-estimated EV of selected bets | +20.0% |
| Hit rate / mean odds | 28.5% / 4.33 |
| ROI by season | 22–23 −10.6%, 23–24 −11.7%, 24–25 +1.8%, 25–26 −12.6% |

The model claimed an average edge of +20% on its selections and realised −8%. That gap is the
clearest result of A3: when the Poisson model disagrees with the market, the model is usually wrong.
The rule favours away underdogs at long odds (mean 4.33), where the return SD per bet is 1.72, so
the CI is wide. The test CI includes both 0 and the −5% margin hurdle, so the strategy cannot be told apart from
a bettor with no edge who pays the margin. The validation profit did not carry over to the test seasons, as expected
when a threshold is chosen on a noisy ROI.

Exploratory test-season sensitivity (not used for selection): ROI ranged from −6.1% (τ = 0.05) to −22.1% (τ = 0.20);
no τ produced a positive test ROI.

## Cross-check with Track B

`selections_test.csv` was run through Ksheer's `src.statistics.run --bets` (Monte Carlo and
bootstrap) without modifying that branch. Their code reproduces the observed ROI (−8.33%) and
market-null expected ROI (−5.15%) exactly. Their date-cluster bootstrap CI [−18.5%, +2.4%] agrees with ours.
Their market-null Monte Carlo gives a 95% simulated interval [−15.4%, +5.6%] and an upper-tail
p = 0.72, so the observed loss is consistent with luck plus the bookmaker margin.

## Limitations

* Independent Poisson ignores the low-score dependence (Dixon–Coles ρ) and under-represents draws at
  the extremes; it uses only goals, not xG, injuries or line-ups, which the market prices.
* 2020–21 (validation) was played without crowds, so home advantage dropped. The time decay adapts, but slowly.
* Bet365 "pre-closing" odds have no uniform timestamp; executable prices and stake limits are not modelled.
* One threshold selected on two seasons is a noisy choice. The honest evidence is the single test-season run.
