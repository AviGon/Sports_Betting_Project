# B4 — Statistical power and sample size

Question: how many independent bets are required to detect hypothetical positive expected ROI?

## Null, alternatives and historical variance

One-sided H0: mean net return ≤0; H1: mean net return δ>0. Significance α=.05, target power .80. Hypothetical δ = 0%, 1%, 2%, 3%, 5%; none is evidence of an actual edge. These calculations study detectability, not a trading strategy.

Use the observed distribution of 4,180 historical Bet365 HOME prices as the empirical population of prospective odds. Prospective observations independently draw a price from this fixed empirical distribution and then one binary win/loss for one selected bet on a separate match. This differs from B3's fixed historical bet list with categorical outcomes. Grouping equal prices and using multinomial counts plus conditional binomial wins simulates this iid mixture exactly without enormous bet-by-path arrays. Mutually exclusive multiple bets are not modeled in this power experiment.

The break-even null win probability at odds o is q0=1/o, giving conditional mean ROI 0 and variance o−1. This **break-even null differs from B3's de-vigged market null**, which usually predicts negative ROI. For a hypothetical edge, use qδ(o)=expit(logit(1/o)+λδ), finding λδ so the empirical mean of o qδ(o)−1 equals δ. This respects [0,1] even for very short odds where simply (1+δ)/o would exceed 1. The alternative's return variance is σδ²=E[o² qδ(o)]−(1+δ)², including price-mixture variation in conditional means. σ0²=E[o−1]. These are plug-in odds-based design variances, not known population constants.

For comparison, actual historical home-bet return sample variance: **1.877602**. Mean conditional variance under the de-vigged market probabilities: **1.886874**. They are reported separately rather than silently using a negative-ROI market null to test zero-ROI profitability.

## CLT calculation and validation

Reject H0 when sample mean > z_.95 σ0/√n. Approximate power = 1−Φ((z_.95 σ0−δ√n)/σδ). Required n = ceiling[(z_.95 σ0+z_.80 σδ)²/δ²]. At δ=0 the rejection probability is approximately α for every n; no finite n reaches 80% power.

Monte Carlo validation uses 4,000 independent experiments per point, base seed 227 (per-point seeds stored in JSON). Wilson 95% intervals quantify simulation precision, not uncertainty in the historical odds population. Validation also checks the null's rejection rate. Curves are CLT approximations; dots are actual simulated detection proportions.

## Computed designs

- 0% ROI: no finite 80%-power sample size; σ0²=1.928278.
- 1% ROI: **119,546 bets** for 80% CLT power; Monte Carlo power 0.805, Wilson 95% CI [0.792, 0.817]; alternative variance 1.944007.
- 2% ROI: **29,969 bets** for 80% CLT power; Monte Carlo power 0.808, Wilson 95% CI [0.796, 0.820]; alternative variance 1.959674.
- 3% ROI: **13,356 bets** for 80% CLT power; Monte Carlo power 0.803, Wilson 95% CI [0.791, 0.815]; alternative variance 1.975276.
- 5% ROI: **4,834 bets** for 80% CLT power; Monte Carlo power 0.802, Wilson 95% CI [0.790, 0.814]; alternative variance 2.006294.

## Limits

Historical odds are a design proxy and may differ from a selected future strategy. Fixed prospective odds would use sums of conditional variances and the exact fixed-price composition; iid mixture calculations include additional variance from price sampling. No uncertainty from strategy fitting, price estimation or edge selection is included. Large odds create skewed payoffs; finite-sample approximation is checked by simulation, not assumed exact. Under positive serial correlation a rough effective sample size is n/[1+2 sum_h rho(h)], or n/[1+(m−1)rho] for equal clusters, so nominal sample-size needs increase by the design effect. Negative dependence can change this direction. Estimate dependence from the actual selection process before applying these iid numbers.

Figure: `reports/figures/power_curve.png`. All curves, validation seeds and simulation precision: `reports/results/power.json`.
