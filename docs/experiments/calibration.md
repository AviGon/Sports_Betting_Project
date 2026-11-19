# B1 — Bookmaker calibration

Question: do Bet365's de-vigged pre-closing probabilities match empirical H/D/A frequencies?

## Mathematics and assumptions

For decimal odds o_ik, p_ik = (1/o_ik) / sum_j(1/o_ij). Only odds enter these probabilities; outcomes are used solely for retrospective evaluation. The multiplicative margin allocation is an assumption and may hide nonuniform outcome margins. Neither p nor fitted calibration curves are claimed to be true probabilities.

For one-hot outcomes y_ik, the multiclass Brier score is mean_i sum_k (p_ik−y_ik)^2 (unscaled, range 0–2). Log loss is −mean_i log p_i,observed (natural logarithm, nats). Lower scores are better. These describe this market, without fitting or comparing Person A's model.

Reliability diagrams use ten fixed equal-width probability bins, separately for home wins, draws and away wins. Markers show observed frequencies against mean forecast probabilities. Frequency intervals use the intercept-only residual mean sandwich, with season clusters, CR1 correction and Student G−1 degrees of freedom; add the bin's mean p to residual endpoints and truncate to [0,1]. Tiny bins and few clusters give unstable uncertainty; these intervals are pointwise, not simultaneous or exact binomial intervals. Rows within a season may be dependent; seasons are assumed independent. Only 11 seasons exist.

Joint logistic calibration fits logit Pr(Y_k=1) = a_k + b_k logit(p_k) one-vs-rest. Perfect calibration corresponds to (a,b)=(0,1). Joint intercept and slope have season-cluster sandwich 95% CIs. These are diagnostic fits to the same historical outcomes, not new calibrated predictions or out-of-sample performance estimates. No joint hypothesis test is inferred from individual coefficient CIs.

## Computed results

4,180 actual matches, 2015–16 through 2025–26. Brier score: **0.568975**. Log loss: **0.960206 nats**.

- H: observed 1853 vs expected 1843.27; frequencies 0.4433 vs 0.4410. Joint intercept 0.012 (95% CI -0.089, 0.114); slope 1.005 (95% CI 0.937, 1.073).
- D: observed 990 vs expected 1004.64; frequencies 0.2368 vs 0.2403. Joint intercept 0.032 (95% CI -0.243, 0.307); slope 1.046 (95% CI 0.803, 1.288).
- A: observed 1337 vs expected 1332.09; frequencies 0.3199 vs 0.3187. Joint intercept 0.027 (95% CI -0.089, 0.143); slope 1.032 (95% CI 0.892, 1.172).

Full bin counts and diagnostics: `reports/results/calibration.json`. Figure: `reports/figures/calibration.png`. Apparent deviations need the multiplicity-controlled tests in B2. These descriptive summaries do not establish profitable tradable opportunities.
