# B3 — Monte Carlo: luck versus skill

Question: could the fixed bet list's realized return arise from randomness under de-vigged market probabilities?

## Design

Strategy: `home_baseline`. The built-in, pre-specified baseline places one unit on every home team; it is not a fitted or optimized strategy. Imported selections use `match_id, selected_side, odds, estimated_ev, realized_return`. Sides are H/D/A, odds are raw decimal quotes >1, stakes are one unit each and returns are net (+o−1 on a win, −1 otherwise). `estimated_ev` is allowed to be missing and is never used in the null simulation. Import validates historical settlement, unknown IDs, duplicate (match_id, side) rows and missing odds. Different sides of one fixture are allowed; duplicate same-side rows are rejected to catch double counting. Only unit stakes are currently supported.

Keep selections and raw odds fixed. For each simulation, draw exactly ONE categorical H/D/A outcome per selected match from its market probabilities. Settle ALL bets on that match using that single outcome. Different matches are simulated independently. This preserves within-match exclusivity but omits serial/team/date dependence between matches. The shared probabilities are a hypothesized market null, not known truths. No historical matches or bet rows are resampled here.

For payoff vector r_i across the three outcomes (summing all bets on fixture i), E[ROI] = sum_i sum_k p_ik r_ik / N_bets. Under independent matches, Var(ROI) = sum_i [sum_k p_ik r_ik² − (sum_k p_ik r_ik)²] / N_bets². Raw odds, rather than fair odds 1/p, determine payouts, so market-null ROI usually reflects the bookmaker margin.

An observed high return is compared to the null distribution with upper-tail Monte Carlo p = (1 + number of simulated ROI ≥ observed)/(B+1). It is a single-strategy conditional diagnostic, not proof of skill. Multiple strategy searches would require another multiplicity/design correction. If imported selections were chosen using these historical outcomes, conditioning on the selected bet list does not correct selection bias; use genuinely prespecified or out-of-sample selections.

## Computed results

- Fixed unit bets: 4,180 across 4,180 matches.
- Selected match dates: 2015-08-08 through 2026-05-24.
- Simulation paths: 10,000; NumPy default_rng seed: 225.
- Market-null expected ROI: **-4.294%**.
- Observed historical ROI: **-3.881%**.
- Central 95% simulation interval: **[-8.460%, -0.103%]**.
- Observed midrank percentile: **57.48**.
- Upper-tail p: 0.42536; approximate Monte Carlo SE 0.00494.
- Simulated mean -4.283%; its Monte Carlo SE 0.0212%.
- Simulated ROI SD 2.124%; theoretical SD 2.125%.

This interval predicts hypothetical outcomes conditional on the fixed market-null model. It is **not** a confidence interval for a strategy's unknown mean return. Historical bootstrap CIs are described separately. Neither a return inside this interval nor failure to reject establishes that the market model is correct. See `reports/results/monte_carlo.json` and the corresponding paths CSV and figure.
