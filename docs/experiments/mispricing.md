# B2 — Systematic market mispricing

## Hypotheses and fixed analysis plan

Each test uses residual R_ik = 1(Y_i=k)−p_ik. H0: the group's mean residual is zero; H1: it is nonzero (two-sided). Positive values indicate more wins than the de-vigged market expected; negative values indicate fewer. These are probability deviations, not return advantages after margin.

Before examining significance, define five probability bands [0,.15), [.15,.30), [.30,.50), [.50,.70), [.70,1); three outcome groups H/D/A; and each team's win outcomes, home and away pooled. Minimum eligibility: 100 outcome rows/matches and 5 seasons. Membership in team tests depends only on fixture identities and coverage, not observed effects. A team contributes at most one win indicator per fixture; draw outcomes are not team wins. The band sample size counts outcome rows, not independent matches.

## Dependence and inference

Primary inference uses season-cluster residual sums and the CR1 variance G/(G−1) × sum_g [sum_i∈g (R_i−mean R)]²/n². Use a Student t reference with G−1 degrees of freedom and two-sided pointwise 95% CIs. All H/D/A outcomes from a match remain in the same season cluster, so mutually exclusive outcomes are never treated as independent. Dates within a season may be dependent. Seasons are assumed independent; only 11 exist, with some team groups having fewer, so finite-cluster inference remains approximate. Mean intervals are not simultaneous.

Date-cluster CIs and p-values are a sensitivity analysis (assumes dates independent). This often yields different precision. No iid binomial test is used for the pooled outcome rows. The tests do not identify the source of a deviation and do not establish a causal or tradable effect.

Benjamini–Hochberg FDR is applied once to **all 28 eligible tests** (bands, outcomes and teams together). Report all raw p-values, BH q-values and effects. No claim is based on an uncorrected p < .05. BH guarantees need independence or suitable positive dependence; overlapping teams and compositional outcomes may violate that assumption. Also report Benjamini–Yekutieli q-values (BH × harmonic number of tests) as a conservative arbitrary-dependence sensitivity. Cluster p-values themselves remain approximations under few clusters. Tests are exploratory and not a held-out confirmation.

## Computed results

**0 of 28 tests have BH q < .05.**

- Brighton: n=342, seasons=9; effect -3.40 pp, 95% CI [-6.94, +0.14]; p=0.057516, BH q=0.75153, BY q=1.
- Newcastle: n=380, seasons=10; effect +3.25 pp, 95% CI [-0.19, +6.70]; p=0.061373, BH q=0.75153, BY q=1.
- Southampton: n=342, seasons=9; effect -5.48 pp, 95% CI [-11.80, +0.84]; p=0.080521, BH q=0.75153, BY q=1.
- p in [0.30,0.50): n=2992, seasons=11; effect -0.97 pp, 95% CI [-2.55, +0.61]; p=0.20287, BH q=0.91642, BY q=1.
- p in [0.50,0.70): n=1617, seasons=11; effect +1.81 pp, 95% CI [-1.64, +5.25]; p=0.26992, BH q=0.91642, BY q=1.
- Arsenal: n=418, seasons=11; effect +2.28 pp, 95% CI [-2.57, +7.14]; p=0.31861, BH q=0.91642, BY q=1.
- Aston Villa: n=304, seasons=8; effect +4.06 pp, 95% CI [-3.30, +11.43]; p=0.23343, BH q=0.91642, BY q=1.
- Bournemouth: n=342, seasons=9; effect +1.01 pp, 95% CI [-1.76, +3.77]; p=0.42548, BH q=0.91642, BY q=1.
- Brentford: n=190, seasons=5; effect +3.22 pp, 95% CI [-4.93, +11.37]; p=0.33429, BH q=0.91642, BY q=1.
- Chelsea: n=418, seasons=11; effect -4.82 pp, 95% CI [-11.76, +2.12]; p=0.15256, BH q=0.91642, BY q=1.
- Crystal Palace: n=418, seasons=11; effect +0.90 pp, 95% CI [-1.34, +3.15]; p=0.39154, BH q=0.91642, BY q=1.
- Leicester: n=342, seasons=9; effect +2.52 pp, 95% CI [-4.35, +9.40]; p=0.42209, BH q=0.91642, BY q=1.
- West Ham: n=418, seasons=11; effect +2.59 pp, 95% CI [-1.54, +6.73]; p=0.19266, BH q=0.91642, BY q=1.
- p in [0.00,0.15): n=1395, seasons=11; effect -0.75 pp, 95% CI [-2.98, +1.48]; p=0.47155, BH q=0.94311, BY q=1.
- p in [0.15,0.30): n=5898, seasons=11; effect +0.23 pp, 95% CI [-0.98, +1.44]; p=0.68212, BH q=0.96184, BY q=1.
- p in [0.70,1.00): n=638, seasons=11; effect -0.52 pp, 95% CI [-3.56, +2.53]; p=0.71292, BH q=0.96184, BY q=1.
- Outcome H: n=4180, seasons=11; effect +0.23 pp, 95% CI [-1.86, +2.33]; p=0.80939, BH q=0.96184, BY q=1.
- Outcome D: n=4180, seasons=11; effect -0.35 pp, 95% CI [-1.96, +1.26]; p=0.63861, BH q=0.96184, BY q=1.
- Burnley: n=304, seasons=8; effect +2.10 pp, 95% CI [-5.30, +9.50]; p=0.52339, BH q=0.96184, BY q=1.
- Everton: n=418, seasons=11; effect -0.32 pp, 95% CI [-3.44, +2.80]; p=0.82443, BH q=0.96184, BY q=1.
- Fulham: n=228, seasons=6; effect +1.33 pp, 95% CI [-7.83, +10.49]; p=0.72435, BH q=0.96184, BY q=1.
- Manchester City: n=418, seasons=11; effect +0.50 pp, 95% CI [-3.62, +4.62]; p=0.79339, BH q=0.96184, BY q=1.
- Tottenham Hotspur: n=418, seasons=11; effect +1.32 pp, 95% CI [-4.02, +6.67]; p=0.59337, BH q=0.96184, BY q=1.
- Wolverhampton Wanderers: n=304, seasons=8; effect +0.93 pp, 95% CI [-4.77, +6.63]; p=0.7108, BH q=0.96184, BY q=1.
- Outcome A: n=4180, seasons=11; effect +0.12 pp, 95% CI [-1.86, +2.10]; p=0.89751, BH q=0.97922, BY q=1.
- Watford: n=228, seasons=6; effect -0.27 pp, 95% CI [-6.13, +5.58]; p=0.90928, BH q=0.97922, BY q=1.
- Liverpool: n=418, seasons=11; effect +0.02 pp, 95% CI [-5.91, +5.95]; p=0.99332, BH q=0.99332, BY q=1.
- Manchester United: n=418, seasons=11; effect +0.06 pp, 95% CI [-4.74, +4.86]; p=0.97864, BH q=0.99332, BY q=1.

All estimates and date-cluster sensitivities: `reports/results/mispricing_tests.csv`. Figure: `reports/figures/mispricing.png`. Failure to reject does not prove market efficiency; rejected deviations need independent replication and executable-odds analysis before any profitability claim. The groups and multiplicity family must remain fixed when interpreting these reported tests.
