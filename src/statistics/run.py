"""Rebuild Ksheer's figures, result JSON and experiment documentation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.prepare import validate
from src.statistics import calibration, mispricing


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def calibration_doc(r):
    text = f"""# B1 — Bookmaker calibration

Question: do Bet365's de-vigged pre-closing probabilities match empirical H/D/A frequencies?

## Mathematics and assumptions

For decimal odds o_ik, p_ik = (1/o_ik) / sum_j(1/o_ij). Only odds enter these probabilities; outcomes are used solely for retrospective evaluation. The multiplicative margin allocation is an assumption and may hide nonuniform outcome margins. Neither p nor fitted calibration curves are claimed to be true probabilities.

For one-hot outcomes y_ik, the multiclass Brier score is mean_i sum_k (p_ik−y_ik)^2 (unscaled, range 0–2). Log loss is −mean_i log p_i,observed (natural logarithm, nats). Lower scores are better. These describe this market, without fitting or comparing Person A's model.

Reliability diagrams use ten fixed equal-width probability bins, separately for home wins, draws and away wins. Markers show observed frequencies against mean forecast probabilities. Frequency intervals use the intercept-only residual mean sandwich, with season clusters, CR1 correction and Student G−1 degrees of freedom; add the bin's mean p to residual endpoints and truncate to [0,1]. Tiny bins and few clusters give unstable uncertainty; these intervals are pointwise, not simultaneous or exact binomial intervals. Rows within a season may be dependent; seasons are assumed independent. Only 11 seasons exist.

Joint logistic calibration fits logit Pr(Y_k=1) = a_k + b_k logit(p_k) one-vs-rest. Perfect calibration corresponds to (a,b)=(0,1). Joint intercept and slope have season-cluster sandwich 95% CIs. These are diagnostic fits to the same historical outcomes, not new calibrated predictions or out-of-sample performance estimates. No joint hypothesis test is inferred from individual coefficient CIs.

## Computed results

{r['n_matches']:,} actual matches, 2015–16 through 2025–26. Brier score: **{r['brier_score']:.6f}**. Log loss: **{r['log_loss']:.6f} nats**.

"""
    for side, v in r["sides"].items():
        c = v["calibration"]
        text += f"- {side}: observed {v['observed_count']} vs expected {v['expected_count']:.2f}; frequencies {v['observed_frequency']:.4f} vs {v['expected_frequency']:.4f}. Joint intercept {c['intercept']:.3f} (95% CI {c['intercept_ci'][0]:.3f}, {c['intercept_ci'][1]:.3f}); slope {c['slope']:.3f} (95% CI {c['slope_ci'][0]:.3f}, {c['slope_ci'][1]:.3f}).\n"
    return text + "\nFull bin counts and diagnostics: `reports/results/calibration.json`. Figure: `reports/figures/calibration.png`. Apparent deviations need the multiplicity-controlled tests in B2. These descriptive summaries do not establish profitable tradable opportunities.\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["calibration", "mispricing", "all"], default="all")
    args = parser.parse_args()
    df = pd.read_csv("data/processed/epl_matches.csv")
    validate(df)
    figures = Path("reports/figures")
    results = Path("reports/results")
    docs = Path("docs/experiments")
    for path in (figures, results, docs):
        path.mkdir(parents=True, exist_ok=True)
    if args.only in ("calibration", "all"):
        r = calibration.analyze(df)
        write_json(results / "calibration.json", r)
        calibration.figure(r, figures / "calibration.png")
        (docs / "calibration.md").write_text(calibration_doc(r))
        print(f"Calibration: Brier={r['brier_score']:.6f}; log loss={r['log_loss']:.6f}")
    if args.only in ("mispricing", "all"):
        r = mispricing.analyze(df)
        write_json(results / "mispricing.json", r)
        pd.DataFrame(r["tests"]).to_csv(results / "mispricing_tests.csv", index=False)
        mispricing.figure(r, figures / "mispricing.png")
        (docs / "mispricing.md").write_text(mispricing_doc(r))
        print(f"Mispricing: {r['n_rejected_bh']} / {r['n_tests']} BH rejections")


def mispricing_doc(r):
    text = f"""# B2 — Systematic market mispricing

## Hypotheses and fixed analysis plan

Each test uses residual R_ik = 1(Y_i=k)−p_ik. H0: the group's mean residual is zero; H1: it is nonzero (two-sided). Positive values indicate more wins than the de-vigged market expected; negative values indicate fewer. These are probability deviations, not return advantages after margin.

Before examining significance, define five probability bands [0,.15), [.15,.30), [.30,.50), [.50,.70), [.70,1); three outcome groups H/D/A; and each team's win outcomes, home and away pooled. Minimum eligibility: {r['min_matches']} outcome rows/matches and {r['min_seasons']} seasons. Membership in team tests depends only on fixture identities and coverage, not observed effects. A team contributes at most one win indicator per fixture; draw outcomes are not team wins. The band sample size counts outcome rows, not independent matches.

## Dependence and inference

Primary inference uses season-cluster residual sums and the CR1 variance G/(G−1) × sum_g [sum_i∈g (R_i−mean R)]²/n². Use a Student t reference with G−1 degrees of freedom and two-sided pointwise 95% CIs. All H/D/A outcomes from a match remain in the same season cluster, so mutually exclusive outcomes are never treated as independent. Dates within a season may be dependent. Seasons are assumed independent; only 11 exist, with some team groups having fewer, so finite-cluster inference remains approximate. Mean intervals are not simultaneous.

Date-cluster CIs and p-values are a sensitivity analysis (assumes dates independent). This often yields different precision. No iid binomial test is used for the pooled outcome rows. The tests do not identify the source of a deviation and do not establish a causal or tradable effect.

Benjamini–Hochberg FDR is applied once to **all {r['n_tests']} eligible tests** (bands, outcomes and teams together). Report all raw p-values, BH q-values and effects. No claim is based on an uncorrected p < .05. BH guarantees need independence or suitable positive dependence; overlapping teams and compositional outcomes may violate that assumption. Also report Benjamini–Yekutieli q-values (BH × harmonic number of tests) as a conservative arbitrary-dependence sensitivity. Cluster p-values themselves remain approximations under few clusters. Tests are exploratory and not a held-out confirmation.

## Computed results

**{r['n_rejected_bh']} of {r['n_tests']} tests have BH q < .05.**

"""
    for row in sorted(r["tests"], key=lambda x: x["q_value_bh"]):
        text += f"- {row['name']}: n={row['n']}, seasons={row['clusters']}; effect {100*row['effect']:+.2f} pp, 95% CI [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}]; p={row['p_value']:.5g}, BH q={row['q_value_bh']:.5g}, BY q={row['q_value_by']:.5g}.\n"
    text += "\nAll estimates and date-cluster sensitivities: `reports/results/mispricing_tests.csv`. Figure: `reports/figures/mispricing.png`. Failure to reject does not prove market efficiency; rejected deviations need independent replication and executable-odds analysis before any profitability claim. The groups and multiplicity family must remain fixed when interpreting these reported tests.\n"
    return text


if __name__ == "__main__":
    main()
