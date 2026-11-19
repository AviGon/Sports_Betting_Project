"""Rebuild Ksheer's figures, result JSON and experiment documentation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.prepare import validate
from src.statistics import calibration


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
    parser.add_argument("--only", choices=["calibration", "all"], default="all")
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


if __name__ == "__main__":
    main()
