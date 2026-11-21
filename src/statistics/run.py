"""Rebuild market statistics, figures and experiment documentation."""
import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import pandas as pd

from src.data import MATCHES_PATH, dataset_sha256
from src.data.prepare import validate
from src.statistics import bootstrap, calibration, mispricing, monte_carlo, power


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def calibration_doc(r):
    text = f"""# Bookmaker calibration

Question: do Bet365's de-vigged pre-closing probabilities match empirical H/D/A frequencies?

## Mathematics and assumptions

For decimal odds o_ik, p_ik = (1/o_ik) / sum_j(1/o_ij). Only odds enter these probabilities; outcomes are used solely for retrospective evaluation. The multiplicative margin allocation is an assumption and may hide nonuniform outcome margins. Neither p nor fitted calibration curves are claimed to be true probabilities.

For one-hot outcomes y_ik, the multiclass Brier score is mean_i sum_k (p_ik−y_ik)^2 (unscaled, range 0–2). Log loss is −mean_i log p_i,observed (natural logarithm, nats). Lower scores are better. These describe this market, without fitting or comparing the Poisson model.

Reliability diagrams use ten fixed equal-width probability bins, separately for home wins, draws and away wins. Markers show observed frequencies against mean forecast probabilities. Frequency intervals use the intercept-only residual mean sandwich, with season clusters, CR1 correction and Student G−1 degrees of freedom; add the bin's mean p to residual endpoints and truncate to [0,1]. Tiny bins and few clusters give unstable uncertainty; these intervals are pointwise, not simultaneous or exact binomial intervals. Rows within a season may be dependent; seasons are assumed independent. Only 11 seasons exist.

Joint logistic calibration fits logit Pr(Y_k=1) = a_k + b_k logit(p_k) one-vs-rest. Perfect calibration corresponds to (a,b)=(0,1). Joint intercept and slope have season-cluster sandwich 95% CIs. These are diagnostic fits to the same historical outcomes, not new calibrated predictions or out-of-sample performance estimates. No joint hypothesis test is inferred from individual coefficient CIs.

## Computed results

{r['n_matches']:,} actual matches, 2015–16 through 2025–26. Brier score: **{r['brier_score']:.6f}**. Log loss: **{r['log_loss']:.6f} nats**.

"""
    for side, v in r["sides"].items():
        c = v["calibration"]
        text += f"- {side}: observed {v['observed_count']} vs expected {v['expected_count']:.2f}; frequencies {v['observed_frequency']:.4f} vs {v['expected_frequency']:.4f}. Joint intercept {c['intercept']:.3f} (95% CI {c['intercept_ci'][0]:.3f}, {c['intercept_ci'][1]:.3f}); slope {c['slope']:.3f} (95% CI {c['slope_ci'][0]:.3f}, {c['slope_ci'][1]:.3f}).\n"
    return text + "\nFull bin counts and diagnostics: `reports/results/calibration.json`. Figure: `reports/figures/calibration.png`. Apparent deviations need the multiplicity-controlled mispricing tests. These descriptive summaries do not establish profitable tradable opportunities.\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["calibration", "mispricing", "monte-carlo", "bootstrap", "returns", "power", "all"], default="all")
    parser.add_argument("--bets", type=Path, help="Bet selection CSV; used only by Monte Carlo/bootstrap")
    parser.add_argument("--label", help="Scientific output label for an imported bet list (e.g. poisson_test)")
    parser.add_argument("--paths", type=int, default=10000, help="Outcome simulation paths (>=2000)")
    parser.add_argument("--bootstrap-paths", type=int, default=5000)
    parser.add_argument("--power-paths", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=225, help="MC seed; bootstrap and power use +1 and +2")
    args = parser.parse_args()
    if args.label and (not args.bets or not re.fullmatch(r"[a-z][a-z0-9_]*", args.label) or args.label == "home_baseline"):
        parser.error("--label needs --bets and a lowercase scientific name other than home_baseline")
    df = pd.read_csv(MATCHES_PATH)
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
        (docs / "calibration.md").write_text(calibration_doc(r), encoding="utf-8")
        print(f"Calibration: Brier={r['brier_score']:.6f}; log loss={r['log_loss']:.6f}")
    if args.only in ("mispricing", "all"):
        r = mispricing.analyze(df)
        write_json(results / "mispricing.json", r)
        pd.DataFrame(r["tests"]).to_csv(results / "mispricing_tests.csv", index=False)
        mispricing.figure(r, figures / "mispricing.png")
        (docs / "mispricing.md").write_text(mispricing_doc(r), encoding="utf-8")
        print(f"Mispricing: {r['n_rejected_bh']} / {r['n_tests']} BH rejections")
    label = (args.label or "imported") if args.bets else "home_baseline"
    suffix = f"_{label}" if args.bets else ""
    if args.only in ("monte-carlo", "bootstrap", "returns", "all"):
        bets = pd.read_csv(args.bets) if args.bets else monte_carlo.home_baseline(df)
        monte_carlo.validate_bets(df, bets)
        if not args.bets:
            bets.to_csv(results / "home_baseline_selections.csv", index=False)
    if args.only in ("monte-carlo", "returns", "all"):
        r, paths = monte_carlo.simulate(df, bets, args.paths, args.seed)
        r["strategy"] = label
        write_json(results / f"monte_carlo{suffix}.json", r)
        pd.DataFrame({"roi": paths}).to_csv(results / f"monte_carlo_paths{suffix}.csv", index=False)
        monte_carlo.figure(r, paths, figures / f"monte_carlo{suffix}.png")
        (docs / f"monte_carlo{suffix}.md").write_text(monte_carlo_doc(r), encoding="utf-8")
        print(f"Monte Carlo ({label}): observed={r['observed_roi']:.4%}; expected={r['expected_roi']:.4%}")
    if args.only in ("bootstrap", "returns", "all"):
        r = {"strategy": label, "intervals": [bootstrap.confidence_interval(df, bets, method=m,
             n_paths=args.bootstrap_paths, seed=args.seed+1)[0] for m in ("ordinary", "match", "date", "season")]}
        write_json(results / f"bootstrap{suffix}.json", r)
        (docs / f"bootstrap{suffix}.md").write_text(bootstrap_doc(r), encoding="utf-8")
        print(f"Bootstrap ({label}): four resampling units")
    if args.only in ("power", "all"):
        r = power.analyze(df, args.power_paths, args.seed+2)
        write_json(results / "power.json", r)
        power.figure(r, figures / "power_curve.png")
        (docs / "statistical_power.md").write_text(power_doc(r), encoding="utf-8")
        print("Power: " + "; ".join(f"{x['edge']:.0%}: n80={x['required_n_80']}" for x in r["rows"]))
    import importlib.metadata
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib", "pytest")},
                "input": MATCHES_PATH.as_posix(),
                "input_sha256": hashlib.sha256(MATCHES_PATH.read_bytes()).hexdigest(),
                "input_lf_sha256": dataset_sha256(MATCHES_PATH),
                "bets_sha256": hashlib.sha256(args.bets.read_bytes()).hexdigest() if args.bets else None,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(results / f"run_metadata{suffix}.json", metadata)


def mispricing_doc(r):
    text = f"""# Systematic market mispricing

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


def monte_carlo_doc(r):
    lo, hi = r["simulation_interval_95"]
    return f"""# Monte Carlo: luck versus skill

Question: could the fixed bet list's realized return arise from randomness under de-vigged market probabilities?

## Design

Strategy: `{r['strategy']}`. The built-in, pre-specified baseline places one unit on every home team; it is not a fitted or optimized strategy. Imported selections use `match_id, selected_side, odds, estimated_ev, realized_return`. Sides are H/D/A, odds are raw decimal quotes >1, stakes are one unit each and returns are net (+o−1 on a win, −1 otherwise). `estimated_ev` is allowed to be missing and is never used in the null simulation. Import validates historical settlement, unknown IDs, duplicate (match_id, side) rows and missing odds. Different sides of one fixture are allowed; duplicate same-side rows are rejected to catch double counting. Only unit stakes are currently supported.

Keep selections and raw odds fixed. For each simulation, draw exactly ONE categorical H/D/A outcome per selected match from its market probabilities. Settle ALL bets on that match using that single outcome. Different matches are simulated independently. This preserves within-match exclusivity but omits serial/team/date dependence between matches. The shared probabilities are a hypothesized market null, not known truths. No historical matches or bet rows are resampled here.

For payoff vector r_i across the three outcomes (summing all bets on fixture i), E[ROI] = sum_i sum_k p_ik r_ik / N_bets. Under independent matches, Var(ROI) = sum_i [sum_k p_ik r_ik² − (sum_k p_ik r_ik)²] / N_bets². Raw odds, rather than fair odds 1/p, determine payouts, so market-null ROI usually reflects the bookmaker margin.

An observed high return is compared to the null distribution with upper-tail Monte Carlo p = (1 + number of simulated ROI ≥ observed)/(B+1). It is a single-strategy conditional diagnostic, not proof of skill. Multiple strategy searches would require another multiplicity/design correction. If imported selections were chosen using these historical outcomes, conditioning on the selected bet list does not correct selection bias; use genuinely prespecified or out-of-sample selections.

## Computed results

- Fixed unit bets: {r['n_bets']:,} across {r['n_matches']:,} matches.
- Selected match dates: {r['date_min']} through {r['date_max']}.
- Simulation paths: {r['n_paths']:,}; NumPy default_rng seed: {r['seed']}.
- Market-null expected ROI: **{r['expected_roi']:.3%}**.
- Observed historical ROI: **{r['observed_roi']:.3%}**.
- Central 95% simulation interval: **[{lo:.3%}, {hi:.3%}]**.
- Observed midrank percentile: **{r['observed_percentile_midrank']:.2f}**.
- Upper-tail p: {r['upper_tail_p_value']:.5f}; approximate Monte Carlo SE {r['upper_tail_mc_se']:.5f}.
- Simulated mean {r['simulation_mean_roi']:.3%}; its Monte Carlo SE {r['mc_se_of_mean']:.4%}.
- Simulated ROI SD {r['simulation_sd_roi']:.3%}; theoretical SD {r['theoretical_sd_roi']:.3%}.

This interval predicts hypothetical outcomes conditional on the fixed market-null model. It is **not** a confidence interval for a strategy's unknown mean return. Historical bootstrap CIs are described separately. Neither a return inside this interval nor failure to reject establishes that the market model is correct. See `reports/results/monte_carlo{('_' + r['strategy'] if r['strategy'] != 'home_baseline' else '')}.json` and the corresponding paths CSV and figure.
"""


def bootstrap_doc(r):
    text = f"""# Historical bootstrap confidence intervals

Strategy: `{r['strategy']}`. These are percentile CIs for a historical mean return, distinct from the Monte Carlo experiment's simulation of new match outcomes under a specified market model. Reuse `src.statistics.bootstrap.confidence_interval(matches, bets, method=..., n_paths=5000, seed=226)` with the shared selection interface. Each row is one unit stake.

Resample with replacement the same number of units as observed:

- `ordinary`: individual bet rows; ignores shared-match dependence, useful only as an iid comparison.
- `match`: complete fixtures; different sides of one fixture stay together. Assumes match clusters independent.
- `date`: complete calendar match dates; retains all bets on the same date, allows contemporaneous dependence and assumes dates independent. This is a date-block bootstrap, not a moving multi-day bootstrap.
- `season`: whole seasons; retains serial dependence within each season. With only {next(x["n_units"] for x in r["intervals"] if x["method"] == "season")} seasons this interval is highly approximate; between-season independence and exchangeability can fail under structural changes (including COVID).

For each draw compute total sampled net returns / total sampled unit stakes. With unequal cluster sizes this is a ratio, not an unweighted average of cluster means. Keep repeated clusters with their full multiplicity. No model refitting or strategy reselection is performed. These CIs condition on historical strategy choices and do not correct data mining, market regime shifts or systematic missingness. Larger clusters need not always produce wider intervals: limited clusters can produce unstable variance estimates.

## Computed 95% percentile intervals

"""
    for row in r["intervals"]:
        text += f"- {row['method']}: [{row['ci'][0]:.3%}, {row['ci'][1]:.3%}], {row['n_units']:,} resampling units, {row['n_paths']:,} draws, seed {row['seed']}.\n"
    note = " For this one-home-bet-per-match baseline, ordinary and match resampling have the same target distribution. Small random differences may arise from unit ordering under the same seed." if r['strategy'] == 'home_baseline' else ""
    return text + f"\nObserved ROI: {r['intervals'][0]['observed_roi']:.3%}.{note} See the result JSON for exact values.\n"


def power_doc(r):
    text = f"""# Statistical power and sample size

Question: how many independent bets are required to detect hypothetical positive expected ROI?

## Null, alternatives and historical variance

One-sided H0: mean net return ≤0; H1: mean net return δ>0. Significance α=.05, target power .80. Hypothetical δ = 0%, 1%, 2%, 3%, 5%; none is evidence of an actual edge. These calculations study detectability, not a trading strategy.

Use the observed distribution of {r['source_n']:,} historical Bet365 HOME prices as the empirical population of prospective odds. Prospective observations independently draw a price from this fixed empirical distribution and then one binary win/loss for one selected bet on a separate match. This differs from the Monte Carlo experiment's fixed historical bet list with categorical outcomes. Grouping equal prices and using multinomial counts plus conditional binomial wins simulates this iid mixture exactly without enormous bet-by-path arrays. Mutually exclusive multiple bets are not modeled in this power experiment.

The break-even null win probability at odds o is q0=1/o, giving conditional mean ROI 0 and variance o−1. This **break-even null differs from the Monte Carlo experiment's de-vigged market null**, which usually predicts negative ROI. For a hypothetical edge, use qδ(o)=expit(logit(1/o)+λδ), finding λδ so the empirical mean of o qδ(o)−1 equals δ. This respects [0,1] even for very short odds where simply (1+δ)/o would exceed 1. The alternative's return variance is σδ²=E[o² qδ(o)]−(1+δ)², including price-mixture variation in conditional means. σ0²=E[o−1]. These are plug-in odds-based design variances, not known population constants.

For comparison, actual historical home-bet return sample variance: **{r['historical_return_variance']:.6f}**. Mean conditional variance under the de-vigged market probabilities: **{r['market_conditional_return_variance_mean']:.6f}**. They are reported separately rather than silently using a negative-ROI market null to test zero-ROI profitability.

## CLT calculation and validation

Reject H0 when sample mean > z_.95 σ0/√n. Approximate power = 1−Φ((z_.95 σ0−δ√n)/σδ). Required n = ceiling[(z_.95 σ0+z_.80 σδ)²/δ²]. At δ=0 the rejection probability is approximately α for every n; no finite n reaches 80% power.

Monte Carlo validation uses {r['n_paths']:,} independent experiments per point, base seed {r['seed']} (per-point seeds stored in JSON). Wilson 95% intervals quantify simulation precision, not uncertainty in the historical odds population. Validation also checks the null's rejection rate. Curves are CLT approximations; dots are actual simulated detection proportions.

## Computed designs

"""
    for row in r["rows"]:
        if row["required_n_80"] is None:
            text += f"- 0% ROI: no finite 80%-power sample size; σ0²={row['null_variance']:.6f}.\n"
        else:
            mc = row["mc_at_required_n"]
            text += f"- {row['edge']:.0%} ROI: **{row['required_n_80']:,} bets** for 80% CLT power; Monte Carlo power {mc['power']:.3f}, Wilson 95% CI [{mc['wilson_ci_95'][0]:.3f}, {mc['wilson_ci_95'][1]:.3f}]; alternative variance {row['alternative_variance']:.6f}.\n"
    return text + "\n## Limits\n\nHistorical odds are a design proxy and may differ from a selected future strategy. Fixed prospective odds would use sums of conditional variances and the exact fixed-price composition; iid mixture calculations include additional variance from price sampling. No uncertainty from strategy fitting, price estimation or edge selection is included. Large odds create skewed payoffs; finite-sample approximation is checked by simulation, not assumed exact. Under positive serial correlation a rough effective sample size is n/[1+2 sum_h rho(h)], or n/[1+(m−1)rho] for equal clusters, so nominal sample-size needs increase by the design effect. Negative dependence can change this direction. Estimate dependence from the actual selection process before applying these iid numbers.\n\nFigure: `reports/figures/power_curve.png`. All curves, validation seeds and simulation precision: `reports/results/power.json`.\n"



if __name__ == "__main__":
    main()
