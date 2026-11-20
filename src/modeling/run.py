"""Run all Track A experiments: python -m src.modeling.run

Order of operations enforces the pre-registration rules:
1. Poisson hyperparameters are chosen by log loss on TRAIN seasons only.
2. The model is refitted walk-forward for every later match date.
3. The EV threshold is chosen on VALIDATION seasons only and written to disk.
4. The frozen rule is applied once to TEST seasons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .data import DATA_PATH, SIDES, TEST, TRAIN, VALIDATION, load_matches
from .evaluate import (PROB_COLS, calibration_table, compare_forecasters, log_loss,
                       naive_frequencies, per_match_scores)
from .poisson import PoissonConfig, walk_forward

OUT = Path("reports/track_a")
XI_GRID = (0.0, 0.001, 0.002, 0.004)
RIDGE_GRID = (0.5, 2.0, 8.0)

# Fixed colours (reference categorical palette): entity -> colour, never by rank.
COLOURS = {"poisson": "#2a78d6", "market": "#eb6834", "naive": "#8a8984"}
LABELS = {"poisson": "Poisson model", "market": "Bookmaker (Bet365, de-vigged)",
          "naive": "Naive frequencies"}


def tune_hyperparameters(matches: pd.DataFrame) -> tuple[PoissonConfig, pd.DataFrame]:
    rows = []
    for xi, ridge in product(XI_GRID, RIDGE_GRID):
        cfg = PoissonConfig(xi=xi, ridge=ridge)
        pred = walk_forward(matches, cfg, predict_seasons=TRAIN).merge(matches[["match_id", "y"]])
        p = pred[PROB_COLS["poisson"]].to_numpy()
        rows.append({"xi": xi, "ridge": ridge, "half_life_days": (np.log(2) / xi) if xi else np.inf,
                     "train_logloss": float(log_loss(p, pred["y"].to_numpy()).mean())})
        print(f"  xi={xi:<6} ridge={ridge:<4} train log loss={rows[-1]['train_logloss']:.5f}")
    table = pd.DataFrame(rows).sort_values("train_logloss").reset_index(drop=True)
    best = table.iloc[0]
    return PoissonConfig(xi=float(best["xi"]), ridge=float(best["ridge"])), table


def plot_setup():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "lines.linewidth": 2, "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "legend.frameon": False,
    })
    return plt


def figure_scores(df: pd.DataFrame, evaluation: dict, path: Path):
    plt = plot_setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    scores = per_match_scores(df).assign(season=df["season"].to_numpy())
    seasons = list(TRAIN + VALIDATION + TEST)
    by = scores.groupby("season").mean().reindex(seasons)
    x = np.arange(len(seasons))
    for name in ("naive", "poisson", "market"):
        ax1.plot(x, by[f"brier_{name}"], marker="o", ms=4, color=COLOURS[name], label=LABELS[name])
    ymax = by[[f"brier_{n}" for n in PROB_COLS]].to_numpy().max() + .02
    ax1.set_ylim(top=ymax)
    spans = ((-.5, len(TRAIN) - .5, "train", None),
             (len(TRAIN) - .5, len(TRAIN) + len(VALIDATION) - .5, "validation", "#f2f1ec"),
             (len(TRAIN) + len(VALIDATION) - .5, len(seasons) - .5, "test", "#e6e5e0"))
    for lo, hi, lab, shade in spans:
        if shade:
            ax1.axvspan(lo, hi, color=shade, zorder=0)
        ax1.text((lo + hi) / 2, ymax - .002, lab, ha="center", va="top", color="#52514e", fontsize=8)
    ax1.set_xlim(-.5, len(seasons) - .5)
    ax1.set_xticks(x, [s[2:] for s in seasons], rotation=45)
    ax1.set_ylabel("Mean Brier score (lower = better)")
    ax1.set_title("Forecast accuracy by season", loc="left", fontsize=10)
    ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)

    comps = [("brier:poisson-market", "Brier: Poisson − bookmaker"),
             ("logloss:poisson-market", "Log loss: Poisson − bookmaker"),
             ("brier:poisson-naive", "Brier: Poisson − naive")]
    test_diff = evaluation["test"]["differences"]
    for i, (key, lab) in enumerate(comps):
        d = test_diff[key]
        ax2.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci_low"]], [d["ci_high"] - d["mean"]]],
                     fmt="o", ms=6, color="#0b0b0b", capsize=3, lw=1.5)
        ax2.annotate(f"{d['mean']:+.4f}", (d["mean"], i), xytext=(0, -12), textcoords="offset points",
                     fontsize=8, color="#52514e", ha="center", va="top")
    ax2.axvline(0, color="#8a8984", lw=1)
    ax2.set_yticks(range(len(comps)), [l.replace(": ", ":\n") for _, l in comps])
    ax2.set_ylim(len(comps) - .4, -.6)
    ax2.set_xlabel("Paired difference, 2022–26 (95% CI)\n> 0 means the first forecaster is worse")
    ax2.set_title("Test-set paired differences", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_calibration(cal: pd.DataFrame, path: Path):
    plt = plot_setup()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)
    for ax, side in zip(axes, SIDES):
        sub = cal[cal.side == side]
        lo = min(sub.mean_pred.min(), sub.ci_low.min()) - .02
        hi = max(sub.mean_pred.max(), sub.ci_high.max()) + .02
        ax.plot([lo, hi], [lo, hi], color="#8a8984", lw=1, ls="--", label="Perfect calibration")
        for name, off in (("poisson", -.003), ("market", .003)):
            s = sub[sub.forecaster == name]
            ax.errorbar(s.mean_pred + off, s.observed, yerr=[s.observed - s.ci_low, s.ci_high - s.observed],
                        fmt="o-", ms=4, lw=1.5, capsize=0, elinewidth=1, color=COLOURS[name],
                        label=LABELS[name])
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_title({"H": "Home win", "D": "Draw", "A": "Away win"}[side], loc="left", fontsize=10)
        ax.set_xlabel("Mean forecast probability (decile bin)")
    axes[0].set_ylabel("Observed frequency (95% Wilson CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, -.06))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_backtest(val_bets, test_bets, grid_val, grid_test, tau, path: Path):
    plt = plot_setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    for bets, name, colour in ((val_bets, "validation", "#8a8984"), (test_bets, "test", COLOURS["poisson"])):
        if bets.empty:
            continue
        cum = bets.sort_values("date")["realized_return"].cumsum().to_numpy()
        ax1.plot(np.arange(1, len(cum) + 1), cum, color=colour, label=f"{name} ({len(cum)} bets)")
        null = bets.sort_values("date")["market_null_ev"].cumsum().to_numpy()
        ax1.plot(np.arange(1, len(null) + 1), null, color=colour, lw=1, ls=":")
    ax1.axhline(0, color="#8a8984", lw=1)
    ax1.set_xlabel("Bet number (chronological)")
    ax1.set_ylabel("Cumulative profit (units staked = 1)")
    ax1.set_title(f"Value bets, EV > {tau:.3f} (dotted: expected under market odds)", loc="left", fontsize=10)
    ax1.legend(fontsize=8, loc="lower left")

    for table, name, colour, off in ((grid_val, "validation (selects τ)", "#8a8984", -.003),
                                     (grid_test, "test (exploratory)", COLOURS["poisson"], .003)):
        t = table[table.n_bets > 0]
        lo = np.array([c[0] for c in t.roi_ci95]); hi = np.array([c[1] for c in t.roi_ci95])
        ax2.errorbar(t.tau + off, t.roi, yerr=[t.roi - lo, hi - t.roi], fmt="o-", ms=4, lw=1.5,
                     elinewidth=1, color=colour, label=name)
    ax2.axhline(0, color="#8a8984", lw=1)
    ax2.axvline(tau, color="#0b0b0b", lw=1, ls="--")
    ax2.set_xlabel("EV threshold τ")
    ax2.set_ylabel("ROI per unit staked (95% CI)")
    ax2.set_title("ROI vs threshold", loc="left", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=int, default=5000, help="bootstrap resamples (default 5000)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--skip-tuning", action="store_true",
                        help="reuse hyperparameters saved in reports/track_a/results/poisson_config.json")
    args = parser.parse_args(argv)
    res_dir, fig_dir = OUT / "results", OUT / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    cfg_path = res_dir / "poisson_config.json"
    if args.skip_tuning and cfg_path.exists():
        cfg = PoissonConfig(**json.loads(cfg_path.read_text(encoding="utf-8"))["config"])
        print(f"A1: reusing {cfg}")
    else:
        print("A1: tuning Poisson hyperparameters on train seasons", TRAIN)
        cfg, tuning = tune_hyperparameters(matches)
        tuning.to_csv(res_dir / "hyperparameter_search.csv", index=False)
        cfg_path.write_text(json.dumps({"config": cfg.__dict__, "selected_on": list(TRAIN),
                                        "criterion": "mean log loss"}, indent=2), encoding="utf-8")
        print(f"A1: selected {cfg}")

    print("A1: walk-forward predictions for", TRAIN[0], "to", TEST[-1])
    pred = walk_forward(matches, cfg, predict_seasons=TRAIN + VALIDATION + TEST)
    pred.to_csv(res_dir / "predictions.csv", index=False, float_format="%.6f")

    df = (matches.merge(pred, on="match_id").merge(naive_frequencies(matches), on="match_id")
          .sort_values("date").reset_index(drop=True))

    print("A2: forecast comparison")
    evaluation = {}
    for name, seasons in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST)):
        evaluation[name] = compare_forecasters(df[df.season.isin(seasons)], args.boot, args.seed)
    scores = per_match_scores(df).assign(season=df.season)
    evaluation["by_season"] = scores.groupby("season").mean().round(5).to_dict(orient="index")
    test = df[df.season.isin(TEST)]
    cal = pd.concat([calibration_table(test, f) for f in ("poisson", "market")])
    cal.to_csv(res_dir / "calibration_test.csv", index=False)
    (res_dir / "model_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print("A3: choosing EV threshold on validation seasons", VALIDATION)
    val = df[df.season.isin(VALIDATION)]
    tau, grid_val = bt.choose_threshold(val, n_boot=args.boot, seed=args.seed)
    (res_dir / "threshold_lock.json").write_text(json.dumps(
        {"tau": tau, "grid": list(bt.TAU_GRID), "min_validation_bets": bt.MIN_VALIDATION_BETS,
         "rule": "max validation ROI, ties -> smaller tau", "selected_on": list(VALIDATION)}, indent=2), encoding="utf-8")
    print(f"A3: locked tau = {tau}")
    val_bets = bt.select_bets(val, tau)
    test_bets = bt.select_bets(test, tau)
    grid_test = pd.DataFrame([{"tau": t, **{k: v for k, v in bt.summarise(bt.select_bets(test, t), args.boot,
                                                                          args.seed).items()
                                             if k in ("n_bets", "roi", "roi_ci95", "profit_units",
                                                      "market_null_expected_roi")}}
                              for t in bt.TAU_GRID])
    backtest = {
        "tau": tau,
        "validation": bt.summarise(val_bets, args.boot, args.seed),
        "test": bt.summarise(test_bets, args.boot, args.seed),
        "test_by_season": {s: bt.summarise(g, args.boot, args.seed) for s, g in test_bets.groupby("season")},
        "threshold_grid_validation": grid_val.to_dict(orient="records"),
        "threshold_grid_test_exploratory": grid_test.to_dict(orient="records"),
    }
    (res_dir / "backtest.json").write_text(json.dumps(backtest, indent=2, default=float), encoding="utf-8")
    bt.selection_export(test_bets).to_csv(res_dir / "selections_test.csv", index=False, float_format="%.6f")
    bt.selection_export(val_bets).to_csv(res_dir / "selections_validation.csv", index=False, float_format="%.6f")

    print("Figures")
    figure_scores(df, evaluation, fig_dir / "a2_forecast_scores.png")
    figure_calibration(cal, fig_dir / "a2_calibration_test.png")
    figure_backtest(val_bets, test_bets, grid_val, grid_test, tau, fig_dir / "a3_backtest.png")

    (res_dir / "run_metadata.json").write_text(json.dumps({
        "input": str(DATA_PATH), "input_sha256": sha256(DATA_PATH), "seed": args.seed,
        "bootstrap_resamples": args.boot, "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "splits": {"train": TRAIN, "validation": VALIDATION, "test": TEST},
    }, indent=2), encoding="utf-8")

    t = evaluation["test"]
    print(json.dumps({"test_scores": t["scores"],
                      "brier_poisson_minus_market": t["differences"]["brier:poisson-market"],
                      "backtest_test": backtest["test"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
