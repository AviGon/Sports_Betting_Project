"""Predeclared residual tests with season clusters and one global BH family."""
import numpy as np
import pandas as pd

from src.data.prepare import PROBS
from src.statistics.common import SIDE_CODES, cluster_mean, fdr_bh, long_outcomes, one_hot

BAND_EDGES = np.array([0., .15, .30, .50, .70, 1.])


def analyze(df, min_matches=100, min_seasons=5):
    long = long_outcomes(df)
    tests = []
    def add(name, family, values, seasons, dates):
        if len(values) < min_matches or pd.Series(seasons).nunique() < min_seasons:
            return
        primary = cluster_mean(values, seasons)
        if primary["clusters"] < min_seasons or primary["n"] < min_matches:
            return
        sensitivity = cluster_mean(values, dates)
        tests.append({"name": name, "group": family, **primary,
                      "date_cluster_se": sensitivity["se"], "date_cluster_p_value": sensitivity["p_value"],
                      "date_ci_low": sensitivity["ci_low"], "date_ci_high": sensitivity["ci_high"],
                      "date_clusters": sensitivity["clusters"]})
    # All outcomes within a fixture and a season share the same cluster.
    for left, right in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        group = long.loc[(long.p >= left) & (long.p < right)]
        add(f"p in [{left:.2f},{right:.2f})", "probability_band", group.residual, group.season, group.date)
    for side in SIDE_CODES:
        group = long.loc[long.side == side]
        add(f"Outcome {side}", "outcome", group.residual, group.season, group.date)
    p = df[PROBS].to_numpy()
    y = one_hot(df.outcome)
    for team in sorted(set(df.home_team) | set(df.away_team)):
        home = df.home_team.eq(team).to_numpy()
        away = df.away_team.eq(team).to_numpy()
        mask = home | away
        residual = np.where(home, y[:, 0]-p[:, 0], y[:, 2]-p[:, 2])
        add(team, "team_win", residual[mask], df.season.to_numpy()[mask], df.date.to_numpy()[mask])
    q = fdr_bh([row["p_value"] for row in tests])
    # BY is a conservative arbitrary-dependence sensitivity, not a second family.
    harmonic = sum(1/i for i in range(1, len(tests)+1))
    for row, adjusted in zip(tests, q):
        row["q_value_bh"] = float(adjusted)
        row["q_value_by"] = float(min(1., adjusted*harmonic))
        row["rejected_bh_05"] = bool(adjusted < .05)
    return {"tests": tests, "n_tests": len(tests), "min_matches": min_matches, "min_seasons": min_seasons,
            "band_edges": BAND_EDGES.tolist(), "n_rejected_bh": sum(t["rejected_bh_05"] for t in tests)}


def figure(result, path):
    from src.statistics.common import plot_setup
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    bands = [t for t in result["tests"] if t["group"] == "probability_band"]
    teams = sorted([t for t in result["tests"] if t["group"] == "team_win"], key=lambda t: t["effect"])
    for ax, rows, title in zip(axes, (bands, teams), ("Favorite–longshot residuals", "Team win residuals")):
        for i, row in enumerate(rows):
            color = "#ad4b35" if row["rejected_bh_05"] else "#27647b"
            ax.plot(100*np.array([row["ci_low"], row["ci_high"]]), [i, i], color=color, linewidth=1.3)
            ax.scatter(100*row["effect"], i, color=color, s=22)
        ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set(yticks=np.arange(len(rows)), yticklabels=[r["name"] for r in rows], title=title,
               xlabel="Observed − implied probability (percentage points)")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"EPL market mispricing · {result['n_tests']} tests in one BH family")
    fig.text(.5, .015, "2015–16 to 2025–26 · season-cluster pointwise 95% CIs · rust indicates BH q < 0.05 · ≥100 rows and ≥5 seasons", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(path)
    plt.close(fig)
