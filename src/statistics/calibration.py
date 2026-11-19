"""Proper multiclass scores and retrospective one-vs-rest calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import t

from src.data.prepare import PROBS
from src.statistics.common import SIDE_CODES, cluster_mean, one_hot, probabilities


def brier_score(p, outcomes):
    """Mean sum of squared class errors; unscaled multiclass range [0,2]."""
    p = probabilities(p)
    y = one_hot(outcomes)
    if p.shape != y.shape:
        raise ValueError("Probability/outcome length mismatch")
    return float(np.square(p-y).sum(axis=1).mean())


def log_loss(p, outcomes):
    p = probabilities(p)
    y = one_hot(outcomes)
    if p.shape != y.shape:
        raise ValueError("Probability/outcome length mismatch")
    return float(-(y*np.log(p)).sum(axis=1).mean())


def logistic_calibration(p, y, clusters):
    """logit P(Y=1) = intercept + slope logit(p), with season CR1 CIs."""
    x = np.column_stack([np.ones(len(p)), logit(p)])
    y = np.asarray(y, dtype=float)
    if len(p) < 30 or y.sum() < 10 or (1-y).sum() < 10 or np.std(x[:, 1]) < 1e-8:
        return {"available": False, "reason": "insufficient variation/events"}
    def loss(beta):
        z = x @ beta
        return np.sum(np.logaddexp(0, z)-y*z)
    def grad(beta):
        return x.T @ (expit(x @ beta)-y)
    result = minimize(loss, [0., 1.], jac=grad, method="BFGS", options={"gtol": 1e-6})
    if not result.success and np.linalg.norm(grad(result.x)) > 1e-4:
        raise RuntimeError(f"Calibration optimizer failed: {result.message}")
    fitted = expit(x @ result.x)
    bread = np.linalg.inv(x.T @ ((fitted*(1-fitted))[:, None]*x))
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    g = codes.max()+1
    if g < 2:
        return {"available": False, "reason": "fewer than two clusters"}
    scores = np.zeros((g, 2))
    np.add.at(scores, codes, x*(y-fitted)[:, None])
    cov = g/(g-1) * bread @ (scores.T @ scores) @ bread
    se = np.sqrt(np.diag(cov))
    radius = t.ppf(.975, g-1)*se
    return {"available": True, "intercept": float(result.x[0]), "slope": float(result.x[1]),
            "intercept_ci": (result.x[0]+np.array([-1, 1])*radius[0]).tolist(),
            "slope_ci": (result.x[1]+np.array([-1, 1])*radius[1]).tolist(), "clusters": int(g)}


def analyze(df, bins=10):
    p = probabilities(df[PROBS])
    y = one_hot(df.outcome)
    result = {"n_matches": len(df), "brier_score": brier_score(p, df.outcome),
              "log_loss": log_loss(p, df.outcome), "bin_edges": np.linspace(0, 1, bins+1).tolist(), "sides": {}}
    edges = np.linspace(0, 1, bins+1)
    for k, side in enumerate(SIDE_CODES):
        rows = []
        codes = np.minimum(np.searchsorted(edges, p[:, k], side="right")-1, bins-1)
        for b in range(bins):
            mask = codes == b
            if not mask.any():
                continue
            stats = cluster_mean((y-p)[mask, k], df.season.to_numpy()[mask])
            pm = float(p[mask, k].mean())
            rows.append({"bin": b, "lower": float(edges[b]), "upper": float(edges[b+1]),
                         "mean_probability": pm, "observed_frequency": float(y[mask, k].mean()),
                         "observed_count": int(y[mask, k].sum()), "expected_count": float(p[mask, k].sum()),
                         "frequency_ci_low": max(0., pm+stats["ci_low"]) if stats["ci_low"] is not None else None,
                         "frequency_ci_high": min(1., pm+stats["ci_high"]) if stats["ci_high"] is not None else None,
                         **stats})
        result["sides"][str(side)] = {"observed_count": int(y[:, k].sum()), "expected_count": float(p[:, k].sum()),
                                         "observed_frequency": float(y[:, k].mean()), "expected_frequency": float(p[:, k].mean()),
                                         "calibration": logistic_calibration(p[:, k], y[:, k], df.season), "bins": rows}
    return result


def figure(result, path):
    from src.statistics.common import plot_setup
    plt = plot_setup()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
    for ax, (side, values) in zip(axes, result["sides"].items()):
        rows = values["bins"]
        ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1, label="Perfect calibration")
        for row in rows:
            if row["frequency_ci_low"] is not None:
                # The model-based residual CI may not contain its estimate after clipping only in pathological bins.
                ax.plot([row["mean_probability"]]*2, [row["frequency_ci_low"], row["frequency_ci_high"]], color="#27647b", alpha=.55)
        ax.scatter([r["mean_probability"] for r in rows], [r["observed_frequency"] for r in rows],
                   s=[12+np.sqrt(r["n"]) for r in rows], color="#27647b", zorder=3, label="Observed (95% CI)")
        ax.set(title={"H": "Home win", "D": "Draw", "A": "Away win"}[side], xlabel="Mean market probability", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Observed outcome frequency")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Bet365 pre-closing calibration · {result['n_matches']:,} EPL matches")
    fig.text(.5, .015, "2015–16 to 2025–26 · equal-width bins · season-cluster Student 95% intervals · marker size scales with count", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(path)
    plt.close(fig)
