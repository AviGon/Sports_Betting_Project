"""Prospective power for hypothetical edges using historical home-odds support."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm

EDGES = (0., .01, .02, .03, .05)


def edge_model(odds, edge):
    """Tilt break-even probabilities to achieve an exact mean net ROI.

    Historical odds form an empirical population for iid prospective bets.
    The break-even null is q0=1/o; this differs from the de-vigged market null.
    A common log-odds shift respects probability bounds even for short prices.
    """
    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 1 or len(odds) == 0 or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("Need a nonempty vector of decimal odds > 1")
    if not np.isfinite(edge) or edge < 0 or edge >= (odds-1).mean():
        raise ValueError("Hypothetical edge is not attainable at these odds")
    q0 = 1/odds
    if edge == 0:
        shift, q = 0., q0
    else:
        z = logit(q0)
        shift = brentq(lambda value: (odds*expit(z+value)-1).mean()-edge, -60., 60., xtol=1e-13)
        q = expit(z+shift)
    null_variance = float((odds-1).mean())
    # Includes variance of conditional means when prospective odds are iid draws.
    alternative_variance = float(np.mean(odds**2*q)-(1+edge)**2)
    return q, {"edge": edge, "logit_shift": float(shift), "null_variance": null_variance,
               "alternative_variance": alternative_variance, "achieved_mean_roi": float((odds*q-1).mean())}


def detection_power(n, edge, null_variance, alternative_variance, alpha=.05):
    if np.any(np.asarray(n) <= 0) or null_variance <= 0 or alternative_variance <= 0 or not 0 < alpha < 1:
        raise ValueError("Positive sample sizes/variances and valid alpha required")
    threshold = norm.ppf(1-alpha)*np.sqrt(null_variance)
    return norm.sf((threshold-edge*np.sqrt(n))/np.sqrt(alternative_variance))


def required_sample_size(edge, null_variance, alternative_variance, target=.8, alpha=.05):
    if edge <= 0:
        return None
    if null_variance <= 0 or alternative_variance <= 0 or not 0 < alpha < target < 1:
        raise ValueError("Invalid power design")
    return int(np.ceil(((norm.ppf(1-alpha)*np.sqrt(null_variance)+norm.ppf(target)*np.sqrt(alternative_variance))/edge)**2))


def monte_carlo_power(odds, edge, n_bets, n_paths=4000, seed=227, alpha=.05):
    if n_bets < 1 or n_paths < 2000:
        raise ValueError("Need positive sample size and >=2,000 paths")
    odds = np.asarray(odds, dtype=float)
    _, model = edge_model(odds, edge)
    prices, frequencies = np.unique(odds, return_counts=True)
    z = logit(1/prices)
    q = expit(z+model["logit_shift"])
    rng = np.random.default_rng(seed)
    critical = norm.ppf(1-alpha)*np.sqrt(model["null_variance"]/n_bets)
    rejects = 0
    for start in range(0, n_paths, 256):
        size = min(256, n_paths-start)
        # Exact iid empirical-price mixture via multinomial price counts.
        # Each prospective observation is one selected bet on its own match.
        counts = rng.multinomial(n_bets, frequencies/frequencies.sum(), size=size)
        wins = rng.binomial(counts, q[None, :])
        roi = (wins @ prices-n_bets)/n_bets
        rejects += int((roi > critical).sum())
    estimate = rejects/n_paths
    # Wilson interval measures simulation precision, not historical uncertainty.
    z95 = norm.ppf(.975)
    center = (estimate+z95**2/(2*n_paths))/(1+z95**2/n_paths)
    radius = z95*np.sqrt(estimate*(1-estimate)/n_paths+z95**2/(4*n_paths**2))/(1+z95**2/n_paths)
    return {"n_bets": int(n_bets), "n_paths": n_paths, "seed": seed, "power": float(estimate),
            "mc_se": float(np.sqrt(estimate*(1-estimate)/n_paths)), "wilson_ci_95": [float(center-radius), float(center+radius)]}


def analyze(matches, n_paths=4000, seed=227):
    odds = matches.odds_home.to_numpy(float)
    n_grid = np.unique(np.round(np.geomspace(100, 200000, 70)).astype(int))
    mc_grid = [500, 2000, 5000, 10000, 25000, 50000, 100000]
    rows = []
    for j, edge in enumerate(EDGES):
        _, model = edge_model(odds, edge)
        n80 = required_sample_size(edge, model["null_variance"], model["alternative_variance"])
        curve = detection_power(n_grid, edge, model["null_variance"], model["alternative_variance"])
        validation = [monte_carlo_power(odds, edge, n, n_paths, seed+j*100+i) for i, n in enumerate(mc_grid)]
        at_target = monte_carlo_power(odds, edge, n80, n_paths, seed+1000+j) if n80 else None
        rows.append({**model, "required_n_80": n80, "n_grid": n_grid.tolist(), "clt_power": curve.tolist(),
                     "mc_validation": validation, "mc_at_required_n": at_target})
    observed_returns = odds*(matches.outcome.to_numpy() == "H")-1
    market_p = matches.p_market_home.to_numpy(float)
    return {"alpha": .05, "target_power": .8, "n_paths": n_paths, "seed": seed, "source_n": len(odds),
            "historical_return_variance": float(observed_returns.var(ddof=1)),
            "market_conditional_return_variance_mean": float(np.mean(odds**2*market_p*(1-market_p))),
            "rows": rows}


def figure(result, path):
    from src.statistics.common import plot_setup
    plt = plot_setup()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["0.5", "#a3b9c2", "#6a98a8", "#27647b", "#ad4b35"]
    for row, color in zip(result["rows"], colors):
        ax.plot(row["n_grid"], row["clt_power"], color=color, label=f"{row['edge']:.0%} true ROI")
        mc = row["mc_validation"]
        ax.scatter([x["n_bets"] for x in mc], [x["power"] for x in mc], color=color, s=14)
    ax.axhline(.8, color="0.5", linestyle=":", label="80% target")
    ax.set(xscale="log", ylim=(0, 1.02), title="Power to detect a hypothetical positive betting edge",
           xlabel="Independent unit bets (log scale)", ylabel="Detection probability")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.text(.5, .015, "One-sided α=5% · curves: CLT · dots: Monte Carlo · historical home-odds distribution, EPL 2015–16 to 2025–26", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
