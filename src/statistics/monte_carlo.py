"""Fixed-selection market-null simulations; one categorical outcome per match."""
import numpy as np
import pandas as pd

from src.data.prepare import PROBS
from src.statistics.common import SIDE_CODES, one_hot, probabilities

BET_FIELDS = ["match_id", "selected_side", "odds", "estimated_ev", "realized_return"]


def settle(odds, selected_side, outcome):
    odds = np.asarray(odds, dtype=float)
    selected = np.asarray(selected_side)
    actual = np.asarray(outcome)
    if not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("Finite decimal odds > 1 required")
    if not np.isin(selected, SIDE_CODES).all() or not np.isin(actual, SIDE_CODES).all():
        raise ValueError("Sides/outcomes must be H/D/A")
    if odds.shape != selected.shape or selected.shape != actual.shape:
        raise ValueError("Settlement arrays must have the same shape")
    return odds*(selected == actual)-1


def home_baseline(matches):
    return pd.DataFrame({"match_id": matches.match_id, "selected_side": "H", "odds": matches.odds_home,
                         "estimated_ev": np.nan,
                         "realized_return": settle(matches.odds_home, np.full(len(matches), "H"), matches.outcome)})


def validate_bets(matches, bets):
    if missing := set(BET_FIELDS)-set(bets.columns):
        raise ValueError(f"Missing bet fields: {sorted(missing)}")
    if not len(bets):
        raise ValueError("Empty bet list")
    if not matches.match_id.is_unique or matches.match_id.isna().any():
        raise ValueError("Match IDs must be nonmissing and unique")
    if bets.match_id.isna().any() or not bets.match_id.isin(matches.match_id).all():
        raise ValueError("Missing/unknown match IDs in bets")
    if bets.duplicated(["match_id", "selected_side"]).any():
        raise ValueError("Duplicate match_id/selected_side bets; aggregate deliberate repeats before import")
    if not bets.selected_side.isin(SIDE_CODES).all():
        raise ValueError("Selected side must be H/D/A")
    odds = pd.to_numeric(bets.odds, errors="coerce").to_numpy(float)
    returns = pd.to_numeric(bets.realized_return, errors="coerce").to_numpy(float)
    ev = pd.to_numeric(bets.estimated_ev, errors="coerce").to_numpy(float)
    invalid_ev = bets.estimated_ev.notna().to_numpy() & ~np.isfinite(ev)
    if invalid_ev.any():
        raise ValueError("estimated_ev must be finite or missing")
    if not np.isfinite(odds).all() or (odds <= 1).any() or not np.isfinite(returns).all():
        raise ValueError("Finite odds >1 and realized returns required")
    # Take columns explicitly to prevent imported metadata from replacing labels.
    joined = bets[BET_FIELDS].copy().merge(matches[["match_id", "date", "season", "outcome"]+PROBS],
                                         on="match_id", how="left", validate="many_to_one", sort=False)
    joined["odds"] = odds
    joined["realized_return"] = returns
    probabilities(joined[PROBS])
    actual = settle(odds, joined.selected_side, joined.outcome)
    if not np.allclose(actual, returns, atol=1e-8, rtol=1e-8):
        raise ValueError("realized_return disagrees with one-unit net settlement")
    return joined


def simulation_inputs(matches, bets):
    joined = validate_bets(matches, bets)
    # Preserve original chronological match order, independent of CSV bet order.
    selected = matches.loc[matches.match_id.isin(joined.match_id)].reset_index(drop=True)
    index = pd.Index(selected.match_id).get_indexer(joined.match_id)
    side_index = pd.Index(SIDE_CODES).get_indexer(joined.selected_side)
    counts = np.bincount(index, minlength=len(selected))
    payoffs = -np.repeat(counts[:, None], 3, axis=1).astype(float)
    np.add.at(payoffs, (index, side_index), joined.odds.to_numpy(float))
    return selected, probabilities(selected[PROBS]), payoffs, joined


def simulate(matches, bets, n_paths=10000, seed=225, batch_size=256):
    if n_paths < 2000 or batch_size < 1:
        raise ValueError("Use at least 2,000 paths and positive batch size")
    selected, p, payoff, joined = simulation_inputs(matches, bets)
    n_bets = len(joined)
    expectation = (p*payoff).sum(axis=1)
    expected_roi = expectation.sum()/n_bets
    variance_roi = ((p*payoff**2).sum(axis=1)-expectation**2).sum()/n_bets**2
    rng = np.random.default_rng(seed)
    cumulative = p.cumsum(axis=1)
    paths = np.empty(n_paths)
    for start in range(0, n_paths, batch_size):
        size = min(batch_size, n_paths-start)
        u = rng.random((size, len(selected)))
        # One uniform -> one categorical outcome for each fixture in each path.
        outcome = (u >= cumulative[:, 0]).astype(int)+(u >= cumulative[:, 1]).astype(int)
        paths[start:start+size] = payoff[np.arange(len(selected))[None, :], outcome].sum(axis=1)/n_bets
    observed = joined.realized_return.mean()
    equal = np.isclose(paths, observed, atol=1e-14, rtol=0).sum()
    # Exact inequalities define tail probabilities; midrank only for percentile.
    percentile = 100*((paths < observed-1e-14).sum()+.5*equal)/n_paths
    upper_p = (1+(paths >= observed).sum())/(n_paths+1)
    result = {"n_bets": n_bets, "n_matches": len(selected), "n_paths": n_paths, "seed": seed,
              "expected_roi": float(expected_roi), "observed_roi": float(observed),
              "simulation_mean_roi": float(paths.mean()), "simulation_sd_roi": float(paths.std(ddof=1)),
              "theoretical_sd_roi": float(np.sqrt(max(0., variance_roi))),
              "mc_se_of_mean": float(paths.std(ddof=1)/np.sqrt(n_paths)),
              "simulation_interval_95": np.quantile(paths, [.025, .975]).tolist(),
              "observed_percentile_midrank": float(percentile), "upper_tail_p_value": float(upper_p),
              "upper_tail_mc_se": float(np.sqrt(upper_p*(1-upper_p)/n_paths))}
    return result, paths


def figure(result, paths, path):
    from src.statistics.common import plot_setup
    plt = plot_setup()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(paths*100, bins=50, density=True, color="#27647b", alpha=.75)
    ax.axvspan(*np.array(result["simulation_interval_95"])*100, color="0.5", alpha=.10, label="Central 95% simulation interval")
    ax.axvline(result["expected_roi"]*100, color="0.2", linestyle="--", label="Market-null expected ROI")
    ax.axvline(result["observed_roi"]*100, color="#ad4b35", linewidth=2, label="Observed ROI")
    ax.set(title="Fixed selections under simulated market outcomes", xlabel="Net return / total unit stakes (%)",
           ylabel="Probability density (per percentage point)")
    ax.legend(fontsize=8)
    fig.text(.5, .02, f"EPL 2015–16 to 2025–26 · {result['n_bets']:,} unit bets · {result['n_paths']:,} paths · seed {result['seed']} · independent matches", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
