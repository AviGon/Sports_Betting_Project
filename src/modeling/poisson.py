"""Experiment A1: independent Poisson goals model fitted walk-forward.

Model (Maher 1982; Dixon & Coles 1997 time weighting):

    home goals  X ~ Poisson(lambda),  log lambda = mu + h + att[home] - def[away]
    away goals  Y ~ Poisson(nu),      log nu     = mu       + att[away] - def[home]

X and Y are conditionally independent given the team strengths, so

    P(X = x, Y = y) = Pois(x; lambda) * Pois(y; nu)
    P(home win) = sum_{x > y} P(x, y),  P(draw) = sum_{x = y},  P(away) = sum_{x < y}.

Parameters are fitted by maximising a time-weighted, ridge-penalised Poisson
log-likelihood using only matches played strictly before the prediction date.
Match weights are exp(-xi * age_in_days) inside a finite look-back window.

Newly promoted teams have little or no recent top-flight data. Instead of
shrinking them toward an average Premier League team, their penalty is centred
on the mean strength of the three weakest established teams of the previous
fit, a pre-match proxy for "a team at the bottom of the league".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass(frozen=True)
class PoissonConfig:
    xi: float = 0.002            # time decay per day (half-life = ln 2 / xi)
    ridge: float = 2.0           # L2 penalty on attack/defence deviations from their prior centre
    window_days: int = 3 * 365   # look-back window
    promoted_prior: bool = True  # centre promoted teams on bottom-3 strength


@dataclass
class PoissonFit:
    teams: list[str]
    mu: float
    home_adv: float
    attack: np.ndarray
    defence: np.ndarray
    converged: bool = True
    index: dict = field(init=False)

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.teams)}

    def rates(self, home: str, away: str) -> tuple[float, float]:
        i, j = self.index[home], self.index[away]
        lam = np.exp(self.mu + self.home_adv + self.attack[i] - self.defence[j])
        nu = np.exp(self.mu + self.attack[j] - self.defence[i])
        return float(lam), float(nu)

    def strength(self) -> pd.Series:
        return pd.Series(self.attack + self.defence, index=self.teams)


def outcome_probs(lam: float, nu: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(H), P(D), P(A) from independent Poisson goal rates (renormalised over 0..max_goals)."""
    g = np.arange(max_goals + 1)
    joint = np.outer(poisson.pmf(g, lam), poisson.pmf(g, nu))
    p = np.array([np.tril(joint, -1).sum(), np.trace(joint), np.triu(joint, 1).sum()])
    return p / p.sum()


def fit_poisson(home_idx, away_idx, home_goals, away_goals, weights, n_teams,
                ridge, centre_att=None, centre_def=None, x0=None):
    """Weighted, penalised MLE. Returns (mu, h, att, def, converged)."""
    hi = np.asarray(home_idx, int)
    ai = np.asarray(away_idx, int)
    xg = np.asarray(home_goals, float)
    yg = np.asarray(away_goals, float)
    w = np.asarray(weights, float)
    ca = np.zeros(n_teams) if centre_att is None else np.asarray(centre_att, float)
    cd = np.zeros(n_teams) if centre_def is None else np.asarray(centre_def, float)

    def objective(theta):
        mu, h = theta[0], theta[1]
        att, dfn = theta[2:2 + n_teams], theta[2 + n_teams:]
        eta_h = mu + h + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lam, nu = np.exp(eta_h), np.exp(eta_a)
        nll = np.sum(w * (lam - xg * eta_h)) + np.sum(w * (nu - yg * eta_a))
        da, dd = att - ca, dfn - cd
        nll += ridge * (da @ da + dd @ dd)
        rh, ra = w * (lam - xg), w * (nu - yg)  # d nll / d eta
        g_att = np.bincount(hi, rh, n_teams) + np.bincount(ai, ra, n_teams) + 2 * ridge * da
        g_def = -np.bincount(ai, rh, n_teams) - np.bincount(hi, ra, n_teams) + 2 * ridge * dd
        grad = np.concatenate([[rh.sum() + ra.sum(), rh.sum()], g_att, g_def])
        return nll, grad

    if x0 is None:
        mean_goals = max(np.average((xg + yg) / 2, weights=w), 1e-3)
        x0 = np.concatenate([[np.log(mean_goals), 0.2], ca, cd])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    th = res.x
    return th[0], th[1], th[2:2 + n_teams], th[2 + n_teams:], bool(res.success)


def fit_at(history: pd.DataFrame, as_of: pd.Timestamp, teams_needed, cfg: PoissonConfig,
           promoted=(), previous: PoissonFit | None = None) -> PoissonFit:
    """Fit on matches strictly before ``as_of`` within the look-back window."""
    hist = history[(history["date"] < as_of) &
                   (history["date"] >= as_of - pd.Timedelta(days=cfg.window_days))]
    teams = sorted(set(hist["home_team"]) | set(hist["away_team"]) | set(teams_needed))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ca, cd = np.zeros(n), np.zeros(n)
    if cfg.promoted_prior and previous is not None and promoted:
        established = [t for t in previous.teams if t not in promoted]
        strength = previous.strength().loc[established].sort_values()
        weakest = [previous.index[t] for t in strength.index[:3]]
        for t in promoted:
            if t in idx:
                ca[idx[t]] = previous.attack[weakest].mean()
                cd[idx[t]] = previous.defence[weakest].mean()
    x0 = None
    if previous is not None:
        att0 = np.array([previous.attack[previous.index[t]] if t in previous.index else ca[i]
                         for i, t in enumerate(teams)])
        def0 = np.array([previous.defence[previous.index[t]] if t in previous.index else cd[i]
                         for i, t in enumerate(teams)])
        x0 = np.concatenate([[previous.mu, previous.home_adv], att0, def0])
    if len(hist) == 0:
        return PoissonFit(teams, np.log(1.4), 0.25, ca, cd, False)
    age = (as_of - hist["date"]).dt.days.to_numpy(float)
    w = np.exp(-cfg.xi * age)
    mu, h, att, dfn, ok = fit_poisson(hist["home_team"].map(idx), hist["away_team"].map(idx),
                                      hist["home_goals"], hist["away_goals"], w, n,
                                      cfg.ridge, ca, cd, x0)
    return PoissonFit(teams, mu, h, att, dfn, ok)


def promoted_teams(matches: pd.DataFrame) -> dict[str, set[str]]:
    """Teams in each season that were not in the previous season (known pre-season from the fixture list)."""
    seasons = sorted(matches["season"].unique())
    members = {s: set(matches.loc[matches.season == s, "home_team"]) for s in seasons}
    out = {seasons[0]: set()}
    for prev, cur in zip(seasons, seasons[1:]):
        out[cur] = members[cur] - members[prev]
    return out


def walk_forward(matches: pd.DataFrame, cfg: PoissonConfig = PoissonConfig(),
                 predict_seasons=None) -> pd.DataFrame:
    """Predict every match in ``predict_seasons`` using a model refitted before each match date.

    All fixtures on the same calendar date share one fit that uses only earlier
    dates, so no same-day result can leak into a prediction.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    promoted = promoted_teams(matches)
    target = matches if predict_seasons is None else matches[matches["season"].isin(predict_seasons)]
    rows, previous = [], None
    for date, day in target.groupby("date", sort=True):
        season = day["season"].iloc[0]
        needed = set(day["home_team"]) | set(day["away_team"])
        fit = fit_at(matches, date, needed, cfg, promoted.get(season, set()), previous)
        previous = fit
        for r in day.itertuples():
            lam, nu = fit.rates(r.home_team, r.away_team)
            p = outcome_probs(lam, nu)
            rows.append((r.match_id, lam, nu, *p, fit.converged))
    return pd.DataFrame(rows, columns=["match_id", "lambda_home", "lambda_away",
                                       "p_poisson_home", "p_poisson_draw", "p_poisson_away",
                                       "fit_converged"])
