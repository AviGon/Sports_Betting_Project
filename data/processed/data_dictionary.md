# Shared EPL dataset contract

One row per completed EPL match, 2015–16 through 2025–26. Read with pandas; `date` is ISO YYYY-MM-DD. Outcome order throughout the code is H, D, A.

- `match_id`: stable `epl_` plus first 20 SHA256 hexadecimal characters of season|canonical home team|canonical away team. Join key for selections and predictions.
- `date`: match date; `kickoff_time`: optional provider wall-clock HH:MM, timezone unspecified. Missing times stay missing.
- `season`: YYYY-YY season label.
- `home_team`, `away_team`: canonical team names, alias mapping in preparation code.
- `home_goals`, `away_goals`: integer full-time goals. Targets, unavailable before the match.
- `outcome`: H/D/A, target label consistent with full-time goals.
- `odds_home`, `odds_draw`, `odds_away`: raw Bet365 pre-closing decimal odds (source B365H/D/A); >1. Use these for settlement, never reciprocal normalized probabilities.
- `p_market_home`, `p_market_draw`, `p_market_away`: reciprocal Bet365 odds divided by their row sum. Positive and sum to 1; these are a market null model, not known true probabilities.
- `raw_<header>`: optional numeric decimal 1X2 quotes under original source names. Suffix H/D/A identifies outcome; an immediately preceding C marks closing (e.g. B365CH). Avg, Max and older BbAv/BbMx identify provider aggregates. PS is Pinnacle; other prefixes retain the provider's naming. Check dataset_manifest.json for actual per-season schemas. No missing quotes are imputed. Exchange values do not account for commission.

Do not pass goals, outcome or future closing quotes into predictive features. Opening snapshots are not guaranteed available at a common lead time; same-day ordering is not guaranteed when time is absent. Person A owns chronological feature construction, model fitting and backtests.

Person A should read `data/processed/epl_matches.csv`, sort chronologically and join on `match_id`. For downstream return inference, export `match_id, selected_side, odds, estimated_ev, realized_return`: selected_side is H/D/A; odds are finite decimal >1; realized_return is net return per unit stake (+odds−1 on a win, −1 on a loss); estimated_ev may be absent/NaN for a baseline. One unit per row. Reject duplicate (match_id, selected_side) rows to prevent accidental double counting; different sides of one match are allowed and simulated jointly.
