"""Download schema-varying football-data CSVs and build the shared contract."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

SIDES = ("home", "draw", "away")
ODDS = [f"odds_{s}" for s in SIDES]
PROBS = [f"p_market_{s}" for s in SIDES]
ALIASES = {"Man United": "Manchester United", "Man City": "Manchester City",
           "Wolves": "Wolverhampton Wanderers", "Nott'm Forest": "Nottingham Forest",
           "Sheffield United": "Sheffield United", "Tottenham": "Tottenham Hotspur"}


def devig(odds):
    """Multiplicative de-vigging; probabilities use odds only, never results."""
    a = np.asarray(odds, dtype=float)
    if a.ndim < 1 or a.shape[-1] != 3 or not np.all(np.isfinite(a) & (a > 1)):
        raise ValueError("Expected finite decimal 1X2 odds > 1")
    implied = 1 / a
    return implied / implied.sum(axis=-1, keepdims=True)


def parse_dates(values):
    # Explicit formats avoid pandas' mixed-format inference across source eras.
    def parse(value):
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return pd.Timestamp(datetime.strptime(str(value), fmt))
            except ValueError:
                pass
        return pd.NaT
    return values.map(parse)


def clean_season(raw, season):
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"}
    if missing := required - set(raw.columns):
        raise ValueError(f"{season}: missing required source columns {sorted(missing)}")
    raw = raw.dropna(how="all").reset_index(drop=True)
    out = pd.DataFrame(index=raw.index)
    out["season"] = season
    out["date"] = parse_dates(raw.Date)
    out["kickoff_time"] = raw.get("Time", pd.Series(pd.NA, index=raw.index)).astype("string")
    for source, target in (("HomeTeam", "home_team"), ("AwayTeam", "away_team")):
        out[target] = raw[source].astype("string").str.strip().replace(ALIASES)
    for source, target in (("FTHG", "home_goals"), ("FTAG", "away_goals")):
        out[target] = pd.to_numeric(raw[source], errors="coerce")
    out["outcome"] = raw.FTR.astype("string").str.strip()
    for source, target in zip(("B365H", "B365D", "B365A"), ODDS):
        out[target] = pd.to_numeric(raw[source], errors="coerce")
    # Recognize actual complete triplets, excluding handicap/goal-market columns.
    # Retain source header spelling so bookmaker/timing identity is unambiguous.
    kept = []
    for col in raw.columns:
        if re.fullmatch(r"[A-Za-z0-9]+H", col) and col[:-1] + "D" in raw and col[:-1] + "A" in raw:
            prefix = col[:-1]
            for side in "HDA":
                name = prefix + side
                a = pd.to_numeric(raw[name], errors="coerce")
                out[f"raw_{name}"] = a.where(np.isfinite(a) & (a > 1))
                kept.append(name)
    reasons = pd.Series("", index=out.index)
    def reject(mask, reason):
        nonlocal reasons
        reasons.loc[mask & reasons.eq("")] = reason
    reject(out.date.isna(), "invalid_date")
    start = int(season[:4])
    # COVID postponed the end of 2019–20 until July 2020.
    end = "2020-08-31" if season == "2019-20" else f"{start+1}-06-30"
    reject(out.date.notna() & ~out.date.between(f"{start}-07-01", end), "outside_season")
    reject(out.home_team.isna() | out.away_team.isna() | out.home_team.eq("") | out.away_team.eq("") | out.home_team.eq(out.away_team), "invalid_teams")
    goals = out[["home_goals", "away_goals"]].to_numpy(float)
    reject(pd.Series((~np.isfinite(goals) | (goals < 0) | (goals != np.floor(goals))).any(axis=1)), "invalid_or_missing_goals")
    reject(~out.outcome.isin(list("HDA")), "invalid_or_missing_outcome")
    expected = np.where(out.home_goals > out.away_goals, "H", np.where(out.home_goals < out.away_goals, "A", "D"))
    reject(out.outcome.ne(expected).fillna(True), "outcome_goal_mismatch")
    a = out[ODDS].to_numpy(float)
    reject(pd.Series((~np.isfinite(a) | (a <= 1)).any(axis=1)), "invalid_or_missing_b365_odds")
    excluded = [{"season": season, "source_row": int(i)+2, "reason": r} for i, r in reasons.items() if r]
    out = out.loc[reasons.eq("")].copy()
    key = ["season", "home_team", "away_team"]
    duplicates = out.duplicated(key, keep=False)
    if duplicates.any():
        for _, group in out.loc[duplicates].groupby(key):
            if len(group.drop_duplicates()) > 1:
                raise ValueError(f"Conflicting duplicate fixture: {group[key].iloc[0].tolist()}")
        for i in out.index[out.duplicated(key)]:
            excluded.append({"season": season, "source_row": int(i)+2, "reason": "exact_duplicate_fixture"})
        out = out.drop_duplicates(key)
    out["match_id"] = out.apply(lambda r: "epl_" + hashlib.sha256(f"{r.season}|{r.home_team}|{r.away_team}".encode()).hexdigest()[:20], axis=1)
    out[PROBS] = devig(out[ODDS])
    out[["home_goals", "away_goals"]] = out[["home_goals", "away_goals"]].astype(int)
    return out, excluded, sorted(kept)


def validate(df):
    assert len(df) > 0, "Empty dataset"
    assert df.match_id.is_unique
    assert not df.duplicated(["season", "home_team", "away_team"]).any()
    assert df.outcome.isin(list("HDA")).all()
    a = df[ODDS].to_numpy(float)
    assert (np.isfinite(a) & (a > 1)).all()
    p = df[PROBS].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p, devig(a))
    assert pd.to_datetime(df.date).is_monotonic_increasing
    g = df[["home_goals", "away_goals"]].to_numpy(float)
    assert (np.isfinite(g) & (g >= 0) & (g == np.floor(g))).all()
    expected = np.where(g[:, 0] > g[:, 1], "H", np.where(g[:, 0] < g[:, 1], "A", "D"))
    assert (df.outcome == expected).all()


def build(root=Path("."), download=False):
    raw_dir = root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames, audits, exclusions = [], [], []
    for year in range(2015, 2026):
        code = f"{year%100:02d}{(year+1)%100:02d}"
        season = f"{year}-{str(year+1)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        path = raw_dir / f"{code}_E0.csv"
        meta_path = path.with_suffix(".json")
        if download or not path.exists():
            with urlopen(url, timeout=60) as response:
                content = response.read()
            # Validate CSV before replacing a valid cached file.
            if b"FTHG" not in content[:4096]:
                raise ValueError(f"Not a match CSV: {url}")
            path.write_bytes(content)
            meta_path.write_text(json.dumps({"downloaded_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
        raw = pd.read_csv(path, encoding="utf-8-sig")
        clean, excluded, kept = clean_season(raw, season)
        # Fail closed: default dataset consists of completed 20-team seasons only.
        fixtures = raw.dropna(subset=["HomeTeam", "AwayTeam"]).drop_duplicates(["HomeTeam", "AwayTeam"])
        if len(fixtures) != 380 or raw.FTR.isin(list("HDA")).sum() < 380:
            raise ValueError(f"{season} is not complete: {len(fixtures)} fixtures; do not silently label as completed")
        frames.append(clean)
        exclusions.extend(excluded)
        audits.append({"season": season, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "retrieval": json.loads(meta_path.read_text()) if meta_path.exists() else {"downloaded_at_utc": "unknown (pre-existing cache)"},
                       "source_rows": len(raw.dropna(how="all")), "included_rows": len(clean),
                       "excluded_rows": len(excluded), "date_min": str(clean.date.min().date()), "date_max": str(clean.date.max().date()),
                       "source_columns": list(raw.columns), "retained_odds_columns": kept,
                       "missing_by_source_column": {k: int(v) for k, v in raw.isna().sum().items() if v},
                       "missing_kickoff": int(clean.kickoff_time.isna().sum())})
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "kickoff_time", "match_id"], na_position="first").reset_index(drop=True)
    front = ["match_id", "date", "home_team", "away_team", "home_goals", "away_goals", "outcome"] + ODDS + PROBS + ["season", "kickoff_time"]
    df = df[front + sorted(set(df.columns) - set(front))]
    validate(df)
    dest = root / "data/processed"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "epl_matches.csv", index=False, date_format="%Y-%m-%d", float_format="%.12g")
    audit = {"seasons": audits, "exclusions": exclusions, "rows": len(df), "partial_2026_27_included": False}
    (dest / "dataset_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame(exclusions, columns=["season", "source_row", "reason"]).to_csv(dest / "exclusions.csv", index=False)
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    lines = ["# Shared dataset audit", "", f"Included {len(df):,} matches from 2015–16 through 2025–26. Partial 2026–27 is excluded.",
             "", "## Coverage and schema inspection", ""]
    for s in audits:
        lines.append(f"- {s['season']}: {s['included_rows']}/380 included; {s['date_min']} to {s['date_max']}; {len(s['source_columns'])} source columns; {s['missing_kickoff']} missing kickoff times; {s['excluded_rows']} exclusions.")
    lines += ["", "## Decisions", "", "Bet365 non-C 1X2 quotes define the primary market. They are the provider's pre-closing/opening snapshot, not verified first-ever quotes or a uniform lead time. Pre-2019 separate closing quotes are generally unavailable. No fallback bookmaker is substituted for missing Bet365 odds.",
              "", "Preserve every recognized H/D/A odds triplet under raw_<source header>, including other bookmakers, averages, maxima and closing quotes. Invalid optional odds become missing. Avg* is the provider's arithmetic odds average, not an average of de-vigged probabilities. Max* combines bookmakers and is not one bookmaker's executable triplet. Exchange quotes omit commission. Missing optional columns stay missing, never imputed.",
              "", "Dates use explicit day/month/year formats. The 2019–20 season window extends through August 2020 to retain COVID-postponed July fixtures. Kickoff times are unzoned provider wall-clock strings; absent times remain missing. Date ties use available time and then match ID; unknown intraday order is not inferred. There are no post-match statistics in the processed features, but goals/outcome are target labels and closing quotes must not enter an opening-price strategy.",
              "", "Team aliases are defined in src/data/prepare.py. Match IDs hash season and canonical home/away teams, remaining stable under fixture rescheduling. Exact duplicate fixtures are removed; conflicting duplicates raise an error. Invalid core fields, missing/invalid Bet365 odds and inconsistent results are excluded with a single first-failure reason and source row in exclusions.csv. All season row counts reconcile to the manifest.",
              "", "Raw downloads are cached and ignored by Git. dataset_manifest.json records URLs, content SHA256, download provenance where known, full observed schemas, missing counts and exclusions. An offline rebuild uses the cache. Refreshing mutable provider files may change data; compare hashes. Source: https://www.football-data.co.uk/englandm.php and https://www.football-data.co.uk/notes.txt.",
              "", f"Total excluded source rows: {len(exclusions)}. See data/processed/exclusions.csv for the complete ledger."]
    (docs / "dataset_audit.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(df)} matches; excluded {len(exclusions)} rows")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Refresh all source CSVs (otherwise use cached files)")
    args = parser.parse_args()
    build(download=args.download)
