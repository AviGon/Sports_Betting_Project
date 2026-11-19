# Shared dataset audit

Included 4,180 matches from 2015–16 through 2025–26. Partial 2026–27 is excluded.

## Coverage and schema inspection

- 2015-16: 380/380 included; 2015-08-08 to 2016-05-17; 65 source columns; 380 missing kickoff times; 0 exclusions.
- 2016-17: 380/380 included; 2016-08-13 to 2017-05-21; 65 source columns; 380 missing kickoff times; 0 exclusions.
- 2017-18: 380/380 included; 2017-08-11 to 2018-05-13; 65 source columns; 380 missing kickoff times; 0 exclusions.
- 2018-19: 380/380 included; 2018-08-10 to 2019-05-12; 62 source columns; 380 missing kickoff times; 0 exclusions.
- 2019-20: 380/380 included; 2019-08-09 to 2020-07-26; 106 source columns; 0 missing kickoff times; 0 exclusions.
- 2020-21: 380/380 included; 2020-09-12 to 2021-05-23; 106 source columns; 0 missing kickoff times; 0 exclusions.
- 2021-22: 380/380 included; 2021-08-13 to 2022-05-22; 106 source columns; 0 missing kickoff times; 0 exclusions.
- 2022-23: 380/380 included; 2022-08-05 to 2023-05-28; 106 source columns; 0 missing kickoff times; 0 exclusions.
- 2023-24: 380/380 included; 2023-08-11 to 2024-05-19; 106 source columns; 0 missing kickoff times; 0 exclusions.
- 2024-25: 380/380 included; 2024-08-16 to 2025-05-25; 120 source columns; 0 missing kickoff times; 0 exclusions.
- 2025-26: 380/380 included; 2025-08-15 to 2026-05-24; 132 source columns; 0 missing kickoff times; 0 exclusions.

## Decisions

Bet365 non-C 1X2 quotes define the primary market. They are the provider's pre-closing/opening snapshot, not verified first-ever quotes or a uniform lead time. Pre-2019 separate closing quotes are generally unavailable. No fallback bookmaker is substituted for missing Bet365 odds.

Preserve every recognized H/D/A odds triplet under raw_<source header>, including other bookmakers, averages, maxima and closing quotes. Standardized odds_consensus_opening_* uses Avg* or the older BbAv* source; odds_consensus_closing_* uses AvgC*; odds_b365_closing_* uses B365C*. Each group records its source prefix and optional de-vigged probabilities only for complete valid triplets. Invalid optional odds become missing. Avg* is the provider's arithmetic odds average, not an average of de-vigged probabilities. Max* combines bookmakers and is not one bookmaker's executable triplet. Exchange quotes omit commission. Missing optional columns stay missing, never imputed.

Dates use explicit day/month/year formats. The 2019–20 season window extends through August 2020 to retain COVID-postponed July fixtures. Kickoff times are unzoned provider wall-clock strings; absent times remain missing. Date ties use available time and then match ID; unknown intraday order is not inferred. There are no post-match statistics in the processed features, but goals/outcome are target labels and closing quotes must not enter an opening-price strategy.

Team aliases are defined in src/data/prepare.py. Match IDs hash season and canonical home/away teams, remaining stable under fixture rescheduling. Exact duplicate fixtures are removed; conflicting duplicates raise an error. Invalid core fields, missing/invalid Bet365 odds and inconsistent results are excluded with a single first-failure reason and source row in exclusions.csv. All season row counts reconcile to the manifest.

Raw downloads are cached and ignored by Git. dataset_manifest.json records URLs, content SHA256, download provenance where known, full observed schemas, missing counts and exclusions. An offline rebuild uses the cache. Refreshing mutable provider files may change data; compare hashes. Source: https://www.football-data.co.uk/englandm.php and https://www.football-data.co.uk/notes.txt.

Total excluded source rows: 0. See data/processed/exclusions.csv for the complete ledger.
