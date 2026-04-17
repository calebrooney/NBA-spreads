---
name: nba-spread-predictions
overview: Train a spread (home_margin) model from `nba.game_logs`, then score every live odds snapshot in `nba.odds` to compute edge vs the current spread and simulate/track intraday betting performance.
todos:
  - id: inspect-current-ml-stubs
    content: Review `nba_spreads/ML/model.py` and `nba_spreads/ML/pipeline.py` stubs (currently untracked) and align structure with existing ingestion + benchmark logic.
    status: pending
  - id: build-training-features
    content: Implement leakage-safe rolling features from `nba.game_logs` (home row target + home/away form features and differences).
    status: pending
  - id: intraday-odds-join
    content: Build snapshot-level join between `nba.odds` and game outcomes; compute edge vs `point` at each snapshot and enforce pre-commence filtering.
    status: pending
  - id: train-walkforward-eval
    content: Train baseline regression model and run walk-forward evaluation; report MAE/RMSE plus ATS/ROI metrics by threshold.
    status: pending
  - id: live-scoring-script
    content: Create a script that pulls latest odds snapshots, computes features, loads model, and outputs betting signals (CSV/JSON; optional DB table).
    status: pending
isProject: false
---

## Goal

Build your own spread predictor trained on `nba.game_logs`, then “compete with odds” by computing **edge vs each live snapshot** in `nba.odds` and backtesting/simulating a strategy (bet when edge is large enough).

## What you already have (we’ll leverage)

- **Outcomes table**: `nba.game_logs` (primary key `(game_id, team)`) includes `home_margin` and advanced team stats.
  - Defined in `[sql/game_logs.sql](/Users/caleb/Desktop/Gambling-Project/sql/game_logs.sql)`.
- **Odds snapshots table**: `nba.odds` stores multiple snapshots per game/book, keyed by `(game_id, bookmaker_key, snapshot_time_pacific)` and FK to `game_logs`.
  - Defined in `[sql/odds.sql](/Users/caleb/Desktop/Gambling-Project/sql/odds.sql)`.
- **Benchmark join logic** already exists in `[notebooks/benchmark.ipynb](/Users/caleb/Desktop/Gambling-Project/notebooks/benchmark.ipynb)` (loads `nba.odds` + `nba.game_logs`, takes latest snapshot per key for a “closing-ish” line).
- **Ingestion jobs** already populate both tables in Neon (`scripts/live_neon_sink.py` and `scripts/ingest_historical_game_logs.py`).

## Key design decisions (based on your answers)

- **Target**: predict `home_margin` (home − away) and compare to spread `point`.
- **Intraday**: score each odds snapshot; treat each `(game_id, bookmaker_key, snapshot_time_*)` as a decision point.

## Core approach

### 1) Build a training dataset from `nba.game_logs`

We need one row per **game**, with features known pre-game, and label = realized `home_margin`.

- Start from the **home row** in `nba.game_logs` (same approach as the benchmark notebook).
- Initial feature set (all already in `nba.game_logs`): `ortg, drtg, pace, ftr, _3par, TS%, TRB%, AST%, STL%, BLK%, eFG%, TOV%, ORB%, FT/FGA` and opponent variants `Opp_*`.
- Convert to **pre-game signals** by using rolling/expanding windows *prior to the game date* (avoid leakage): e.g. team last-5, last-10, season-to-date.
  - For each stat, compute both **team form** and **opponent form**, and differences (home − away).
- Include schedule/context features derivable from `game_logs`: rest days since last game, back-to-back flag, travel proxy (optional later).

### 2) Join odds snapshots for evaluation (intraday)

We’ll create an “odds decision table” that joins each snapshot to the correct game outcome.

- Use `nba.odds` rows where `team` corresponds to the **home side** (your pipeline already tends to keep home-side odds rows when preparing).
- Join on `game_id` to the home-game outcome (and use `home_margin` as label).
- For each snapshot, compute:
  - `model_pred_margin` (expected home margin)
  - `edge_points = model_pred_margin - spread_point`
  - If you want ATS-style decision: bet **home** when `edge_points >= threshold`, bet **away** when `edge_points <= -threshold`.

### 3) Model choice (start simple, iterate)

Start with a strong baseline that’s hard to beat:

- **Regularized linear regression** (Ridge/Lasso/ElasticNet) on engineered features.
- Then try **tree boosting** (LightGBM / XGBoost) if you want non-linear effects.

Loss/metrics:

- Primary: MAE/RMSE on `home_margin`.
- Betting: ATS win%, average CLV proxy (optional), and simulated unit ROI using snapshot spreads.

Validation:

- Use **time-based splits** (walk-forward): train on early seasons, validate on later dates.
- Optional: per-season folds to avoid overfitting to a single year.

### 4) Backtest / simulation for intraday decisions

Because you picked intraday, we’ll be explicit about the decision rule:

- Define a “cooldown” per game+book (e.g. only one bet per side per game, take first signal that crosses threshold, or take best edge prior to tip-off).
- Only allow bets **before `commence_time`**.
- Grade ATS using `home_margin` vs `point` at that snapshot.
- Output a report:
  - ROI by threshold, by book, by month
  - Number of bets, hit rate, average edge at bet time

### 5) Live scoring pipeline

Add a lightweight script that:

- Pulls today’s/upcoming odds snapshots from Neon
- Builds the latest pre-game features (from `nba.game_logs` rolling windows)
- Loads a saved model artifact
- Writes out “signals” (CSV/JSON) and optionally inserts a `nba.model_signals` table.

## Where this will live in your repo

- New modeling code in the untracked folder you already created: `[nba_spreads/ML/](/Users/caleb/Desktop/Gambling-Project/nba_spreads/ML/)`
  - `nba_spreads/ML/pipeline.py`: dataset building (features + joins) and walk-forward evaluation
  - `nba_spreads/ML/model.py`: model training/loading + prediction
- Optional new SQL: `sql/model_signals.sql` if you want a dedicated table.
- Optional notebook: `notebooks/model_backtest.ipynb` for exploration.

## Concrete data contracts (so everything lines up)

- `nba.game_logs` per team-game; we’ll derive a **game-level** dataset by selecting home rows and joining in opponent rolling stats.
- `nba.odds` per snapshot; we’ll evaluate intraday by treating each snapshot as a candidate bet, filtered to `commence_time` in the future.

## Risks / gotchas we’ll guard against

- **Leakage**: rolling features must exclude the current game.
- **Timezone/date mismatch**: your `game_id` logic already converts odds `commence_time` to US timezone before taking date; we’ll keep that convention.
- **Snapshot multiplicity**: intraday backtests can “double count” the same game; we’ll implement a strict per-game selection rule.

## Output you’ll get

- A trained model artifact (plus feature config).
- A reproducible walk-forward backtest report.
- A live scoring script that emits actionable edges from current odds snapshots.

