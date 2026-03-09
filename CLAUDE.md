# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
CLI tool that analyzes Winner 16 football betting forms from winner.co.il. Multiple AI models (OpenAI, Google, Anthropic, DeepSeek) plus ML models (CatBoost, XGBoost) and a Poisson/Dixon-Coles statistical model each produce a prediction column; results include consensus, union column for multi-column betting with cost estimate, and optional auto-submission to winner.co.il.

## Commands
- Run: `uv run toto`
- Dry run (mock data, no scraping): `uv run toto --dry-run`
- Skip research (stats + news): `uv run toto --no-research`
- Premium models: `uv run toto --premium`
- Schedule mode (poll + auto-run): `uv run toto --schedule`
- Auto-submit: `uv run toto --send-auto`
- Mark form submitted: `uv run toto --mark-submitted <form_number>`
- Backtest against past results: `uv run toto --test`
- Download historical match data: `uv run toto --download-data` (add `--force-download` to re-download)
- Download Israeli data only: `uv run toto --download-israel`
- Scrape historical odds from OddsPortal: `uv run toto --scrape-odds`
- Train ML models: `uv run toto --train-model`
- Sync deps: `uv sync`
- Lint: `uv run ruff check src/`
- Format: `uv run ruff format src/`
- Tests: `uv run pytest`

## Tech Stack
- Python 3.11+, uv package manager, hatchling build
- pydantic-ai for AI agents with structured output (`list[MatchPrediction]`)
- Selenium (Chrome) for web scraping (winner.co.il is a JS SPA)
- Perplexity sonar-pro for match research (stats, news, odds via web search)
- CatBoost, XGBoost, scikit-learn, pandas for ML prediction models
- Rich for console output, Click for CLI
- pydantic-settings for config (loads from `.env`)

## Architecture

```
src/toto_ai/
  main.py          # CLI entry point (Click), UTF-8 setup, .env loading
  pipeline.py      # Orchestration: fetch -> research -> ML -> AI -> display -> submit
  config.py        # pydantic-settings Settings class (all env vars)
  console.py       # Shared Rich Console instance
  perplexity.py    # Perplexity model factory (sonar-pro) + deep research
  tracker.py       # SubmissionTracker — JSON file tracking submitted forms
  scraper/         # Selenium scraper for winner.co.il + form validation
    models.py      # Match, WinnerForm pydantic models
    winner_scraper.py  # WinnerScraper + create_mock_form()
  stats/           # Match research via Perplexity sonar-pro
    models.py      # MatchStats, H2H, TeamForm, Standing, PoissonProbabilities
    perplexity_stats.py # PerplexityStatsCollector — per-match web research
    poisson.py     # Poisson/Dixon-Coles statistical model (pure Python, no ML deps)
  news/            # Supplemental news sources
    one_scraper.py # one.co.il Israeli sports news supplement
  data_collector/  # Historical match data
    football_data_downloader.py  # Async downloader for football-data.co.uk CSVs
    api_football_downloader.py   # Israeli league data from API-Football (fixtures, stats, odds)
    oddsportal_scraper.py        # Selenium scraper for OddsPortal historical odds
  ml/              # ML prediction models
    feature_engineering.py  # Feature pipeline, rating features, inference builder
    pi_ratings.py    # Pi-Rating team strength (goal-difference model)
    glicko2.py       # Glicko-2 ratings (Bayesian with uncertainty/volatility)
    catboost_model.py  # CatBoost training + inference
    xgboost_model.py   # XGBoost training + inference
    train.py         # Training entry point (trains both models)
  analyzer/        # AI prediction engine
    models.py      # MatchPrediction, FullColumn (column_type: ai/statistical/ml), FullReport
    agent.py       # Model configs, agent creation, parallel execution
    prompts.py     # System prompt + match data prompt builder
    tools.py       # has_data_gaps utility
  display/         # Output formatting
    console.py     # Rich table display
    file_writer.py # Markdown report writer (saves to reports/)
  submitter/       # Auto-submission to winner.co.il
    models.py      # SubmissionColumn, SubmissionResult
    winner_submitter.py # Selenium-based form submission + column dedup
  notifier/        # Email notification
    email_sender.py # Gmail SMTP report sender
```

## Key Design Patterns

- **Pipeline flow** (`pipeline.py`): fetch form -> validate -> research matches (Perplexity) -> supplement with one.co.il -> verify data coverage -> Poisson/CatBoost/XGBoost enrichment -> AI analysis -> display -> email -> submit. In `--schedule` mode this loops with polling.
- **Per-match research** (`perplexity_stats.py`): Each match gets a single Perplexity sonar-pro call that returns H2H, form, standings, injuries, news, and odds as structured output. Runs 16 calls with 4x concurrency via asyncio semaphore.
- **AI models run in parallel** via `asyncio.gather` in `analyzer/agent.py`. Each model returns `list[MatchPrediction]` (structured output). Failed models are filtered out.
- **6-way consensus**: 3 AI model columns + Poisson + CatBoost + XGBoost columns all participate in consensus voting. Each column is a `FullColumn` with `column_type` ("ai", "statistical", or "ml").
- **Standard vs Premium tiers**: `MODELS_STANDARD` and `MODELS_PREMIUM` in `agent.py` define model lists. Premium adds reasoning/thinking settings per model.
- **DeepSeek** uses OpenAI-compatible provider with custom base URL.
- **ML model tiers**: Rich model (main leagues, ~60 features including match stats) vs Simple model (all leagues, ~15 features). League code determines which model is used at inference.
- **Rating systems**: Pi-Ratings (goal-difference based) and Glicko-2 (Bayesian with uncertainty/volatility) are computed from historical data and shared across CatBoost and XGBoost as features.
- **Inference-only features**: 17 features (rest days, H2H, injuries, xG, standings) are NaN during training and populated from live MatchStats at prediction time.
- **Poisson/Dixon-Coles** (`stats/poisson.py`): Pure-Python statistical baseline using standings/form/xG data with Dixon-Coles low-score correction. No ML dependencies.
- **Form validation** (`_validate_form`): Guards against scraping UI navigation text or maintenance pages instead of real matches.
- **Submission tracker** (`tracker.py`): JSON file at `data/submissions.json` prevents re-submitting the same form in schedule mode.

## Data Directories
- `data/football_data/` — historical match CSVs from football-data.co.uk (35+ divisions, 32 years)
- `data/models/` — trained ML models (`catboost_rich.cbm`, `catboost_simple.cbm`, `xgboost_rich.json`, `xgboost_simple.json`, `pi_ratings.pkl`, `glicko2_ratings.pkl`, `xgb_label_encoders.pkl`)

## Environment Variables
All configured via `.env` file (see `.env.example`). Key vars:
- `PERPLEXITY_API_KEY` — required for match research (Perplexity sonar-pro)
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` — AI providers
- `WINNER_USERNAME`, `WINNER_PASSWORD` — winner.co.il login for auto-submission
- `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENT` — Gmail SMTP notification
- `HEADLESS` — run Chrome headless (default: true)
- `FOOTBALL_DATA_DIR` — historical data path (default: `data/football_data`)
- `MODEL_DIR` — trained models path (default: `data/models`)

## Important Notes
- Windows path encoding: Hebrew characters in paths cause issues. `.env` is loaded without explicit path (`load_dotenv()`) to avoid this.
- stdout is reconfigured to UTF-8 on Windows at startup (`main.py:10-12`).
- The scraper targets a JS SPA — Selenium with Chrome is required, not simple HTTP requests.
- `ruff` is configured with `line-length = 100` and `target-version = "py311"`.
