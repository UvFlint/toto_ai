# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
CLI tool that analyzes Winner 16 football betting forms from winner.co.il. Multiple AI models (OpenAI, Google, Anthropic, DeepSeek) each produce a prediction column; results include consensus, union column for multi-column betting with cost estimate, and optional auto-submission to winner.co.il.

## Commands
- Run: `uv run toto`
- Dry run (mock data, no scraping): `uv run toto --dry-run`
- Skip research (stats + news): `uv run toto --no-research`
- Premium models: `uv run toto --premium`
- Schedule mode (poll + auto-run): `uv run toto --schedule`
- Auto-submit: `uv run toto --send-auto`
- Mark form submitted: `uv run toto --mark-submitted <form_number>`
- Sync deps: `uv sync`
- Lint: `uv run ruff check src/`
- Format: `uv run ruff format src/`
- Tests: `uv run pytest`

## Tech Stack
- Python 3.11+, uv package manager, hatchling build
- pydantic-ai for AI agents with structured output (`list[MatchPrediction]`)
- Selenium (Chrome) for web scraping (winner.co.il is a JS SPA)
- Perplexity sonar-pro for match research (stats, news, odds via web search)
- Rich for console output, Click for CLI
- pydantic-settings for config (loads from `.env`)

## Architecture

```
src/toto_ai/
  main.py          # CLI entry point (Click), UTF-8 setup, .env loading
  pipeline.py      # Orchestration: fetch -> research -> AI -> display -> submit
  config.py        # pydantic-settings Settings class (all env vars)
  console.py       # Shared Rich Console instance
  perplexity.py    # Perplexity model factory (sonar-pro) + deep research
  tracker.py       # SubmissionTracker — JSON file tracking submitted forms
  scraper/         # Selenium scraper for winner.co.il + form validation
    models.py      # Match, WinnerForm pydantic models
    winner_scraper.py  # WinnerScraper + create_mock_form()
  stats/           # Match research via Perplexity sonar-pro
    models.py      # MatchStats, H2H, TeamForm, Standing, etc.
    perplexity_stats.py # PerplexityStatsCollector — per-match web research
  news/            # Supplemental news sources
    one_scraper.py # one.co.il Israeli sports news supplement
  analyzer/        # AI prediction engine
    models.py      # MatchPrediction, FullColumn, FullReport, ConsensusMatch
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

- **Pipeline flow** (`pipeline.py`): fetch form -> validate -> research matches (Perplexity) -> supplement with one.co.il -> verify data coverage -> AI analysis -> display -> email -> submit. In `--schedule` mode this loops with polling.
- **Per-match research** (`perplexity_stats.py`): Each match gets a single Perplexity sonar-pro call that returns H2H, form, standings, injuries, news, and odds as structured output. Runs 16 calls with 4x concurrency via asyncio semaphore.
- **AI models run in parallel** via `asyncio.gather` in `analyzer/agent.py`. Each model returns `list[MatchPrediction]` (structured output). Failed models are filtered out.
- **Standard vs Premium tiers**: `MODELS_STANDARD` and `MODELS_PREMIUM` in `agent.py` define model lists. Premium adds reasoning/thinking settings per model.
- **DeepSeek** uses OpenAI-compatible provider with custom base URL.
- **Form validation** (`_validate_form`): Guards against scraping UI navigation text or maintenance pages instead of real matches.
- **Submission tracker** (`tracker.py`): JSON file at `data/submissions.json` prevents re-submitting the same form in schedule mode.

## Environment Variables
All configured via `.env` file (see `.env.example`). Key vars:
- `PERPLEXITY_API_KEY` — required for match research (Perplexity sonar-pro)
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` — AI providers
- `WINNER_USERNAME`, `WINNER_PASSWORD` — winner.co.il login for auto-submission
- `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENT` — Gmail SMTP notification
- `HEADLESS` — run Chrome headless (default: true)

## Important Notes
- Windows path encoding: Hebrew characters in paths cause issues. `.env` is loaded without explicit path (`load_dotenv()`) to avoid this.
- stdout is reconfigured to UTF-8 on Windows at startup (`main.py:10-12`).
- The scraper targets a JS SPA — Selenium with Chrome is required, not simple HTTP requests.
- `ruff` is configured with `line-length = 100` and `target-version = "py311"`.
