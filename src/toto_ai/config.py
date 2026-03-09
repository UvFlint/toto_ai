from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AI Models
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    # API-Football v3
    API_FOOTBALL_API_KEY: str = ""
    API_FOOTBALL_HOST: str = "v3.football.api-sports.io"
    API_FOOTBALL_USE_RAPIDAPI: bool = False
    API_FOOTBALL_MAX_CONCURRENT: int = 4
    TEAM_CACHE_FILE: str = "data/team_cache.json"

    # Scraping
    HEADLESS: bool = True
    SCRAPER_TIMEOUT: int = 30000

    # Winner.co.il login (for form submission)
    WINNER_USERNAME: str = ""
    WINNER_PASSWORD: str = ""

    # Email notification (Gmail SMTP)
    EMAIL_SENDER: str = ""
    EMAIL_PASSWORD: str = ""  # Gmail App Password
    EMAIL_RECIPIENT: str = "yuvalm121212@gmail.com"

    # Understat xG enrichment
    UNDERSTAT_ENABLED: bool = True

    # API-Football match statistics (opt-in, costs extra API calls)
    API_FOOTBALL_FETCH_MATCH_STATS: bool = True

    # Submission tracker
    TRACKER_FILE: str = "data/submissions.json"

    # Calibration (post-odds multiplier + review history)
    CALIBRATION_FILE: str = "data/calibration.json"

    # Israeli league data collection (API-Football)
    API_FOOTBALL_ISRAEL_SEASONS: str = "2019,2020,2021,2022,2023,2024,2025"
    API_FOOTBALL_ISRAEL_FETCH_STATS: bool = True

    # Historical data collection
    FOOTBALL_DATA_DIR: str = "data/football_data"

    # CatBoost ML models
    MODEL_DIR: str = "data/models"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
