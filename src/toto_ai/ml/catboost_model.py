"""CatBoost match result classifier — train, load, predict.

Two models:
- **Rich**: trained on main leagues with match stats + odds features
- **Simple**: trained on all leagues with goals + odds features only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from toto_ai.analyzer.models import FullColumn, MatchPrediction
from toto_ai.console import console
from toto_ai.ml.feature_engineering import (
    CAT_FEATURES,
    DRAW_CAT_FEATURES,
    DRAW_FEATURES,
    DRAW_LEAGUE_GROUP_CAT_FEATURES,
    DRAW_LEAGUE_GROUP_FEATURES,
    LEAGUE_GROUP_CAT_FEATURES,
    LEAGUE_GROUP_FEATURES,
    LEAGUE_GROUPS,
    RICH_FEATURES,
    SIMPLE_FEATURES,
    build_inference_features,
    engineer_draw_features,
    engineer_rich_features,
    engineer_simple_features,
    league_code_to_group,
    load_extra_league_data,
    load_main_league_data,
    match_to_league_code,
)
from toto_ai.ml.glicko2 import (
    GlickoStore,
    compute_glicko2_features,
    load_glicko_store,
    save_glicko_store,
)
from toto_ai.ml.pi_ratings import RatingStore, compute_pi_features, load_ratings, save_ratings
from toto_ai.scraper.models import Match
from toto_ai.stats.models import MatchStats

RICH_MODEL_FILE = "catboost_rich.cbm"
SIMPLE_MODEL_FILE = "catboost_simple.cbm"
DRAW_MODEL_FILE = "catboost_draw.cbm"
PI_RATINGS_FILE = "pi_ratings.pkl"
GLICKO_RATINGS_FILE = "glicko2_ratings.pkl"
DRAW_LEAGUE_GROUPS_DIR = "draw_league_groups"

# Class labels for CatBoost output
CLASS_NAMES = ["H", "D", "A"]  # 0=Home, 1=Draw, 2=Away
PREDICTION_MAP = {0: "1", 1: "X", 2: "2"}


def _prepare_pool(
    df: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
) -> tuple[Pool, pd.Series]:
    """Build a CatBoost Pool from a DataFrame, handling missing features."""
    available = [f for f in features if f in df.columns]
    X = df[available].copy()

    # Fill categorical NaN with "unknown" (CatBoost needs string cats)
    for cf in cat_features:
        if cf in X.columns:
            X[cf] = X[cf].fillna("unknown").astype(str)

    cat_idx = [i for i, f in enumerate(available) if f in cat_features]
    pool = Pool(X, label=df["target"], cat_features=cat_idx)
    return pool, df["target"]


def train_rich_model(
    data_dir: str | Path,
) -> tuple[CatBoostClassifier, dict, RatingStore, GlickoStore]:
    """Train the rich model on main league data with full match stats."""
    console.print("[bold]Loading main league data...[/bold]")
    df = load_main_league_data(data_dir)
    if df.empty:
        raise RuntimeError("No main league data found. Run --download-data first.")

    console.print(f"  Loaded {len(df):,} matches from main leagues")
    df, pi_store, glicko_store = engineer_rich_features(df)

    # Drop rows with no rolling features (first few matches per team)
    df = df.dropna(subset=["h_roll_gf", "a_roll_gf"])
    console.print(f"  After feature engineering: {len(df):,} matches")

    # Train/val split: last season as validation
    seasons = sorted(df["season"].unique())
    val_season = seasons[-1] if len(seasons) > 1 else None

    if val_season:
        train_df = df[df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        # Random split fallback
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

    console.print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_pool, _ = _prepare_pool(train_df, RICH_FEATURES, CAT_FEATURES)
    val_pool, _ = _prepare_pool(val_df, RICH_FEATURES, CAT_FEATURES)

    # Class weights to handle draw imbalance
    class_counts = train_df["target"].value_counts()
    total = len(train_df)
    class_weights = {c: total / (3 * count) for c, count in class_counts.items()}

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        class_weights=list(class_weights.get(i, 1.0) for i in range(3)),
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        use_best_model=True,
    )

    model.fit(train_pool, eval_set=val_pool)

    # Eval metrics
    val_preds = model.predict(val_pool)
    val_true = val_df["target"].values
    accuracy = np.mean(val_preds.flatten().astype(int) == val_true)
    metrics = {"accuracy": float(accuracy), "train_size": len(train_df), "val_size": len(val_df)}

    return model, metrics, pi_store, glicko_store


def train_simple_model(
    data_dir: str | Path,
) -> tuple[CatBoostClassifier, dict, RatingStore, GlickoStore]:
    """Train the simple model on all available data (main + extra leagues)."""
    console.print("[bold]Loading all league data...[/bold]")
    main_df = load_main_league_data(data_dir)
    extra_df = load_extra_league_data(data_dir)

    frames = [df for df in [main_df, extra_df] if not df.empty]
    if not frames:
        raise RuntimeError("No data found. Run --download-data first.")

    # Use only common columns for simple model
    common_cols = [
        "league",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "home_odds",
        "draw_odds",
        "away_odds",
        "date",
    ]
    normalized = []
    for df in frames:
        available = [c for c in common_cols if c in df.columns]
        normalized.append(df[available])

    df = pd.concat(normalized, ignore_index=True)
    console.print(f"  Loaded {len(df):,} total matches")

    df, pi_store, glicko_store = engineer_simple_features(df)
    df = df.dropna(subset=["h_roll_gf", "a_roll_gf"])
    console.print(f"  After feature engineering: {len(df):,} matches")

    # Train/val split
    seasons = sorted(df["season"].unique())
    val_season = seasons[-1] if len(seasons) > 1 else None

    if val_season:
        train_df = df[df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

    console.print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_pool, _ = _prepare_pool(train_df, SIMPLE_FEATURES, CAT_FEATURES)
    val_pool, _ = _prepare_pool(val_df, SIMPLE_FEATURES, CAT_FEATURES)

    class_counts = train_df["target"].value_counts()
    total = len(train_df)
    class_weights = {c: total / (3 * count) for c, count in class_counts.items()}

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        class_weights=list(class_weights.get(i, 1.0) for i in range(3)),
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        use_best_model=True,
    )

    model.fit(train_pool, eval_set=val_pool)

    val_preds = model.predict(val_pool)
    val_true = val_df["target"].values
    accuracy = np.mean(val_preds.flatten().astype(int) == val_true)
    metrics = {"accuracy": float(accuracy), "train_size": len(train_df), "val_size": len(val_df)}

    return model, metrics, pi_store, glicko_store


def train_and_save(data_dir: str | Path, model_dir: str | Path) -> dict:
    """Train both models and save to disk. Returns rich model metrics."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Training Rich CatBoost Model[/bold blue]")
    console.print("=" * 50)
    rich_model, rich_metrics, pi_store, glicko_store = train_rich_model(data_dir)
    rich_path = model_dir / RICH_MODEL_FILE
    rich_model.save_model(str(rich_path))
    console.print(f"[green]Rich model saved → {rich_path}[/green]")
    console.print(f"  Validation accuracy: [bold]{rich_metrics['accuracy']:.1%}[/bold]")

    console.print("\n[bold blue]Training Simple CatBoost Model[/bold blue]")
    console.print("=" * 50)
    simple_model, simple_metrics, simple_pi_store, simple_glicko_store = train_simple_model(
        data_dir
    )
    simple_path = model_dir / SIMPLE_MODEL_FILE
    simple_model.save_model(str(simple_path))
    console.print(f"[green]Simple model saved → {simple_path}[/green]")
    console.print(f"  Validation accuracy: [bold]{simple_metrics['accuracy']:.1%}[/bold]")

    # Save pi-ratings (use simple store as it covers all leagues)
    pi_path = model_dir / PI_RATINGS_FILE
    # Merge: simple store has more teams, enrich with rich store for main league teams
    merged_store = {**simple_pi_store, **pi_store}
    save_ratings(merged_store, pi_path)
    console.print(f"[green]Pi-ratings saved → {pi_path} ({len(merged_store)} teams)[/green]")

    # Save Glicko-2 ratings
    glicko_path = model_dir / GLICKO_RATINGS_FILE
    merged_glicko = {**simple_glicko_store, **glicko_store}
    save_glicko_store(merged_glicko, glicko_path)
    console.print(f"[green]Glicko-2 saved → {glicko_path} ({len(merged_glicko)} teams)[/green]")

    console.print("\n[bold green]Both models trained successfully![/bold green]")
    return rich_metrics


def train_league_group_models(
    data_dir: str | Path, model_dir: str | Path, global_rich_accuracy: float = 0.0
) -> None:
    """Train country-group CatBoost models from pre-engineered main league data."""
    import json

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    groups_dir = model_dir / "league_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Training League-Group CatBoost Models[/bold blue]")
    console.print("=" * 50)

    # Load and engineer features once (global pi-ratings/glicko)
    console.print("[bold]Loading main league data for league-group training...[/bold]")
    df = load_main_league_data(data_dir)
    if df.empty:
        console.print("[yellow]No main league data found, skipping league-group models.[/yellow]")
        return

    df, _, _ = engineer_rich_features(df)
    df = df.dropna(subset=["h_roll_gf", "a_roll_gf"])
    console.print(f"  Engineered {len(df):,} matches total")

    index: dict[str, object] = {
        "global_rich_accuracy": global_rich_accuracy,
        "groups": {},
    }

    for group_name, league_codes in LEAGUE_GROUPS.items():
        group_df = df[df["league"].isin(league_codes)].copy()
        if len(group_df) < 100:
            console.print(
                f"  [yellow]{group_name}: too few matches ({len(group_df)}), skipping[/yellow]"
            )
            continue

        # Convert season to ordinal numeric (0..N)
        seasons_sorted = sorted(group_df["season"].unique())
        season_map = {s: i for i, s in enumerate(seasons_sorted)}
        group_df["season"] = group_df["season"].map(season_map).astype(float)

        # Drop league column (constant within group)
        if "league" in group_df.columns:
            group_df = group_df.drop(columns=["league"])

        # Train/val split: last season
        val_season_idx = max(season_map.values())
        train_df = group_df[group_df["season"] != val_season_idx]
        val_df = group_df[group_df["season"] == val_season_idx]

        if len(val_df) < 10:
            train_df = group_df.sample(frac=0.85, random_state=42)
            val_df = group_df.drop(train_df.index)

        # Prepare pools (no categorical features for league-group models)
        available = [f for f in LEAGUE_GROUP_FEATURES if f in group_df.columns]
        X_train = train_df[available].copy()
        X_val = val_df[available].copy()

        train_pool = Pool(X_train, label=train_df["target"])
        val_pool = Pool(X_val, label=val_df["target"])

        # Class weights
        class_counts = train_df["target"].value_counts()
        total = len(train_df)
        class_weights = {c: total / (3 * count) for c, count in class_counts.items()}

        model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            class_weights=list(class_weights.get(i, 1.0) for i in range(3)),
            early_stopping_rounds=50,
            verbose=0,
            random_seed=42,
            use_best_model=True,
        )

        model.fit(train_pool, eval_set=val_pool)

        val_preds = model.predict(val_pool)
        val_true = val_df["target"].values
        accuracy = float(np.mean(val_preds.flatten().astype(int) == val_true))

        # Save model
        group_dir = groups_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        model_path = group_dir / "catboost.cbm"
        model.save_model(str(model_path))

        # Save metadata
        metadata = {
            "leagues": league_codes,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "accuracy": accuracy,
            "max_season_ordinal": int(val_season_idx),
        }
        with open(group_dir / "catboost_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for code in league_codes:
            index["groups"][code] = group_name

        beat_global = accuracy > global_rich_accuracy
        status = "[bold green]BETTER[/bold green]" if beat_global else "[dim]worse[/dim]"
        console.print(
            f"  [green]{group_name:12s}[/green] "
            f"train={len(train_df):>6,} val={len(val_df):>5,} "
            f"acc={accuracy:.1%} ({status} than global {global_rich_accuracy:.1%})"
        )

    # Save index (catboost-specific, includes global accuracy for hybrid filtering)
    with open(groups_dir / "catboost_index.json", "w") as f:
        json.dump(index, f, indent=2)

    console.print(
        f"[green]League-group CatBoost models saved ({len(index['groups'])} league codes)[/green]"
    )


# --- League-group model cache ---
_catboost_group_cache: dict[str, CatBoostClassifier | None] = {}


def _load_league_group_catboost(
    model_dir: str | Path, group_name: str
) -> CatBoostClassifier | None:
    """Lazily load a league-group CatBoost model if it beats the global rich model."""
    import json

    if group_name in _catboost_group_cache:
        return _catboost_group_cache[group_name]

    model_dir = Path(model_dir)
    path = model_dir / "league_groups" / group_name / "catboost.cbm"
    if not path.exists():
        _catboost_group_cache[group_name] = None
        return None

    # Check if group model accuracy beats global rich
    meta_path = model_dir / "league_groups" / group_name / "catboost_metadata.json"
    index_path = model_dir / "league_groups" / "catboost_index.json"
    if meta_path.exists() and index_path.exists():
        meta = json.loads(meta_path.read_text())
        idx = json.loads(index_path.read_text())
        global_acc = idx.get("global_rich_accuracy", 0.0)
        if meta.get("accuracy", 0.0) <= global_acc:
            _catboost_group_cache[group_name] = None
            return None

    model = CatBoostClassifier()
    model.load_model(str(path))
    _catboost_group_cache[group_name] = model
    return model


def load_models(
    model_dir: str | Path,
) -> tuple[
    CatBoostClassifier | None,
    CatBoostClassifier | None,
    RatingStore | None,
    GlickoStore | None,
]:
    """Load trained models, pi-ratings, and Glicko-2 ratings from disk."""
    model_dir = Path(model_dir)
    rich_model = simple_model = None
    pi_store = None
    glicko_store = None

    rich_path = model_dir / RICH_MODEL_FILE
    if rich_path.exists():
        rich_model = CatBoostClassifier()
        rich_model.load_model(str(rich_path))

    simple_path = model_dir / SIMPLE_MODEL_FILE
    if simple_path.exists():
        simple_model = CatBoostClassifier()
        simple_model.load_model(str(simple_path))

    pi_path = model_dir / PI_RATINGS_FILE
    if pi_path.exists():
        pi_store = load_ratings(pi_path)

    glicko_path = model_dir / GLICKO_RATINGS_FILE
    if glicko_path.exists():
        glicko_store = load_glicko_store(glicko_path)

    return rich_model, simple_model, pi_store, glicko_store


def _extract_form_stats(form) -> tuple[float | None, float | None, float | None]:
    """Extract goals for/against per game and points per game from TeamForm."""
    if not form or not form.recent_matches:
        return None, None, None
    n = len(form.recent_matches)
    gf = form.goals_for / n
    ga = form.goals_against / n
    pts = (form.wins * 3 + form.draws) / n
    return gf, ga, pts


def _extract_standing_stats(standing) -> tuple[float | None, float | None]:
    """Extract goals per game and points per game from Standing."""
    if not standing or standing.played < 3:
        return None, None
    gpg = standing.goals_for / standing.played
    ppg = standing.points / standing.played
    return gpg, ppg


def _extract_form_match_stats(form) -> dict:
    """Extract average match stats (shots, corners, etc.) from TeamFormStats."""
    if not form or not form.form_stats or form.form_stats.matches_with_stats == 0:
        return {}
    fs = form.form_stats
    return {
        "shots": fs.avg_shots_total,
        "sot": fs.avg_shots_on_target,
        "corners": fs.avg_corners,
        "fouls": fs.avg_fouls,
        "possession": fs.avg_possession,
        "blocked_shots": fs.avg_blocked_shots,
        "gk_saves": fs.avg_gk_saves,
        "pass_accuracy": fs.avg_pass_accuracy,
        "yellow": fs.avg_yellow_cards,
        "red": fs.avg_red_cards,
        "offsides": fs.avg_offsides,
        "shots_insidebox": fs.avg_shots_insidebox,
        "shots_outsidebox": fs.avg_shots_outsidebox,
        "total_passes": fs.avg_total_passes,
    }


def _extract_h2h_features(stats: MatchStats | None) -> dict:
    """Extract head-to-head features from MatchStats."""
    if not stats or not stats.h2h:
        return {}
    h2h = stats.h2h
    total = h2h.home_wins + h2h.draws + h2h.away_wins
    if total == 0:
        return {}
    return {
        "h2h_home_win_rate": h2h.home_wins / total,
        "h2h_draw_rate": h2h.draws / total,
        "h2h_total_matches": total,
    }


def _extract_xg_features(stats: MatchStats | None) -> dict:
    """Extract xG features from Understat TeamXG data."""
    if not stats:
        return {}
    result = {}
    if stats.home_xg and stats.home_xg.matches_played >= 3:
        result["home_xg_per_game"] = stats.home_xg.xg_per_game
        result["home_xga_per_game"] = stats.home_xg.xga_per_game
        result["home_npxgd"] = stats.home_xg.npxgd
    if stats.away_xg and stats.away_xg.matches_played >= 3:
        result["away_xg_per_game"] = stats.away_xg.xg_per_game
        result["away_xga_per_game"] = stats.away_xg.xga_per_game
        result["away_npxgd"] = stats.away_xg.npxgd
    if "home_xg_per_game" in result and "away_xg_per_game" in result:
        result["xg_diff"] = result["home_xg_per_game"] - result["away_xg_per_game"]
    return result


def _extract_news_features(stats: MatchStats | None) -> dict:
    """Extract news-derived features from MatchNewsAnalysis."""
    if not stats or not stats.news_analysis:
        return {}
    na = stats.news_analysis
    return {
        "news_home_impact": na.home_impact_score,
        "news_away_impact": na.away_impact_score,
        "news_net_impact": na.net_impact,
        "news_max_weight": na.max_item_weight,
        "news_has_x_factor": 1.0 if na.has_x_factor else 0.0,
        "news_home_absence_count": float(na.team_absence_count("home")),
        "news_away_absence_count": float(na.team_absence_count("away")),
    }


def _extract_standing_features(stats: MatchStats | None) -> dict:
    """Extract league standing position features."""
    if not stats:
        return {}
    result = {}
    if stats.home_standing:
        result["home_position"] = stats.home_standing.position
    if stats.away_standing:
        result["away_position"] = stats.away_standing.position
    if stats.home_standing and stats.away_standing:
        result["position_diff"] = stats.home_standing.position - stats.away_standing.position
    return result


def predict_match(
    match: Match,
    stats: MatchStats | None,
    rich_model: CatBoostClassifier | None,
    simple_model: CatBoostClassifier | None,
    pi_store: RatingStore | None = None,
    glicko_store: GlickoStore | None = None,
) -> tuple[dict[str, float], str] | None:
    """Predict match probabilities using the best available model.

    Returns ({"1": p, "X": p, "2": p}, model_variant) or None if no model.
    """
    if rich_model is None and simple_model is None:
        return None

    league_code = match_to_league_code(match.league, match.country)

    # Extract features from live stats
    h_odds = d_odds = a_odds = None
    if stats and stats.odds:
        h_odds = stats.odds.home_odds or None
        d_odds = stats.odds.draw_odds or None
        a_odds = stats.odds.away_odds or None

    h_gf, h_ga, h_pts = _extract_form_stats(stats.home_form if stats else None)
    a_gf, a_ga, a_pts = _extract_form_stats(stats.away_form if stats else None)

    h_gpg, h_ppg = _extract_standing_stats(stats.home_standing if stats else None)
    a_gpg, a_ppg = _extract_standing_stats(stats.away_standing if stats else None)

    # Rich match stats from form
    h_ms = _extract_form_match_stats(stats.home_form if stats else None)
    a_ms = _extract_form_match_stats(stats.away_form if stats else None)

    # Pi-rating and Glicko-2 features
    pi_feats = compute_pi_features(pi_store, match.home_team, match.away_team) if pi_store else {}
    glicko_feats = (
        compute_glicko2_features(glicko_store, match.home_team, match.away_team)
        if glicko_store
        else {}
    )

    # Inference-only features from MatchStats
    h2h_feats = _extract_h2h_features(stats)
    xg_feats = _extract_xg_features(stats)
    standing_feats = _extract_standing_features(stats)
    news_feats = _extract_news_features(stats)

    features = build_inference_features(
        match_league=league_code,
        home_odds=h_odds,
        draw_odds=d_odds,
        away_odds=a_odds,
        home_form_gf=h_gf,
        home_form_ga=h_ga,
        home_form_pts=h_pts,
        away_form_gf=a_gf,
        away_form_ga=a_ga,
        away_form_pts=a_pts,
        home_season_gpg=h_gpg,
        home_season_ppg=h_ppg,
        away_season_gpg=a_gpg,
        away_season_ppg=a_ppg,
        home_shots=h_ms.get("shots"),
        away_shots=a_ms.get("shots"),
        home_sot=h_ms.get("sot"),
        away_sot=a_ms.get("sot"),
        home_corners=h_ms.get("corners"),
        away_corners=a_ms.get("corners"),
        home_fouls=h_ms.get("fouls"),
        away_fouls=a_ms.get("fouls"),
        home_possession=h_ms.get("possession"),
        away_possession=a_ms.get("possession"),
        home_blocked_shots=h_ms.get("blocked_shots"),
        away_blocked_shots=a_ms.get("blocked_shots"),
        home_gk_saves=h_ms.get("gk_saves"),
        away_gk_saves=a_ms.get("gk_saves"),
        home_pass_accuracy=h_ms.get("pass_accuracy"),
        away_pass_accuracy=a_ms.get("pass_accuracy"),
        home_yellow=h_ms.get("yellow"),
        away_yellow=a_ms.get("yellow"),
        home_red=h_ms.get("red"),
        away_red=a_ms.get("red"),
        home_offsides=h_ms.get("offsides"),
        away_offsides=a_ms.get("offsides"),
        home_shots_insidebox=h_ms.get("shots_insidebox"),
        away_shots_insidebox=a_ms.get("shots_insidebox"),
        home_shots_outsidebox=h_ms.get("shots_outsidebox"),
        away_shots_outsidebox=a_ms.get("shots_outsidebox"),
        home_total_passes=h_ms.get("total_passes"),
        away_total_passes=a_ms.get("total_passes"),
        home_rest_days=stats.home_rest_days if stats else None,
        away_rest_days=stats.away_rest_days if stats else None,
        home_injury_count=len(stats.home_injuries) if stats else None,
        away_injury_count=len(stats.away_injuries) if stats else None,
        **pi_feats,
        **glicko_feats,
        **h2h_feats,
        **xg_feats,
        **standing_feats,
        **news_feats,
    )

    # Try league-group model first
    from toto_ai.config import settings
    from toto_ai.data_collector.football_data_downloader import ALL_DIVISIONS

    group_name = league_code_to_group(league_code)
    group_model = (
        _load_league_group_catboost(settings.MODEL_DIR, group_name) if group_name else None
    )

    if group_model is not None:
        model = group_model
        feat_list = LEAGUE_GROUP_FEATURES
        cat_feats = LEAGUE_GROUP_CAT_FEATURES
        variant = f"CatBoost ({group_name})"
        # Convert season to numeric for league-group model
        features["season"] = 0.0  # ordinal placeholder for unseen season
    elif rich_model is not None and league_code in ALL_DIVISIONS:
        model = rich_model
        feat_list = RICH_FEATURES
        cat_feats = CAT_FEATURES
        variant = "CatBoost (rich)"
    elif simple_model is not None:
        model = simple_model
        feat_list = SIMPLE_FEATURES
        cat_feats = CAT_FEATURES
        variant = "CatBoost (simple)"
    else:
        return None

    # Build feature vector in correct order
    row = {}
    for f in feat_list:
        val = features.get(f)
        if f in cat_feats:
            row[f] = str(val) if val is not None else "unknown"
        else:
            row[f] = float(val) if val is not None else np.nan
    row_df = pd.DataFrame([row])

    cat_idx = [i for i, f in enumerate(feat_list) if f in cat_feats]
    pool = Pool(row_df, cat_features=cat_idx)

    proba = model.predict_proba(pool)[0]  # [p_home, p_draw, p_away]
    return {
        "1": round(float(proba[0]), 4),
        "X": round(float(proba[1]), 4),
        "2": round(float(proba[2]), 4),
    }, variant


def build_catboost_column(
    matches: list[Match],
    stats: list[MatchStats],
) -> FullColumn:
    """Build a FullColumn from CatBoost predictions for consensus voting.

    Mirrors the pattern from ``build_poisson_column()`` in stats/poisson.py.
    """
    from toto_ai.config import settings

    rich_model, simple_model, pi_store, glicko_store = load_models(settings.MODEL_DIR)
    if rich_model is None and simple_model is None:
        console.print(
            "[yellow]CatBoost: No trained models found. Run --train-model first.[/yellow]"
        )
        return FullColumn(model_name="CatBoost", column_type="ml")

    predictions: list[MatchPrediction] = []

    for match in matches:
        s = next((st for st in stats if st.home_team == match.home_team), None)
        result = predict_match(match, s, rich_model, simple_model, pi_store, glicko_store)

        if result is None:
            continue

        probs, variant = result
        best = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence = probs[best]

        home = s.home_team_english or match.home_team if s else match.home_team
        away = s.away_team_english or match.away_team if s else match.away_team

        predictions.append(
            MatchPrediction(
                match_number=match.match_number,
                home_team=home,
                away_team=away,
                prediction=best,
                confidence=round(confidence, 3),
                reasoning=(
                    f"{variant}: P(1)={probs['1']:.1%} P(X)={probs['X']:.1%} P(2)={probs['2']:.1%}"
                ),
                key_factors=[variant, "Historical match data"],
            )
        )

    console.print(f"[green]CatBoost model: {len(predictions)} predictions generated[/green]")
    return FullColumn(model_name="CatBoost", predictions=predictions, column_type="ml")


def enrich_stats_with_catboost(
    matches: list[Match],
    stats: list[MatchStats],
) -> None:
    """Attach CatBoost probabilities to each MatchStats for LLM prompt enrichment."""
    from toto_ai.config import settings

    rich_model, simple_model, pi_store, glicko_store = load_models(settings.MODEL_DIR)
    if rich_model is None and simple_model is None:
        return

    for match in matches:
        s = next((st for st in stats if st.home_team == match.home_team), None)
        if s is None:
            continue

        result = predict_match(match, s, rich_model, simple_model, pi_store, glicko_store)
        if result is not None:
            probs, _ = result
            s.catboost_probs = probs


# --- Draw Detection Binary Classifier ---


def train_draw_model(
    data_dir: str | Path,
) -> tuple[CatBoostClassifier, dict, RatingStore, GlickoStore]:
    """Train binary draw detector on all available data (main + extra leagues)."""
    console.print("[bold]Loading all league data for draw classifier...[/bold]")
    main_df = load_main_league_data(data_dir)
    extra_df = load_extra_league_data(data_dir)

    frames = [df for df in [main_df, extra_df] if not df.empty]
    if not frames:
        raise RuntimeError("No data found. Run --download-data first.")

    common_cols = [
        "league", "season", "home_team", "away_team",
        "home_goals", "away_goals", "result",
        "home_odds", "draw_odds", "away_odds", "date",
    ]
    normalized = []
    for df in frames:
        available = [c for c in common_cols if c in df.columns]
        normalized.append(df[available])

    df = pd.concat(normalized, ignore_index=True)
    console.print(f"  Loaded {len(df):,} total matches")

    df, pi_store, glicko_store = engineer_draw_features(df)
    df = df.dropna(subset=["h_roll_gf", "a_roll_gf"])
    console.print(f"  After feature engineering: {len(df):,} matches")

    draw_count = int((df["target"] == 1).sum())
    draw_pct = 100.0 * draw_count / len(df)
    console.print(f"  Draw rate: {draw_pct:.1f}% ({draw_count:,} draws)")

    seasons = sorted(df["season"].unique())
    val_season = seasons[-1] if len(seasons) > 1 else None
    if val_season:
        train_df = df[df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

    console.print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    class_counts = train_df["target"].value_counts()
    total = len(train_df)
    class_weights = {c: total / (2 * count) for c, count in class_counts.items()}

    train_pool, _ = _prepare_pool(train_df, DRAW_FEATURES, DRAW_CAT_FEATURES)
    val_pool, _ = _prepare_pool(val_df, DRAW_FEATURES, DRAW_CAT_FEATURES)

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        class_weights=list(class_weights.get(i, 1.0) for i in range(2)),
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        use_best_model=True,
    )
    model.fit(train_pool, eval_set=val_pool)

    val_preds = model.predict(val_pool)
    val_true = val_df["target"].values
    accuracy = float(np.mean(val_preds.flatten().astype(int) == val_true))

    # Draw recall: fraction of actual draws correctly predicted as draw
    draw_mask = val_true == 1
    draw_recall = float(np.mean(val_preds.flatten().astype(int)[draw_mask] == 1)) if draw_mask.any() else 0.0

    metrics = {
        "accuracy": accuracy,
        "draw_recall": draw_recall,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "val_draw_rate": draw_pct,
    }
    return model, metrics, pi_store, glicko_store


def train_draw_league_group_models(
    data_dir: str | Path, model_dir: str | Path, global_draw_accuracy: float = 0.0
) -> None:
    """Train per-country draw classifiers; only save if they beat the global model."""
    import json

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    groups_dir = model_dir / DRAW_LEAGUE_GROUPS_DIR
    groups_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Training League-Group Draw Classifiers[/bold blue]")
    console.print("=" * 50)

    df = load_main_league_data(data_dir)
    if df.empty:
        console.print("[yellow]No main league data, skipping draw league-group models.[/yellow]")
        return

    df, _, _ = engineer_draw_features(df)
    df = df.dropna(subset=["h_roll_gf", "a_roll_gf"])

    index: dict[str, object] = {"global_draw_recall": global_draw_accuracy, "groups": {}}

    for group_name, league_codes in LEAGUE_GROUPS.items():
        group_df = df[df["league"].isin(league_codes)].copy()
        if len(group_df) < 100:
            console.print(f"  [yellow]{group_name}: too few matches ({len(group_df)}), skipping[/yellow]")
            continue

        seasons_sorted = sorted(group_df["season"].unique())
        season_map = {s: i for i, s in enumerate(seasons_sorted)}
        group_df["season"] = group_df["season"].map(season_map).astype(float)
        val_season_idx = season_map.get(seasons_sorted[-1], 0)
        if "league" in group_df.columns:
            group_df = group_df.drop(columns=["league"])

        if val_season_idx > 0:
            train_df = group_df[group_df["season"] != float(val_season_idx)]
            val_df = group_df[group_df["season"] == float(val_season_idx)]
        else:
            train_df = group_df.sample(frac=0.8, random_state=42)
            val_df = group_df.drop(train_df.index)

        if len(train_df) < 50 or len(val_df) < 20:
            continue

        class_counts = train_df["target"].value_counts()
        total = len(train_df)
        class_weights = {c: total / (2 * count) for c, count in class_counts.items()}

        train_pool, _ = _prepare_pool(train_df, DRAW_LEAGUE_GROUP_FEATURES, DRAW_LEAGUE_GROUP_CAT_FEATURES)
        val_pool, _ = _prepare_pool(val_df, DRAW_LEAGUE_GROUP_FEATURES, DRAW_LEAGUE_GROUP_CAT_FEATURES)

        model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            class_weights=list(class_weights.get(i, 1.0) for i in range(2)),
            early_stopping_rounds=30,
            verbose=0,
            random_seed=42,
            use_best_model=True,
        )
        model.fit(train_pool, eval_set=val_pool)

        val_preds = model.predict(val_pool)
        val_true = val_df["target"].values
        val_preds_flat = val_preds.flatten().astype(int)
        accuracy = float(np.mean(val_preds_flat == val_true))
        draw_mask = val_true == 1
        draw_recall = float(np.mean(val_preds_flat[draw_mask] == 1)) if draw_mask.any() else 0.0

        group_dir = groups_dir / group_name
        group_dir.mkdir(exist_ok=True)
        model.save_model(str(group_dir / "catboost.cbm"))

        metadata = {
            "leagues": league_codes,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "accuracy": accuracy,
            "draw_recall": draw_recall,
            "max_season_ordinal": int(val_season_idx),
        }
        with open(group_dir / "catboost_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for code in league_codes:
            index["groups"][code] = group_name

        beat_global = draw_recall > global_draw_accuracy
        status = "[bold green]BETTER[/bold green]" if beat_global else "[dim]worse[/dim]"
        console.print(
            f"  [green]{group_name:12s}[/green] "
            f"train={len(train_df):>6,} val={len(val_df):>5,} "
            f"acc={accuracy:.1%} recall={draw_recall:.1%} ({status} than global recall {global_draw_accuracy:.1%})"
        )

    with open(groups_dir / "catboost_index.json", "w") as f:
        json.dump(index, f, indent=2)
    console.print(f"[green]Draw league-group models saved ({len(index['groups'])} league codes)[/green]")


# --- Draw model cache ---
_draw_group_cache: dict[str, CatBoostClassifier | None] = {}


def _load_draw_league_group_model(
    model_dir: str | Path, group_name: str
) -> CatBoostClassifier | None:
    """Lazily load a draw league-group model if it beats the global draw model."""
    import json

    if group_name in _draw_group_cache:
        return _draw_group_cache[group_name]

    model_dir = Path(model_dir)
    path = model_dir / DRAW_LEAGUE_GROUPS_DIR / group_name / "catboost.cbm"
    if not path.exists():
        _draw_group_cache[group_name] = None
        return None

    meta_path = model_dir / DRAW_LEAGUE_GROUPS_DIR / group_name / "catboost_metadata.json"
    index_path = model_dir / DRAW_LEAGUE_GROUPS_DIR / "catboost_index.json"
    if meta_path.exists() and index_path.exists():
        meta = json.loads(meta_path.read_text())
        idx = json.loads(index_path.read_text())
        global_acc = idx.get("global_draw_recall", 0.0)
        if meta.get("draw_recall", 0.0) <= global_acc:
            _draw_group_cache[group_name] = None
            return None

    model = CatBoostClassifier()
    model.load_model(str(path))
    _draw_group_cache[group_name] = model
    return model


def load_draw_model(model_dir: str | Path) -> CatBoostClassifier | None:
    """Load the global draw classifier from disk."""
    path = Path(model_dir) / DRAW_MODEL_FILE
    if not path.exists():
        return None
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def predict_draw_probability(
    match: Match,
    stats: MatchStats | None,
    draw_model: CatBoostClassifier | None,
    pi_store: RatingStore | None = None,
    glicko_store: GlickoStore | None = None,
) -> float | None:
    """Return P(draw) in [0, 1] using the draw detection classifier, or None if unavailable."""
    from toto_ai.config import settings

    league_code = match_to_league_code(match.league, match.country)

    # Try per-league group model first
    group_name = league_code_to_group(league_code)
    group_model = (
        _load_draw_league_group_model(settings.MODEL_DIR, group_name) if group_name else None
    )

    if group_model is not None:
        model = group_model
        feat_list = DRAW_LEAGUE_GROUP_FEATURES
        cat_feats = DRAW_LEAGUE_GROUP_CAT_FEATURES
        use_ordinal_season = True
    elif draw_model is not None:
        model = draw_model
        feat_list = DRAW_FEATURES
        cat_feats = DRAW_CAT_FEATURES
        use_ordinal_season = False
    else:
        return None

    # Extract features (reuse existing helpers)
    h_odds = d_odds = a_odds = None
    if stats and stats.odds:
        h_odds = stats.odds.home_odds or None
        d_odds = stats.odds.draw_odds or None
        a_odds = stats.odds.away_odds or None

    h_gf, h_ga, h_pts = _extract_form_stats(stats.home_form if stats else None)
    a_gf, a_ga, a_pts = _extract_form_stats(stats.away_form if stats else None)
    h_gpg, h_ppg = _extract_standing_stats(stats.home_standing if stats else None)
    a_gpg, a_ppg = _extract_standing_stats(stats.away_standing if stats else None)

    pi_feats = compute_pi_features(pi_store, match.home_team, match.away_team) if pi_store else {}
    glicko_feats = (
        compute_glicko2_features(glicko_store, match.home_team, match.away_team)
        if glicko_store
        else {}
    )
    h2h_feats = _extract_h2h_features(stats)
    xg_feats = _extract_xg_features(stats)
    standing_feats = _extract_standing_features(stats)

    # Compute implied probabilities inline
    implied_draw = None
    if h_odds and d_odds and a_odds:
        inv_total = 1.0 / h_odds + 1.0 / d_odds + 1.0 / a_odds
        implied_draw = (1.0 / d_odds) / inv_total
        implied_home = (1.0 / h_odds) / inv_total
        implied_away = (1.0 / a_odds) / inv_total
    else:
        implied_home = implied_away = None

    raw_features: dict = {
        "implied_draw": implied_draw,
        "implied_home": implied_home,
        "implied_away": implied_away,
        "draw_odds": d_odds,
        "home_odds": h_odds,
        "away_odds": a_odds,
        "h_roll_gf": h_gf,
        "h_roll_ga": h_ga,
        "h_roll_pts": h_pts,
        "a_roll_gf": a_gf,
        "a_roll_ga": a_ga,
        "a_roll_pts": a_pts,
        "h_roll_draw_rate": None,  # not available at inference time
        "a_roll_draw_rate": None,
        "h_season_gpg": h_gpg,
        "h_season_ppg": h_ppg,
        "a_season_gpg": a_gpg,
        "a_season_ppg": a_ppg,
        "h_season_draw_rate": None,
        "a_season_draw_rate": None,
        "league": league_code,
        "season": 0.0 if use_ordinal_season else "current",
        "h2h_draw_rate": h2h_feats.get("h2h_draw_rate"),
        "h2h_total_matches": h2h_feats.get("h2h_total_matches"),
        "home_rest_days": stats.home_rest_days if stats else None,
        "away_rest_days": stats.away_rest_days if stats else None,
        "xg_diff": xg_feats.get("xg_diff"),
        "home_injury_count": float(len(stats.home_injuries)) if stats else None,
        "away_injury_count": float(len(stats.away_injuries)) if stats else None,
        "position_diff": standing_feats.get("position_diff"),
        **pi_feats,
        **glicko_feats,
    }

    row = {}
    for f in feat_list:
        val = raw_features.get(f)
        if f in cat_feats:
            row[f] = str(val) if val is not None else "unknown"
        else:
            row[f] = float(val) if val is not None else np.nan

    row_df = pd.DataFrame([row])
    cat_idx = [i for i, f in enumerate(feat_list) if f in cat_feats]
    pool = Pool(row_df, cat_features=cat_idx)

    proba = model.predict_proba(pool)[0]  # [p_no_draw, p_draw]
    return round(float(proba[1]), 4)


def enrich_stats_with_draw_probability(
    matches: list[Match],
    stats: list[MatchStats],
) -> None:
    """Attach draw probability to each MatchStats using the draw detection classifier."""
    from toto_ai.config import settings

    draw_model = load_draw_model(settings.MODEL_DIR)
    _, _, pi_store, glicko_store = load_models(settings.MODEL_DIR)

    if draw_model is None:
        console.print("[yellow]Draw classifier: no model found. Run --train-model first.[/yellow]")
        return

    for match in matches:
        s = next((st for st in stats if st.home_team == match.home_team), None)
        if s is None:
            continue
        prob = predict_draw_probability(match, s, draw_model, pi_store, glicko_store)
        if prob is not None:
            s.draw_prob = prob
