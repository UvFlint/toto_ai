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
    RICH_FEATURES,
    SIMPLE_FEATURES,
    build_inference_features,
    engineer_rich_features,
    engineer_simple_features,
    load_extra_league_data,
    load_main_league_data,
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
PI_RATINGS_FILE = "pi_ratings.pkl"
GLICKO_RATINGS_FILE = "glicko2_ratings.pkl"

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


def train_and_save(data_dir: str | Path, model_dir: str | Path) -> None:
    """Train both models and save to disk."""
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


def _match_to_league_code(match: Match) -> str:
    """Try to map a Match to a football-data.co.uk division code."""
    from toto_ai.data_collector.football_data_downloader import ALL_DIVISIONS

    league_lower = (match.league or "").lower()
    country_lower = (match.country or "").lower()

    for code, name in ALL_DIVISIONS.items():
        if name.lower() in league_lower or league_lower in name.lower():
            return code

    # Country-based fallback for top divisions
    country_map = {
        "england": "E0",
        "germany": "D1",
        "italy": "I1",
        "spain": "SP1",
        "france": "F1",
        "scotland": "SC0",
        "netherlands": "N1",
        "belgium": "B1",
        "portugal": "P1",
        "turkey": "T1",
        "greece": "G1",
    }
    for country, code in country_map.items():
        if country in country_lower or country in league_lower:
            return code

    return match.country or match.league or "unknown"


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

    league_code = _match_to_league_code(match)

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
        home_rest_days=stats.home_rest_days if stats else None,
        away_rest_days=stats.away_rest_days if stats else None,
        home_injury_count=len(stats.home_injuries) if stats else None,
        away_injury_count=len(stats.away_injuries) if stats else None,
        **pi_feats,
        **glicko_feats,
        **h2h_feats,
        **xg_feats,
        **standing_feats,
    )

    # Choose model: rich if the league is in main divisions and rich model exists
    from toto_ai.data_collector.football_data_downloader import ALL_DIVISIONS

    use_rich = rich_model is not None and league_code in ALL_DIVISIONS
    if use_rich:
        model = rich_model
        feat_list = RICH_FEATURES
        variant = "CatBoost (rich)"
    elif simple_model is not None:
        model = simple_model
        feat_list = SIMPLE_FEATURES
        variant = "CatBoost (simple)"
    else:
        return None

    # Build feature vector in correct order
    row = {}
    for f in feat_list:
        val = features.get(f)
        if f in CAT_FEATURES:
            row[f] = str(val) if val is not None else "unknown"
        else:
            row[f] = float(val) if val is not None else np.nan
    row_df = pd.DataFrame([row])

    cat_idx = [i for i, f in enumerate(feat_list) if f in CAT_FEATURES]
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
