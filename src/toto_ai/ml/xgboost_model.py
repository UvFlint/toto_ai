"""XGBoost match result classifier — train, load, predict.

Two models (mirroring CatBoost):
- **Rich**: trained on main leagues with match stats + odds features
- **Simple**: trained on all leagues with goals + odds features only
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from toto_ai.analyzer.models import FullColumn, MatchPrediction
from toto_ai.console import console
from toto_ai.ml.catboost_model import (
    GLICKO_RATINGS_FILE,
    PI_RATINGS_FILE,
    _extract_form_match_stats,
    _extract_form_stats,
    _extract_h2h_features,
    _extract_standing_features,
    _extract_standing_stats,
    _extract_xg_features,
    _match_to_league_code,
)
from toto_ai.ml.glicko2 import (
    GlickoStore,
    compute_glicko2_features,
    load_glicko_store,
    save_glicko_store,
)
from toto_ai.ml.pi_ratings import RatingStore, compute_pi_features, load_ratings, save_ratings
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
from toto_ai.scraper.models import Match
from toto_ai.stats.models import MatchStats

RICH_MODEL_FILE = "xgboost_rich.json"
SIMPLE_MODEL_FILE = "xgboost_simple.json"
LABEL_ENCODERS_FILE = "xgb_label_encoders.pkl"

PREDICTION_MAP = {0: "1", 1: "X", 2: "2"}


def _encode_categoricals(
    df: pd.DataFrame,
    cat_features: list[str],
    encoders: dict | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns for XGBoost.

    When fit=True, fits new encoders. Otherwise uses provided encoders.
    Unseen categories at inference time are mapped to a fallback code.
    """
    from sklearn.preprocessing import LabelEncoder

    if encoders is None:
        encoders = {}

    df = df.copy()
    for col in cat_features:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("unknown").astype(str)

        if fit:
            le = LabelEncoder()
            # Add "unknown" to ensure it's always in the encoder
            all_vals = list(df[col].unique()) + ["unknown"]
            le.fit(all_vals)
            encoders[col] = le

        le = encoders[col]
        # Map unseen values to "unknown"
        known = set(le.classes_)
        df[col] = df[col].map(lambda x, k=known: x if x in k else "unknown")
        df[col] = le.transform(df[col])

    return df, encoders


def _prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    encoders: dict | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, dict]:
    """Prepare X, y, sample_weights for XGBoost training."""
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df["target"].values

    X, encoders = _encode_categoricals(X, cat_features, encoders, fit=fit)

    # Class weights as sample weights
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    weight_map = {c: total / (3 * count) for c, count in class_counts.items()}
    sample_weights = np.array([weight_map[yi] for yi in y])

    return X, y, sample_weights, encoders


def train_rich_model(
    data_dir: str | Path,
) -> tuple[XGBClassifier, dict, dict, RatingStore, GlickoStore]:
    """Train the rich model on main league data. Returns (model, metrics, encoders, pi_store, glicko_store)."""
    console.print("[bold]Loading main league data...[/bold]")
    df = load_main_league_data(data_dir)
    if df.empty:
        raise RuntimeError("No main league data found. Run --download-data first.")

    console.print(f"  Loaded {len(df):,} matches from main leagues")
    df, pi_store, glicko_store = engineer_rich_features(df)
    df = df.dropna(subset=["h_roll_gf", "a_roll_gf"])
    console.print(f"  After feature engineering: {len(df):,} matches")

    # Train/val split: last season as validation
    seasons = sorted(df["season"].unique())
    val_season = seasons[-1] if len(seasons) > 1 else None

    if val_season:
        train_df = df[df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

    console.print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    X_train, y_train, w_train, encoders = _prepare_xy(
        train_df, RICH_FEATURES, CAT_FEATURES, fit=True
    )
    X_val, y_val, _, _ = _prepare_xy(
        val_df, RICH_FEATURES, CAT_FEATURES, encoders=encoders, fit=False
    )

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        early_stopping_rounds=50,
        random_state=42,
        verbosity=1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_preds = model.predict(X_val)
    accuracy = np.mean(val_preds == y_val)
    metrics = {"accuracy": float(accuracy), "train_size": len(train_df), "val_size": len(val_df)}

    return model, metrics, encoders, pi_store, glicko_store


def train_simple_model(
    data_dir: str | Path,
    encoders: dict | None = None,
) -> tuple[XGBClassifier, dict, dict, RatingStore, GlickoStore]:
    """Train the simple model on all data. Returns (model, metrics, encoders, pi_store, glicko_store)."""
    console.print("[bold]Loading all league data...[/bold]")
    main_df = load_main_league_data(data_dir)
    extra_df = load_extra_league_data(data_dir)

    frames = [df for df in [main_df, extra_df] if not df.empty]
    if not frames:
        raise RuntimeError("No data found. Run --download-data first.")

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

    seasons = sorted(df["season"].unique())
    val_season = seasons[-1] if len(seasons) > 1 else None

    if val_season:
        train_df = df[df["season"] != val_season]
        val_df = df[df["season"] == val_season]
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

    console.print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Fit new encoders if none provided (standalone training)
    fit = encoders is None
    X_train, y_train, w_train, encoders = _prepare_xy(
        train_df, SIMPLE_FEATURES, CAT_FEATURES, encoders=encoders, fit=fit
    )
    X_val, y_val, _, _ = _prepare_xy(
        val_df, SIMPLE_FEATURES, CAT_FEATURES, encoders=encoders, fit=False
    )

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        early_stopping_rounds=50,
        random_state=42,
        verbosity=1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_preds = model.predict(X_val)
    accuracy = np.mean(val_preds == y_val)
    metrics = {"accuracy": float(accuracy), "train_size": len(train_df), "val_size": len(val_df)}

    return model, metrics, encoders, pi_store, glicko_store


def train_and_save(data_dir: str | Path, model_dir: str | Path) -> None:
    """Train both XGBoost models and save to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Training Rich XGBoost Model[/bold blue]")
    console.print("=" * 50)
    rich_model, rich_metrics, encoders, pi_store, glicko_store = train_rich_model(data_dir)
    rich_path = model_dir / RICH_MODEL_FILE
    rich_model.save_model(str(rich_path))
    console.print(f"[green]Rich model saved → {rich_path}[/green]")
    console.print(f"  Validation accuracy: [bold]{rich_metrics['accuracy']:.1%}[/bold]")

    console.print("\n[bold blue]Training Simple XGBoost Model[/bold blue]")
    console.print("=" * 50)
    simple_model, simple_metrics, encoders, simple_pi_store, simple_glicko_store = (
        train_simple_model(data_dir, encoders=encoders)
    )
    simple_path = model_dir / SIMPLE_MODEL_FILE
    simple_model.save_model(str(simple_path))
    console.print(f"[green]Simple model saved → {simple_path}[/green]")
    console.print(f"  Validation accuracy: [bold]{simple_metrics['accuracy']:.1%}[/bold]")

    # Save label encoders
    enc_path = model_dir / LABEL_ENCODERS_FILE
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    console.print(f"[green]Label encoders saved → {enc_path}[/green]")

    # Save pi-ratings (merge both stores, simple covers more teams)
    pi_path = model_dir / PI_RATINGS_FILE
    merged_store = {**simple_pi_store, **pi_store}
    save_ratings(merged_store, pi_path)
    console.print(f"[green]Pi-ratings saved → {pi_path} ({len(merged_store)} teams)[/green]")

    # Save Glicko-2 ratings
    glicko_path = model_dir / GLICKO_RATINGS_FILE
    merged_glicko = {**simple_glicko_store, **glicko_store}
    save_glicko_store(merged_glicko, glicko_path)
    console.print(f"[green]Glicko-2 saved → {glicko_path} ({len(merged_glicko)} teams)[/green]")

    console.print("\n[bold green]Both XGBoost models trained successfully![/bold green]")


def load_models(
    model_dir: str | Path,
) -> tuple[
    XGBClassifier | None,
    XGBClassifier | None,
    dict | None,
    RatingStore | None,
    GlickoStore | None,
]:
    """Load trained models, encoders, pi-ratings, and Glicko-2 ratings from disk."""
    model_dir = Path(model_dir)
    rich_model = simple_model = None
    encoders = None
    pi_store = None
    glicko_store = None

    rich_path = model_dir / RICH_MODEL_FILE
    if rich_path.exists():
        rich_model = XGBClassifier()
        rich_model.load_model(str(rich_path))

    simple_path = model_dir / SIMPLE_MODEL_FILE
    if simple_path.exists():
        simple_model = XGBClassifier()
        simple_model.load_model(str(simple_path))

    enc_path = model_dir / LABEL_ENCODERS_FILE
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)  # noqa: S301

    pi_path = model_dir / PI_RATINGS_FILE
    if pi_path.exists():
        pi_store = load_ratings(pi_path)

    glicko_path = model_dir / GLICKO_RATINGS_FILE
    if glicko_path.exists():
        glicko_store = load_glicko_store(glicko_path)

    return rich_model, simple_model, encoders, pi_store, glicko_store


def predict_match(
    match: Match,
    stats: MatchStats | None,
    rich_model: XGBClassifier | None,
    simple_model: XGBClassifier | None,
    encoders: dict | None,
    pi_store: RatingStore | None = None,
    glicko_store: GlickoStore | None = None,
) -> tuple[dict[str, float], str] | None:
    """Predict match probabilities using the best available model."""
    if rich_model is None and simple_model is None:
        return None
    if encoders is None:
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

    # Choose model
    from toto_ai.data_collector.football_data_downloader import ALL_DIVISIONS

    use_rich = rich_model is not None and league_code in ALL_DIVISIONS
    if use_rich:
        model = rich_model
        feat_list = RICH_FEATURES
        variant = "XGBoost (rich)"
    elif simple_model is not None:
        model = simple_model
        feat_list = SIMPLE_FEATURES
        variant = "XGBoost (simple)"
    else:
        return None

    # Build feature vector
    row = {}
    for f in feat_list:
        val = features.get(f)
        if f in CAT_FEATURES:
            str_val = str(val) if val is not None else "unknown"
            le = encoders.get(f)
            if le is not None:
                known = set(le.classes_)
                str_val = str_val if str_val in known else "unknown"
                row[f] = int(le.transform([str_val])[0])
            else:
                row[f] = 0
        else:
            row[f] = float(val) if val is not None else np.nan

    row_df = pd.DataFrame([row])
    proba = model.predict_proba(row_df)[0]

    return {
        "1": round(float(proba[0]), 4),
        "X": round(float(proba[1]), 4),
        "2": round(float(proba[2]), 4),
    }, variant


def build_xgboost_column(
    matches: list[Match],
    stats: list[MatchStats],
) -> FullColumn:
    """Build a FullColumn from XGBoost predictions for consensus voting."""
    from toto_ai.config import settings

    rich_model, simple_model, encoders, pi_store, glicko_store = load_models(settings.MODEL_DIR)
    if rich_model is None and simple_model is None:
        console.print("[yellow]XGBoost: No trained models found. Run --train-model first.[/yellow]")
        return FullColumn(model_name="XGBoost", column_type="ml")

    predictions: list[MatchPrediction] = []

    for match in matches:
        s = next((st for st in stats if st.home_team == match.home_team), None)
        result = predict_match(match, s, rich_model, simple_model, encoders, pi_store, glicko_store)

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

    console.print(f"[green]XGBoost model: {len(predictions)} predictions generated[/green]")
    return FullColumn(model_name="XGBoost", predictions=predictions, column_type="ml")


def enrich_stats_with_xgboost(
    matches: list[Match],
    stats: list[MatchStats],
) -> None:
    """Attach XGBoost probabilities to each MatchStats for LLM prompt enrichment."""
    from toto_ai.config import settings

    rich_model, simple_model, encoders, pi_store, glicko_store = load_models(settings.MODEL_DIR)
    if rich_model is None and simple_model is None:
        return

    for match in matches:
        s = next((st for st in stats if st.home_team == match.home_team), None)
        if s is None:
            continue

        result = predict_match(match, s, rich_model, simple_model, encoders, pi_store, glicko_store)
        if result is not None:
            probs, _ = result
            s.xgboost_probs = probs
