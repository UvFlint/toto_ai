"""Standalone training entry point for CatBoost models."""

from __future__ import annotations

from toto_ai.ml.catboost_model import train_and_save as catboost_train_and_save
from toto_ai.ml.catboost_model import train_league_group_models as catboost_train_groups


def run_training(data_dir: str, model_dir: str) -> None:
    """Train CatBoost ML models and save them."""
    cb_metrics = catboost_train_and_save(data_dir=data_dir, model_dir=model_dir)
    # Train league-group specific models (only used at inference if they beat global)
    catboost_train_groups(
        data_dir=data_dir, model_dir=model_dir, global_rich_accuracy=cb_metrics["accuracy"]
    )
