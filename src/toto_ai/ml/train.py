"""Standalone training entry point for CatBoost and XGBoost models."""

from __future__ import annotations

from toto_ai.ml.catboost_model import train_and_save as catboost_train_and_save
from toto_ai.ml.xgboost_model import train_and_save as xgboost_train_and_save


def run_training(data_dir: str, model_dir: str) -> None:
    """Train all ML models (CatBoost + XGBoost) and save them."""
    catboost_train_and_save(data_dir=data_dir, model_dir=model_dir)
    xgboost_train_and_save(data_dir=data_dir, model_dir=model_dir)
