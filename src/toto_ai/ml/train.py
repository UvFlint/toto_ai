"""Standalone training entry point for CatBoost models."""

from __future__ import annotations

from pathlib import Path

from toto_ai.console import console
from toto_ai.ml.catboost_model import train_and_save as catboost_train_and_save
from toto_ai.ml.catboost_model import train_draw_league_group_models, train_draw_model
from toto_ai.ml.catboost_model import train_league_group_models as catboost_train_groups

DRAW_MODEL_FILE = "catboost_draw.cbm"


def run_training(data_dir: str, model_dir: str) -> None:
    """Train CatBoost ML models and save them."""
    cb_metrics = catboost_train_and_save(data_dir=data_dir, model_dir=model_dir)
    # Train league-group specific models (only used at inference if they beat global)
    catboost_train_groups(
        data_dir=data_dir, model_dir=model_dir, global_rich_accuracy=cb_metrics["accuracy"]
    )

    # Train draw detection binary classifier
    console.print("\n[bold blue]Training Draw Detection Classifier[/bold blue]")
    console.print("=" * 50)
    draw_model, draw_metrics, _, _ = train_draw_model(data_dir=data_dir)
    draw_path = Path(model_dir) / DRAW_MODEL_FILE
    draw_model.save_model(str(draw_path))
    console.print(f"[green]Draw classifier saved → {draw_path}[/green]")
    console.print(f"  Validation accuracy: [bold]{draw_metrics['accuracy']:.1%}[/bold]")
    console.print(f"  Draw recall: [bold]{draw_metrics['draw_recall']:.1%}[/bold]")

    # Train per-league draw models (compare by draw recall, not accuracy)
    train_draw_league_group_models(
        data_dir=data_dir,
        model_dir=model_dir,
        global_draw_accuracy=draw_metrics["draw_recall"],
    )

    console.print("\n[bold green]All models trained successfully![/bold green]")
