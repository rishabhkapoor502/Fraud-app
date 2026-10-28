#!/usr/bin/env python3
"""Train the fraud detection ANN and save model artifacts.

Usage:
    python train_model.py
    python train_model.py --data data/creditcard.csv --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DATASET_PATH, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from src.preprocessing import DatasetNotFoundError
from src.validation import DataValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit card fraud ANN.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH, help="Path to creditcard.csv")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Starting training pipeline...")
    print(f"Dataset: {args.data}", flush=True)
    if not args.data.exists():
        print(
            f"ERROR: Dataset not found at {args.data}. "
            "Download the ULB Credit Card Fraud Detection CSV from Kaggle "
            "and save it as data/creditcard.csv.",
            file=sys.stderr,
        )
        return 1

    from src.training import train_pipeline

    try:
        artifacts = train_pipeline(
            dataset_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    test_metrics = artifacts["test_metrics"]
    print("\nTraining complete. Hold-out test metrics:")
    for key in ("precision", "recall", "f1_score", "roc_auc", "pr_auc"):
        print(f"  {key}: {test_metrics[key]:.4f}")
    print(f"  threshold: {test_metrics['threshold']:.6f}")
    print("Artifacts saved under models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
