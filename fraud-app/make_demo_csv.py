#!/usr/bin/env python3
"""Create a small demo CSV from the model's held-out TEST split.

Run from the project root (next to app.py), with the venv active:

    python make_demo_csv.py

It reads data/creditcard.csv, rebuilds the same train/validation/test split
used by train_model.py, and samples rows ONLY from the test part. These rows
were never used to train the model or choose the threshold, so a demo on them
is honest. The output keeps the Class column so Batch Analysis can compare
predictions with the real labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import RANDOM_STATE, TARGET_COLUMN
from src.preprocessing import clean_dataset, load_raw_dataset, stratified_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a demo CSV from the test split.")
    parser.add_argument("--genuine", type=int, default=90, help="genuine rows to include")
    parser.add_argument("--fraud", type=int, default=10, help="fraud rows to include")
    parser.add_argument("--out", type=Path, default=Path("demo_test_transactions.csv"))
    args = parser.parse_args()

    cleaned = clean_dataset(load_raw_dataset())
    splits = stratified_splits(cleaned)

    test = splits["x_test"].copy()
    test[TARGET_COLUMN] = splits["y_test"]

    genuine = test[test[TARGET_COLUMN] == 0]
    fraud = test[test[TARGET_COLUMN] == 1]
    n_genuine = min(args.genuine, len(genuine))
    n_fraud = min(args.fraud, len(fraud))

    demo = pd.concat(
        [
            genuine.sample(n_genuine, random_state=RANDOM_STATE),
            fraud.sample(n_fraud, random_state=RANDOM_STATE),
        ]
    )
    demo = demo.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    demo.to_csv(args.out, index=False)

    print(f"Saved {len(demo)} rows to {args.out} ({n_genuine} genuine, {n_fraud} fraud).")
    print("All rows come from the held-out test split.")


if __name__ == "__main__":
    main()
