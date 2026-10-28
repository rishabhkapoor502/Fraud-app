"""SQLite persistence for prediction history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DATABASE_PATH, FEATURE_COLUMNS


def _connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or DATABASE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(path: Path | None = None) -> None:
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                probability REAL NOT NULL,
                risk_score REAL NOT NULL,
                threshold REAL NOT NULL,
                amount REAL,
                time_value REAL,
                features_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def log_prediction(
    *,
    source: str,
    predicted_class: int,
    prediction_label: str,
    probability: float,
    risk_score: float,
    threshold: float,
    amount: float | None,
    time_value: float | None,
    features: dict,
    path: Path | None = None,
) -> int:
    initialize_database(path)
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                source, predicted_class, prediction_label, probability,
                risk_score, threshold, amount, time_value, features_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                int(predicted_class),
                prediction_label,
                float(probability),
                float(risk_score),
                float(threshold),
                None if amount is None else float(amount),
                None if time_value is None else float(time_value),
                json.dumps(features),
                created_at,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def log_prediction_frame(
    results: pd.DataFrame,
    source: str,
    path: Path | None = None,
    max_rows: int | None = None,
) -> int:
    if max_rows is not None:
        results = results.head(max_rows)
    count = 0
    for _, row in results.iterrows():
        log_prediction(
            source=source,
            predicted_class=int(row["predicted_class"]),
            prediction_label=str(row["prediction_label"]),
            probability=float(row["fraud_probability"]),
            risk_score=float(row["risk_score"]),
            threshold=float(row["threshold_used"]),
            amount=float(row["Amount"]),
            time_value=float(row["Time"]),
            features={col: float(row[col]) for col in FEATURE_COLUMNS},
            path=path,
        )
        count += 1
    return count


def fetch_predictions(limit: int = 200, path: Path | None = None) -> pd.DataFrame:
    initialize_database(path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT id, source, predicted_class, prediction_label, probability,
                   risk_score, threshold, amount, time_value, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def prediction_stats(path: Path | None = None) -> dict:
    initialize_database(path)
    with _connect(path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        fraud = connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE predicted_class = 1"
        ).fetchone()[0]
        latest = connection.execute(
            "SELECT created_at FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "total": int(total),
        "flagged": int(fraud),
        "legitimate": int(total) - int(fraud),
        "latest": None if latest is None else latest[0],
    }


def clear_predictions(path: Path | None = None) -> None:
    initialize_database(path)
    with _connect(path) as connection:
        connection.execute("DELETE FROM predictions")
        connection.commit()
