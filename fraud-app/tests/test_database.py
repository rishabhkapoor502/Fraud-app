from src.config import FEATURE_COLUMNS
from src.database import clear_predictions, fetch_predictions, log_prediction, prediction_stats


def test_log_and_fetch_predictions(tmp_path):
    db_path = tmp_path / "history.db"
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    row_id = log_prediction(
        source="single",
        predicted_class=1,
        prediction_label="Potentially fraudulent",
        probability=0.91,
        risk_score=91.0,
        threshold=0.4,
        amount=120.5,
        time_value=15.0,
        features=features,
        path=db_path,
    )
    assert row_id >= 1
    stats = prediction_stats(path=db_path)
    assert stats["total"] == 1
    assert stats["flagged"] == 1
    rows = fetch_predictions(path=db_path)
    assert len(rows) == 1
    assert rows.iloc[0]["amount"] == 120.5
    clear_predictions(path=db_path)
    assert prediction_stats(path=db_path)["total"] == 0
