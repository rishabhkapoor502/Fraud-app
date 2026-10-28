from src.config import DATASET_PATH
from src.preprocessing import DatasetNotFoundError, load_raw_dataset


def test_missing_dataset_has_clear_error():
    if DATASET_PATH.exists():
        df = load_raw_dataset()
        assert not df.empty
        return
    try:
        load_raw_dataset()
        raise AssertionError("Expected DatasetNotFoundError when CSV is absent")
    except DatasetNotFoundError as exc:
        assert "creditcard.csv" in str(exc)
