"""Project paths, feature names, and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_FILENAME = "creditcard.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

MODEL_PATH = MODELS_DIR / "fraud_ann.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ARTIFACTS_PATH = MODELS_DIR / "training_artifacts.json"
CURVES_PATH = MODELS_DIR / "evaluation_curves.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"

DATABASE_PATH = DATA_DIR / "prediction_history.db"

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *PCA_FEATURES, "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
# Remaining 70% is used for training after the two hold-out splits.

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
