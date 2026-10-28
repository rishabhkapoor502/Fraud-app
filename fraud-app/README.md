# Credit Card Fraud Detection Using Deep Learning

College project that trains a feed-forward artificial neural network on the ULB Credit Card Fraud Detection dataset and serves predictions through a Streamlit application.

The application **does not retrain** when it starts. It loads a model and preprocessing artifacts saved by `train_model.py`. Predictions are **estimates**, not confirmed fraud.

## Project layout

```
app.py                 Streamlit entry point
train_model.py         Training pipeline
src/                   Preprocessing, model, evaluation, prediction, SQLite
app_pages/             Dashboard and other UI pages
data/                  Place creditcard.csv here
models/                Saved ANN, scaler, metrics
tests/                 pytest suite
```

## Requirements

- Python 3.10–3.12 is recommended for TensorFlow wheels. Newer versions may work if TensorFlow provides a build.
- About 150 MB disk space for the dataset plus model files.

## Setup

```bash
cd credit-card-fraud-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

1. Download [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file at `data/creditcard.csv`.

Do not invent a substitute CSV. Training will refuse to run if the file is missing.

## Train the model

```bash
python train_model.py
```

Optional arguments:

```bash
python train_model.py --data data/creditcard.csv --epochs 30 --batch-size 2048
```

This command:

- Cleans the dataset
- Creates stratified train / validation / test splits
- Fits a scaler on **training features only**
- Trains a Keras ANN with class weights and callbacks
- Chooses a classification threshold on **validation** F1-score
- Evaluates precision, recall, F1, ROC-AUC, PR-AUC, and the confusion matrix on the **test** set
- Writes files under `models/`

Training on CPU typically takes several minutes.

## Run the web application

```bash
streamlit run app.py
```

Pages:

1. Dashboard
2. Single Transaction
3. Batch Analysis
4. Model Performance
5. Transaction History
6. About Project

If the dataset or trained model is missing, the UI shows a clear error instead of fake metrics.

## Tests

```bash
pytest
```

These tests cover validation, splits, scaling, threshold helpers, and SQLite history. They do not invent ULB dataset metrics.

## Features used by the model

Inputs: `Time`, `V1`–`V28`, `Amount`.  
`Class` is the label only. It is never passed into the network.

## Limitation statement

This is an academic demonstration. A high fraud probability is not proof of crime and must not be used as the only reason to block a customer.
