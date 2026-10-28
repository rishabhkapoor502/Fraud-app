# Project Report: Credit Card Fraud Detection Using Deep Learning

## 1. Introduction

Credit-card fraud is rare compared with legitimate spending, but a missed fraud case is costly. This project builds a complete academic system that:

- Trains a feed-forward artificial neural network (ANN) on the public ULB Credit Card Fraud Detection dataset.
- Evaluates the model with metrics that remain meaningful under class imbalance.
- Serves predictions in a Streamlit application without retraining on every launch.

The software is intended for a college demonstration on a local machine. It is not a production fraud engine.

## 2. Problem statement

Given a transaction described by `Time`, anonymised PCA features `V1`–`V28`, and `Amount`, estimate the probability that the transaction belongs to class 1 (fraud) rather than class 0 (legitimate). A decision threshold converts that probability into a binary flag.

Accuracy is not an adequate objective. A model that always predicts “legitimate” would appear accurate because fraud is uncommon.

## 3. Dataset

Source: [ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The CSV must be stored as `data/creditcard.csv`. Typical properties of this dataset (from the dataset description, not from fabricated local results):

- Transactions of European cardholders in September 2013.
- Two days of activity.
- PCA-transformed confidential features `V1`–`V28`.
- `Time`: seconds elapsed between a transaction and the first transaction in the file.
- `Amount`: transaction amount.
- `Class`: 1 = fraud, 0 = legitimate.

This report does not quote trained precision, recall, or AUC figures unless they were produced by an actual run of `train_model.py` on the machine.

## 4. Methodology

### 4.1 Cleaning

Rows with missing values are dropped. Exact duplicate rows are removed. The label is checked so that only 0/1 values remain.

### 4.2 Splits

A stratified train / validation / test split is used (approximately 70% / 15% / 15%). Stratification keeps the fraud ratio similar in each part. The test set is held out before threshold selection.

### 4.3 Scaling without leakage

`StandardScaler` is fitted on the training features only, then applied to validation, test, and live inference data.

### 4.4 Class imbalance

Keras `class_weight="balanced"` equivalents are computed from the training labels. The minority (fraud) class receives a larger weight in the loss. Synthetic oversampling is not required for this demonstration and is not used, so no extra fake fraud rows are created.

### 4.5 Model

A fully connected network maps 30 scaled inputs through layers of 64, 32, and 16 ReLU units, with batch normalisation and dropout, to a sigmoid output. The loss is binary cross-entropy. Optimisation uses Adam. Callbacks:

- Early stopping on validation PR-AUC, restoring the best weights.
- Reduce learning rate on plateau of validation loss.

### 4.6 Threshold

A threshold is chosen on **validation** predicted probabilities by maximising F1-score. The test set is not used for this choice.

### 4.7 Evaluation

On the test set the project reports:

- Precision, recall, F1-score
- ROC-AUC and PR-AUC
- Confusion matrix
- Accuracy only as a secondary, easily misleading figure

PR-AUC is emphasised because it is more informative than ROC-AUC when positives are rare.

### 4.8 Persistence

The trained Keras model, scaler, threshold, split summary, test metrics, and ROC/PR curve points are written under `models/`. The Streamlit app loads these files and never fits the network at startup.

## 5. Application

The web interface (Aegis Fraud Monitor) includes:

1. **Dashboard** — dataset profile (if the CSV is present), model status, training loss (if saved), and local prediction counts.
2. **Single Transaction** — form for the 30 features.
3. **Batch Analysis** — CSV upload with column and type checks, on-screen results, and download.
4. **Model Performance** — stored test metrics and curves, not invented numbers.
5. **Transaction History** — SQLite log of scores produced by the app.
6. **About Project** — objective, data, method, stack, and limits.

Every scoring page states that outputs are estimates, not confirmed fraud.

## 6. Testing

`pytest` covers:

- Missing columns, empty files, non-numeric values, and dropped `Class` columns
- Stratified splits and scaler shapes
- Threshold and confusion-matrix helpers
- SQLite insert, fetch, and clear

Training is only executed when `data/creditcard.csv` is available.

## 7. Results

After a successful local training run, copy the printed test metrics from the terminal or from `models/training_artifacts.json` into this section. Do not paste placeholder scores.

If training has not been run, the result is: **no empirical metrics yet**.

## 8. Limitations and ethics

- PCA features hide original variables, so explanations are limited.
- A 2013 European sample may not match another country or year.
- False positives can harm genuine customers; false negatives miss fraud.
- The tool must not be presented as legal evidence.
- Students should not upload real customer card data to public clouds.

## 9. Conclusion

The project demonstrates a leakage-aware deep-learning pipeline for imbalanced fraud classification and a complete local web demo. The important academic points are the split design, scaling discipline, imbalance handling, threshold choice on validation data, and honest reporting when files are missing.
