# Viva questions and answers

Use these during a college viva. Numbers for precision, recall, or AUC should be read from `models/training_artifacts.json` after training, not memorised from this file.

## 1. What is the objective of the project?

To estimate whether a credit-card transaction is legitimate or potentially fraudulent using a feed-forward ANN trained on the ULB dataset, and to demonstrate the model in a local Streamlit application.

## 2. Which features does the model use?

`Time`, `V1` through `V28`, and `Amount`. `Class` is the target label only.

## 3. Why are V1–V28 not human-readable?

The dataset publisher applied PCA to protect confidentiality. The components are linear combinations of original variables that were not released.

## 4. Why not use accuracy as the main metric?

Fraud is rare. A classifier that always predicts class 0 can have very high accuracy while catching no fraud. Precision, recall, F1, ROC-AUC, and especially PR-AUC are more informative.

## 5. What is precision in this setting?

Of the transactions flagged as fraud, the fraction that are actually fraud.

## 6. What is recall?

Of the actual fraud cases, the fraction the model flags.

## 7. Why report both ROC-AUC and PR-AUC?

ROC-AUC summarises ranking quality across thresholds using TPR and FPR. PR-AUC focuses on the positive class and is usually more telling under severe imbalance.

## 8. How is class imbalance handled?

Balanced class weights are computed from the training labels so that fraud errors contribute more to the loss. The validation threshold is then tuned with F1 rather than accuracy.

## 9. Why three splits instead of two?

Train fits the network and the scaler. Validation is used for early stopping and threshold selection. Test is used once for an unbiased estimate after those choices are frozen.

## 10. What would count as data leakage here?

Fitting the scaler on the full dataset, using `Class` as an input, tuning the threshold on the test set, or dropping duplicates after splitting in a way that mixes information across sets.

## 11. Why StandardScaler?

`Time` and `Amount` have different magnitudes from the PCA features. Gradient-based training is more stable when inputs are standardised. The scaler is fitted on train only.

## 12. Why a feed-forward ANN rather than logistic regression or a tree?

The project requirement is to implement a deep-learning classifier. An ANN can model non-linear combinations of the PCA features. Simpler models remain valid baselines in a longer study but are not the deployed model here.

## 13. Describe the architecture.

Input of 30 features; dense 64 + batch norm + dropout; dense 32 + batch norm + dropout; dense 16 + dropout; sigmoid output.

## 14. Why sigmoid and binary cross-entropy?

The task is binary classification. Sigmoid maps to a probability in (0, 1). Binary cross-entropy penalises confident wrong probabilities.

## 15. What callbacks are used?

Early stopping on validation PR-AUC with restore-best-weights, and reduce-on-plateau for the learning rate using validation loss.

## 16. How is the decision threshold chosen?

Predicted probabilities on the validation set are swept using the precision-recall curve. The threshold with the highest F1-score is saved and later applied to test data and to the app.

## 17. Does the Streamlit app train the model?

No. Training is a separate command. The app loads `models/fraud_ann.keras` and the scaler. If they are missing, it shows an error.

## 18. How does batch upload validation work?

The CSV must be non-empty, parseable, and must contain the 30 feature columns with numeric values and no missing entries. A `Class` column, if present, is dropped.

## 19. Where is prediction history stored?

In a local SQLite database at `data/prediction_history.db`.

## 20. What are the ethical limits of the demo?

Outputs are statistical estimates. They are not proof of fraud, not a legal finding, and not suitable as the only reason to block a genuine customer.

## 21. What happens if `creditcard.csv` is absent?

`train_model.py` exits with a clear error. The dashboard reports that the dataset is missing. No synthetic ULB data is generated.

## 22. Why not SMOTE in this project?

Class weights keep the original rows and avoid manufacturing extra fraud samples. That is simpler to defend in a viva and sufficient for the demonstration.

## 23. Can Time and Amount leak customer identity?

They are weak identifiers by themselves, but students still should not mix this demo with real production cardholder files.

## 24. What would you improve next?

A cost-sensitive threshold, comparison with gradient boosting, calibration plots, and a time-based split that respects chronology more strictly than a random stratified split.

## 25. Why is a random split not perfect for this data?

Transactions are ordered in time. A chronological split can better mimic scoring future payments. Stratified random splitting is still used here because it is standard, simple, and preserves class balance in a student project.
