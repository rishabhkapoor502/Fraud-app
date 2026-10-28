# Dataset placement

Download the ULB **Credit Card Fraud Detection** dataset from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Save the CSV in this folder with this exact name:

```
data/creditcard.csv
```

The file must contain the columns `Time`, `V1` through `V28`, `Amount`, and `Class`.

Optional Kaggle CLI download from the project root:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data
```
