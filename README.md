# Corporate Default Prediction & Data Separation (`zbiór_4.csv`)

This repository contains the Exploratory Data Analysis (EDA), leakage-safe corporate partitioning, and feature preparation for corporate default prediction.

---

## 1. Generated Partitions & Metadata

| File | Shape | Default Count | Purpose |
| :--- | :--- | :--- | :--- |
| **`development_set.csv`** | 2,401 rows $\times$ 173 cols | 144 defaults (6.00%) | 80% Development partition with 5 inner CV folds |
| **`test_holdout_set.csv`** | 599 rows $\times$ 172 cols | 36 defaults (6.01%) | 20% Sealed holdout test set for final unbiased benchmark |
| **`selected_features.json`** | Metadata JSON | — | Feature lists: top 20 modeling features, 168 retained, 49 dropped |
| **`eda_and_data_separation.ipynb`** | Notebook | — | Complete executable analysis from raw data to saved splits |
| **`interpretable_model.ipynb`** | Notebook | — | Interpretable Scorecard & Decision Tree (XAI, metrics, stability) |

### Key Columns in `development_set.csv`:
* **`default`**: Binary target (0 = healthy firm, 1 = bankrupt / default).
* **`cv_fold`**: Integer fold assignment (`0, 1, 2, 3, 4`).
* **`company_group_id`**: Composite corporate identifier. All consecutive annual filings of the same company share the same ID and are strictly assigned to the same fold.

---

## 2. How to Use the Fold Separations

To ensure zero data leakage, iterate through the 5 cross-validation folds using the `cv_fold` column:

```python
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Load the development set and top feature list
dev_df = pd.read_csv('development_set.csv')
with open('selected_features.json', 'r') as f:
    features = json.load(f)['top_selected_features']

# 2. Iterate across the 5 cross-validation folds
for fold in range(5):
    # Training folds (4 folds ~ 1,920 rows)
    train_split = dev_df[dev_df['cv_fold'] != fold]
    X_train = train_split[features]
    y_train = train_split['default']

    # Validation fold (1 fold ~ 480 rows)
    val_split = dev_df[dev_df['cv_fold'] == fold]
    X_val = val_split[features]
    y_val = val_split['default']

    # Preprocessing (fitted strictly on training split)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
```

---


## 3. Setup & Installation

```bash
pip install -r requirements.txt
```

