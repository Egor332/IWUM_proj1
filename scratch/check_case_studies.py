import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

df_dev = pd.read_csv('data/development_set.csv')
df_test = pd.read_csv('data/test_holdout_set.csv')

MODEL_FEATURES = [
    'wsk_ROA',                             
    'wsk_pokrycie_zob_kr_gotowkowe_1',    
    'wsk_ogolnego_zadluzenia_1',           
    'Kapital_wlasny',                      
    'wsk_poziom_kapitalu_obrotowego_netto', 
    'wsk_rotacja_aktywow_1',               
    'wsk_liczba_dni_istnienia',             
    'wsk_index_nowe'                        
]

class MonotonicWoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=4):
        self.n_bins = n_bins
        self.bin_cutoffs_ = {}
        self.woe_maps_ = {}
        self.bin_stats_ = {}

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        self.columns_ = list(X.columns)
        for col in self.columns_:
            s = X[col].replace([np.inf, -np.inf], np.nan)
            try:
                _, bin_edges = pd.qcut(s.dropna(), q=self.n_bins, retbins=True, duplicates='drop')
            except Exception:
                _, bin_edges = pd.cut(s.dropna(), bins=self.n_bins, retbins=True)
            
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            self.bin_cutoffs_[col] = bin_edges
            
            binned = pd.cut(s, bins=bin_edges).astype(str)
            binned = binned.mask(X[col] == np.inf, 'INF_POS')
            binned = binned.mask(X[col] == -np.inf, 'INF_NEG')
            binned = binned.mask(X[col].isna(), 'MISSING')
            
            ct = pd.crosstab(binned, y)
            if 0 not in ct.columns:
                ct[0] = 0
            if 1 not in ct.columns:
                ct[1] = 0
            good = ct[0] + 0.5
            bad = ct[1] + 0.5
            d_good = good / good.sum()
            d_bad = bad / bad.sum()
            woe_map = np.log(d_good / d_bad).to_dict()
            self.woe_maps_[col] = woe_map
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        X_out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            s = X[col].replace([np.inf, -np.inf], np.nan)
            binned = pd.cut(s, bins=self.bin_cutoffs_[col]).astype(str)
            binned = binned.mask(X[col] == np.inf, 'INF_POS')
            binned = binned.mask(X[col] == -np.inf, 'INF_NEG')
            binned = binned.mask(X[col].isna(), 'MISSING')
            woe_val = binned.map(self.woe_maps_[col]).fillna(0.0)
            X_out[col] = woe_val
        return X_out

woe_encoder = MonotonicWoETransformer(n_bins=4).fit(df_dev[MODEL_FEATURES], df_dev['default'])
X_dev_woe = woe_encoder.transform(df_dev[MODEL_FEATURES])
X_test_woe = woe_encoder.transform(df_test[MODEL_FEATURES])

lr_scorecard = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
lr_scorecard.fit(X_dev_woe, df_dev['default'])

PDO = 20.0
FACTOR = PDO / np.log(2.0)
OFFSET = 600.0 - FACTOR * np.log(50.0)

m_features = len(MODEL_FEATURES)
beta_0 = lr_scorecard.intercept_[0]
betas = lr_scorecard.coef_[0]

points_matrix = pd.DataFrame(index=df_test.index)
for j, col in enumerate(MODEL_FEATURES):
    points_matrix[col] = -FACTOR * (betas[j] * X_test_woe[col] + beta_0 / m_features)

test_p_sc = lr_scorecard.predict_proba(X_test_woe)[:, 1]
total_credit_scores = (OFFSET + points_matrix.sum(axis=1)).round(1)

df_test_scored = df_test.copy()
df_test_scored['Credit_Score'] = total_credit_scores
df_test_scored['Predicted_Prob'] = test_p_sc

case_healthy = df_test_scored[df_test_scored['default'] == 0].sort_values('Predicted_Prob').index[0]
case_bankrupt = df_test_scored[df_test_scored['default'] == 1].sort_values('Predicted_Prob', ascending=False).index[0]
case_borderline = df_test_scored.iloc[(df_test_scored['Predicted_Prob'] - 0.15).abs().argsort()[:1]].index[0]

for name, idx in [("Healthy", case_healthy), ("Bankrupt", case_bankrupt), ("Borderline", case_borderline)]:
    print(f"=== {name} Company (Index {idx}) ===")
    print(f"Score: {df_test_scored.loc[idx, 'Credit_Score']}, Prob: {df_test_scored.loc[idx, 'Predicted_Prob']:.2%}")
    row_pts = points_matrix.loc[idx].round(1)
    for feat in MODEL_FEATURES:
        raw_val = df_test.loc[idx, feat]
        woe_val = X_test_woe.loc[idx, feat]
        pts = row_pts[feat]
        print(f"  {feat}: raw={raw_val:.2f}, WoE={woe_val:.3f} -> Points = {pts:+.1f}")
    print(f"  Offset = {OFFSET:.1f}")
    print(f"  Sum of Points + Offset = {(OFFSET + row_pts.sum()):.1f}")
    print()
