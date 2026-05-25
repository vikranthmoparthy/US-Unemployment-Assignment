import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('us_macro_engineered.csv', index_col='Date', parse_dates=True)

X_train_full = df.drop(columns=['Target_Unemployment_1m_ahead', 'Unemployment_Rate'])
y_raw_full = df['Target_Unemployment_1m_ahead']

thresholds = [
    3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 
    4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
]

tscv = TimeSeriesSplit(n_splits=5, gap=3)

final_models = {}

for tau in thresholds:
    y_binary = (y_raw_full > tau).astype(int)
    
    if y_binary.sum() < 15:
        continue

    fold_brier_scores = []
    
    for train_index, test_index in tscv.split(X_train_full):
        X_train_fold, X_test_fold = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train_fold, y_test_fold = y_binary.iloc[train_index], y_binary.iloc[test_index]
        
        train_size = int(len(X_train_fold) * 0.8)
        
        X_fit = X_train_fold.iloc[:train_size]
        y_fit = y_train_fold.iloc[:train_size]
        X_cal = X_train_fold.iloc[train_size:]
        y_cal = y_train_fold.iloc[train_size:]
        
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
            continue
            
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=1000))
        ])
        
        pipeline.fit(X_fit, y_fit)
        
        calibrated_clf = CalibratedClassifierCV(estimator=pipeline, method='sigmoid', cv='prefit')
        calibrated_clf.fit(X_cal, y_cal)
        
        y_prob = calibrated_clf.predict_proba(X_test_fold)[:, 1]
        fold_brier_scores.append(brier_score_loss(y_test_fold, y_prob))
            
    split_idx = int(len(X_train_full) * 0.8)
    X_fit_final, y_fit_final = X_train_full.iloc[:split_idx], y_binary.iloc[:split_idx]
    X_cal_final, y_cal_final = X_train_full.iloc[split_idx:], y_binary.iloc[split_idx:]
    
    if len(np.unique(y_fit_final)) < 2 or len(np.unique(y_cal_final)) < 2:
        continue
        
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=1000))
    ])
    final_pipeline.fit(X_fit_final, y_fit_final)
    
    final_calibrated_clf = CalibratedClassifierCV(estimator=final_pipeline, method='sigmoid', cv='prefit')
    final_calibrated_clf.fit(X_cal_final, y_cal_final)
    
    final_models[tau] = {
        'model': final_calibrated_clf,
        'brier': np.mean(fold_brier_scores) if fold_brier_scores else None
    }

print("\n--- Final Predictions vs Market Implied ---")
may_2026_features = X_train_full.iloc[-1:].copy() 

print(f"{'Strike':<10} | {'Raw Prob':<10} | {'Smoothed Prob':<15} | {'Brier Score'}")
print("-" * 60)

raw_predictions = {}
brier_scores = {}

for tau in thresholds:
    if tau in final_models:
        model = final_models[tau]['model']
        raw_predictions[tau] = model.predict_proba(may_2026_features)[:, 1][0] * 100
        brier_scores[tau] = final_models[tau]['brier']

if raw_predictions:
    threshold_list = sorted(list(raw_predictions.keys()))
    raw_prob_list = [raw_predictions[t] for t in threshold_list]

    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
    smoothed_probs = ir.fit_transform(threshold_list, raw_prob_list)
    adjusted_predictions = dict(zip(threshold_list, smoothed_probs))

    for tau in threshold_list:
        raw = raw_predictions[tau]
        adj = adjusted_predictions[tau]
        brier = brier_scores[tau]
        
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        
        print(f"> {tau:.1f}%     | {raw:>5.2f}%     | {adj:>6.2f}%          | {brier_str}")
else:
    pass