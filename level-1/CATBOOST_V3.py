"""
KEBA AI CODING CONTEST 2026 - CATBOOST ONLY
===========================================
Same procedure as XGB+LGB, but using CatBoost alone

"""

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("KEBA AI CODING CONTEST 2026 - CATBOOST ONLY")
print("="*80)

# Load and split
train_df = pd.read_csv('level_1_train.csv', sep=';')
test_df = pd.read_csv('level_1_test_submission.csv', sep=';')

train_df['ds'] = pd.to_datetime(train_df['ds'])
test_df['ds'] = pd.to_datetime(test_df['ds'])
train_df = train_df.sort_values('ds').reset_index(drop=True)
test_df = test_df.sort_values('ds').reset_index(drop=True)

# Proper chronological split by DATE
split_date = pd.Timestamp('2025-03-15 23:59:59')

train_split = train_df[train_df['ds'] <= split_date].copy()
val_split = train_df[train_df['ds'] > split_date].copy()

print(f"\n[1/6] Data split (chronological by date)")
print(f"  Train: {len(train_split):,} samples ({train_split['ds'].min()} to {train_split['ds'].max()})")
print(f"    Charging rate: {train_split['y_b'].mean():.4f}")
print(f"  Val:   {len(val_split):,} samples ({val_split['ds'].min()} to {val_split['ds'].max()})")
print(f"    Charging rate: {val_split['y_b'].mean():.4f}")
print(f"  Test:  {len(test_df):,} samples ({test_df['ds'].min()} to {test_df['ds'].max()})")

# Features
print(f"\n[2/6] Feature engineering...")

def add_features(df):
    df = df.copy()
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(8, 17).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_holiday'] = df['month'].isin([11, 12]).astype(int)
    
    if 'y_b' in df.columns:
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df.groupby('unique_id')['y_b'].shift(lag).fillna(0)
    else:
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = 0
    
    return df

train_split = add_features(train_split)
val_split = add_features(val_split)
test_df = add_features(test_df)

# User features (from train only)
global_mean = train_split['y_b'].mean()
user_rate = train_split.groupby('unique_id')['y_b'].mean().to_dict()

for df in [train_split, val_split, test_df]:
    df['user_rate'] = df['unique_id'].map(user_rate).fillna(global_mean)

cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'is_holiday', 'user_rate', 'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12']

X_train = train_split[cols].fillna(0)
y_train = train_split['y_b']
X_val = val_split[cols].fillna(0)
y_val = val_split['y_b']
X_test = test_df[cols].fillna(0)

print(f"  {len(cols)} features created")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

# Train CatBoost ONLY
print(f"\n[3/6] Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.01, depth=6,
    subsample=0.8, random_state=42, verbose=False,
    scale_pos_weight=scale_pos
)
cat_model.fit(X_train, y_train)

cat_train = cat_model.predict_proba(X_train)[:, 1]
cat_val = cat_model.predict_proba(X_val)[:, 1]

cat_train_brier = brier_score_loss(y_train, cat_train)
cat_val_brier = brier_score_loss(y_val, cat_val)

print(f"  Train Brier: {cat_train_brier:.5f}")
print(f"  Val Brier:   {cat_val_brier:.5f}")

# Validation
print(f"\n[4/6] Validation results")
baseline_brier = brier_score_loss(y_val, np.full(len(y_val), y_train.mean()))

print(f"  CatBoost Brier: {cat_val_brier:.5f}")
print(f"  Baseline:       {baseline_brier:.5f}")
print(f"  Target:         0.0249")
print(f"  Result:         {'Beats target' if cat_val_brier < 0.0249 else 'Does not beat target'}")

# Test predictions
print(f"\n[5/6] Test predictions...")
cat_test = cat_model.predict_proba(X_test)[:, 1]

test_pred = np.clip(cat_test, 0.0001, 0.9999)

print(f"  Mean: {test_pred.mean():.6f}")
print(f"  Std:  {test_pred.std():.6f}")
print(f"  Min:  {test_pred.min():.6f}")
print(f"  Max:  {test_pred.max():.6f}")

# Save
print(f"\n[6/6] Saving submission...")
submission_df = test_df[['unique_id', 'ds']].copy()
submission_df['y_prob'] = test_pred
submission_df.to_csv('level_1_catboost_submission.csv', sep=';', index=False)

print(f"  Saved: level_1_catboost_submission.csv")

print("\n" + "="*80)
print("FINAL RESULTS - CATBOOST ONLY")
print("="*80)
print(f"""
VALIDATION:
  CatBoost Brier: {cat_val_brier:.5f}
  Baseline:       {baseline_brier:.5f}
  Target:         0.0249
  
COMPARISON:
  Improvement vs baseline: {(baseline_brier - cat_val_brier):.5f}
  vs target: {'Beats target' if cat_val_brier < 0.0249 else 'Below target by ' + str(round(cat_val_brier - 0.0249, 5))}

TEST PREDICTIONS:
  Mean: {test_pred.mean():.6f}
  Std:  {test_pred.std():.6f}
  Range: [{test_pred.min():.6f}, {test_pred.max():.6f}]

SUBMISSION: level_1_catboost_submission.csv
""")
print("="*80)