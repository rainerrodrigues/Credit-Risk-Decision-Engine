import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_score, 
    recall_score
)
# sklearn.set_config(transform_output="pandas")

os.makedirs('Models', exist_ok=True)

DATA_PATHS = {
    'train': 'data/data_imputation_train.csv',
    'val': 'data/data_imputation_val.csv',
    'test': 'data/data_imputation_test.csv'
}

AUTO_ACCEPT_THRESHOLD = 0.25
AUTO_REJECT_THRESHOLD = 0.60

LEAKAGE_COLS = ['int_rate', 'sub_grade', 'grade', 'issue_d']

def load_and_clean(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['target'] + [c for c in LEAKAGE_COLS if c in df.columns])
    y = df['target']
    return X, y

def build_pipeline(X_sample):
    num_cols = X_sample.select_dtypes(include=['number']).columns
    cat_cols = X_sample.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 33,          
    'max_depth': 7,             
    'n_estimators': 256,        
    'learning_rate': 0.1272,    
    'reg_alpha': 0.3134,        
    'reg_lambda': 0.4182,       
    'min_child_samples': 28,    
    'random_state': 42
}

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**final_params))
    ])

def classify_loan(prob):
    if prob < AUTO_ACCEPT_THRESHOLD:
        return 'AUTOMATIC ACCEPT', 'Low risk profile.'
    elif prob > AUTO_REJECT_THRESHOLD:
        return 'AUTOMATIC REJECT', 'High default probability.'
    else:
        return 'MANUAL REVIEW', 'Moderate risk; needs underwriting.'
    
def evaluate_model(name, y_true, y_probs, threshold=0.5):
    """Prints a comprehensive suite of metrics for a given split."""
    y_pred = (y_probs >= threshold).astype(int)

    for name, (X, y) in splits.items():
        probs = pipeline.predict_proba(X)[:, 1]
        # Evaluate at the rejection line to see the 'True' business impact
        evaluate_model(name, y, probs, threshold=AUTO_REJECT_THRESHOLD)
    
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs) 
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    print(f"\n--- {name.upper()} METRICS ---")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"PR-AUC (Avg Prec):  {ap:.4f}")
    print(f"F1-Score:           {f1:.4f} (at {threshold} threshold)")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")

if __name__ == "__main__":
    # 1. Load Data
    X_train, y_train = load_and_clean(DATA_PATHS['train'])
    X_val, y_val = load_and_clean(DATA_PATHS['val'])
    X_test, y_test = load_and_clean(DATA_PATHS['test'])

    pipeline = build_pipeline(X_train)
    print("Starting final model training...")
    pipeline.fit(X_train, y_train)
    
    joblib.dump(pipeline, 'Models/finalized_credit_model.joblib')
    print("Model saved to Models/finalized_credit_model.joblib")

    splits = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }

    for name, (X, y) in splits.items():
        probs = pipeline.predict_proba(X)[:, 1]
        evaluate_model(name, y, probs)

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_results = pd.DataFrame({'prob': test_probs, 'target': y_test})
    test_results['segment'] = test_results['prob'].apply(lambda x: classify_loan(x)[0])
    
    print("\n--- TEST DATA SEGMENT DISTRIBUTION ---")
    print(test_results['segment'].value_counts(normalize=True).map(lambda n: f"{n:.2%}"))
