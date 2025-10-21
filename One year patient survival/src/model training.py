"""model_training.py
Functions to train a stacked ensemble, evaluate it, save artifacts (model, scaler, feature columns).
"""
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from typing import Tuple, Dict


def prepare_data(df: pd.DataFrame, target_col: str = 'Survived_1_year', test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype('float32')
    X_test_scaled = scaler.transform(X_test).astype('float32')

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_stacked_classifier(n_jobs: int = -1, small: bool = True):
    # small=True reduces tree counts for quicker experimentation
    base_models = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nb', BernoulliNB())
    ]

    if small:
        base_models += [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('xgb', XGBClassifier(eval_metric='logloss', n_estimators=50, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=3))
        ]
    else:
        base_models += [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('xgb', XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42)),
            ('knn', KNeighborsClassifier())
        ]

    meta = LogisticRegression(max_iter=1000, random_state=42)
    stacked = StackingClassifier(estimators=base_models, final_estimator=meta, cv=3, n_jobs=n_jobs)
    return stacked


def train_and_evaluate(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, y_train: pd.Series, y_test: pd.Series, model=None):
    if model is None:
        model = build_stacked_classifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return model, metrics, y_proba


def save_artifacts(model, scaler, feature_columns: pd.Index, out_dir: str = 'models'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, f'{out_dir}/stacked_model.joblib')
    joblib.dump(scaler, f'{out_dir}/scaler.joblib')
    # save feature names so the app can reindex inputs
    pd.Series(list(feature_columns)).to_csv(f'{out_dir}/feature_columns.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--artifacts', default='models')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)
    model = build_stacked_classifier()
    model, metrics, y_proba = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, model)

    print('Metrics:')
    for k, v in metrics.items():
        if k == 'confusion_matrix':
            print(k)
            print(v)
        else:
            print(f"{k}: {v:.4f}")

    save_artifacts(model, scaler, X_train.columns, args.artifacts)
    print(f"Saved artifacts to {args.artifacts}")