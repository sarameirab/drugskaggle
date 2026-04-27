


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.decomposition import PCA  # <-- Added PCA import
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from joblib import Parallel, delayed
import lightgbm as lgb

RANDOM_FOREST_CLASSIFIER_THRESHOLD = 0.402  # From precision-recall curve analysis on the validation set (see plot_pr_threshold.png)

class DrugInteractionRandomForestClassifier:
    def __init__(self, threshold=RANDOM_FOREST_CLASSIFIER_THRESHOLD):
        self._model = None
        self.threshold = threshold
        self._binary_cols = None

    def train(self, X_train, df: pd.DataFrame, binary_cols: list, n_splits: int = 5):
        """
        Train the binary side-effect classifier using a Random Forest with n_splits-fold CV.

        Each fold reports OOF micro-F1. A final model is retrained on all data
        after CV. Returns a fitted MultiOutputClassifier.
        """
        from sklearn.ensemble import RandomForestClassifier

        print(f"\nTraining Binary side-effect classifier (Random Forest, {n_splits}-fold CV)...")
        y_binary = df[binary_cols].fillna(0).values.astype(int)

        base_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=20,
            random_state=42,
        )

        if n_splits != 0:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            oof_preds = np.zeros_like(y_binary)
            fold_f1s = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"\n  Fold {fold + 1}/{n_splits}  ({len(tr_idx)} train / {len(val_idx)} val rows)")
                X_tr  = X_train.iloc[tr_idx]  if hasattr(X_train, "iloc") else X_train[tr_idx]
                X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
                y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]

                fold_model = MultiOutputClassifier(clone(base_clf), n_jobs=-1)
                fold_model.fit(X_tr, y_tr)

                fold_preds = fold_model.predict(X_val)
                oof_preds[val_idx] = fold_preds

                fold_f1 = f1_score(y_val.ravel(), fold_preds.ravel(), average="micro")
                fold_f1s.append(fold_f1)
                print(f"  Fold {fold + 1} micro-F1: {fold_f1:.4f}")

            overall_f1 = f1_score(y_binary.ravel(), oof_preds.ravel(), average="micro")
            print(f"\n  OOF micro-F1 : {overall_f1:.4f}")
            print(f"  Per-fold F1  : {[f'{f:.4f}' for f in fold_f1s]}")

            print("\n  Training final model on all data...")
        self._model = MultiOutputClassifier(base_clf, n_jobs=-1)
        self._model.fit(X_train, y_binary)

        print(f"  Binary RF model trained on {len(binary_cols)} side effects.")
        return self._model
    
    def predict_proba(self, X):
        """Predict probabilities for binary side effects using the trained model."""
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self._model.predict_proba(X)
    
    def predict(self, X):
        """Predict binary side effects using the trained model."""
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        probs = np.column_stack([p[:, 1] for p in self._model.predict_proba(X)]) 
        return (probs > self.threshold).astype(int)
    