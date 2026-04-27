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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from utils import binary_zero_breakdown

LGBM_CLASSIFIER_THRESHOLD = 0.331  # From precision-recall curve analysis on the validation set (see plot_pr_threshold.png)


def _fit_binary_estimator(base_clf, X_tr, y_tr_col, X_es, y_es_col):
    """Fit one LGBMClassifier with early stopping. Module-level for joblib pickling."""
    clf = clone(base_clf)
    clf.fit(
        X_tr, y_tr_col,
        eval_set=[(X_es, y_es_col)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return clf


def _fit_binary_estimator_fixed(base_clf, X_tr, y_tr_col):
    """Fit one LGBMClassifier with a fixed n_estimators (no early stopping). Module-level for joblib pickling."""
    clf = clone(base_clf)
    clf.fit(X_tr, y_tr_col)
    return clf



class DrugInteractionLGBMClassifier:
    def __init__(self, threshold=LGBM_CLASSIFIER_THRESHOLD):
        self._model = None
        self.threshold = threshold
        self._binary_cols = None

    def train(self, X_train, df: pd.DataFrame, binary_cols: list, n_splits: int = 5):
        # Override to ensure predict_proba returns probabilities for both classes.
        """
        Train the binary side-effect classifier using n_splits-fold cross-validation.

        Each fold uses its validation split for early stopping and OOF evaluation.
        The average best iteration across folds determines n_estimators for the
        final model, which is then retrained on all data without early stopping.
        Returns a fitted MultiOutputClassifier.
        """
        print(f"\nTraining Binary side-effect classifier ({n_splits}-fold CV)...")
        y_binary = df[binary_cols].fillna(0).values.astype(int)

        base_model = lgb.LGBMClassifier(
            n_estimators=500,  # upper bound; early stopping selects the right number per fold
            scale_pos_weight=2,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            verbose=-1,
        )

        if n_splits != 0:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            oof_preds = np.zeros_like(y_binary)
            fold_f1s = []
            best_iterations = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"\n  Fold {fold + 1}/{n_splits}  ({len(tr_idx)} train / {len(val_idx)} val rows)")
                X_tr  = X_train.iloc[tr_idx]  if hasattr(X_train, "iloc") else X_train[tr_idx]
                X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
                y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]

                estimators = Parallel(n_jobs=20)(
                    delayed(_fit_binary_estimator)(base_model, X_tr, y_tr[:, i], X_val, y_val[:, i])
                    for i in range(len(binary_cols))
                )

                fold_preds = np.column_stack([clf.predict(X_val) for clf in estimators])
                oof_preds[val_idx] = fold_preds

                fold_f1 = f1_score(y_val.ravel(), fold_preds.ravel(), average="micro")
                fold_f1s.append(fold_f1)

                avg_iter = int(np.mean([
                    clf.best_iteration_ for clf in estimators
                    if getattr(clf, "best_iteration_", None) is not None
                ] or [500]))
                best_iterations.append(avg_iter)
                print(f"  Fold {fold + 1} micro-F1: {fold_f1:.4f}  avg best_iter: {avg_iter}")

            overall_f1 = f1_score(y_binary.ravel(), oof_preds.ravel(), average="micro")
            final_n_estimators = int(np.mean(best_iterations))
            print(f"\n  OOF micro-F1 : {overall_f1:.4f}")
            print(f"  Per-fold F1  : {[f'{f:.4f}' for f in fold_f1s]}")
            print(f"  Final n_estimators (mean best_iter across folds): {final_n_estimators}")

                # Retrain on all data with fixed n_estimators — no ES split needed.
            print("\n  Training final model on all data...")
            model = lgb.LGBMClassifier(
                n_estimators=final_n_estimators,
                scale_pos_weight=2,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=5,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
                verbose=-1,
            )

            estimators = Parallel(n_jobs=20)(
                delayed(_fit_binary_estimator_fixed)(model, X_train, y_binary[:, i])
                for i in range(len(binary_cols))
            )

            self._model = MultiOutputClassifier(model, n_jobs=20)
            self._model.estimators_ = estimators
            self._model.classes_ = [clf.classes_ for clf in estimators]
            self._model.n_outputs_ = len(binary_cols)

            print(f"  Binary model trained on {len(binary_cols)} side effects.")
        else:
            self._model = MultiOutputClassifier(base_model, n_jobs=-1)
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
    