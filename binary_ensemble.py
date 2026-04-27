import numpy as np


def get_pos_proba(model, X) -> np.ndarray:
    """
    Extract a (n_samples, n_outputs) matrix of positive-class probabilities
    from any model whose predict_proba() returns a list of per-output arrays.
    Works with MultiOutputClassifier and BinaryEnsemble alike.
    """
    return np.column_stack([p[:, 1] for p in model.predict_proba(X)])


class BinaryEnsemble:
    """
    Wraps multiple fitted MultiOutputClassifiers.

    predict_proba() averages positive-class probabilities (used for PR curve analysis).
    predict() applies each model's own threshold independently, then takes a
    majority vote across models (mean of binary preds >= 0.5).
    """

    def __init__(self, models: list, weights: list = None, threshold: float = 0.5):
        self.models    = models
        self.weights   = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.threshold = threshold

    def predict_proba(self, X) -> list:
        """
        Returns a list of (n_samples, 2) arrays — one per output column —
        with averaged positive-class probabilities across all member models.
        """
        all_probas = [m.predict_proba(X) for m in self.models]
        n_outputs = len(all_probas[0])
        averaged = []
        for i in range(n_outputs):
            avg_pos = sum(w * p[i][:, 1] for w, p in zip(self.weights, all_probas))
            averaged.append(np.column_stack([1.0 - avg_pos, avg_pos]))
        return averaged

    def predict(self, X) -> np.ndarray:
        """
        Thresholds the averaged ensemble probabilities at self.threshold.
        Returns a (n_samples, n_outputs) int array.
        """
        probs = get_pos_proba(self, X)   # uses predict_proba → averaged probabilities
        return (probs >= self.threshold).astype(int)
    
    def train(self, X_train, df, binary_cols: list):
        """
        Train each model in the ensemble on the same data. Models are expected
        to have a train() method that accepts (X_train, df, binary_cols).
        """
        for model in self.models:
            model.train(X_train, df, binary_cols)