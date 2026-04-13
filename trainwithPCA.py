"""
Drug-Drug Interaction Prediction Pipeline
Multi-task: Severity (classification) + Binary side effects (multi-label) + PRR (regression)
Updated with PCA for drastically faster training times.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.decomposition import PCA  # <-- Added PCA import
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from joblib import Parallel, delayed
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator


# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUBMISSION_PATH = f"{DATA_DIR}/submission.csv"

TEXT_COLS = [
    "Mechanism", "Pharmacodynamics", "Metabolism",
    "Absorption", "Toxicity", "Indication", "Warning",
]
SMILES_COLS = ["SMILES_A", "SMILES_B"]
MORGAN_BITS = 1024
TFIDF_MAX_FEATURES = 150  # per field pair

# ── Feature Engineering ────────────────────────────────────────────────────────

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)

def morgan_fingerprint(smiles: str) -> np.ndarray:
    """Convert a SMILES string to a Morgan (ECFP4) fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return np.zeros(MORGAN_BITS, dtype=np.float32)
    
    # Use the generator to get the fingerprint
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def cyp450_overlap(cyp_a: str, cyp_b: str) -> dict:
    """
    Extract CYP450 enzyme tokens from both drugs and compute overlap features.
    Shared CYP enzymes are a primary driver of metabolic DDIs.
    """
    def extract_cyps(text):
        if not isinstance(text, str):
            return set()
        return set(re.findall(r"CYP\w+", text.upper()))

    set_a = extract_cyps(cyp_a)
    set_b = extract_cyps(cyp_b)
    union = set_a | set_b
    intersection = set_a & set_b
    jaccard = len(intersection) / len(union) if union else 0.0
    return {
        "cyp_overlap_count": len(intersection),
        "cyp_jaccard": jaccard,
        "cyp_a_count": len(set_a),
        "cyp_b_count": len(set_b),
    }


def build_features(df: pd.DataFrame, tfidf_vectorizers: dict = None, fit: bool = True):
    """
    Build the full feature matrix for a dataframe.
    If fit=True, fits the TF-IDF vectorizers and returns them.
    If fit=False, uses pre-fitted vectorizers (for test set).
    """
    feature_blocks = []

    # 1. Morgan fingerprints for Drug A and Drug B
    print("  Computing Morgan fingerprints...")
    fp_a = np.vstack(df["SMILES_A"].apply(morgan_fingerprint))
    fp_b = np.vstack(df["SMILES_B"].apply(morgan_fingerprint))
    # Also compute the element-wise XOR (difference) and AND (similarity) of the two fingerprints
    fp_diff = np.abs(fp_a - fp_b)
    fp_and = fp_a * fp_b
    feature_blocks.extend([fp_a, fp_b, fp_diff, fp_and])

    # 2. CYP450 overlap features
    print("  Computing CYP450 overlap features...")
    cyp_feats = df.apply(
        lambda row: cyp450_overlap(row.get("CYP450_Enzymes_A", ""), row.get("CYP450_Enzymes_B", "")),
        axis=1,
    )
    cyp_df = pd.DataFrame(list(cyp_feats))
    feature_blocks.append(cyp_df.values.astype(np.float32))

    # 3. TF-IDF on concatenated text pairs
    print("  Computing TF-IDF features...")
    if fit:
        tfidf_vectorizers = {}
    for col in TEXT_COLS:
        col_a = f"{col}_A"
        col_b = f"{col}_B"
        if col_a not in df.columns or col_b not in df.columns:
            continue
        # Concatenate Drug A and Drug B text for the field — captures combined pharmacology
        combined = (df[col_a].fillna("") + " [SEP] " + df[col_b].fillna(""))
        if fit:
            vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
            tfidf_mat = vec.fit_transform(combined).toarray().astype(np.float32)
            tfidf_vectorizers[col] = vec
        else:
            vec = tfidf_vectorizers[col]
            tfidf_mat = vec.transform(combined).toarray().astype(np.float32)
        feature_blocks.append(tfidf_mat)

    X = np.hstack(feature_blocks)
    print(f"  Feature matrix shape: {X.shape}")
    return X, tfidf_vectorizers


# ── Label Preparation ──────────────────────────────────────────────────────────

def get_target_columns(df: pd.DataFrame):
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    prr_cols = [c for c in df.columns if c.startswith("Target_PRR_")]
    return binary_cols, prr_cols


# ── Training ───────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, tfidf_vecs=None, pca=None, fit: bool = True):
    """
    Build features and apply PCA.
    If fit=True, fits tfidf_vecs and pca from df and returns them.
    If fit=False, transforms df using the provided tfidf_vecs and pca.
    Returns (X, tfidf_vecs, pca).
    """
    X, tfidf_vecs = build_features(df, tfidf_vectorizers=tfidf_vecs, fit=fit)
    if fit:
        print("\nApplying PCA...")
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)
    print(f"  Feature matrix shape after PCA: {X.shape}")
    return X, tfidf_vecs, pca


def train_severity(X_train: np.ndarray, df: pd.DataFrame):
    """Train the severity classifier. Returns (model, label_encoder)."""
    print("\nTraining Severity classifier...")
    le = LabelEncoder()
    y_severity = le.fit_transform(df["Severity"].fillna("Moderate"))
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=5,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_severity)
    print(classification_report(y_severity, model.predict(X_train), target_names=le.classes_))
    return model, le


def _fit_binary_estimator(base_clf, X_tr, y_tr_col, X_es, y_es_col):
    """Fit one LGBMClassifier with early stopping. Module-level for joblib pickling."""
    clf = clone(base_clf)
    clf.fit(
        X_tr, y_tr_col,
        eval_set=[(X_es, y_es_col)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return clf


def train_binary(X_train: np.ndarray, df: pd.DataFrame, binary_cols: list, es_frac: float = 0.15):
    """
    Train the binary side-effect classifier with early stopping.

    Splits X_train into a fitting set and an internal early-stopping validation
    set (es_frac of rows). This is a second validation set — separate from the
    outer df_val used for final evaluation in __main__.
    Returns a fitted MultiOutputClassifier.
    """
    print("\nTraining Binary side-effect classifier...")
    y_binary = df[binary_cols].fillna(0).values.astype(int)

    # Internal early-stopping split (distinct from the outer evaluation df_val)
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train, y_binary, test_size=es_frac, random_state=42
    )
    print(f"  Early-stopping split: {X_tr.shape[0]} fit rows / {X_es.shape[0]} early-stop rows")

    base_clf = lgb.LGBMClassifier(
        n_estimators=500,  # upper bound; early stopping will select the right number
        scale_pos_weight=2,  # heavily weight positive class to combat extreme imbalance
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
        delayed(_fit_binary_estimator)(base_clf, X_tr, y_tr[:, i], X_es, y_es[:, i])
        for i in range(len(binary_cols))
    )

    # Reconstruct a MultiOutputClassifier with the manually fitted estimators
    # so the rest of the pipeline (predict, binary_zero_breakdown) is unchanged.
    model = MultiOutputClassifier(base_clf, n_jobs=20)
    model.estimators_ = estimators
    model.classes_ = [clf.classes_ for clf in estimators]
    model.n_outputs_ = len(binary_cols)

    print(f"  Binary model trained on {len(binary_cols)} side effects.")
    return model


def train_prr(X_train: np.ndarray, df: pd.DataFrame, prr_cols: list):
    """Train the PRR regressor. Returns model."""
    print("\nTraining PRR regressor...")
    y_prr = df[prr_cols].fillna(0).values.astype(np.float32)
    # Tweedie regression is designed for zero-inflated, right-skewed positive
    # data — exactly the PRR distribution profile. Unlike MSE, it doesn't get
    # dominated by the zero-majority and naturally handles the heavy right tail.
    # tweedie_variance_power=1.5 sits between Poisson (1.0) and Gamma (2.0).
    model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            verbose=1,
        ),
        n_jobs=-1,
    )
    model.fit(X_train, y_prr)
    print(f"  PRR model trained on {len(prr_cols)} targets.")
    return model


def train_all(train_path: str = TRAIN_PATH) -> dict:
    """Train all three models end-to-end. Returns the full models dict."""
    print("Loading training data...")
    df = pd.read_csv(train_path)
    print(f"  Train shape: {df.shape}")

    binary_cols, prr_cols = get_target_columns(df)

    print("\nBuilding features...")
    X_train, tfidf_vecs, pca = prepare_features(df, fit=True)

    severity_model, le = train_severity(X_train, df)
    binary_model       = train_binary(X_train, df, binary_cols)
    prr_model          = train_prr(X_train, df, prr_cols)

    return {
        "severity_model": severity_model,
        "binary_model":   binary_model,
        "prr_model":      prr_model,
        "label_encoder":  le,
        "tfidf_vecs":     tfidf_vecs,
        "pca":            pca,
        "binary_cols":    binary_cols,
        "prr_cols":       prr_cols,
    }

def binary_zero_breakdown(binary_preds: np.ndarray, df_data: pd.DataFrame, binary_cols: list):
    """
    Confusion-style breakdown of the binary side-effect classifier,
    both in aggregate and per side effect (sorted by Recall ascending).
    """
    y_true = df_data[binary_cols].fillna(0).values.astype(int)
    y_pred = binary_preds.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = y_true.size

    print(f"\n{'='*52}")
    print(f"  Binary Classifier Breakdown  ({total:,} cells total)")
    print(f"{'='*52}")
    print(f"  True  Positive (true=1, pred=1) : {tp:>8,}  ({100*tp/total:5.1f}%)")
    print(f"  True  Negative (true=0, pred=0) : {tn:>8,}  ({100*tn/total:5.1f}%)")
    print(f"  False Positive (true=0, pred=1) : {fp:>8,}  ({100*fp/total:5.1f}%)")
    print(f"  False Negative (true=1, pred=0) : {fn:>8,}  ({100*fn/total:5.1f}%)")
    print(f"{'='*52}")
    actual_pos = tp + fn
    if actual_pos > 0:
        print(f"  Recall  (TP / all true=1)       : {tp/actual_pos:.3f}")
    if (tp + fp) > 0:
        print(f"  Precision (TP / all pred=1)     : {tp/(tp+fp):.3f}")
    print(f"{'='*52}")

    rows = []
    for i, col in enumerate(binary_cols):
        t = y_true[:, i]
        p = y_pred[:, i]
        c_tp = int(((t == 1) & (p == 1)).sum())
        c_tn = int(((t == 0) & (p == 0)).sum())
        c_fp = int(((t == 0) & (p == 1)).sum())
        c_fn = int(((t == 1) & (p == 0)).sum())
        n_pos = c_tp + c_fn
        recall    = c_tp / n_pos       if n_pos > 0          else float("nan")
        precision = c_tp / (c_tp+c_fp) if (c_tp + c_fp) > 0 else float("nan")
        rows.append({
            "Side Effect": col.replace("Target_Binary_", ""),
            "TP": c_tp, "TN": c_tn, "FP": c_fp, "FN": c_fn,
            "True Positives": n_pos,
            "Recall":         round(recall, 3),
            "Precision":      round(precision, 3),
        })

    per_effect = (
        pd.DataFrame(rows)
        .sort_values("Recall")
        .reset_index(drop=True)
    )
    print("\nPer-side-effect breakdown (sorted by Recall ascending):")
    print(per_effect.to_string(index=False))


def prr_zero_breakdown(preds: dict, df_data: pd.DataFrame, prr_cols: list):
    """
    Print a confusion-style breakdown of predicted vs true PRR zeros/non-zeros
    across all (row, side-effect) entries.

        True=0, Pred=0  → True Negative  (correctly silent)
        True=0, Pred>0  → False Positive  (predicted a signal where none exists)
        True>0, Pred=0  → False Negative  (missed a real signal)
        True>0, Pred>0  → True Positive   (correctly detected a signal)
    """
    y_true = df_data[prr_cols].fillna(0).values.astype(np.float32)
    y_pred = preds["prr"].astype(np.float32)

    true_pos = ((y_true > 0) & (y_pred > 0)).sum()
    true_neg = ((y_true == 0) & (y_pred == 0)).sum()
    false_pos = ((y_true == 0) & (y_pred > 0)).sum()
    false_neg = ((y_true > 0) & (y_pred == 0)).sum()
    total = y_true.size

    print(f"\n{'='*52}")
    print(f"  PRR Zero / Non-Zero Breakdown  ({total:,} cells total)")
    print(f"{'='*52}")
    print(f"  True  Positive (true>0, pred>0) : {true_pos:>8,}  ({100*true_pos/total:5.1f}%)")
    print(f"  True  Negative (true=0, pred=0) : {true_neg:>8,}  ({100*true_neg/total:5.1f}%)")
    print(f"  False Positive (true=0, pred>0) : {false_pos:>8,}  ({100*false_pos/total:5.1f}%)")
    print(f"  False Negative (true>0, pred=0) : {false_neg:>8,}  ({100*false_neg/total:5.1f}%)")
    print(f"{'='*52}")
    actual_pos = (y_true > 0).sum()
    if actual_pos > 0:
        recall = true_pos / actual_pos
        print(f"  Signal Recall (TP / all true>0) : {recall:.3f}")
    print(f"{'='*52}")

    # ── Per-side-effect breakdown ──────────────────────────────────────────────
    rows = []
    for i, col in enumerate(prr_cols):
        t = y_true[:, i]
        p = y_pred[:, i]
        tp = int(((t > 0) & (p > 0)).sum())
        tn = int(((t == 0) & (p == 0)).sum())
        fp = int(((t == 0) & (p > 0)).sum())
        fn = int(((t > 0) & (p == 0)).sum())
        n_pos = tp + fn
        recall = tp / n_pos if n_pos > 0 else float("nan")
        rows.append({
            "Side Effect":    col.replace("Target_PRR_", ""),
            "TP":  tp, "TN": tn, "FP": fp, "FN": fn,
            "True Positives": n_pos,
            "Recall":         round(recall, 3),
        })

    per_effect = (
        pd.DataFrame(rows)
        .sort_values("Recall")
        .reset_index(drop=True)
    )
    print("\nPer-side-effect breakdown (sorted by Recall ascending):")
    print(per_effect.to_string(index=False))


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(models: dict, df: pd.DataFrame) -> dict:
    """
    Run all three models on df and return a dict of raw prediction arrays.

    Returns:
        severity_preds : 1-D array of string labels
        binary_preds   : 2-D int array  (n_rows × n_binary_cols)
        prr_preds      : 2-D float array (n_rows × n_prr_cols), clipped to >= 0
    """
    print("\nBuilding features...")
    X, _ = build_features(df, tfidf_vectorizers=models["tfidf_vecs"], fit=False)
    X = models["pca"].transform(X)

    print("Predicting...")
    severity_preds = models["label_encoder"].inverse_transform(
        models["severity_model"].predict(X)
    )
    binary_preds = models["binary_model"].predict(X)
    prr_preds    = np.clip(models["prr_model"].predict(X), 0, None)
    # Zero-gate: if the binary model says a side effect doesn't occur, its PRR
    # must be 0. Tweedie never outputs exactly 0, so without this every cell
    # would be non-zero regardless of whether there's a real signal.
    prr_preds[binary_preds == 0] = 0.0

    return {
        "severity": severity_preds,
        "binary":   binary_preds,
        "prr":      prr_preds,
    }


# ── PRR Visualisation ─────────────────────────────────────────────────────────

def visualize_prr(preds: dict, df_data: pd.DataFrame, prr_cols: list, out_path: str = "prr_analysis.png"):
    """
    Four-panel diagnostic plot for PRR predictions vs ground truth.

      1. Distribution of all true PRR values (non-zero only)
      2. Predicted vs True scatter on masked (non-zero truth) entries
      3. Per-side-effect RMSE bar chart (top 20 worst)
      4. Residual distribution (pred - true) on masked entries
    """
    y_true = df_data[prr_cols].fillna(0).values.astype(np.float32)
    y_pred = preds["prr"].astype(np.float32)
    mask   = y_true > 0

    true_masked = y_true[mask]
    pred_masked = y_pred[mask]
    residuals   = pred_masked - true_masked

    # Per-column RMSE (only over masked entries in each column)
    per_col_rmse = []
    for i, col in enumerate(prr_cols):
        col_mask = y_true[:, i] > 0
        if col_mask.sum() > 0:
            rmse = np.sqrt(np.mean((y_pred[:, i][col_mask] - y_true[:, i][col_mask]) ** 2))
        else:
            rmse = 0.0
        per_col_rmse.append((col.replace("Target_PRR_", ""), rmse))
    per_col_rmse.sort(key=lambda x: x[1], reverse=True)
    top20_labels, top20_rmse = zip(*per_col_rmse)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PRR Prediction Analysis", fontsize=14, fontweight="bold")

    # 1. Distribution of true PRR (non-zero)
    ax = axes[0, 0]
    ax.hist(true_masked, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_title("True PRR Distribution (non-zero entries)")
    ax.set_xlabel("True PRR")
    ax.set_ylabel("Count")
    ax.axvline(np.median(true_masked), color="tomato", linestyle="--", label=f"Median {np.median(true_masked):.2f}")
    ax.legend()

    # 2. Predicted vs True scatter
    ax = axes[0, 1]
    ax.scatter(true_masked, pred_masked, alpha=0.3, s=8, color="steelblue")
    lim = max(true_masked.max(), pred_masked.max())
    ax.plot([0, lim], [0, lim], color="tomato", linestyle="--", linewidth=1, label="Perfect")
    ax.set_title("Predicted vs True PRR (non-zero entries)")
    ax.set_xlabel("True PRR")
    ax.set_ylabel("Predicted PRR")
    ax.legend()

    # 3. Per-side-effect RMSE (top 20 worst)
    ax = axes[1, 0]
    ax.barh(range(len(top20_rmse)), top20_rmse[::-1], color="steelblue")
    ax.set_yticks(range(len(top20_rmse)))
    ax.set_yticklabels(top20_labels[::-1], fontsize=8)
    ax.set_title("Top 20 Side Effects by PRR RMSE")
    ax.set_xlabel("RMSE")

    # 4. Residual distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="tomato", linestyle="--", linewidth=1, label="Zero error")
    ax.axvline(np.mean(residuals), color="orange", linestyle="--", linewidth=1, label=f"Mean {np.mean(residuals):.2f}")
    ax.set_title("Residuals (Predicted - True) on non-zero entries")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"PRR analysis plot saved to: {out_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(preds, df_data) -> float:
    """
    Compute the competition's Hardcore Clinical Score on a labeled validation set.

    Score = 0.4 * Severity_MacroF1
          + 0.3 * SideEffects_MicroF1
          + 0.3 * (1 / (1 + MaskedRMSE_PRR))

    The PRR RMSE is computed only over entries where the true PRR > 0 (masked).
    Returns the final composite score (0.0 – 1.0).
    """
    # ── 1. Severity – Macro F1 ─────────────────────────────────────────────────
    y_sev_true  = df_data["Severity"].fillna("Moderate")
    f1_severity = f1_score(y_sev_true, preds["severity"], average="macro")

    # ── 2. Side Effects – Micro F1 (flattened across all columns) ─────────────
    y_bin_true = df_data[binary_cols].fillna(0).values.astype(int)
    f1_binary  = f1_score(y_bin_true.ravel(), preds["binary"].ravel(), average="micro")

    # ── 3. PRR – Masked RMSE → Inverse score ──────────────────────────────────
    y_prr_true = df_data[prr_cols].fillna(0).values.astype(np.float32)
    mask = y_prr_true > 0
    if mask.sum() > 0:
        masked_rmse = np.sqrt(np.mean((preds["prr"][mask] - y_prr_true[mask]) ** 2))
    else:
        masked_rmse = 0.0
    prr_score = 1.0 / (1.0 + masked_rmse)

    # ── Final composite score ──────────────────────────────────────────────────
    score = 0.4 * f1_severity + 0.3 * f1_binary + 0.3 * prr_score

    print(f"\n{'='*52}")
    print(f"  Hardcore Clinical Score  (val n={len(df_data)})")
    print(f"{'='*52}")
    print(f"  Severity  Macro  F1  : {f1_severity:.4f}   (weight 40%)")
    print(f"  Side Eff. Micro  F1  : {f1_binary:.4f}   (weight 30%)")
    print(f"  PRR Masked RMSE      : {masked_rmse:.4f}")
    print(f"  PRR Score 1/(1+RMSE) : {prr_score:.4f}   (weight 30%)")
    print(f"{'='*52}")
    print(f"  FINAL SCORE          : {score:.4f}")
    print(f"{'='*52}")
    return score


# ── Submission ─────────────────────────────────────────────────────────────────

def submit(models, preds, df_test, out_path: str = SUBMISSION_PATH):
    print("Building submission file...")
    sub = pd.DataFrame({"Pair_ID": df_test["Pair_ID"]})
    sub["Severity"] = preds["severity"]
    sub = pd.concat([
        sub,
        pd.DataFrame(preds["binary"], columns=models["binary_cols"]),
        pd.DataFrame(preds["prr"],    columns=models["prr_cols"]),
    ], axis=1)

    sub.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}  ({sub.shape[0]} rows, {sub.shape[1]} cols)")
    return sub


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hold out 15 % of train for evaluation so the score reflects unseen data.
    # df_full = pd.read_csv(TRAIN_PATH)
    # df_val   = df_full.sample(frac=0.15, random_state=42)
    # df_train = df_full.drop(df_val.index)
    # df_train.to_csv(f"{DATA_DIR}/_train_split.csv", index=False)

    # models = train_all(f"{DATA_DIR}/_train_split.csv")
    # binary_cols = models["binary_cols"]
    # prr_cols    = models["prr_cols"]

    # validation_preds = predict(models, df_val)
    # print("\nEvaluating on validation set...")
    # evaluate(validation_preds, df_val)
    # visualize_prr(validation_preds, df_val, prr_cols)
    # prr_zero_breakdown(validation_preds, df_val, prr_cols)

    # print(f"\nLoading test data from {TEST_PATH}...")
    # df_test = pd.read_csv(TEST_PATH)
    # print(f"  Test shape: {df_test.shape}")
    # test_preds = predict(models, df_test)
    # sub = submit(models, test_preds, df_test)
    # print("\nDone. Preview:")
    # print(sub.head())

    # ── PRR-only mode ─────────────────────────────────────────────────────────
    print("Loading data...")
    df_full  = pd.read_csv(TRAIN_PATH)
    df_val   = df_full.sample(frac=0.15, random_state=42)
    df_train = df_full.drop(df_val.index)
    binary_cols, prr_cols = get_target_columns(df_full)

    print("\nBuilding features...")
    X_train, tfidf_vecs, pca = prepare_features(df_train, fit=True)
    X_val, _, _              = prepare_features(df_val, tfidf_vecs=tfidf_vecs, pca=pca, fit=False)

    # Train binary on all rows — used for zero-gating at inference.
    binary_model = train_binary(X_train, df_train, binary_cols)
    print("\nPredicting PRR on validation set...")
    binary_preds = binary_model.predict(X_val)
    print("\nZero breakdown on validation set...")
    binary_zero_breakdown(binary_preds, df_val, binary_cols)
    print("\nPredicting PRR on training set...")
    binary_preds_training = binary_model.predict(X_train)
    print("\nZero breakdown on training set...")
    binary_zero_breakdown(binary_preds_training, df_train, binary_cols)





    # Train PRR only on rows where at least one side effect is non-zero.
    # This aligns the training distribution with the masked RMSE metric, which
    # only scores non-zero entries. The binary model handles the zero/non-zero
    # decision at inference; the PRR model only needs to learn magnitudes.
    # y_prr_train  = df_train[prr_cols].fillna(0).values.astype(np.float32)
    # active_rows  = (y_prr_train > 0).any(axis=1)
    # print(f"\n  PRR training on {active_rows.sum()} active rows (of {len(df_train)} total)")
    # prr_model = train_prr(X_train[active_rows], df_train.iloc[active_rows], prr_cols)


    # prr_preds    = np.clip(prr_model.predict(X_val), 0, None)
    # prr_preds[binary_preds == 0] = 0.0
    # preds = {"prr": prr_preds}

    # visualize_prr(preds, df_val, prr_cols)
    # prr_zero_breakdown(binary_preds, df_val, prr_cols)

    # y_prr_true  = df_val[prr_cols].fillna(0).values.astype(np.float32)
    # mask        = y_prr_true > 0
    # masked_rmse = np.sqrt(np.mean((prr_preds[mask] - y_prr_true[mask]) ** 2))
    # print(f"\n  PRR Masked RMSE      : {masked_rmse:.4f}")
    # print(f"  PRR Score 1/(1+RMSE) : {1/(1+masked_rmse):.4f}")
