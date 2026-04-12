"""
Drug-Drug Interaction Prediction Pipeline
Multi-task: Severity (classification) + Binary side effects (multi-label) + PRR (regression)
Updated with PCA for drastically faster training times.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.decomposition import PCA  # <-- Added PCA import
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem


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

def morgan_fingerprint(smiles: str, n_bits: int = MORGAN_BITS) -> np.ndarray:
    """Convert a SMILES string to a Morgan (ECFP4) fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
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

def train(train_path: str = TRAIN_PATH):
    print("Loading training data...")
    df = pd.read_csv(train_path)

    # Note: I have commented this out so it trains on your full 15,000 row dataset.
    # df = df.sample(frac=0.01, random_state=42) 
    print(f"  Train shape: {df.shape}")

    binary_cols, prr_cols = get_target_columns(df)

    print("\nBuilding features...")
    X_train, tfidf_vecs = build_features(df, fit=True)

    # ========== PCA DIMENSIONALITY REDUCTION ==========
    print("\nApplying PCA to reduce dimensions (this will speed up LightGBM significantly)...")
    pca = PCA(n_components=500, random_state=42)
    X_train = pca.fit_transform(X_train)
    print(f"  New feature matrix shape after PCA: {X_train.shape}")
    # ==================================================

    # ── Target 1: Severity (multi-class classification) ────────────────────────
    print("\nTraining Severity classifier...")
    le = LabelEncoder()
    y_severity = le.fit_transform(df["Severity"].fillna("Moderate"))

    severity_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth = 5,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=1,
    )
    severity_model.fit(X_train, y_severity)
    train_preds = severity_model.predict(X_train)
    print(classification_report(y_severity, train_preds, target_names=le.classes_))

    # ── Target 2: Binary side effects (multi-label classification) ─────────────
    print("Training Binary side-effect classifier...")
    y_binary = df[binary_cols].fillna(0).values.astype(int)

    binary_model = MultiOutputClassifier(
        lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            max_depth = 5,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            verbose=1,
        ),
        n_jobs=-1,
    )
    binary_model.fit(X_train, y_binary)
    print(f"  Binary model trained on {len(binary_cols)} side effects.")

    # ── Target 3: PRR regression (multi-output) ────────────────────────────────
    print("Training PRR regressor...")
    y_prr = df[prr_cols].fillna(0).values.astype(np.float32)

    prr_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth = 5,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            verbose=1,
        ),
        n_jobs=-1,
    )
    prr_model.fit(X_train, y_prr)
    print(f"  PRR model trained on {len(prr_cols)} targets.")

    return {
        "severity_model": severity_model,
        "binary_model": binary_model,
        "prr_model": prr_model,
        "label_encoder": le,
        "tfidf_vecs": tfidf_vecs,
        "pca": pca,               # <-- Added PCA to the returned dictionary
        "binary_cols": binary_cols,
        "prr_cols": prr_cols,
    }


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

    return {
        "severity": severity_preds,
        "binary":   binary_preds,
        "prr":      prr_preds,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(preds) -> float:
    """
    Compute the competition's Hardcore Clinical Score on a labeled validation set.

    Score = 0.4 * Severity_MacroF1
          + 0.3 * SideEffects_MicroF1
          + 0.3 * (1 / (1 + MaskedRMSE_PRR))

    The PRR RMSE is computed only over entries where the true PRR > 0 (masked).
    Returns the final composite score (0.0 – 1.0).
    """
    # ── 1. Severity – Macro F1 ─────────────────────────────────────────────────
    y_sev_true  = df_val["Severity"].fillna("Moderate")
    f1_severity = f1_score(y_sev_true, preds["severity"], average="macro")

    # ── 2. Side Effects – Micro F1 (flattened across all columns) ─────────────
    y_bin_true = df_val[binary_cols].fillna(0).values.astype(int)
    f1_binary  = f1_score(y_bin_true.ravel(), preds["binary"].ravel(), average="micro")

    # ── 3. PRR – Masked RMSE → Inverse score ──────────────────────────────────
    y_prr_true = df_val[prr_cols].fillna(0).values.astype(np.float32)
    mask = y_prr_true > 0
    if mask.sum() > 0:
        masked_rmse = np.sqrt(np.mean((preds["prr"][mask] - y_prr_true[mask]) ** 2))
    else:
        masked_rmse = 0.0
    prr_score = 1.0 / (1.0 + masked_rmse)

    # ── Final composite score ──────────────────────────────────────────────────
    score = 0.4 * f1_severity + 0.3 * f1_binary + 0.3 * prr_score

    print(f"\n{'='*52}")
    print(f"  Hardcore Clinical Score  (val n={len(df_val)})")
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

def submit(preds, out_path: str = SUBMISSION_PATH):
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
    df_full = pd.read_csv(TRAIN_PATH)
    df_val   = df_full.sample(frac=0.15, random_state=42)
    df_train = df_full.drop(df_val.index)
    df_train.to_csv(f"{DATA_DIR}/_train_split.csv", index=False)

    models = train(f"{DATA_DIR}/_train_split.csv")
    binary_cols = models["binary_cols"]
    prr_cols    = models["prr_cols"]

    preds = predict(models, df_val)
    evaluate(preds)

    print(f"\nLoading test data from {TEST_PATH}...")
    df_test = pd.read_csv(TEST_PATH)
    print(f"  Test shape: {df_test.shape}")
    preds = predict(models, df_test)

    sub = submit(preds)
    print("\nDone. Preview:")
    print(sub.head())