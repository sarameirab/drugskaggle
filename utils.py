import numpy as np
import pandas as pd


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
