
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate

def cross_validate_pipeline(pipeline, X, y, cv, scorers: str, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probability of the positive class for every row, from the
        fold model that did not train on it. `cv` must be a partition (each row validated
        exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    dummy_preds = scores["estimator"][0].predict_proba(X.iloc[[0]])
    n_classes = dummy_preds.shape[1]
    if n_classes > 2:
        oof_proba = np.empty((len(X), n_classes))
        for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
            oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])
    else:
        oof_proba = np.empty(len(X))
        for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
            oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, oof_proba

def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)

def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}

def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    # It now handles multiclass argmax prediction vs binary thresholding
    if oof_proba.ndim > 1 and oof_proba.shape[1] > 2: # if it's a matrix and has more than 2 classes
        y_pred = np.argmax(oof_proba, axis=1) # picks the class index with the highest probability across the columns for each row.
    else:
        probs = oof_proba[:,1] if oof_proba.ndim > 1 else oof_proba
        y_pred = (probs >= threshold).astype(int) # extracts the positive class column and applies the cutoff threshold (>= 0.5) to generate 0 or 1 predictions.

    # personalized labels based on unique classes found
    n_classes = len(np.unique(y_true))
    class_labels = [f"true {i}" for i in range(n_classes)]
    pred_labels = [f"predicted {i}" for i in range(n_classes)]
    # creates matching text labels for rows (true 0, true 1, true 2) and columns (predicted 0, etc.) regardless of whether it's 2 or 3 classes.

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=class_labels, columns=pred_labels)
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race", threshold: float = 0.5) -> str:
    """
    False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's out-of-fold predictions and for COMPAS's own
    score (`score_text` other than "Low" counts as predicted to reoffend), on the same rows.
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values

    # handle model predictions (binary vs multiclass)
    if oof_proba.ndim > 1 and oof_proba.shape[1] > 2:
        df["y_pred_model"] = np.argmax(oof_proba, axis=1)
    else:
        probs = oof_proba[:, 1] if oof_proba.ndim > 1 else oof_proba
        df["y_pred_model"] = (probs >= threshold).astype(int)

    # handle COMPAS score (score_text)
    # check if score_text is already mapped to int
    if "score_text" in df.columns:
        if df["score_text"].dtype == object or isinstance(df["score_text"].iloc[0], str):
            # binary text check vs multiclass text check
            if len(np.unique(y_true)) > 2:
                # For 3 classes: define "High" (class 2) as positive or map text
                df["y_pred_compas"] = (df["score_text"] == "High").astype(int)
            else:
                df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)
        else:
            # If score_text is already numeric
            df["y_pred_compas"] = (df["score_text"] == 2).astype(int) if len(np.unique(y_true)) > 2 else (df["score_text"] > 0).astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            if len(np.unique(y_true)) > 2:    # 3 classes: Negatives = not truly High (y_true != 2), Flagged = predicted High (== 2)
                negatives = g[g["y_true"] != 2]
                if len(negatives) == 0:
                    continue
                fpr = (negatives[col] == 2).mean()
            else:    # binary: negatives = didn't reoffend (y_true == 0), flagged = predicted positive (==1)
                negatives = g[g["y_true"] == 0]
                if len(negatives) == 0:
                    continue
                fpr = (negatives[col] == 1).mean()
                
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)

def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers