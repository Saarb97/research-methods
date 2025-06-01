"""
Multiclass-ready re-write of the original XGBoost + SMOTE-Tomek pipeline.
"""
# ───────────────────────────────── Imports ────────────────────────────────────
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt       # noqa – kept for backwards-compat

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, accuracy_score
)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

import optuna, xgboost as xgb, torch


# ─────────────────────────── Global configuration ────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

MAX_BOOST_ROUNDS          = 20000
EARLY_STOPPING_PATIENCE   = 50
N_OPTUNA_TRIALS           = 50
OPTUNA_DIRECTION          = "maximize"
OPTUNA_N_JOBS             = 1

VALIDATION_SPLIT_SIZE     = 0.15
N_CV_SPLITS               = 5
RANDOM_STATE              = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
if device.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# ────────────────────────────── Utils & helpers ──────────────────────────────
def build_xgb_params(trial, n_classes: int) -> dict:
    """Return a parameter dictionary suitable for binary or multiclass."""
    base = {
        "tree_method": "hist",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", .5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "seed": RANDOM_STATE,
        "device": device.type,
    }

    if n_classes == 2:
        base.update(objective="binary:logistic", eval_metric="auc")
    else:
        base.update(objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss")   # or "auc" if XGB≥1.7
    return base


def roc_auc_multiclass(y_true, y_score):
    """Macro-average ROC-AUC that falls back to binary properly."""
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_score[:, 1])
    return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")


def pr_auc_multiclass(y_true, y_score):
    """Macro average-precision (area under PR curve) for n ≥ 2 classes."""
    classes = np.unique(y_true)
    ap_scores = []
    for c in classes:
        y_true_bin = (y_true == c).astype(int)
        if y_true_bin.sum() == 0:
            continue
        ap_scores.append(average_precision_score(y_true_bin, y_score[:, c]))
    return np.mean(ap_scores) if ap_scores else np.nan


def apply_smote_tomek(X_train, y_train, random_state=42):
    """Apply SMOTE-Tomek; returns resampled X, y (y as Series)."""
    y_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    if not np.issubdtype(y_series.dtype, np.integer):
        y_series = y_series.astype(int)

    class_counts = y_series.value_counts()
    if len(class_counts) < 2:
        return X_train, y_series                         # nothing to balance

    minority = class_counts.min()
    k_neighbors = min(minority - 1, 5)
    if k_neighbors < 1:
        return X_train, y_series

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    tomek = TomekLinks(sampling_strategy="majority")
    smt   = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
    X_res, y_res = smt.fit_resample(X_train, y_series)
    return X_res, pd.Series(y_res, name=y_series.name)


# ────────────────────────────── Optuna objective ─────────────────────────────
def objective(trial, X_train_fold, y_train_fold):
    """Hyper-param optimisation objective (binary or multiclass)."""
    n_classes = y_train_fold.nunique()
    params    = build_xgb_params(trial, n_classes)

    # split for early-stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_fold, y_train_fold,
        test_size=VALIDATION_SPLIT_SIZE,
        stratify=y_train_fold,
        random_state=RANDOM_STATE + trial.number,
    )

    if y_val.nunique() < 2:                     # AUC needs both classes
        raise optuna.exceptions.TrialPruned("Val set single class")

    # resample only the training part
    X_tr_res, y_tr_res = apply_smote_tomek(X_tr, y_tr, random_state=RANDOM_STATE)

    model = xgb.XGBClassifier(
        **params,
        n_estimators=MAX_BOOST_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,
    )
    model.fit(X_tr_res, y_tr_res,
              eval_set=[(X_val, y_val)],
              verbose=False)

    y_val_proba = model.predict_proba(X_val)
    auc = roc_auc_multiclass(y_val.values, y_val_proba)
    return auc


# ─────────────────────── Data loading & preparation ──────────────────────────
def load_and_prepare_data(file_name, text_col_name, target_col_name):
    df = pd.read_csv(file_name)
    cols_to_drop = [c for c in (text_col_name, "cluster", "named_entities") if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    X = df.drop(columns=[target_col_name])
    y = df[target_col_name]
    return X, y


# ──────────────────────── Model training helpers ─────────────────────────────
def train_xgboost_with_SMOTE(X_train_fold, y_train_fold):
    """Tune + train one final model for a single fold."""
    def _obj(trial): return objective(trial, X_train_fold, y_train_fold)

    # Bail-out if minority class too small for the internal split
    if y_train_fold.value_counts().min() < int(np.ceil(1 / VALIDATION_SPLIT_SIZE)):
        return None

    study = optuna.create_study(direction=OPTUNA_DIRECTION,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(_obj, n_trials=N_OPTUNA_TRIALS,
                   n_jobs=OPTUNA_N_JOBS, timeout=1800)

    best_params = study.best_params
    n_classes   = y_train_fold.nunique()
    final_params = build_xgb_params(study.best_trial, n_classes) | best_params

    # internal split for early stopping of the final model
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_fold, y_train_fold,
        test_size=VALIDATION_SPLIT_SIZE,
        stratify=y_train_fold,
        random_state=RANDOM_STATE + 1000)

    X_tr_res, y_tr_res = apply_smote_tomek(X_tr, y_tr, random_state=RANDOM_STATE)

    final_model = xgb.XGBClassifier(
        **final_params,
        n_estimators=MAX_BOOST_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,
    )
    final_model.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)
    return final_model


# ─────────────────────────────── K-fold loop ─────────────────────────────────
def train_with_smote_kfold(X, y, cluster_idx):
    kf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, roc_aucs, pr_aucs, f1s, feats, reports = [], [], [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model = train_xgboost_with_SMOTE(X_tr, y_tr)
        if model is None:
            accs.append(np.nan); roc_aucs.append(np.nan); pr_aucs.append(np.nan); f1s.append(np.nan)
            feats.append(np.full(X.shape[1], np.nan)); reports.append({})
            continue

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        roc_aucs.append(roc_auc_multiclass(y_te.values, y_proba))
        pr_aucs.append(pr_auc_multiclass(y_te.values, y_proba))
        f1s.append(classification_report(y_te, y_pred, output_dict=True)["weighted avg"]["f1-score"])
        feats.append(model.feature_importances_)
        reports.append(classification_report(y_te, y_pred, output_dict=True))

    return (
        np.nanmean(accs), np.nanmean(roc_aucs), np.nanmean(pr_aucs),
        np.nanmean(f1s), reports, np.nanmean(feats, axis=0)
    )


# ──────────────────────────── Driver function ────────────────────────────────
def main_kfold(data_files_loc, num_of_clusters, output_loc,
               text_col_name, target_col_name):

    os.makedirs(output_loc, exist_ok=True)
    summary, all_importances = [], []

    MIN_SAMPLES_PER_CLASS = max(N_CV_SPLITS,
                                int(np.ceil(1 / (VALIDATION_SPLIT_SIZE *
                                                 (N_CV_SPLITS - 1) / N_CV_SPLITS)))
                                if N_CV_SPLITS > 1 else 1)

    for c in range(num_of_clusters):
        print(f"\n── Cluster {c} ─────────────────────────────────────────")
        path = os.path.join(data_files_loc, f"{c}_data.csv")
        if not os.path.exists(path):
            print("No file – skipping.")
            continue

        X, y = load_and_prepare_data(path, text_col_name, target_col_name)
        counts = y.value_counts()
        print("Class counts:", counts.to_dict())

        if counts.min() < MIN_SAMPLES_PER_CLASS or counts.nunique() < 2:
            print("Too few samples per class – skipping.")
            continue

        acc, auc, pr, f1, reports, fi = train_with_smote_kfold(X, y, c)
        summary.append({
            "cluster": c,
            "accuracy": acc,
            "roc_auc_macro": auc,
            "pr_auc_macro": pr,
            "weighted_f1": f1,
        })
        all_importances.append({"cluster": c, **dict(zip(X.columns, fi))})

    pd.DataFrame(summary).to_csv(os.path.join(output_loc, "summary.csv"), index=False)
    pd.DataFrame(all_importances).to_csv(os.path.join(output_loc, "feature_importances.csv"), index=False)
    print("\nSaved results →", output_loc)


# ─────────────────────────────── Example usage ───────────────────────────────
if __name__ == "__main__":
    DATA_DIR   = "clusters_csv"
    OUT_DIR    = "XGBoost_Output"
    N_CLUSTERS = 20
    TEXT_COL   = "text"
    TARGET_COL = "sentiment"

    main_kfold(DATA_DIR, N_CLUSTERS, OUT_DIR, TEXT_COL, TARGET_COL)
