# ───────────────────────────────── Imports ────────────────────────────────────
import os, time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, accuracy_score
)
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support
)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

import optuna, xgboost as xgb, torch
from pathlib import Path
import joblib


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
                    eval_metric="auc")   # or "auc" if XGB≥1.7
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
def train_with_smote_kfold(X, y, cluster_idx, models_dir=None):
    kf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, roc_aucs, pr_aucs, f1s, feats, reports = [], [], [], [], [], []
    fold_pred_dfs = []
    
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model = train_xgboost_with_SMOTE(X_tr, y_tr)
        if model is None:
            accs.append(np.nan); roc_aucs.append(np.nan); pr_aucs.append(np.nan); f1s.append(np.nan)
            feats.append(np.full(X.shape[1], np.nan)); reports.append({})
            continue
        
        # Save the first model
        if fold == 0:
          if models_dir is not None:
            fname = models_dir / f"cluster{cluster_idx}_fold{fold}.pkl"
            joblib.dump({"model": final_model,          # saves the estimator
                         "features": list(X_tr.columns)}, fname)
        
        # ---------- predictions ----------
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        # ---- guarantee a 2-column matrix in the binary case ----
        if y_proba.ndim == 1:   # some builds return shape (n,)
            y_proba = np.vstack([1 - y_proba, y_proba]).T
        
        prob_cols = [f"p_{c}" for c in range(y_proba.shape[1])]
        
        # ---- build BOTH frames on the SAME index, then concat ----
        df_meta  = pd.DataFrame(
            {
                "row_id": X_te.index,        # keeps global row id
                "true"  : y_te.values,
                "pred"  : y_pred,
                "cluster": cluster_idx
            },
            index=X_te.index                # identical index
        )
        df_proba = pd.DataFrame(y_proba, columns=prob_cols, index=X_te.index)
        fold_pred_dfs.append(pd.concat([df_meta, df_proba], axis=1))

        # ---------- metrics ----------
        accs.append(accuracy_score(y_te, y_pred))
        roc_aucs.append(roc_auc_multiclass(y_te.values, y_proba))
        pr_aucs.append(pr_auc_multiclass(y_te.values, y_proba))
        f1s.append(classification_report(y_te, y_pred,
                                         output_dict=True, zero_division=0)["weighted avg"]["f1-score"])
        feats.append(model.feature_importances_)
        reports.append(classification_report(y_te, y_pred, output_dict=True, zero_division=0))

    return (
        np.nanmean(accs), np.nanmean(roc_aucs), np.nanmean(pr_aucs),
        np.nanmean(f1s), reports, np.nanmean(feats, axis=0),
        pd.concat(fold_pred_dfs, ignore_index=True) 
    )

# ──────────────────────────── Driver function ────────────────────────────────
def main_kfold(data_files_loc, num_of_clusters, output_loc,
               text_col_name, target_col_name):

    os.makedirs(output_loc, exist_ok=True)
    models_root = Path(output_loc) / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    summary, all_importances, all_preds = [], [], []

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

        acc, auc, pr, f1, reports, fi, preds_df = train_with_smote_kfold(X, y, c, models_dir=models_root)
        summary.append({
            "cluster": c,
            "accuracy": acc,
            "roc_auc_macro": auc,
            "pr_auc_macro": pr,
            "weighted_f1": f1,
        })
        all_importances.append({"cluster": c, **dict(zip(X.columns, fi))})
        all_preds.append(preds_df)

    # ---------- per-cluster CSVs ----------
    pd.DataFrame(summary).to_csv(os.path.join(output_loc, "summary.csv"), index=False)
    pd.DataFrame(all_importances).to_csv(os.path.join(output_loc, "feature_importances.csv"), index=False)

    # ---------- GLOBAL METRICS ----------
    if all_preds:  # protect against empty list
        # ---------------------------------------------------------------------------
        # 1.  Basic vectors
        # ---------------------------------------------------------------------------
        global_preds = (
            pd.concat(all_preds, ignore_index=True)
              .sort_values("row_id")
        )
        y_true = global_preds["true"].values
        y_hat  = global_preds["pred"].values
        labels = np.unique(np.concatenate([y_true, y_hat]))   # all classes seen
        
        # ---------------------------------------------------------------------------
        # 2.  Accuracy & weighted-F1 (same for binary / multiclass)
        # ---------------------------------------------------------------------------
        acc_glob = accuracy_score(y_true, y_hat)
        f1_glob  = classification_report(
            y_true, y_hat, output_dict=True, zero_division=0
        )["weighted avg"]["f1-score"]
        
        # ---------------------------------------------------------------------------
        # 3.  Probability-based metrics (macro ROC-AUC & PR-AUC)
        #     – only if you stored predict_proba columns named “p_<class>”
        # ---------------------------------------------------------------------------
        prob_cols = [c for c in global_preds.columns if c.startswith("p_")]
        if prob_cols:                                # might be missing if you skipped proba
            proba_full = global_preds[prob_cols].values
        
            if proba_full.shape[1] == 2:             # binary
                auc_glob = roc_auc_score(y_true, proba_full[:, 1])
                pr_glob  = average_precision_score(y_true, proba_full[:, 1])
            else:                                    # multiclass
                auc_glob = roc_auc_score(
                    y_true, proba_full, multi_class="ovr", average="macro"
                )
                pr_glob  = average_precision_score(
                    y_true, proba_full, average="macro"
                )
        else:
            auc_glob = np.nan
            pr_glob  = np.nan
        
        # ---------------------------------------------------------------------------
        # 4.  Micro-averaged confusion-matrix counts
        # ---------------------------------------------------------------------------
        if len(labels) == 2:                         # binary → 2×2 matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=labels).ravel()
        else:                                        # multiclass → stack one-vs-rest 2×2 matrices
            mcm = multilabel_confusion_matrix(y_true, y_hat, labels=labels)
            tn = mcm[:, 0, 0].sum()
            fp = mcm[:, 0, 1].sum()
            fn = mcm[:, 1, 0].sum()
            tp = mcm[:, 1, 1].sum()
        
        # ---------------------------------------------------------------------------
        # 5.  Micro metrics (identical formulas for binary & multiclass)
        # ---------------------------------------------------------------------------
        precision     = tp / (tp + fp) if tp + fp else 0.0          # PPV
        recall        = tp / (tp + fn) if tp + fn else 0.0          # TPR
        specificity   = tn / (tn + fp) if tn + fp else 0.0          # TNR
        fpr           = fp / (fp + tn) if fp + tn else 0.0
        fnr           = fn / (fn + tp) if fn + tp else 0.0
        fdr           = fp / (fp + tp) if fp + tp else 0.0
        npv           = tn / (tn + fn) if tn + fn else 0.0
        
        bal_acc = balanced_accuracy_score(y_true, y_hat)
        mcc     = matthews_corrcoef(y_true, y_hat)
        
        # ---------------------------------------------------------------------------
        # 6.  Optional macro / weighted averages (often helpful for class imbalance)
        # ---------------------------------------------------------------------------
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_hat, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_hat, average="weighted", zero_division=0
        )
        
        # ---------------------------------------------------------------------------
        # 7.  Dump everything to a TXT file
        # ---------------------------------------------------------------------------
        os.makedirs(output_loc, exist_ok=True)
        metrics_txt = os.path.join(output_loc, "global_metrics.txt")
        
        with open(metrics_txt, "w") as fout:
            fout.write("=== Whole-dataset (out-of-fold) metrics ===\n")
            fout.write(f"Samples                : {len(y_true)}\n\n")
        
            fout.write(f"True Positives  (TP)   : {tp}\n")
            fout.write(f"False Positives (FP)   : {fp}\n")
            fout.write(f"True Negatives  (TN)   : {tn}\n")
            fout.write(f"False Negatives (FN)   : {fn}\n\n")
        
            fout.write(f"Accuracy               : {acc_glob:.4f}\n")
            fout.write(f"Weighted F1            : {f1_glob:.4f}\n")
            fout.write(f"Micro Precision (PPV)  : {precision:.4f}\n")
            fout.write(f"Micro Recall           : {recall:.4f}\n")
            fout.write(f"Specificity (TNR)      : {specificity:.4f}\n")
            fout.write(f"False-Positive Rate    : {fpr:.4f}\n")
            fout.write(f"False-Negative Rate    : {fnr:.4f}\n")
            fout.write(f"False Discovery Rate   : {fdr:.4f}\n")
            fout.write(f"Negative Predictive Val: {npv:.4f}\n")
            fout.write(f"Balanced Accuracy      : {bal_acc:.4f}\n")
            fout.write(f"Matthews Corr. Coef.   : {mcc:.4f}\n\n")
        
            fout.write("--- Macro averages ---\n")
            fout.write(f"Macro Precision        : {prec_macro:.4f}\n")
            fout.write(f"Macro Recall           : {rec_macro:.4f}\n")
            fout.write(f"Macro F1               : {f1_macro:.4f}\n\n")
        
            fout.write("--- Weighted averages ---\n")
            fout.write(f"Weighted Precision     : {prec_weighted:.4f}\n")
            fout.write(f"Weighted Recall        : {rec_weighted:.4f}\n")
            fout.write(f"Weighted F1            : {f1_weighted:.4f}\n\n")
        
            fout.write(f"Macro ROC-AUC          : {auc_glob:.4f}\n")
            fout.write(f"Macro PR-AUC           : {pr_glob:.4f}\n")
        
        # quick console preview
        print(open(metrics_txt).read())

        # ---------------------------------------------------------------------------
        # 8. save raw OOF predictions for later analysis
        # ---------------------------------------------------------------------------
        global_preds.to_csv(os.path.join(output_loc, "oof_predictions.csv"), index=False)
        
        print("Saved results →", output_loc)



# ─────────────────────────────── Example usage ───────────────────────────────
if __name__ == "__main__":
    DATA_DIR   = "clusters_csv"
    OUT_DIR    = "XGBoost_Output"
    N_CLUSTERS = 20
    TEXT_COL   = "text"
    TARGET_COL = "sentiment"

    main_kfold(DATA_DIR, N_CLUSTERS, OUT_DIR, TEXT_COL, TARGET_COL)
