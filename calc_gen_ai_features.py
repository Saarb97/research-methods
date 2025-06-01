#!/usr/bin/env python3
"""
Zero-shot sentence→hypothesis scoring with MoritzLaurer DeBERTa-v3 models.

Fix: earlier code stacked the SAME logit twice, so every probability became 0.50.
We now select the real 'entailment' column after softmax.
"""

import os, time, torch, pandas as pd, numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Global configuration ──────────────────────────────────────────────────────
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model / tokenizer once
nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nli_model.to(device).eval()                       # eval() = no dropout
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

# Cache label IDs once for speed & safety
LABEL2ID   = nli_model.config.label2id
ENTAIL_ID  = LABEL2ID.get("entailment")
# 2-label models use "not_entailment"; 3-label MNLI-style use "contradiction"
NEG_ID     = LABEL2ID.get("not_entailment", LABEL2ID.get("contradiction"))

if ENTAIL_ID is None:
    raise ValueError(f"{MODEL_NAME} has no 'entailment' label in config.label2id")

# ── Core helper ───────────────────────────────────────────────────────────────
def _compute_probabilities(sentences, hypotheses):
    """
    Return P(entailment) for each (sentence, hypothesis) pair.

    Output shape: list[ dict{hypothesis: prob, …} ]  ← aligned with sentences
    """
    if not sentences or not hypotheses:
        return [{} for _ in sentences]

    # Build premise–hypothesis pairs
    premise, hypo = zip(*[(s, h) for s in sentences for h in hypotheses])

    inputs = tokenizer(
        list(premise), list(hypo),
        padding=True, truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits        # (N, num_labels)

    # Softmax once across ALL labels (works for 2- or 3-label models)
    probs_entail = torch.softmax(logits, dim=-1)[:, ENTAIL_ID]   # (N,)

    # Reshape back → (len(sentences), len(hypotheses))
    probs_entail = probs_entail.view(len(sentences), len(hypotheses)).cpu().tolist()

    return [dict(zip(hypotheses, row)) for row in probs_entail]

# ── File-level processing (unchanged except for new _compute_probabilities) ──
def _process_file_with_classification(file_index, ai_features,
                                      data_files_location, text_col_name,
                                      hypothesis_chunk_size=10):
    path = os.path.join(data_files_location, f"{file_index}_data.csv")
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("Missing:", path); return
    except pd.errors.EmptyDataError:
        print("Empty file:", path); return

    if text_col_name not in data.columns:
        print(f"{path}: no '{text_col_name}' column"); return

    sentences = data[text_col_name].fillna("").astype(str).tolist()
    all_scores = []

    for sent in sentences:
        if not sent.strip():                       # blank line
            all_scores.append({})
            continue

        row_scores = {}
        for i in range(0, len(ai_features), hypothesis_chunk_size):
            chunk = ai_features[i:i+hypothesis_chunk_size]
            row_scores.update(_compute_probabilities([sent], chunk)[0])
        all_scores.append(row_scores)

    scores_df = pd.DataFrame(all_scores)
    out_df    = pd.concat([data, scores_df], axis=1)
    out_df.to_csv(path, index=False)
    print(f"{file_index}: wrote {len(scores_df.columns)} score cols to {path}")

def parallel_process_files(tasks_list):
    """Sequential loop (easy to swap for multiprocessing)."""
    for kw in tasks_list:
        _process_file_with_classification(**kw)

# ── Utility: read the hypotheses CSV ──────────────────────────────────────────
def _check_ai_features_file(ai_features_file_location):
    clustered = pd.read_csv(ai_features_file_location)
    cols      = clustered.columns.tolist()
    return clustered, cols

# ── PUBLIC API: keep name & args unchanged! ───────────────────────────────────
def deberta_for_llm_features(ai_features_file_location: str,
                             data_files_location: str,
                             text_col_name: str,
                             hypothesis_chunk_size: int = 10) -> None:
    """
    Reads clustered hypotheses, scores every <cluster>_data.csv in `data_files_location`,
    and writes P(entailment) columns back into each file.
    """
    print("Torch:", torch.__version__,
          "| CUDA:", torch.cuda.is_available(),
          "| Device:", device if not torch.cuda.is_available()
                               else torch.cuda.get_device_name(0))

    clustered, cols = _check_ai_features_file(ai_features_file_location)

    tasks = []
    for col in cols:                    # each column header is the cluster_id
        feats = clustered[str(col)].dropna().astype(str).tolist()
        if not feats:
            print(f"{col}: no hypotheses, skipping"); continue
        tasks.append({
            "file_index": str(col),
            "ai_features": feats,
            "data_files_location": data_files_location,
            "text_col_name": text_col_name,
            "hypothesis_chunk_size": hypothesis_chunk_size
        })

    if tasks:
        parallel_process_files(tasks)
    else:
        print("No clusters to process!")

# ── Stand-alone execution (unchanged defaults) ───────────────────────────────
if __name__ == "__main__":
    AI_FEATURES_CSV = "clustered_ai_features.csv"
    DATA_DIR        = "clusters_csv"
    TEXT_COL        = "text"
    CHUNK_SIZE      = 10

    t0 = time.time()
    deberta_for_llm_features(AI_FEATURES_CSV, DATA_DIR, TEXT_COL, CHUNK_SIZE)
    print(f"Done in {time.time()-t0:.1f}s")
