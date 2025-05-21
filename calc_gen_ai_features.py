import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer, util # Not used in DeBERTa path
from transformers import pipeline # Keep if legacy classifier is ever used
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import time
import torch
import os
# from multiprocessing import Pool, cpu_count # Keep if true parallelism is restored
from torch.multiprocessing import set_start_method # Keep for multiprocessing setup
# from torch.utils.data import DataLoader, Dataset # No longer needed for the new approach
# from itertools import chain # Keep, might be used in _compute_probabilities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# It's good practice to load model and tokenizer once globally if they don't change
# Consider enabling quantization for significant memory savings
#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.float16,
#)

nli_model = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    #quantization_config=bnb_config, # Enable this for 4-bit quantization
    # device_map="auto" # Recommended with BitsAndBytes for automatic device placement
)
# If not using device_map="auto", then explicit .to(device) is needed.
# However, with quantization_config, .to(device) might behave differently or be handled by device_map.
# If device_map is not used, and you encounter issues, try nli_model.to(device) after loading.
# For now, let's assume device_map="auto" (you'd add it above) or BitsAndBytes handles it.
# If you don't use device_map="auto" and quantization_config, then this is correct:
if not hasattr(nli_model, 'hf_device_map'): # Simple check if device_map might have been used
    nli_model.to(device)

tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

# Legacy classifier - keep if you might switch back or use it elsewhere
# classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=0 if torch.cuda.is_available() else -1)


def _compute_probabilities(sentences, hypotheses):
    """
    Computes probabilities for a batch of sentences and hypotheses.
    """
    if not sentences or not hypotheses:
        return []

    # Ensure hypotheses is a flat list of strings.
    # The way ai_features is generated should already ensure this.
    if not all(isinstance(hypothesis, str) for hypothesis in hypotheses):
        malformed_hypotheses = [h for h in hypotheses if not isinstance(h, str)][:5]
        raise ValueError(f"Invalid hypotheses structure. All hypotheses must be strings. Found: {malformed_hypotheses} in {hypotheses}")

    # Prepare premise-hypothesis pairs
    premise_hypothesis_pairs = []
    for sentence in sentences:
        for hypothesis in hypotheses:
            premise_hypothesis_pairs.append((sentence, hypothesis))
    
    # Unzip for tokenizer
    tokenization_sentences, tokenization_hypotheses = zip(*premise_hypothesis_pairs)

    try:
        inputs = tokenizer(
            list(tokenization_sentences), # list of sentences, repeated
            list(tokenization_hypotheses), # list of hypotheses, corresponding to sentences
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length
        ).to(device)
    except Exception as e:
        print("Error during tokenization.")
        print("Sentences chunk:", sentences)
        print("Hypotheses chunk:", hypotheses)
        raise e

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    entail_contradiction_logits = logits[:, [0, 1]] # MoritzLaurer/deberta-v3-large-zeroshot-v2.0 maps entailment to label 2 (index 1 after reordering to [contradiction, entailment]) or specific index based on config.
                                                   # The original code used probs[:,1] for "hypothesis true", assuming index 1 is entailment.
                                                   # Default for this model: {"contradiction": 0, "neutral": 1, "entailment": 2}
                                                   # The pipeline likely reorders/selects entailment and contradiction.
                                                   # For direct model use: logits for "entailment" (usually index 2) vs "contradiction" (index 0)
                                                   # MoritzLaurer's pipeline uses entailment vs. contradiction.
                                                   # Logits shape: (batch_size, num_classes) num_classes is usually 3 (contra, neutral, entail)
                                                   # We need probability of entailment.
                                                   # Default order: 0: contradiction, 1: neutral, 2: entailment
                                                   # The pipeline implementation does:
                                                   # entail_logits = outputs.logits[:, self.entailment_id]
                                                   # contradiction_logits = outputs.logits[:, self.contradiction_id]
                                                   # So we need to pick the correct indices for entailment and contradiction from the model's config.
                                                   # For "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", entailment_id=2, contradiction_id=0 (from its config)
    
    # Correctly extract entailment and contradiction logits based on common practice for NLI models like this one
    # Typically, label2id: {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    entail_logits = logits[:, nli_model.config.label2id.get('entailment', 2)]
    contradiction_logits = logits[:, nli_model.config.label2id.get('contradiction', 0)]

    # Stack them for softmax: effectively [contradiction_logit, entailment_logit] for each pair
    stacked_logits = torch.stack([contradiction_logits, entail_logits], dim=1)
    probs = stacked_logits.softmax(dim=1)
    probs_hypothesis_true = probs[:, 1] # Probability of entailment

    # Reshape to (num_sentences, num_hypotheses_in_chunk)
    probs_hypothesis_true_reshaped = probs_hypothesis_true.view(len(sentences), len(hypotheses)).tolist()


    probabilities_output = []
    for i, sentence in enumerate(sentences):
        probabilities_output.append(dict(zip(hypotheses, probs_hypothesis_true_reshaped[i])))

    del inputs, logits, entail_logits, contradiction_logits, stacked_logits, probs, probs_hypothesis_true, probs_hypothesis_true_reshaped
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probabilities_output


def _process_file_with_classification(file_index, ai_features, data_files_location, text_col_name, hypothesis_chunk_size=10):
    """Process a single file, compute scores by chunking hypotheses, and save results."""
    file_name = os.path.join(data_files_location, f'{file_index}_data.csv')
    try:
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File not found: {file_name}. Skipping.")
        return
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_name}. Skipping.")
        return

    if text_col_name not in data.columns:
        print(f"Text column '{text_col_name}' not found in {file_name}. Skipping.")
        return
    
    cluster_text_list = data[text_col_name].fillna('').astype(str).tolist()

    start_proc_time = time.time()
    print(f'Starting processing for cluster {file_index} with {len(cluster_text_list)} sentences and {len(ai_features)} features. Max Hypotheses per call: {hypothesis_chunk_size}')

    all_scores_for_file = []

    for idx, sentence_text in enumerate(cluster_text_list):
        # print(f"  Processing sentence {idx+1}/{len(cluster_text_list)} in cluster {file_index}: {sentence_text[:50]}...") # Verbose
        if not sentence_text.strip():
            all_scores_for_file.append({})
            continue

        sentence_scores = {}
        for i in range(0, len(ai_features), hypothesis_chunk_size):
            hypotheses_chunk = ai_features[i:i + hypothesis_chunk_size]
            if not hypotheses_chunk:
                continue
            
            try:
                # _compute_probabilities expects a list of sentences. Here, it's a list with one sentence.
                chunk_probabilities_list = _compute_probabilities([sentence_text], hypotheses_chunk)
                if chunk_probabilities_list: # It returns a list of dicts; we expect one dict for the one sentence
                    sentence_scores.update(chunk_probabilities_list[0])
            except Exception as e:
                print(f"Error computing probabilities for sentence chunk in cluster {file_index} (sentence {idx+1}, hypothesis chunk {i//hypothesis_chunk_size + 1}): {e}")
                # Fill with NaN or skip these hypotheses for this sentence
                for hypo in hypotheses_chunk:
                    sentence_scores[hypo] = np.nan 

        all_scores_for_file.append(sentence_scores)

    if not all_scores_for_file and not cluster_text_list: # Both text and scores are empty
        print(f"No text to process in cluster {file_index}.")
        # Save the original data if it was just empty text column, or do nothing
        # data.to_csv(file_name, index=False) # Resave if necessary
        return 
    elif not all_scores_for_file and cluster_text_list: # Text existed, but all might have been skipped (e.g. all empty strings)
        scores_df = pd.DataFrame(index=data.index, columns=ai_features if ai_features else None) # Create empty df with correct columns
    else:
        scores_df = pd.DataFrame(all_scores_for_file)

    # Align DataFrame index if necessary and concatenate
    if not scores_df.empty:
        scores_df.index = data.index[:len(scores_df)] 
    
    # Ensure all original ai_features columns exist in scores_df, filling missing ones (e.g. due to errors) with NaN
    for feature in ai_features:
        if feature not in scores_df.columns:
            scores_df[feature] = np.nan
    
    # Reorder score_df columns to match ai_features order if desired, though not strictly necessary
    # if ai_features:
    #    scores_df = scores_df.reindex(columns=ai_features, fill_value=np.nan)


    result_df = pd.concat([data, scores_df], axis=1)
    
    try:
        result_df.to_csv(file_name, index=False)
    except Exception as e:
        print(f"Error saving results to {file_name}: {e}")

    end_proc_time = time.time()
    print(f'Finished processing cluster {file_index} in {end_proc_time - start_proc_time:.2f}s. Processed {len(cluster_text_list)} sentences.')


def parallel_process_files(tasks_list):
    """
    Processes tasks. Currently sequential, but can be adapted for true parallelism.
    A task in tasks_list is expected to be a dictionary with necessary arguments.
    """
    # Parallelism is currently disabled in the provided code.
    # If you want to enable it with multiprocessing.Pool:
    # num_processes = min(cpu_count(), 2) # Or your desired number
    # print(f"Using {num_processes} processes for parallel execution.")
    # with Pool(processes=num_processes) as pool:
    #     pool.map(process_single_file_wrapper, tasks_list) # process_single_file_wrapper would unpack dict task_args

    for task_args in tasks_list:
        _process_file_with_classification(
            file_index=task_args["file_index"],
            ai_features=task_args["ai_features"],
            data_files_location=task_args["data_files_location"],
            text_col_name=task_args["text_col_name"],
            hypothesis_chunk_size=task_args.get("hypothesis_chunk_size", 10) # Default if not provided
        )

# Wrapper function if you use multiprocessing.Pool.map, which takes a single argument
# def process_single_file_wrapper(task_args):
# return _process_file_with_classification(
# file_index=task_args["file_index"],
# ai_features=task_args["ai_features"],
# data_files_location=task_args["data_files_location"],
# text_col_name=task_args["text_col_name"],
# hypothesis_chunk_size=task_args.get("hypothesis_chunk_size", 10)
#     )


def _check_ai_features_file(ai_features_file_location: str):
    try:
        print(f'Reading AI features file: {ai_features_file_location}')
        clustered_ai_features = pd.read_csv(ai_features_file_location) # Removed encoding, add back if needed
        cols = clustered_ai_features.columns.tolist()
        for col in cols:
            int(col) 
        return clustered_ai_features, cols
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: AI features file not found at {ai_features_file_location}.")
    # ... (other specific exceptions from your original code) ...
    except Exception as e:
        raise Exception(f"An unexpected error occurred while checking AI features file: {e}")


def deberta_for_llm_features(
    ai_features_file_location: str,
    data_files_location: str,
    text_col_name: str, # Added text_col_name parameter
    hypothesis_chunk_size: int = 10 # Added hypothesis_chunk_size with a default
) -> None:
    # set_start_method("spawn", force=True) # Usually needed if using CUDA with multiprocessing
    print('Torch info for running DeBERTa:')
    print(f'Is CUDA available? {torch.cuda.is_available()}')
    print(f'Torch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'Torch Device: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    try:
        clustered_ai_features, cols = _check_ai_features_file(ai_features_file_location)
        tasks_with_features = []
        for col in cols: # These 'cols' are expected to be file_index prefixes
            try:
                # Ensure col is treated as a string key for the DataFrame if it's not already
                ai_features_for_col = clustered_ai_features[str(col)].dropna().astype(str).tolist()
                if not ai_features_for_col:
                    print(f"Warning: No AI features found for cluster {col}. Skipping this cluster's processing logic if it depends on features.")
                
                tasks_with_features.append({
                    "file_index": str(col), # Ensure file_index is a string if cols are numbers
                    "ai_features": ai_features_for_col,
                    "data_files_location": data_files_location,
                    "text_col_name": text_col_name,
                    "hypothesis_chunk_size": hypothesis_chunk_size
                })
            except KeyError:
                print(f"Warning: Column {col} not found in AI features file. Skipping.")
            except ValueError: # From int(col) in _check_ai_features_file if cols are not int-like
                print(f"Non-numeric cluster identifier in AI features file column name: {col}. Treating as string identifier.")
                # Adapt as needed if column names are not purely numeric strings
                ai_features_for_col = clustered_ai_features[col].dropna().astype(str).tolist()
                tasks_with_features.append({
                    "file_index": str(col),
                    "ai_features": ai_features_for_col,
                    "data_files_location": data_files_location,
                    "text_col_name": text_col_name,
                    "hypothesis_chunk_size": hypothesis_chunk_size
                })


        if not tasks_with_features:
            print("No tasks to process based on the AI features file.")
            return

        parallel_process_files(tasks_with_features)
    except Exception as e:
        print(f"An error occurred in deberta_for_llm_features: {e}") # More specific error message
        raise # Re-raise after printing


if __name__ == '__main__':
    # It's good practice to set the start_method for multiprocessing at the very beginning of the main block
    # if you intend to use multiprocessing with CUDA.
    #try:
    #    set_start_method("spawn", force=True)
    #except RuntimeError:
    #    print("set_start_method has already been called or context is not appropriate (e.g. not main module).")


    print('--- DeBERTa Zero-Shot Classification ---')
    
    # Configuration
    ai_features_csv_path = 'clustered_ai_features.csv' # Path to your CSV with AI features/hypotheses
    data_input_folder = 'clusters_csv' # Folder containing the data CSVs to be processed
    text_column_in_data_csv = 'text'   # The name of the column containing text to classify in your data CSVs
    # Adjust hypothesis_chunk_size based on your VRAM capacity and number/length of hypotheses.
    # Start small (e.g., 5-10) and increase if you have VRAM to spare.
    chunk_size_for_hypotheses = 10

    print(f"AI Features File: {ai_features_csv_path}")
    print(f"Data Folder: {data_input_folder}")
    print(f"Text Column Name in data files: {text_column_in_data_csv}")
    print(f"Hypothesis Chunk Size: {chunk_size_for_hypotheses}")
    print('--- Starting processing ---')
    
    start_time = time.time()
    try:
        deberta_for_llm_features(
            ai_features_file_location=ai_features_csv_path,
            data_files_location=data_input_folder,
            text_col_name=text_column_in_data_csv,
            hypothesis_chunk_size=chunk_size_for_hypotheses
        )
    except Exception as e:
        print(f"Main execution failed: {e}")
    finally:
        end_time = time.time()
        print(f'--- Finished processing all tasks. Total time: {end_time - start_time:.2f}s ---')