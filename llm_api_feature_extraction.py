import json
import os

import dspy
import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class SubthemeExtraction(dspy.Signature):
    """Extract exactly 25 distinct subthemes from the provided text."""
    instruction: str = dspy.InputField(desc="The instruction.")
    texts: str = dspy.InputField(desc="The texts to analyze.")
    # Output field: The themes and subthemes as a nested dictionary
    themes: dict[str, list[str]] = dspy.OutputField(
        desc=(
            "A dictionary where each key is a main theme (as a string), and the value is a list of five "
            "specific subthemes related to that theme. Exactly five themes are required, and each theme must have exactly five subthemes."
        ),
        validator=lambda themes: (
                isinstance(themes, dict) and
                len(themes) == 5 and
                all(len(subthemes) == 5 for subthemes in themes.values())
        )
    )


class SubthemeExtractor(dspy.Module):
    signature = SubthemeExtraction

    def __init__(self):
        super().__init__()
        self.chain_of_thought = dspy.ChainOfThoughtWithHint(SubthemeExtraction)

    def forward(self, texts, instruction):
        return self.chain_of_thought(texts=texts, instruction=instruction)


def _count_cluster_files(clusters_files_loc: str):
    # Match files with the pattern "<cluster>_data.csv"
    cluster_files = [
        f for f in os.listdir(clusters_files_loc)
        if os.path.isfile(os.path.join(clusters_files_loc, f)) and f.endswith("_data.csv")
    ]
    return len(cluster_files)


def _count_tokens(prompt: str, model: str="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))


def _create_cluster_prompt(prompt: str, text_file_loc: str, text_col_name: str, themes: list = None):
    data = pd.read_csv(text_file_loc)
    cluster_text = data[text_col_name]

    # Create base full prompt
    full_prompt = prompt

    # If themes are provided, add them to the prompt
    if themes:
        themes_list_one_per_line = '\n'.join(themes)
        full_prompt += f"\nThemes:\n{themes_list_one_per_line}"

    # Add cluster texts to the prompt
    full_prompt += f"\nCluster texts:\n" + '\n'.join(cluster_text.astype(str))

    return full_prompt


def _create_texts_string(text_file_loc, text_col_name = 'text'):
    data = pd.read_csv(text_file_loc)
    cluster_text = data[text_col_name]
    full_prompt = '\n'.join(cluster_text.astype(str))
    return full_prompt

# direct GPT prompting - Unused
def _get_subthemes(client: OpenAI, full_prompt: str, model: str="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,  # Replace with your model
        messages=[
            {
                "role": "system",
                "content": "Your purpose is to analyze all the texts in the prompt and return subthemes and not high level themes."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "twenty_five_subthemes_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "twenty_five_subthemes": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["twenty_five_subthemes"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        store=True
    )

    # Parse the JSON content from the response
    content = response.choices[0].message.content
    print(f'full response:\n {response}')
    parsed_content = json.loads(content)  # Convert JSON string to Python dictionary

    # Extract the subthemes list
    subthemes = parsed_content.get("twenty_five_subthemes", [])
    print(f'subthemes amount received: {len(subthemes)}')
    return subthemes

# direct GPT prompting - Unused
def llm_feature_extraction_for_cluster_csv(client: OpenAI, text_file_loc, text_col_name, model="gpt-4o-mini"):

    first_stage_instruction = (
        "Analyze this series of stories and questions to identify five main recurring themes. "
        "These themes should reflect patterns in characters' actions, traits, and dynamics, capturing both explicit and subtle ideas across the scenarios. "
        "After identifying the five main themes, expand on each theme by identifying five specific and coherent subthemes that provide more detail. "
        "Ensure each theme has exactly five subthemes. Do not skip or summarize steps.\n\n"
        "Present the output in this exact structured format:\n"
        "- Theme 1\n"
        "  - Subtheme 1.1\n"
        "  - Subtheme 1.2\n"
        "  - Subtheme 1.3\n"
        "  - Subtheme 1.4\n"
        "  - Subtheme 1.5\n"
        "- Theme 2\n"
        "  - Subtheme 2.1\n"
        "  - Subtheme 2.2\n"
        "  - Subtheme 2.3\n"
        "  - Subtheme 2.4\n"
        "  - Subtheme 2.5\n"
        "... \n\n"
        "Ensure the final output includes exactly five themes, each with five subthemes, formatted as described above. "
        "This dataset is for testing ML model fairness, so avoid general descriptions or mentions of individuals."
    )

    # If the LLM returned only 5 main themes without going deeper into the sub themes, a second prompt is sent.
    second_stage_instruction = (
        "The following analysis has already identified five main recurring themes. "
        "Your task is to expand on each theme by identifying exactly five specific and coherent subthemes that provide more detail. "
        "These subthemes should reflect deeper patterns, situations, or dynamics associated with each main theme, capturing both explicit and subtle ideas. "
        "Avoid general descriptions or overly broad categories; focus on concrete and precise subthemes related to the data provided.\n\n"
        "Present the output in this exact structured format:\n"
        "- Theme 1\n"
        "  - Subtheme 1.1\n"
        "  - Subtheme 1.2\n"
        "  - Subtheme 1.3\n"
        "  - Subtheme 1.4\n"
        "  - Subtheme 1.5\n"
        "- Theme 2\n"
        "  - Subtheme 2.1\n"
        "  - Subtheme 2.2\n"
        "  - Subtheme 2.3\n"
        "  - Subtheme 2.4\n"
        "  - Subtheme 2.5\n"
        "... \n\n"
        "Ensure the output includes subthemes for all five themes, with exactly five subthemes per theme. "
        "This dataset is for testing ML model fairness, so avoid general descriptions or mentions of individuals."
    )

    first_full_prompt = _create_cluster_prompt(first_stage_instruction, text_file_loc, 'text')
    token_count = _count_tokens(first_full_prompt, model)

    TOKEN_LIMIT = 128_000
    if token_count > TOKEN_LIMIT:
        print(f"Cluster at {text_file_loc} is too large for {model} context window of {TOKEN_LIMIT} "
              f"with {token_count} tokens.")


    repeats = 1
    sub_themes = _get_subthemes(client, first_full_prompt)
    subthemes_len = len(sub_themes)

    # Another option is to look for the "steps" option and define steps to follow - get themes -> get subthemes
    while subthemes_len < 25 and repeats <= 5:
        # In this case Only the main themes were received -> extracting sub-themes.
        if len(sub_themes) == 5:
            second_full_prompt = _create_cluster_prompt(second_stage_instruction, text_file_loc, 'text')
            sub_themes = _get_subthemes(client, second_full_prompt, model)
            subthemes_len = len(sub_themes)
            repeats += 1

        # when atleast 25 sub-themes are received it is an acceptable result.
        elif len(sub_themes) >= 25:
            print(f'{len(sub_themes)} themes were given: \n {sub_themes}')
            subthemes_len = len(sub_themes)

        else:
            sub_themes = _get_subthemes(client, first_full_prompt)
            subthemes_len = len(sub_themes)
            repeats += 1

    return sub_themes

# direct GPT prompting - Unused
def llm_feature_extraction_for_clusters_folder(client, clusters_files_loc: str, text_col_name: str,
                                               model: str = "gpt-4o-mini") -> pd.DataFrame:

    num_of_clusters = _count_cluster_files(clusters_files_loc)
    llm_features_pd = pd.DataFrame()

    # Loop through all clusters from 0_data.csv to (num_of_clusters - 1)_data.csv
    for i in range(num_of_clusters):
        cluster_file = os.path.join(clusters_files_loc, f"{i}_data.csv")  # Construct file name
        try:
            features = llm_feature_extraction_for_cluster_csv(client, cluster_file, text_col_name, model)
            # llm_features_pd[f"{i}"] = features
            temp_df = pd.DataFrame({f"{i}": features})
            # Concatenate with the main DataFrame, aligning indexes and allowing for NaN values
            llm_features_pd = pd.concat([llm_features_pd, temp_df], axis=1)

        except Exception as e:
            raise(f"Error processing cluster {i}: {e}")

    if len(llm_features_pd.columns) == 0:
        raise Exception(f"No features were generated for all clusters.")
    elif num_of_clusters != len(llm_features_pd.columns):
        print(f"generated features for {len(llm_features_pd.columns)} clusters out of {num_of_clusters} clusters.")
    else:
        print(f"generated features successfully for {num_of_clusters} clusters.")
    return llm_features_pd


def llm_feature_extraction_for_cluster_csv_dspy(text_file_loc, text_col_name, model="gpt-4o-mini"):
    # Token limit set for gpt 4/o/o1/o-mini/o1-mini (including output)
    TOKEN_LIMIT_PER_PROMPT = 128_000  # Full model context window
    RESERVED_TOKENS = 8_000  # Reserved for internal model use, unknown factors
    OUTPUT_TOKEN_ESTIMATE = 2_500  # Increased estimate for safety margin for the JSON output with 25 subthemes

    instruction = (
        "Analyze this series of stories and questions to identify exactly five main recurring themes. "
        "These themes should comprehensively reflect patterns in characters' actions, traits, and dynamics, "
        "capturing both explicit and subtle ideas across the scenarios. "
        "After identifying the five main themes, expand on each theme by identifying exactly five specific and coherent subthemes "
        "that provide more detail and depth. Ensure each theme is accompanied by precisely five subthemes, with no overlaps or omissions. "
        "Do not skip or summarize steps, and ensure the output strictly adheres to the required structure."
    )

    df = pd.read_csv(text_file_loc)

    if text_col_name not in df.columns:
        raise ValueError(f"Column '{text_col_name}' not found in the input file: {text_file_loc}")

    instruction_tokens = _count_tokens(instruction, model)
    # Calculate available token space specifically for the text content of a chunk
    available_text_token_space = TOKEN_LIMIT_PER_PROMPT - instruction_tokens - RESERVED_TOKENS - OUTPUT_TOKEN_ESTIMATE

    if available_text_token_space <= 0:
        raise ValueError(
            f"Instruction tokens ({instruction_tokens}) + reserved ({RESERVED_TOKENS}) + output estimate ({OUTPUT_TOKEN_ESTIMATE}) "
            f"exceed or meet the token limit ({TOKEN_LIMIT_PER_PROMPT}). No space for text."
        )

    all_texts_from_col = df[text_col_name].astype(str).tolist()
    all_results = []
    current_chunk_texts_list = []
    current_chunk_tokens_count = 0
    chunk_number = 0
    newline_token_count = _count_tokens("\n", model) # Usually 1, count it once

    for text_item_content in all_texts_from_col:
        text_item_tokens = _count_tokens(text_item_content, model)

        # Handle individual text items that are too large on their own
        if text_item_tokens > available_text_token_space:
            print(f"Warning: Text item in {text_file_loc} (length: {len(text_item_content)}) is too long ({text_item_tokens} tokens) "
                  f"to fit in any chunk with the current instruction and will be skipped. "
                  f"Available space for text: {available_text_token_space} tokens.")
            # Option: Truncate here if necessary, or log this specific item
            # For now, we skip it
            continue

        # Check if adding the current text item (plus a newline if it's not the first in chunk) would exceed the limit
        potential_new_tokens = text_item_tokens
        if current_chunk_texts_list: # If chunk is not empty, a newline will be added before this item
            potential_new_tokens += newline_token_count

        if current_chunk_tokens_count + potential_new_tokens <= available_text_token_space:
            current_chunk_texts_list.append(text_item_content)
            current_chunk_tokens_count += potential_new_tokens
        else:
            # Process the current chunk as it's full
            if current_chunk_texts_list:
                chunk_number += 1
                print(f"Processing chunk {chunk_number} for {text_file_loc} with {current_chunk_tokens_count} text tokens...")
                chunk_text_as_string = '\n'.join(current_chunk_texts_list)

                # Final check for this specific chunk's total tokens
                final_chunk_prompt_tokens = instruction_tokens + _count_tokens(chunk_text_as_string, model)
                if final_chunk_prompt_tokens + RESERVED_TOKENS + OUTPUT_TOKEN_ESTIMATE > TOKEN_LIMIT_PER_PROMPT:
                    print(f"CRITICAL: Chunk {chunk_number} for {text_file_loc} still calculated to exceed total token limit "
                          f"({final_chunk_prompt_tokens} + {RESERVED_TOKENS} + {OUTPUT_TOKEN_ESTIMATE} > {TOKEN_LIMIT_PER_PROMPT}). "
                          f"Skipping this chunk to prevent API error. This indicates an issue in token calculation or an extremely dense chunk.")
                else:
                    try:
                        extractor = SubthemeExtractor()
                        result = extractor(texts=chunk_text_as_string, instruction=instruction)
                        for key, value_list in result.themes.items():
                            all_results.extend(value_list)
                    except Exception as e:
                        print(f"Error processing chunk {chunk_number} for {text_file_loc} with DSPy: {e}")


            # Start a new chunk with the current text_item
            current_chunk_texts_list = [text_item_content]
            current_chunk_tokens_count = text_item_tokens # No newline token for the first item in a new chunk

    # Process the last remaining chunk, if any
    if current_chunk_texts_list:
        chunk_number += 1
        print(f"Processing final chunk {chunk_number} for {text_file_loc} with {current_chunk_tokens_count} text tokens...")
        chunk_text_as_string = '\n'.join(current_chunk_texts_list)

        final_chunk_prompt_tokens = instruction_tokens + _count_tokens(chunk_text_as_string, model)
        if final_chunk_prompt_tokens + RESERVED_TOKENS + OUTPUT_TOKEN_ESTIMATE > TOKEN_LIMIT_PER_PROMPT:
             print(f"CRITICAL: Final chunk {chunk_number} for {text_file_loc} still calculated to exceed total token limit "
                   f"({final_chunk_prompt_tokens} + {RESERVED_TOKENS} + {OUTPUT_TOKEN_ESTIMATE} > {TOKEN_LIMIT_PER_PROMPT}). "
                   f"Skipping this chunk.")
        else:
            try:
                extractor = SubthemeExtractor()
                result = extractor(texts=chunk_text_as_string, instruction=instruction)
                for key, value_list in result.themes.items():
                    all_results.extend(value_list)
            except Exception as e:
                print(f"Error processing final chunk {chunk_number} for {text_file_loc} with DSPy: {e}")


    if not all_results and df.empty:
        print(f"No texts found in {text_file_loc} to process.")
    elif not all_results and not df.empty:
        print(f"No results were generated from {text_file_loc}. This could be due to all texts being skipped or an LLM issue.")

    return all_results


def llm_feature_extraction_for_clusters_folder_dspy(clusters_files_loc: str, text_col_name: str,
                                               model: str = "gpt-4o-mini") -> pd.DataFrame:

    num_of_clusters = _count_cluster_files(clusters_files_loc)
    llm_features_pd = pd.DataFrame()

    # Loop through all clusters from 0_data.csv to (num_of_clusters - 1)_data.csv
    for i in range(num_of_clusters):
        cluster_file = os.path.join(clusters_files_loc, f"{i}_data.csv")  # Construct file name
        try:
            features = llm_feature_extraction_for_cluster_csv_dspy(cluster_file, text_col_name, model)
            # llm_features_pd[f"{i}"] = features
            temp_df = pd.DataFrame({f"{i}": features})
            # Concatenate with the main DataFrame, aligning indexes and allowing for NaN values
            llm_features_pd = pd.concat([llm_features_pd, temp_df], axis=1)

        except Exception as e:
            raise(f"Error processing cluster {i}: {e}")

    if len(llm_features_pd.columns) == 0:
        raise Exception(f"No features were generated for all clusters.")
    elif num_of_clusters != len(llm_features_pd.columns):
        print(f"generated features for {len(llm_features_pd.columns)} clusters out of {num_of_clusters} clusters.")
    else:
        print(f"generated features successfully for {num_of_clusters} clusters.")
    return llm_features_pd


if __name__ == '__main__':
    text_file_loc = 'clusters csv'
    text_col_name = 'text'
    api_key = os.getenv('OPENAI_KEY')
    # client = OpenAI(api_key=api_key)

    # features = llm_feature_extraction_for_cluster_csv(client, text_file_loc, text_col_name)
    # print(features)
    # features_pd = llm_feature_extraction_for_clusters_folder(client, text_file_loc, text_col_name, model="gpt-4o-mini")

    lm = dspy.LM('openai/gpt-4o-mini',
                 api_key='ADD KEY HERE',
                 cache=False)
    dspy.configure(lm=lm)

    results = llm_feature_extraction_for_cluster_csv_dspy('testfolder2\\0_data.csv', text_col_name, model="gpt-4o-mini")
    print(results)
    print(f'amount of results: {len(results)}')
    # features_pd = llm_feature_extraction_for_clusters_folder_dspy(text_file_loc, text_col_name, model="gpt-4o-mini")
    # ai_features_file_name = os.path.join(text_file_loc, 'ai_features_df2.csv')
    # features_pd.to_csv(ai_features_file_name, index=False)
