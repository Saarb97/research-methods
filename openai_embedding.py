import json
import re
import time
import os
import pandas as pd
import tiktoken
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv('OPENAI_KEY')
# Initialize the OpenAI client
# Note: Initializing the client outside the function is fine,
# as it's generally thread-safe for making requests.
client = OpenAI(api_key=api_key)

# Define the embedding model and related parameters
embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
# The maximum tokens for text-embedding-3-small is 8191, using 8000 as a safe limit
max_tokens = 8000

# Get the encoding for token counting
encoding = tiktoken.get_encoding(embedding_encoding)

def get_embedding(text, model=embedding_model):
    """
    Generates an embedding for a given text string using the specified OpenAI model.

    Args:
        text (str): The input text string to embed.
        model (str): The name of the OpenAI embedding model to use.

    Returns:
        list: A list of floats representing the embedding vector, or None if an error occurs.
    """
    max_retries = 3  # Number of retries in case of rate limit error
    retry_delay = 60  # Pause for 60 seconds on rate limit error

    attempt = 0
    while attempt < max_retries:
        try:
            text = text.replace("\n", " ")
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding

        except Exception as e:
            error_message = str(e).lower()

            if 'rate limit' in error_message or '429' in error_message:
                print(f"Rate limit hit. Attempt {attempt + 1}/{max_retries}. Pausing for {retry_delay} seconds...")
                time.sleep(retry_delay)
                attempt += 1
            else:
                print(f"Error generating embedding")
                return None

    print("Exceeded maximum retries after rate limit errors. Skipping this text.")
    return None

def process_single_row_for_embedding(row_idx, text, delay_per_request=0.02):
    """
    Helper function to process a single text entry, truncate if needed,
    and get its embedding. Includes a small delay after the API call.
    Designed for use with ThreadPoolExecutor.

    Args:
        row_idx (int): The original index of the row in the DataFrame.
        text (str): The text content from the specified column.
        delay_per_request (float): A small delay in seconds to wait after each API call.

    Returns:
        tuple: A tuple containing the original row index and the generated embedding (list or None).
    """
    # Ensure text is a string before processing
    if not isinstance(text, str):
        # print(f"Warning: Row {row_idx} in text column is not a string. Skipping.") # Avoid excessive prints
        return row_idx, None # Return index and None for non-string entries

    # Truncate text if it exceeds the maximum token limit
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        # print(f"Warning: Text in row {row_idx} exceeds {max_tokens} tokens. Truncating.") # Avoid excessive prints
        text = encoding.decode(tokens[:max_tokens])

    # Generate embedding for the (potentially truncated) text
    embedding = get_embedding(text, model=embedding_model)

    # Add a small delay after the API call to help smooth out request rate
    if delay_per_request > 0:
        time.sleep(delay_per_request)

    return row_idx, embedding


def process_dataframe_with_embeddings_v2(df: pd.DataFrame, text_column_name: str, output_filename: str, max_workers: int = 50, max_requests_per_minute: int = 3000, time_window: int = 62, delay_per_request: float = 0.02):
    """
    Processes a pandas DataFrame to generate embeddings for a specified text column
    using parallel processing with rate limiting and a small delay per request.
    It truncates texts exceeding the maximum token limit and saves only the
    'embedding' column to a Parquet file.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column_name (str): The name of the column containing the text to embed.
        output_filename (str): The name of the Parquet file to save the DataFrame with
                               only the 'embedding' column.
        max_workers (int): The maximum number of threads to use for parallel processing.
        max_requests_per_minute (int): The maximum number of API requests allowed per minute.
        time_window (int): The time window in seconds for rate limiting (usually 60).
        delay_per_request (float): A small delay in seconds to wait after each API call
                                   within a worker thread.

    Returns:
        pd.DataFrame: A DataFrame containing only the 'embedding' column.
                      Returns an empty DataFrame if the text column is not found.
    """
    # Ensure the specified text column exists
    if text_column_name not in df.columns:
        print(f"Error: Text column '{text_column_name}' not found in the DataFrame.")
        return pd.DataFrame() # Return an empty df if column not found

    print(f"Processing DataFrame with text column: '{text_column_name}' using {max_workers} workers and {delay_per_request}s delay per request...")

    # Initialize a list to store embeddings, pre-filled with None to maintain order
    embeddings = [None] * len(df)
    processed_count = 0

    # Use ThreadPoolExecutor for parallel API calls
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        # We use df.iterrows() to get both index and row data
        # Pass the delay_per_request to the helper function
        futures = {executor.submit(process_single_row_for_embedding, idx, row[text_column_name], delay_per_request): idx for idx, row in df.iterrows()}

        completed_requests = 0
        # Process results as they complete
        for future in as_completed(futures):
            row_idx, embedding = future.result()
            embeddings[row_idx] = embedding # Place embedding at the correct original index
            completed_requests += 1
            processed_count += 1

            # Print progress periodically
            if processed_count % 100 == 0 or processed_count == len(df):
                 print(f"Processed {processed_count}/{len(df)} rows")

            # Rate limiting logic (still useful as a safeguard, though per-request delay helps)
            elapsed_time = time.time() - start_time
            # Check if we've exceeded the request limit within the time window
            # Note: The per-request delay helps avoid hitting this often,
            # but it's good to keep as a fallback.
            if completed_requests >= max_requests_per_minute and elapsed_time < time_window:
                sleep_time = time_window - elapsed_time
                print(f"Rate limit ({max_requests_per_minute} req/{time_window}s) reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Reset the counter and timer after sleeping
                completed_requests = 0
                start_time = time.time()
            elif elapsed_time >= time_window:
                 # If time window has passed, reset counter and timer without sleeping
                 completed_requests = 0
                 start_time = time.time()


    # Create a new DataFrame containing only the embeddings
    # The embeddings list is already in the correct order due to indexing
    embeddings_df = pd.DataFrame({'embedding': embeddings})

    # Save the resulting DataFrame (containing only embeddings) to a Parquet file
    try:
        # Ensure the directory for the output file exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Save as parquet
        embeddings_df.to_parquet(output_filename, index=False)
        print(f"Processing completed successfully. Results saved to '{output_filename}'.")
    except Exception as e:
        print(f"Error saving DataFrame to Parquet: {e}")

    return embeddings_df

file_path = 'Twitter_covid /preproccessed_finalSentimentdata2.csv' 
try:
    df = pd.read_csv(file_path)

    text_col = 'text' # Replace with your actual text column name
    output_file = 'Twitter_covid /full_embeddings.parquet' # Desired output file name

    # Process the DataFrame to get embeddings
    # Added delay_per_request parameter
    embeddings_only_df = process_dataframe_with_embeddings_v2(df, text_col, output_file, max_workers=50, max_requests_per_minute=3000, delay_per_request=0.03) # Added delay

    # The embeddings_only_df now contains only the 'embedding' column
    print("\nDataFrame containing only embeddings (first 5 rows):")
    print(embeddings_only_df.head())

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred during processing: {e}")
