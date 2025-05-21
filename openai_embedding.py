import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Function to get embeddings with rate limit handling
def get_embedding(text, model):
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
                print(f"Error generating embedding for text: {text[:50]}... -> {e}")
                return None

    print("Exceeded maximum retries after rate limit errors. Skipping this text.")
    return None


def process_row(row_idx, text, embedding_model):
    embedding = get_embedding(text, model=embedding_model)
    return row_idx, embedding

def get_dataset_embeddings(file_path, client: OpenAI,text_col_name: str, model="text-embedding-3-small", embedding_encoding="cl100k_base"):
    # API settings
    max_tokens = 8150  # maximum tokens allowed per request

    df = pd.read_csv(file_path)
    encoding = tiktoken.get_encoding(embedding_encoding)

    # Filter the DataFrame
    df = df[[text_col_name]]
    df["n_tokens"] = df[text_col_name].apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]

    # Multithreading settings
    max_requests_per_minute = 2900 # 3000 is the actual limit
    max_threads = 50  # Number of concurrent threads
    time_window = 70  # Time window in seconds

    # Initialize a list for embeddings
    embeddings = [None] * len(df)

    # Process rows using ThreadPoolExecutor
    start_time = time.time()
    completed_requests = 0

    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(process_row, idx, row[text_col_name], model) for idx, row in df.iterrows()]

        for future in as_completed(futures):
            row_idx, embedding = future.result()
            embeddings[row_idx] = embedding
            completed_requests += 1
            print(f"Processed row {row_idx + 1}/{len(df)}")

            # Rate limiting to 3000 requests per minute
            if completed_requests % max_requests_per_minute == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < time_window:
                    sleep_time = time_window - elapsed_time
                    print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                start_time = time.time()  # Reset after the sleep

    # Add embeddings to the DataFrame
    df['embedding'] = embeddings
    # the comma seperated values causes bugs when saving as a CSV file
    df['embedding'] = df['embedding'].apply(lambda x: '|'.join(map(str, x)))

    # To revert:
    # df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.split('|'))))

    # Save the resulting DataFrame
    # df.to_csv('deepkeep_openai_fairness_test2.csv', index=False)
    return df


if __name__ == '__main__':
    # text_file_loc = 'sarcasm_dataset'
    text_col_name = 'text'

    api_key = os.getenv('OPENAI_KEY')
    client = OpenAI(api_key=api_key)
    #file_path = os.path.join('sarcasm_dataset', 'sarcasm_with_preds.csv')
    df = get_dataset_embeddings('twitter sentiment/processed/twitter_training.csv', client, text_col_name)
    df.to_csv('twitter sentiment/processed/twitter_training_openai_embeddings2.csv', index=False)

