import pandas as pd
import os
import warnings
from tqdm import tqdm
import lingam
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for results and data
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
feature_importance_path = os.path.join(results_dir, 'feature_importance_scores.csv')

# Build absolute paths to the data files
sentiment_file_path = os.path.join(script_dir, 'imdb', 'test_set_predictions.csv')
current_file_path = os.path.join(script_dir, 'imdb', 'test_embeddings_parallel_v3.parquet')

print(f"Looking for embeddings file at: {current_file_path}")
print(f"Looking for sentiment file at: {sentiment_file_path}")

# Load only the required columns
try:
    embeddings_df = pd.read_parquet(current_file_path)
    sentiment_df = pd.read_csv(sentiment_file_path, usecols=['performance'])
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Please ensure the 'imdb' directory exists in the same directory as the script,")
    print("and contains the required files ('test_embeddings_parallel_v3.parquet', 'test_set_predictions.csv').")
    exit() # Exit if files aren't found

# Ensure matching lengths
if len(embeddings_df) != len(sentiment_df):
    raise ValueError("The number of rows in embeddings and sentiment files do not match!")

# Add sentiment back to embeddings
embeddings_df['performance'] = sentiment_df['performance']

# Expand the embedding list into separate columns
embedding_features = pd.DataFrame(embeddings_df['embedding'].tolist(), index=embeddings_df.index)

# Add performance back to the expanded DataFrame
embedding_features['performance'] = embeddings_df['performance']

# Display the resulting DataFrame
print("Initial DataFrame head:")
print(embedding_features.head())

def causal_fs(x, y):
    """
    Perform causal feature selection using LiNGAM model.
    
    Args:
        x: Feature matrix
        y: Target variable
        
    Returns:
        Tuple of (selected indices, feature importance scores, processed dataframe)
    """
    warnings.filterwarnings("ignore")

    df = pd.DataFrame(x)
    # Normalize features
    df = (df-df.min())/(df.max()-df.min())
    # Insert target
    df.insert(loc=0, column='target', value=y)
    columns = range(0,len(df.columns))
    feature_importance = []
    batch_size=3

    # --- First Pass ---
    for i in tqdm(range((len(columns)//batch_size) + 1), desc="First Pass"):
        model = lingam.ICALiNGAM(2,1)
        untill = min(len(columns), (1+(i+1)*batch_size))
        curr = df.iloc[:,[columns[0]] + list(columns[(1+i*batch_size):untill])]
        if len(curr.columns)<=1:
            continue
        try:
            _ = model.fit(curr)
        except Exception:
            feature_importance = np.concatenate((feature_importance, [0]*(len(curr.columns)-1)), axis=0)
            continue
        if len(feature_importance) != 0:
            sub_feat_importance = np.maximum(np.absolute(model.adjacency_matrix_[0, 1:]), np.absolute(model.adjacency_matrix_[1:, 0]))
            feature_importance = np.concatenate((feature_importance, sub_feat_importance), axis=0)
        else:
            feature_importance = np.maximum(np.absolute(model.adjacency_matrix_[0, 1:]), np.absolute(model.adjacency_matrix_[1:, 0]))

    feature_importance = np.concatenate(([0],feature_importance), axis=0)
    indecies = np.argwhere(feature_importance!=0)
    indecies = list(indecies.flatten())

    feature_importance_absolute = np.absolute(feature_importance)

    print(f"Indices after first pass: {indecies}")
    batch_size=3

    # --- Second Pass ---
    for indice in tqdm(indecies, desc="Second Pass"):
        feature_importance = []

        for i in range((len(columns)//batch_size) + 1):
            model = lingam.ICALiNGAM(2, 1)
            untill = min(len(columns), ((i+1)*batch_size))
            elem_list = list(columns[(i*batch_size):untill])
            to_add = False
            zero_index_to_add = -1

            if indice in elem_list:
                to_add = True
                zero_index_to_add = elem_list.index(indice)
                elem_list.remove(indice)

            if not elem_list and not to_add:
                num_zeros_in_batch = len(list(columns[(i*batch_size):untill]))
                if num_zeros_in_batch > 0:
                    feature_importance = np.concatenate((feature_importance, [0]*num_zeros_in_batch), axis=0)
                continue

            cols_to_fit = [columns[indice]] + elem_list

            if len(cols_to_fit) <= 1:
                num_zeros_in_batch = len(list(columns[(i*batch_size):untill]))
                if num_zeros_in_batch > 0:
                    feature_importance = np.concatenate((feature_importance, [0]*num_zeros_in_batch), axis=0)
                continue

            try:
                _ = model.fit(df.iloc[:, cols_to_fit])
            except Exception as e:
                num_zeros = len(elem_list)
                if to_add:
                    num_zeros += 1
                feature_importance = np.concatenate((feature_importance, [0]*num_zeros), axis=0)
                print(f'Exception occurred for indice {indice} in batch {i}. Added {num_zeros} zeros. Error: {e}')
                continue

            current_adjacency_matrix_part = np.maximum(np.absolute(model.adjacency_matrix_[0, 1:]), np.absolute(model.adjacency_matrix_[1:, 0]))

            if to_add:
                current_adjacency_matrix_part = np.insert(current_adjacency_matrix_part, zero_index_to_add, 0)

            if len(feature_importance) != 0:
                feature_importance = np.concatenate((feature_importance, current_adjacency_matrix_part), axis=0)
            else:
                feature_importance = current_adjacency_matrix_part

        feature_importance = np.array(feature_importance)
        feature_importance = np.absolute(feature_importance)

        if feature_importance_absolute.shape != feature_importance.shape:
            print(f"!!! Shape mismatch before np.maximum for indice {indice} !!!")
            print(f"feature_importance_absolute shape: {feature_importance_absolute.shape}")
            print(f"feature_importance shape: {feature_importance.shape}")
            if feature_importance.shape[0] < feature_importance_absolute.shape[0]:
                print("Padding feature_importance with zeros.")
                feature_importance = np.pad(feature_importance, (0, feature_importance_absolute.shape[0] - feature_importance.shape[0]))

        feature_importance_absolute = np.maximum(feature_importance_absolute, feature_importance*feature_importance_absolute[indice])

    feature_importance_absolute[0] = 0
    warnings.resetwarnings()

    final_selected_indices = np.where(feature_importance_absolute[1:] > 0)[0]
    final_importance_scores = feature_importance_absolute[1:]

    return final_selected_indices, final_importance_scores, df.drop('target', axis=1)

def run_improved_kmeans_evaluation(X, n_clusters=100):
    """
    Run k-means clustering and evaluate using internal validation metrics only
    (metrics that don't require ground truth labels)
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters to form
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        # Run k-means++ (explicitly use k-means++ initialization)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate internal validation metrics (don't require ground truth)
        sil_score = silhouette_score(X_scaled, cluster_labels)
        chi = calinski_harabasz_score(X_scaled, cluster_labels)
        dbi = davies_bouldin_score(X_scaled, cluster_labels)
        
        # Calculate inertia (sum of squared distances to closest centroid)
        inertia = kmeans.inertia_
        
        # Print debug info
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster distribution: {np.bincount(cluster_labels)[:5]}... (showing first 5)")
        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Calinski-Harabasz Index: {chi:.4f}")
        print(f"  Davies-Bouldin Index: {dbi:.4f} (lower is better)")
        print(f"  Inertia: {inertia:.4f}")
        
        return {
            'silhouette_score': sil_score, 
            'calinski_harabasz_index': chi,
            'davies_bouldin_index': dbi,
            'inertia': inertia,
            'cluster_labels': cluster_labels
        }
    except Exception as e:
        print(f"Error in clustering with {n_clusters} clusters: {e}")
        return {
            'silhouette_score': 0, 
            'calinski_harabasz_index': 0,
            'davies_bouldin_index': float('inf'),
            'inertia': float('inf'),
            'cluster_labels': None
        }

# --- Prepare data ---
X_df = embedding_features.drop('performance', axis=1)
x = X_df.values
y = embedding_features['performance'].values

# --- Check if feature importance scores already exist ---
if os.path.exists(feature_importance_path):
    print(f"\nFound existing feature importance scores at {feature_importance_path}. Loading...")
    feature_importance_df = pd.read_csv(feature_importance_path)
    feature_importance = feature_importance_df['importance_score'].values
    
    # Check if the dimensions match our current dataset
    if len(feature_importance) == x.shape[1]:
        print("Loaded feature importance scores match the current dataset dimensions.")
        indices = np.where(feature_importance > 0)[0]
        print(f"Number of selected features from loaded file: {len(indices)}")
    else:
        print(f"Warning: Loaded feature importance dimensions ({len(feature_importance)}) don't match current dataset ({x.shape[1]}).")
        print("Running causal feature selection from scratch...")
        indices, feature_importance, processed_df = causal_fs(x, y)
        
        # Save the new feature importance scores
        feature_importance_df = pd.DataFrame({
            'feature_index': np.arange(len(feature_importance)),
            'importance_score': feature_importance
        })
        feature_importance_df.to_csv(feature_importance_path, index=False)
        print(f"New feature importance scores saved to {feature_importance_path}")
else:
    print(f"\nNo existing feature importance scores found. Starting causal feature selection with {x.shape[1]} features.")
    indices, feature_importance, processed_df = causal_fs(x, y)
    
    # Save the feature importance scores
    feature_importance_df = pd.DataFrame({
        'feature_index': np.arange(len(feature_importance)),
        'importance_score': feature_importance
    })
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"Feature importance scores saved to {feature_importance_path}")

# --- Display results ---
print("\n--- Feature Selection Results ---")
print(f"Final Feature Importance length: {len(feature_importance)}")
print(f"Number of selected features: {len(indices)}")

# --- Sort indices by feature importance ---
sorted_indices = np.argsort(-feature_importance)  # Descending order

# --- Running K-means clustering with different feature set sizes ---
print("\n--- Running K-means clustering with different feature set sizes ---")
k_values = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1536]
clustering_results = []

# Initialize result storage
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
inertia_values = []

for k in tqdm(k_values, desc="Evaluating feature sets"):
    # Select top k features
    top_k_indices = sorted_indices[:k]
    X_selected = x[:, top_k_indices]
    
    print(f"\nEvaluating with top {k} features:")
    # Use 100 clusters as specified
    n_clusters = 100
    results = run_improved_kmeans_evaluation(X_selected, n_clusters=n_clusters)
    
    # Store results
    results['k'] = k
    clustering_results.append(results)
    
    silhouette_scores.append(results['silhouette_score'])
    calinski_harabasz_scores.append(results['calinski_harabasz_index'])
    davies_bouldin_scores.append(results['davies_bouldin_index'])
    inertia_values.append(results['inertia'])
    
    print(f"k={k}: Silhouette={results['silhouette_score']:.4f}, CHI={results['calinski_harabasz_index']:.4f}, " +
          f"DBI={results['davies_bouldin_index']:.4f} (lower is better), Inertia={results['inertia']:.2f}")

# Optional: If performance values are meaningful for another analysis (not as ground truth)
# you could analyze how performance relates to clusters discovered
if 'performance' in embedding_features.columns:
    # For the best k (based on silhouette score)
    best_k_index = np.argmax(silhouette_scores)
    best_k = k_values[best_k_index]
    best_clustering = clustering_results[best_k_index]
    
    if best_clustering['cluster_labels'] is not None:
        # Create a dataframe with cluster assignments and performance
        cluster_analysis = pd.DataFrame({
            'cluster': best_clustering['cluster_labels'],
            'performance': embedding_features['performance'].values
        })
        
        # Analyze relationship between clusters and performance
        cluster_performance = cluster_analysis.groupby('cluster')['performance'].agg(['mean', 'std', 'count'])
        print("\n--- Performance across clusters (using feature set size k =", best_k, ") ---")
        print(cluster_performance.sort_values('count', ascending=False).head(10))
        
        # Optional plot of cluster vs performance distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='cluster', y='performance', data=cluster_analysis.loc[cluster_analysis['cluster'].isin(range(20))])
        plt.title(f'Performance Distribution by Cluster (Top 20 clusters, k={best_k})')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cluster_performance_distribution.jpg'), dpi=300)
        print(f"Cluster performance plot saved to {os.path.join(results_dir, 'cluster_performance_distribution.jpg')}")

# Save clustering results
clustering_results_df = pd.DataFrame([
    {
        'k': result['k'],
        'silhouette_score': result['silhouette_score'],
        'calinski_harabasz_index': result['calinski_harabasz_index'],
        'davies_bouldin_index': result['davies_bouldin_index'],
        'inertia': result['inertia']
    }
    for result in clustering_results
])
clustering_results_df.to_csv(os.path.join(results_dir, 'clustering_results.csv'), index=False)
print(f"Clustering results saved to {os.path.join(results_dir, 'clustering_results.csv')}")

# --- CREATE AND SAVE PLOTS ---
# Create figure for metrics
plt.figure(figsize=(12, 10))

# Plot silhouette scores (higher is better)
plt.subplot(2, 2, 1)
plt.plot(k_values, silhouette_scores, 'o-', color='blue')
plt.title('Silhouette Score vs. Number of Features')
plt.xlabel('Number of Features (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

# Plot Calinski-Harabasz scores (higher is better)
plt.subplot(2, 2, 2)
plt.plot(k_values, calinski_harabasz_scores, 'o-', color='orange')
plt.title('Calinski-Harabasz Index vs. Number of Features')
plt.xlabel('Number of Features (k)')
plt.ylabel('CHI')
plt.grid(True)

# Plot Davies-Bouldin scores (lower is better)
plt.subplot(2, 2, 3)
plt.plot(k_values, davies_bouldin_scores, 'o-', color='green')
plt.title('Davies-Bouldin Index vs. Number of Features (lower is better)')
plt.xlabel('Number of Features (k)')
plt.ylabel('DBI')
plt.grid(True)

# Plot Inertia (lower is better but watch for diminishing returns)
plt.subplot(2, 2, 4)
plt.plot(k_values, inertia_values, 'o-', color='purple')
plt.title('Inertia vs. Number of Features (lower is better)')
plt.xlabel('Number of Features (k)')
plt.ylabel('Inertia')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'clustering_evaluation.jpg'), dpi=300)
print(f"Evaluation plot saved to {os.path.join(results_dir, 'clustering_evaluation.jpg')}")

# Create a separate figure for the normalized combined metrics
plt.figure(figsize=(12, 6))

# Normalize metrics for better comparison
def normalize_metric(metric_values, higher_is_better=True):
    # Convert list to numpy array to allow array operations
    metric_values = np.array(metric_values)
    min_val = np.min(metric_values)
    max_val = np.max(metric_values)
    if max_val == min_val:
        return np.zeros_like(metric_values)
    normalized = (metric_values - min_val) / (max_val - min_val)
    # If lower values are better, invert the normalization
    if not higher_is_better:
        normalized = 1 - normalized
    return normalized

# Convert lists to numpy arrays before normalization
norm_silhouette = normalize_metric(np.array(silhouette_scores), higher_is_better=True)
norm_chi = normalize_metric(np.array(calinski_harabasz_scores), higher_is_better=True)
norm_dbi = normalize_metric(np.array(davies_bouldin_scores), higher_is_better=False)
norm_inertia = normalize_metric(np.array(inertia_values), higher_is_better=False)

plt.plot(k_values, norm_silhouette, 'o-', color='blue', label='Normalized Silhouette')
plt.plot(k_values, norm_chi, 'o-', color='orange', label='Normalized CHI')
plt.plot(k_values, norm_dbi, 'o-', color='green', label='Normalized DBI (inverted)')
plt.plot(k_values, norm_inertia, 'o-', color='purple', label='Normalized Inertia (inverted)')
plt.title('Normalized Clustering Metrics vs. Number of Features')
plt.xlabel('Number of Features (k)')
plt.ylabel('Normalized Score (higher is better)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'combined_metrics.jpg'), dpi=300)
print(f"Combined metrics plot saved to {os.path.join(results_dir, 'combined_metrics.jpg')}")

# Plot feature importance distribution
plt.figure(figsize=(12, 6))
sns.histplot(feature_importance, bins=30, kde=True)
plt.title('Distribution of Feature Importance Scores')
plt.xlabel('Importance Score')
plt.ylabel('Count')
plt.savefig(os.path.join(results_dir, 'feature_importance_distribution.jpg'), dpi=300)
print(f"Feature importance distribution plot saved to {os.path.join(results_dir, 'feature_importance_distribution.jpg')}")

# Plot top 50 feature importance scores
plt.figure(figsize=(14, 8))
top_50_indices = sorted_indices[:50]
top_50_scores = feature_importance[top_50_indices]
plt.bar(range(50), top_50_scores)
plt.title('Top 50 Feature Importance Scores')
plt.xlabel('Feature Rank')
plt.ylabel('Importance Score')
plt.savefig(os.path.join(results_dir, 'top_50_features.jpg'), dpi=300)
print(f"Top 50 features plot saved to {os.path.join(results_dir, 'top_50_features.jpg')}")

print("\n--- Analysis complete! ---")