import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.linear_model import LogisticRegression
import numpy as np
from xgboost_clusters import train_xgboost_with_SMOTE
from sklearn.model_selection import train_test_split
import shap
import os
import multiprocessing
from functools import partial
import traceback
from tqdm import tqdm
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import traceback
import time

def load_and_prepare_data(file_name, text_col_name):
    """Load data from a CSV file and prepare it for modeling."""
    data = pd.read_csv(file_name)

    # Define the columns to drop before analysing results
    columns_to_drop = [text_col_name, 'cluster', 'named_entities', 'Unnamed: 0']

    # Filter columns that exist in the data
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=existing_columns_to_drop, inplace=True)
    return data

def get_shap_feature_importance(data_file_name, text_col_name, target_col_name):
    """
    Get SHAP feature importance with additional safeguards against empty data.
    """
    try:
        data = load_and_prepare_data(data_file_name, text_col_name)
        X = data.drop(columns=[target_col_name])
        y = data[target_col_name]
        
        # Check for empty data
        if X.empty or y.empty:
            print(f"Empty data in {data_file_name}")
            return None
            
        # Check for sufficient class distribution
        if len(y.unique()) < 2:
            print(f"Only one class present in {data_file_name}")
            return None
            
        # Configure XGBoost to use appropriate number of threads
        # Let XGBoost use all available cores for this single task
        model = train_xgboost_with_SMOTE(X, y)
        
        # Guard against possible NaN values from model
        if model is None:
            print(f"XGBoost model training failed for {data_file_name}")
            return None
        
        # Raw log odds feature importance
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Get model predictions
        y_pred = model.predict(X)
        
        # Filter indices where the model predicts class 0 and the true label is also 0 (True Negatives)
        true_negatives_indices = (y_pred == 0) & (y == 0)
        
        # Check if we have any true negatives
        if sum(true_negatives_indices) == 0:
            print(f"No true negatives in predictions for {data_file_name}")
            return None
            
        # Get the indices of true negatives
        class_0_indices = np.where(true_negatives_indices)[0]
        
        # Get the corresponding SHAP values for the samples predicted as class 0
        class_0_shap_values = shap_values[class_0_indices]
        
        # Check if we have any SHAP values
        if len(class_0_shap_values) == 0:
            print(f"No SHAP values for class 0 in {data_file_name}")
            return None
            
        # Calculate median SHAP values for each feature
        class_0_median_shap_values = np.nanmedian(class_0_shap_values, axis=0)
        
        # Check for NaN values in median SHAP values
        if np.isnan(class_0_median_shap_values).any():
            print(f"NaN values in median SHAP values for {data_file_name}")
            class_0_median_shap_values = np.nan_to_num(class_0_median_shap_values)
            
        shap_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'SHAP_Importance': class_0_median_shap_values
        }).sort_values(ascending=True, by=['SHAP_Importance'])
        
        return shap_importance_df
    except Exception as e:
        print(f"Error in SHAP calculation: {e}")
        traceback.print_exc()
        return None

def calc_statistical_information(data_df, test_type, target_col_name):
    # Initialize dictionaries to store the median values and test results
    median_0 = {}
    median_1 = {}
    t_statistics = {}
    p_values = {}
    significant = {}

    # Loop through each column except target_col_name
    for column in data_df.columns:
        if column != target_col_name:
            # Get groups for analysis
            group_0 = data_df[data_df[target_col_name] == 0][column]
            group_1 = data_df[data_df[target_col_name] == 1][column]
            
            # Calculate medians with safeguards
            if len(group_0) > 0:
                median_0[column] = group_0.median()
            else:
                median_0[column] = float('nan')
                
            if len(group_1) > 0:
                median_1[column] = group_1.median()
            else:
                median_1[column] = float('nan')

            # Perform test with proper error handling
            try:
                if len(group_0) > 0 and len(group_1) > 0 and group_0.var() > 0 and group_1.var() > 0:
                    if test_type == 't-test':
                        t_stat, p_val = ttest_ind(group_0, group_1, nan_policy='omit')
                    elif test_type == 'Mann-Whitney U test':
                        t_stat, p_val = mannwhitneyu(group_0, group_1, alternative='two-sided')
                    else:
                        print('Unknown test, expected "t-test" or "Mann-Whitney U test", using Mann-Whitney as default')
                        t_stat, p_val = mannwhitneyu(group_0, group_1, alternative='two-sided')
                else:
                    t_stat, p_val = float('nan'), float('nan')
            except Exception as e:
                print(f"Error in statistical test for column {column}: {e}")
                t_stat, p_val = float('nan'), float('nan')

            # Store test results
            t_statistics[column] = t_stat
            p_values[column] = p_val
            significant[column] = p_val < 0.05 if not np.isnan(p_val) else False

    # Create a new DataFrame to store the median values
    median_df = pd.DataFrame([median_0, median_1], index=['median_0', 'median_1'])

    # Append test results to the DataFrame
    median_df.loc['t_statistic'] = t_statistics
    median_df.loc['p_value'] = p_values
    median_df.loc['significant'] = significant

    return median_df

def run_logistic_regression(X, y, num_of_features):
    try:
        # should experiment with 'lbfgs' solver
        log_reg = LogisticRegression(max_iter=10000, solver='saga').fit(X, y)

        # y is a binary hence location [0]
        coefficients = log_reg.coef_[0]
        intercept = log_reg.intercept_[0]

        # Calculate the odds ratio for each feature
        odds_ratios = np.exp(coefficients)

        # Convert odds ratios to probability increase
        prob_increase = (odds_ratios - 1) * 100

        # Create a DataFrame to display the results
        feature_effects = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': coefficients,
            'Odds Ratio': odds_ratios,
            'Probability Increase (%)': prob_increase
        })

        # Sort the DataFrame by the probability increase (lowest to highest)
        sorted_features = feature_effects.sort_values(by='Probability Increase (%)').head(num_of_features)
        sorted_features['Probability Increase (%)'] = sorted_features['Probability Increase (%)'].abs()

        return sorted_features
    except Exception as e:
        print(f"Error in logistic regression: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=['Feature', 'Coefficient', 'Odds Ratio', 'Probability Increase (%)'])

def analyse_results(results_folder_location: str, num_of_clusters: int, destination: str, text_col_name: str, target_col_name: str) -> None:
    """
    Analyze cluster results sequentially, allowing XGBoost and SHAP to use internal multithreading.
    
    Args:
        results_folder_location: Path to the folder containing cluster CSV files
        num_of_clusters: Number of clusters to analyze
        destination: Path to save the analysis results
        text_col_name: Name of the text column to exclude
        target_col_name: Name of the target column
    """
    # Make sure the destination directory exists
    os.makedirs(destination, exist_ok=True)
    
    # Track successes and failures
    successful_clusters = 0
    failed_clusters = 0
    errors = {}
    
    print(f"Starting sequential analysis of {num_of_clusters} clusters")
    start_time = time.time()
    
    # Create a tqdm progress bar
    for i in tqdm(range(num_of_clusters), desc="Analyzing clusters", unit="cluster"):
        cluster_start_time = time.time()
        
        # Using tqdm.write to avoid interfering with progress bar
        tqdm.write(f'Cluster {i} out of {num_of_clusters} analysis began')
        
        try:
            data_file_name = os.path.join(results_folder_location, f'{i}_data.csv')
            data_df = load_and_prepare_data(data_file_name, text_col_name)

            # Check if data is too small or imbalanced
            class_counts = data_df[target_col_name].value_counts()
            if len(class_counts) < 2 or min(class_counts) < 5:
                tqdm.write(f"Cluster {i} skipped: insufficient data or extreme imbalance")
                errors[i] = "Insufficient data or extreme class imbalance"
                failed_clusters += 1
                continue

            test_types = ['t-test', 'Mann-Whitney U test']
            median_df = calc_statistical_information(data_df, test_types[1], target_col_name)

            # Filter the significant features
            significant_features = median_df.columns[median_df.loc['significant'] == 1]
            
            if len(significant_features) == 0:
                tqdm.write(f"Cluster {i}: No significant features found")
                errors[i] = "No significant features found"
                failed_clusters += 1
                continue

            # Separate features and target
            X = data_df[significant_features]
            y = data_df[target_col_name]

            num_of_features = 10
            tqdm.write(f"Cluster {i}: Running logistic regression")
            statistical_important_features = run_logistic_regression(X, y, num_of_features)
            statistical_important_features.to_csv(os.path.join(destination, f'{i}_statistical.csv'), index=False)

            tqdm.write(f"Cluster {i}: Starting SHAP analysis")
            shap_feature_importance = get_shap_feature_importance(data_file_name, text_col_name, target_col_name)
            if shap_feature_importance is not None:
                shap_feature_importance.head(num_of_features).to_csv(os.path.join(destination, f'{i}_shap.csv'), index=False)
                successful_clusters += 1
            else:
                tqdm.write(f"Cluster {i}: SHAP analysis returned None")
                errors[i] = "SHAP analysis failed"
                failed_clusters += 1

            cluster_time = time.time() - cluster_start_time
            tqdm.write(f"Cluster {i} completed in {cluster_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f'Error analysing results for cluster {i}: {e}\n{traceback.format_exc()}'
            tqdm.write(error_msg)
            tqdm.write(f'Error could be from very imbalanced cluster.')
            tqdm.write(f'skipping cluster {i}.')
            errors[i] = str(e)
            failed_clusters += 1
    
    total_time = time.time() - start_time
    print(f'\nAnalysis completed in {total_time:.2f} seconds.')
    print(f'Successfully processed {successful_clusters} clusters.')
    print(f'Failed to process {failed_clusters} clusters.')
    
    # Print errors if any
    if failed_clusters > 0:
        print("\nError summary:")
        for cluster_idx, error in errors.items():
            print(f"Cluster {cluster_idx}: {error[:200]}..." if len(error) > 200 else f"Cluster {cluster_idx}: {error}")

if __name__ == '__main__':

    analyse_results(f'clusters csv', 20,f'results')

    # for i in range(20):  # Loop from 0_data.csv to 19_data.csv
    #     print(f'Cluster {i}')
    #     data_file_name = f'clusters csv\\{i}_data.csv'
    #     statistics_file_name = f'clusters csv\\{i}_statistics.csv'
    #     data_df = load_and_prepare_data(data_file_name)
    #     statistics_df = pd.read_csv(statistics_file_name)
    #
    #     test_types = ['t-test', 'Mann-Whitney U test']
    #     median_df = calc_statistical_information(data_df, test_types[1])
    #
    #     # Display the combined DataFrame
    #     #print(median_df)
    #     #median_df.to_csv(f'clusters csv\\{i}_results_analysis.csv', index=True)
    #
    #     # Filter the significant features
    #     significant_features = median_df.columns[median_df.loc['significant'] == 1]
    #     # print(significant_features)
    #     # Separate features and target
    #     X = data_df[significant_features]
    #     y = data_df[target_col_name]
    #
    #     num_of_features = 10
    #     statistical_important_features = run_logistic_regression(X, y, num_of_features)
    #     statistical_important_features.to_csv(f'results\\{i}_statistical.csv', index=False)
    #
    #     shap_feature_importance = get_shap_feature_importance(data_file_name).head(num_of_features)
    #     #print(shap_feature_importance)
    #     shap_feature_importance.to_csv(f'results\\{i}_shap.csv', index=False)