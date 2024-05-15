import pandas as pd
import numpy as np

def summarize_binary_value_counts_with_threshold(df):
    """
    Summarizes the value counts of binary categorical features in the DataFrame, excluding the 'Status' feature.
    Also calculates the minimum number of instances required to likely have
    each category present in every fold of a k-fold cross-validation.
    
    Adds a column 'ImbalanceRatio' indicating the ratio of the minority to majority category counts,
    where a lower value indicates a greater imbalance.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    
    Returns:
    - summary_df: DataFrame summarizing the features, categories, counts, and imbalance ratio.
    """
    summary_list = []
    for column in df.columns:
        # Exclude the 'Status' feature from the calculation
        if column == "Status":
            continue
        
        value_counts = df[column].value_counts()
        # Proceed only if the feature has exactly two unique categories
        if len(value_counts) <= 2:
            minority_count = value_counts.min()
            majority_count = value_counts.max()
            imbalance_ratio = (minority_count / majority_count)*100
            for category, count in value_counts.items():
                summary_list.append({
                    'Feature': column,
                    'Category': category,
                    'Count': count,
                    'ImbalanceRatio': round(imbalance_ratio, 2)
                })
    
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
    else:
        summary_df = pd.DataFrame(columns=['Feature', 'Category', 'Count', 'ImbalanceRatio'])
    
    return summary_df


def remove_highly_imbalanced_binary_features(df, imbalance_threshold):
    """
    Remove binary features from a DataFrame based on an imbalance threshold, where the ImbalanceRatio
    (minority count divided by majority count) is below the specified threshold, excluding the 'Status' feature.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - imbalance_threshold: The minimum imbalance ratio a binary feature must have to be retained.
    
    Returns:
    - A DataFrame with highly imbalanced binary features removed.
    """
    df_filtered = df.copy()
    
    for column in df.columns:
        # Exclude the 'Status' feature from being processed
        if column == "Status":
            continue
        
        if df[column].dtype == 'object' or isinstance(df[column].dtype, pd.CategoricalDtype):
            value_counts = df[column].value_counts()
            
            # Proceed only if the feature has exactly two unique categories
            if len(value_counts) == 2:
                imbalance_ratio = (value_counts.min() / value_counts.max())*100
                
                # Check if the ImbalanceRatio is below the threshold
                if imbalance_ratio < imbalance_threshold:
                    df_filtered.drop(column, axis=1, inplace=True)
                    print(f"Binary feature '{column}' removed due to imbalance ratio below threshold: {imbalance_ratio:.2f}%")

    return df_filtered



