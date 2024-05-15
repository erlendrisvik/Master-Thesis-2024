from encoding import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
mcv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=173637)


def extract_alpha_coxnet(results_coxnet, coefficient, l1_ratio_value, feature_names):
    """
    Extracts and aggregates data from CoxNet modeling results based on a specified L1 ratio value.

    This function filters the CoxNet results for a given L1 ratio and computes the average coefficients for each unique alpha value within that L1 ratio. 
    It then identifies the non-zero coefficients, constructs a dataframe mapping alphas to the number of selected features, and merges it with the input 
    dataframe to obtain various model metrics.

    Parameters:
    - results_coxnet (DataFrame): A pandas DataFrame containing the CoxNet modeling results.
    - coefficient (dict): A dictionary where the keys are tuples of (alpha, L1 ratio) and the values are lists of coefficient arrays.
    - l1_ratio_value (float): The L1 ratio value used to filter the results.
    - feature_names (list of str): A list containing the names of the features in the dataset.

    Returns:
    - alphas (list): A list of alpha values extracted from the filtered DataFrame.
    - l1_ratio (float): The L1 ratio value used for filtering.
    - conc_train (list): A list of concordance indices for the training set corresponding to each alpha value.
    - conc_test (list): A list of concordance indices for the test set corresponding to each alpha value.
    - brier (list): A list of Brier scores corresponding to each alpha value.
    - num_features (list): A list indicating the number of features with non-zero coefficients for each alpha value.
    - max_features (int): The maximum number of features with non-zero coefficients found across all alpha values.
    - features_alpha (dict): A dictionary mapping each alpha to a list of feature names with non-zero coefficients.

    If no data is found for the specified L1 ratio, the function returns empty lists and the L1 ratio value provided.
    """
    
    # Filter the DataFrame to only include rows where L1 Ratio equals the specified value
    results_l1_ratio = results_coxnet[results_coxnet['L1 Ratio'] == l1_ratio_value]
    
    # Ensure l1_ratio is extracted correctly
    if not results_l1_ratio.empty:
        l1_ratio = results_l1_ratio['L1 Ratio'].iloc[0]
    else:
        return [], l1_ratio_value, [], [], [], [], 0  # Return empty lists and the l1_ratio_value if no data found
    
    alphas = results_l1_ratio['Alpha'].unique()
    non_zero_features = []
    features_alpha = {}
    
    # Loop through all unique alpha values
    for alpha in alphas:
        # Retrieve the list of coefficient arrays for the current alpha
        coefficients_list = coefficient.get((alpha, l1_ratio), [])
        if coefficients_list:
            coefficients_array = np.array(coefficients_list)
            mean_coefficients = np.mean(coefficients_array, axis=0)
            non_zero_features.append(np.sum(mean_coefficients != 0))
            
            # Identify non-zero coefficients
            non_zero_indices = np.where(mean_coefficients != 0)[0]
            non_zero_feature_names = [feature_names[idx] for idx in non_zero_indices]
            features_alpha[alpha] = non_zero_feature_names
        else:
            non_zero_features.append(0)
    
    # Create a DataFrame with alphas and the number of non-zero features
    alpha_features_df = pd.DataFrame({
        'Alpha': alphas,
        'Number of Features': non_zero_features
    })
    
    # Merge the results with the alpha_features_df DataFrame
    combined_df = pd.merge(results_l1_ratio, alpha_features_df, on='Alpha', how='inner')
    combined_df = combined_df.sort_values(by='Alpha', ascending=True).reset_index(drop=True)
    
    # Extract and return the required lists
    alphas = combined_df['Alpha'].to_list()
    conc_train = combined_df['Conc train'].to_list()
    conc_test = combined_df['Conc test'].to_list()
    brier = combined_df['Brier Score'].to_list()
    num_features = combined_df['Number of Features'].to_list()
    max_features = combined_df['Number of Features'].iloc[0] if not combined_df.empty else 0
    
    return alphas, l1_ratio, conc_train, conc_test, brier, num_features, max_features, features_alpha

def extract_alpha_features_coxph(results_coxph, coefficient):
    """
    Extracts and organizes key metrics from CoxPH model results, including alpha values, concordance 
    indices, Brier scores, and the count of non-zero features associated with each alpha value. 
    Computes the mean coefficients for each alpha, and counts the non-zero features to assess model sparsity.

    Parameters:
    - results_coxph (pd.DataFrame): DataFrame containing CoxPH model results with varying alpha values.
    - coefficient (dict): A dictionary where keys are alpha values and values are lists of coefficient 
      arrays corresponding to each alpha.

    Returns:
    - tuple: Contains lists of alpha values, training concordance indices, test concordance indices, 
      Brier scores, number of non-zero features for each alpha, and the maximum number of non-zero 
      features across all alphas.
    
    The function performs the following operations:
    1. Loops through all unique alpha values to calculate the mean coefficients and count non-zero features.
    2. Creates a DataFrame mapping alpha values to their respective count of non-zero features.
    3. Merges this DataFrame with the original CoxPH results.
    4. Sorts and resets the index of the combined DataFrame.
    5. Extracts and returns the desired metrics in the form of lists.
    """

    # Filter the DataFrame to only include rows where L1 Ratio equals the specified value
    alphas = results_coxph['Alpha'].unique()
    non_zero_features = []
    
    # Loop through all unique alpha values
    for alpha in alphas:
        # Retrieve the list of coefficient arrays for the current alpha
        coefficients_list = coefficient.get((alpha), [])
        if coefficients_list:
            coefficients_array = np.array(coefficients_list)
            mean_coefficients = np.mean(coefficients_array, axis=0)
            non_zero_features.append(np.sum(mean_coefficients != 0))
        else:
            non_zero_features.append(0)
    
    # Create a DataFrame with alphas and the number of non-zero features
    alpha_features_df = pd.DataFrame({
        'Alpha': alphas,
        'Number of Features': non_zero_features
    })
    
    # Merge the results with the alpha_features_df DataFrame
    combined_df = pd.merge(results_coxph, alpha_features_df, on='Alpha', how='inner')
    combined_df = combined_df.sort_values(by='Alpha', ascending=True).reset_index(drop=True)
    
    # Extract and return the required lists
    alphas = combined_df['Alpha'].to_list()
    conc_train = combined_df['Conc train'].to_list()
    conc_test = combined_df['Conc test'].to_list()
    brier = combined_df['Brier Score'].to_list()
    num_features = combined_df['Number of Features'].to_list()
    max_features = combined_df['Number of Features'].iloc[0] if not combined_df.empty else 0
    
    return alphas, conc_train, conc_test, brier, num_features, max_features


