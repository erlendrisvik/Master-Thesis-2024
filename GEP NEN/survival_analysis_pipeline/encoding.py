import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from multitargetencoder import MultiTargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# For KKN imputation
def x_y_baseline(df):
    """
    Prepares the dataset for model training by applying encoding (for ordinal and
    binary columns) to the input DataFrame,converting the 'status' column to a boolean representation, 
    and separating the dataset into features (X) and target (y) components. Additionally, it returns 
    a tuple of status values, and a list of target columns identified during the baseline encoding process.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset to be processed.

    Returns:
    - X (pd.DataFrame): The feature matrix after removing the 'time' and 'status' columns.
    - y (numpy.recarray): The target array containing 'status' as boolean and 'time'.
    - tuple_y (list): A list of the 'status' values from the 'y' array, where each status is
      represented as a boolean.
    - target_columns (list): A list of nominal columns that were identified and processed during
      the encoding step.

    The function performs the following steps:
    1. Applies baseline encoding to the DataFrame.
    2. Converts the 'status' column from numeric to boolean, where 0 represents 'Alive' (False)
       and 1 represents 'Dead' (True).
    3. Separates the DataFrame into a features matrix 'X' by dropping the 'time' and 'status'
       columns, and a target 'y' that combines 'status' and 'time'.
    4. Extracts a list of boolean 'status' values from 'y' to create 'tuple_y'.
    """
    df, target_columns  = ordinal_binary_encoding(df)
    df['status'] = df['status'].map({0: False, 1: True})  # 0: Alive, 1: Dead
    X = df.drop(['time', 'status'], axis=1)
    y = df[['status', 'time']].to_records(index=False)
    tuple_y = [i for i, _ in y]
    return X, y, tuple_y, target_columns


def ordinal_binary_encoding(df):
    """
    Perform baseline encoding on the DataFrame, including one-hot encoding for specified binary columns,
    manual encoding for binary columns with NaN values, label encoding for specific columns with custom mappings,
    and conversion of boolean columns to integer.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be encoded.

    Returns:
    - tuple: A tuple containing the encoded DataFrame and a list of target columns present in the DataFrame.
    """

    # One-hot encoding for specified columns
    binary_columns = ['Primary Tumour Resected', 
                      'Prior Other Cancer', 'Sex',  
                      'Haemoglobin', 'WBC', 'Mets(Bone)', 'Mets(LN)', 'Mets(Liver)', 'Mets(Lung)', 
                      'Mets(Other)', 'Co-morbidity', 'Hist Exam Metastasis', 'Hist Exam Primary Tumour',
                      'Loc Adv Resectable Disease', 'Mets(LN Retro)', 'Mets(LN Regional)', 
                      'Mets(LN Distant)', 'Living Alone', 'Stage grouped', 'Treatment Intention']
    
    # Only include columns present in the DataFrame
    binary_columns_to_encode = [col for col in binary_columns if col in df.columns]
    df = pd.get_dummies(df, columns=binary_columns_to_encode, drop_first=True)

    # Manual encoding for binary columns with NaN
    manual_encoding_columns = {
        'Dev of Bone Mets': {'No': 0, 'Yes': 1},
        'Dev of Brain Mets': {'No': 0, 'Yes': 1},
        'Co-morbidity Severity': {'1': 0, '> 1': 1},
        'Platelets': {'Normal': 0, '>400x10^9/L': 1},
        'M-stage': {'M0': 0, 'M1': 1},
        'TNM-staging': {'Clinical': 0, 'Pathological': 1},
        'Creatinine': {'Normal': 0, '> Normal': 1},
        'Radical Surgery': {'No': 0, 'Yes': 1},
        'NEC/MANEC': {'NEC': 0, 'MANEC': 1},
        'Reintroduction with Cisplatin+Etoposide': {'No': 0, 'Yes': 1},
        'Metastatic Disease': {'No': 0, 'Yes': 1},
        '<8 Wks Since Blood Tests': {'No': 0, 'Yes': 1}}

    for col, mapping in manual_encoding_columns.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Label encoding for specific columns with custom mappings
    encoding_maps = {
        'Chromogranin A2': {"Normal": 0, '>Normal <= 2UNL': 1, '> 2UNL': 2},
        'Differentiation': {'Highly Differentiated': 0, 'Intermediate': 1, 'Poorly Differentiated': 2},
        'Chromogranin A': {'Negative': 0, 'Partly Positive': 1, 'Strongly Positive': 2},
        'NSE': {'Normal': 0, '>Normal <= 2UNL': 1, '> 2UNL': 2},
        'Synaptophysin': {'Negative': 0, 'Partly Positive': 1, 'Strongly Positive': 2},
        'WHO Perf Stat': {'WHO 0': 0, 'WHO 1': 1, 'WHO 2': 2, 'WHO 3': 3, 'WHO 4': 4},
        'ALP': {'Normal': 0, '>Normal <= 3 UNL': 1, '>3 UNL': 2},
        'LDH': {'Normal': 0, '>Normal <= 2UNL': 1, '> 2UNL': 2},
        'CD-56': {'Negative': 0, 'Partly Positive': 1, 'Strongly Positive': 2}, 
        'T-stage': {'Tx': 0, 'T0': 1, 'T2': 2, 'T3': 3, 'T4': 4}, 
        'N-stage': {'Nx': 0, 'N0': 1, 'N1': 2, '>N1': 3},
        'SRI': {'Negative': 0, '< Liver': 1, '> Liver': 2},
        'Octreoscan': {'Negative': 0, '< Liver': 1, '> Liver': 2},
    }

    for col, map in encoding_maps.items():
        if col in df.columns:
            df[col] = df[col].map(map)

    # Convert boolean to 0/1 for columns present in the DataFrame
    boolean_columns = df.select_dtypes(include=['bool']).columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    target_columns = ['Primary Tumour', 'Chemotherapy Type', "Best Response (RECIST)", 
                      'Tumour Morphology', 'Smoking', 'Treatment Stopped']
    # Filter target_columns to those present in df
    target_columns = [col for col in target_columns if col in df.columns]
    
    return df, target_columns


def Preprocessing(X_train, X_test, y_train, target_columns):
    """
    Applies target encoding, combines encoded columns, scales, and imputes missing values for training and testing datasets.
    
    Parameters:
    - X_train: DataFrame containing the training features
    - X_test: DataFrame containing the testing features
    - y_train: DataFrame or Series containing the training target variable
    - target_columns: list of column names in X_train/X_test to be target encoded
    
    Returns:
    - X_train: The preprocessed training dataset
    - X_test: The preprocessed testing dataset
    """
    binary_columns_with_nan = [
        'Co-morbidity Severity', 'Platelets', 'Dev of Bone Mets', '<8 Wks Since Blood Tests',
        'Creatinine', 'Dev of Brain Mets', 'Radical Surgery', 'Metastatic Disease',
        'M-stage', 'TNM-staging', 'NEC/MANEC', 'Reintroduction with Cisplatin+Etoposide']
    
    binary_columns_with_nan = [col for col in binary_columns_with_nan if col in X_train.columns and col in X_test.columns]

    ## Target Encoder
    le = MultiTargetEncoder() 

    common_target_columns = [col for col in target_columns if col in X_train.columns and col in X_test.columns]
    X_train_encoded = le.fit_transform(X_train[common_target_columns], y_train["time"])
    X_test_encoded = le.transform(X_test[common_target_columns])

    # Combine the encoded columns back with the rest of the dataset
    # Ensure indexes are aligned
    X_train_encoded = X_train_encoded.set_index(X_train.index)
    X_test_encoded = X_test_encoded.set_index(X_test.index)

    # Combine the encoded columns back with the rest of the dataset
    X_train_combined = X_train.drop(columns=target_columns).join(X_train_encoded)
    X_test_combined = X_test.drop(columns=target_columns).join(X_test_encoded)
    

    # Scale before KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    # KNNImputer
    imputer = KNNImputer(n_neighbors=10)
    X_train_imputed = imputer.fit_transform(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)

    # scaler tilbake (hele datasettet)
    X_train_rescaled = scaler.inverse_transform(X_train_imputed)
    X_test_rescaled = scaler.inverse_transform(X_test_imputed)

    # create a dataframe
    X_train_encoded_df = pd.DataFrame(X_train_rescaled, index=X_train_combined.index, columns=X_train_combined.columns)
    X_test_encoded_df = pd.DataFrame(X_test_rescaled, index=X_test_combined.index, columns=X_test_combined.columns)


    # transform binary that was encoded like 0 and 1, and not one-hot
    X_train_rounded = X_train_encoded_df[binary_columns_with_nan].round()
    X_test_rounded = X_test_encoded_df[binary_columns_with_nan].round()

    manual_encoding_columns = {
        'Dev of Bone Mets': {0: 'No', 1: 'Yes'},
        'Dev of Brain Mets': {0: 'No', 1: 'Yes'},
        'Co-morbidity Severity': {0: 1, 1: '> 1'},
        'Platelets': {0: 'Normal', 1: '>400x10^9/L'},
        'Creatinine': {0: 'Normal', 1: '> Normal'},
        'M-stage': {0: 'M0', 1: 'M1'},
        'TNM-staging': {0: 'NEC', 1: 'MANEC'},
        'NEC/MANEC': {0: 'M0', 1: 'M1'},
        'Reintroduction with Cisplatin+Etoposide': {0: 'No', 1: 'Yes'},
        'Metastatic Disease': {0: 'No', 1: 'Yes'},
        '<8 Wks Since Blood Tests': {0: 'No', 1: 'Yes'},
        'Radical Surgery': {0: 'No', 1: 'Yes'}}
    
    for col, mapping in manual_encoding_columns.items():
        if col in X_train.columns:
            X_train_rounded[col] = X_train_rounded[col].map(mapping)
            X_test_rounded[col] = X_test_rounded[col].map(mapping)
    
    # get_dummies on the binaries
    train_one_hot_encoded_df = pd.get_dummies(X_train_rounded, drop_first=True)
    train_one_hot_encoded_df = train_one_hot_encoded_df.apply(lambda col: col.map(lambda x: 1 if x is True else 0 if x is False else x))
    test_one_hot_encoded_df = pd.get_dummies(X_test_rounded, drop_first=True)
    test_one_hot_encoded_df = test_one_hot_encoded_df.apply(lambda col: col.map(lambda x: 1 if x is True else 0 if x is False else x))

    # merge binaries with NaN with the other columns
    X_train_merged_df = X_train_encoded_df.drop(columns=binary_columns_with_nan).join(train_one_hot_encoded_df)
    X_test_merged_df = X_test_encoded_df.drop(columns=binary_columns_with_nan).join(test_one_hot_encoded_df)

    # Scaler the whole dataset again
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_merged_df)
    X_test_scaled = scaler.transform(X_test_merged_df)

    # create a dataframe after scaling
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_merged_df.index, columns=X_train_merged_df.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_merged_df.index, columns=X_test_merged_df.columns)

    # copy
    X_train = X_train_scaled_df.copy()
    X_test = X_test_scaled_df.copy()

    return X_train, X_test

# For multiple imputation
def x_y_multiple(df):
    """
    Prepares the dataset for model training by applying encoding (for ordinal and
    binary columns), then separating the dataset into features (X) and targets (y), where 'status' 
    is converted to boolean values ('Alive' to False, 'Dead' to True), and 'time' is retained. 
    Additionally, it extracts a tuple list of status values and returns the processed feature matrix,
    target array, status tuple list, and a list of target columns processed during encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset to be processed.

    Returns:
    - X (pd.DataFrame): The features matrix obtained by removing 'time' and 'status' columns.
    - y (numpy.recarray): The target array with 'status' as boolean and 'time'.
    - tuple_y (list): A list of 'status' values extracted from 'y', represented as booleans.
    - target_columns (list): A list of nominal columns that were identified and processed during
      the encoding step.

    The function performs these operations:
    1. Applies baseline encoding to the DataFrame and extracts the encoded DataFrame along with
       identified target columns.
    2. Converts the 'status' column from a string representation ('Alive', 'Dead') to boolean values.
    3. Prepares the features matrix 'X' by excluding 'time' and 'status' columns.
    4. Combines 'status' and 'time' into a target array 'y'.
    5. Extracts boolean 'status' values from 'y' to form 'tuple_y'.
    """

    df, target_columns = ordinal_binary_encoding(df)
    df['status'] = df['status'].map({'Alive': False, 'Dead': True})  # 0: Alive, 1: Dead
    X = df.drop(['time', 'status'], axis=1)
    y = df[['status', 'time']].to_records(index=False)
    tuple_y = [i for i, _ in y]
    return X, y, tuple_y, target_columns


def Preprocessing_without_imputing(X_train, X_test, y_train, target_columns):
    """
    Applies target encoding, combines encoded columns, scales, and imputes missing values for training and testing datasets.
    
    Parameters:
    - X_train: DataFrame containing the training features
    - X_test: DataFrame containing the testing features
    - y_train: DataFrame or Series containing the training target variable
    - target_columns: list of column names in X_train/X_test to be target encoded
    
    Returns:
    - X_train: The preprocessed training dataset
    - X_test: The preprocessed testing dataset
    """

    ## Target Encoder
    le = MultiTargetEncoder() 
    X_train_encoded = le.fit_transform(X_train[target_columns], y_train["time"])
    X_test_encoded = le.transform(X_test[target_columns])

    # Combine the encoded columns back with the rest of the dataset
    # Ensure indexes are aligned
    X_train_encoded = X_train_encoded.set_index(X_train.index)
    X_test_encoded = X_test_encoded.set_index(X_test.index)

    # Combine the encoded columns back with the rest of the dataset
    X_train_combined = X_train.drop(columns=target_columns).join(X_train_encoded)
    X_test_combined = X_test.drop(columns=target_columns).join(X_test_encoded)

    # Scale before KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    X_train_encoded_df = pd.DataFrame(X_train_scaled, index=X_train_combined.index, columns=X_train_combined.columns)
    X_test_encoded_df = pd.DataFrame(X_test_scaled, index=X_test_combined.index, columns=X_test_combined.columns)

    return X_train_encoded_df, X_test_encoded_df


def preprocess_data(df, nan_column_threshold):
    """
    Preprocesses the given DataFrame by removing columns with a high percentage of NaN values based on the specified
    threshold, and then drops any rows containing NaN values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to preprocess.
    - nan_column_threshold (float): The threshold percentage for dropping columns. Columns with a higher percentage
      of NaN values than this threshold will be dropped.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame with columns and rows containing high amounts of NaN values removed.
      The dataframe has no missing values. 
    """
    nan_percentage_per_column = (df.isnull().sum() / len(df)) * 100  # Calculate percentage of missing values per column
    nan_percentage_per_column = nan_percentage_per_column.sort_values(ascending=False)
    
    # Drop columns based on threshold
    columns_to_drop = nan_percentage_per_column[nan_percentage_per_column > nan_column_threshold].index

    print("Dropping columns:")
    # Print out the table with dropped columns and their percentage of missing values
    print(pd.DataFrame({'Column Name': columns_to_drop, 'Percentage Missing': nan_percentage_per_column[columns_to_drop]}))

    df = df.drop(columns=columns_to_drop)
    
    # Drop rows with any NaN values
    df = df.dropna()
    print("Shape after preprocessing:", df.shape)
    
    return df



def correlation_plot(X_train, X_test, threshold=0.85):
    """
    Plots the correlation matrix for features in X_train and X_test combined
    that have a correlation magnitude higher than the specified threshold, 
    focusing only on the most highly correlated pairs.

    Parameters:
    - X_train: pandas DataFrame, training data.
    - X_test: pandas DataFrame, testing data.
    - threshold: float, the correlation threshold to identify highly correlated features.
    """
    # Combine training and testing data
    X = pd.concat([X_train, X_test])
    print(f"Total features before filtering: {len(X.columns)}")
    
    # Calculate the correlation matrix
    corr_matrix = X.corr()

    # Find pairs of features that have high absolute correlation
    high_corr_pairs = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
    high_corr_pairs = high_corr_pairs[high_corr_pairs > threshold]
    high_corr_pairs = high_corr_pairs[high_corr_pairs < 1]  # Exclude perfect correlation (self-correlation)

    # Extract unique features from these pairs
    unique_features = np.unique(np.hstack([high_corr_pairs.index.get_level_values(0), high_corr_pairs.index.get_level_values(1)]))
    
    if len(unique_features) == 0:
        print("No features meet the correlation threshold to plot.")
        return  # Exit the function if no features meet the criteria

    # Plotting the correlation matrix for these features
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.loc[unique_features, unique_features], annot=True, cmap='coolwarm', fmt=".2f")
    #plt.title(f'Correlation Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)  # Proper rotation for y-ticks
    plt.show()

# Used in correlation_PCA_plots.ipynb
def plot_scores(X_train, X_test):
    """
    Generates a scatter plot of the first two principal components after performing PCA on the merged datasets of
    X_train and X_test. Each point in the scatter plot is annotated with the patient number, providing a visual 
    representation of the data's dimensionality reduction.

    Parameters:
    - X_train (pd.DataFrame): The training dataset.
    - X_test (pd.DataFrame): The testing dataset.

    Outputs:
    - A scatter plot showing the PCA scores for the first two principal components, with each point annotated with 
      its corresponding patient number.
    """

    # Merge X_train and X_test
    X = pd.concat([X_train, X_test])

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    # Compute PCA scores
    scores = pca.transform(X)

    # Get patient numbers
    patient_numbers = pd.concat([X_train.index.to_series(), X_test.index.to_series()])

    # Create a scatter plot for scores
    plt.figure(figsize=(8, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5)
    
    # Annotate each point with its patient number
    for i, txt in enumerate(patient_numbers):
        plt.annotate(txt, (scores[i, 0], scores[i, 1]), fontsize=8)
    
    plt.xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)))
    plt.ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)))
    plt.title('PCA Scores Plot')
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.axvline(0, color='grey', lw=1, linestyle='--')
    plt.grid(True)
    
    plt.show()

def plot_loadings(X_train, X_test):
    """
    Generates a scatter plot of the PCA loadings for the first two principal components after performing PCA on the
    merged datasets of X_train and X_test. Each point in the scatter plot is annotated with the corresponding feature
    name, providing a visual representation of the contribution of each feature to the principal components.

    Parameters:
    - X_train (pd.DataFrame): The training dataset.
    - X_test (pd.DataFrame): The testing dataset.

    Outputs:
    - A scatter plot showing the PCA loadings for the first two principal components, with each point annotated with
      its corresponding feature name.
    """

    X = pd.concat([X_train, X_test])

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    # Compute loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Get feature names
    feature_names = X.columns

    # Create a scatter plot for loadings
    plt.figure(figsize=(8, 6))
    plt.scatter(loadings[:, 0], loadings[:, 1], alpha=0.5)
    
    # Annotate each point with its feature name
    for i, feature in enumerate(feature_names):
        plt.annotate(feature, (loadings[i, 0], loadings[i, 1]), fontsize=8)
    
    plt.xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)))
    plt.ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)))
    plt.title('PCA Loadings Plot')
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.axvline(0, color='grey', lw=1, linestyle='--')
    plt.grid(True)
    
    plt.show()


