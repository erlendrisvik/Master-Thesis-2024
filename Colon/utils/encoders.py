import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import TargetEncoder


class MultiLabelEncoder():
    """ 
    Class to extend sklearn's LabelEncoder to handle multiple columns.
    Each column is encoded separately and is assigned their own LabelEncoder instance.
    """
    
    def fit_transform(self, X):
        """
        Fit the label encoders on data to map categories to integers.

        Parameters:
        -----------
        X : numpy.ndarray
            Categorical data features.

        Returns:
        --------
        numpy.ndarray: The transformed features.
        """

        self.encoders = [LabelEncoder() for _ in range(X.shape[1])]
        encoded_cols = []

        for i in range(X.shape[1]):
            column = np.asarray(X[:, i]).flatten()

            # get mask of NaNs to remove temporarily
            mask = pd.isnull(column)        
            #non_nan_column = column[~mask]
            non_nan_column = np.array(column[~mask], dtype = str)

            encoded_column = self.encoders[i].fit_transform(non_nan_column)

            # Prepare the final encoded column with NaNs in place
            full_encoded_column = np.full(column.shape, np.nan, dtype=object)
            full_encoded_column[~mask] = encoded_column
            encoded_cols.append(full_encoded_column)

        self.transformed = np.array(encoded_cols, dtype = object).T
        return self.transformed

    def inverse_transform(self, X):
        """
        Transform the given test data using the fitted encoders.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The integers to be transformed back.

        Returns:
        --------
        numpy.ndarray: The categories mapped back to strings.
        """

        decoded_cols = []

        for i in range(X.shape[1]):
            #column = X[:, i]
            column = np.asarray(X[:, i]).flatten()

            # get mask of NaNs to remove temporarily
            mask = pd.isnull(column)
            non_nan_column = np.array(column[~mask], dtype = int)
            decoded_column = self.encoders[i].inverse_transform(non_nan_column)

            # Prepare the final encoded column with NaNs in place
            full_encoded_column = np.full(column.shape, np.nan, dtype=object)
            full_encoded_column[~mask] = decoded_column
            decoded_cols.append(full_encoded_column)

        self.inverse_transformed = np.array(decoded_cols).T
        return self.inverse_transformed

class MultiTargetEncoder():
    """ 
    Class to extend sklearn's TargetEncoder to handle multiple columns.
    Each column is encoded separately and is assigned their own TargetEncoder instance.
    This is to minimize the loss when removing rows of NA.
    """
    

    def fit_transform(self, X_train, y_train):
            """
            Fit the target encoders on the training data and transform the features.

            Parameters:
            -----------
            X_train : pandas.DataFrame
                The training data features.
                
            y_train : pandas.Series
                The training data target variable.

            Returns:
            --------
            pandas.DataFrame: The transformed features.
            """

            index = X_train.index
            self.colnames = X_train.columns
            self.encoders = [TargetEncoder(smooth = "auto", target_type = "continuous") for _ in range(X_train.shape[1])]

            X_train = X_train.to_numpy()
            y_train = np.array(y_train)
            encoded_cols = []

            for i in range(X_train.shape[1]):
                column = np.asarray(X_train[:, i]).flatten()

                # get mask of NaNs to remove temporarily
                mask = pd.isnull(column)        
                non_nan_column = np.array(column[~mask], dtype = str).reshape(-1, 1)

                encoded_column = self.encoders[i].fit_transform(X = non_nan_column, y = y_train[~mask]).flatten()
                
                # Prepare the final encoded column with NaNs in place
                full_encoded_column = np.full(column.shape, np.nan, dtype=object)
                full_encoded_column[~mask] = encoded_column
                encoded_cols.append(full_encoded_column)

            self.transformed = np.array(encoded_cols, dtype = np.float64).T
            self.transformed = pd.DataFrame(self.transformed, columns = self.colnames, index = index)

            return self.transformed
    
    def transform(self, X_test):
        """
        Transform the given test data using the fitted encoders.

        Parameters:
        -----------
        X_test : pandas.DataFrame:
            The test data to be transformed.

        Returns:
        --------
        pandas.DataFrame: The transformed test data.
        """

        index = X_test.index
        X_test = X_test.to_numpy()

        encoded_cols = []

        for i in range(X_test.shape[1]):
            column = np.asarray(X_test[:, i]).flatten()

            # get mask of NaNs to remove temporarily
            mask = pd.isnull(column)        
            non_nan_column = np.array(column[~mask], dtype = str).reshape(-1, 1)

            encoded_column = self.encoders[i].transform(X = non_nan_column).flatten()
            
            # Prepare the final encoded column with NaNs in place
            full_encoded_column = np.full(column.shape, np.nan, dtype=object)
            full_encoded_column[~mask] = encoded_column
            encoded_cols.append(full_encoded_column)

        self.transformed = np.array(encoded_cols, dtype = np.float64).T
        self.transformed = pd.DataFrame(self.transformed, columns = self.colnames, index = index)

        return self.transformed
    