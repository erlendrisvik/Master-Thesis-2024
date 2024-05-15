from sklearn.preprocessing import TargetEncoder
import numpy as np
import pandas as pd

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
            X_train (pandas.DataFrame): The training data features.
            y_train (pandas.Series): The training data target variable.

            Returns:
            --------
            pandas.DataFrame: The transformed features.
            """

            index = X_train.index
            self.colnames = X_train.columns
            self.encoders = [TargetEncoder(smooth = "auto", target_type = "continuous", random_state=173637) for _ in range(X_train.shape[1])]

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
        X_test (pandas.DataFrame): The test data to be transformed.

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
    