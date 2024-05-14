import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MultiLabelEncoder():
    """ 
    Class to extend sklearn's LabelEncoder to handle multiple columns.
    Each column is encoded separately and is assigned their own LabelEncoder instance.
    """
    
    def fit_transform(self, X):
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
    