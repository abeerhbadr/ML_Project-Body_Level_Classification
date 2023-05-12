import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# load the dataset
class Preprocessor:
    """
    TODO: consider converting y to one hot encoding
    """
    def __init__(self):
        # load from body_level_classification_train.csv
        self.df = pd.read_csv('../Dataset/body_level_classification_train.csv')

    def head(self):
        return self.df.head()
    
    def preprocess(self):
        """
        One hot encoding for categorical columns
        0, 1, 2, 3 for Body_Level
        """
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        categorical_columns.remove('Body_Level')
        # one hot for categorical columns
        df_h = pd.get_dummies(self.df, columns=categorical_columns)

        # make body_level as the last column
        df_h.drop('Body_Level', axis=1, inplace=True)
        df_h['Body_Level'] = self.df['Body_Level']
        df_h['Body_Level'] = pd.factorize(df_h.Body_Level)[0] 

        return df_h
    
    def SMOTE(self, X:np.ndarray, y:np.ndarray):
        smote = SMOTE(random_state=0)
        X_smote, y_smote = smote.fit_resample(X, y)
        return X_smote, y_smote
    
    def split(self, df_h):
        """
        Split the dataset into train and test
        """
        X = df_h.iloc[:, :-1]
        y = df_h.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test