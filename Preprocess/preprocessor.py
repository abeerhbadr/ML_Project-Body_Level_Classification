import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

def ReadData(pth='./Dataset/body_level_classification_train.csv', label='Body_Level'):
    df = pd.read_csv(pth)
    # make label column as the last column
    df_h = df[label].values
    df.drop(label, axis=1, inplace=True)
    df[label] = df_h
    #
    return df

# aggregate columns, BMI index, ...
def Aggregate(df, discretize=False):
    weight = df['Weight'].values
    height = df['Height'].values    
    # BMI
    df['BMI'] = weight / (height * height)
    if discretize:
        df['BMI'] = df['BMI'].apply(lambda x: 0 if x < 18.5 else (1 if x < 25 else (2 if x < 30 else 3)))
    # keep Body_Level as the last column
    df = df[[col for col in df.columns if col != 'Body_Level'] + ['Body_Level']]
    return df 

# encode while preserving ordinal feature of data!
def LabelOrdinalEncode(df):
    def encode_ordinal(df1, col, mapping):
        df1[col] = df1[col].map(mapping)
    #
    data_encoded = df.copy()
    #
    encode_ordinal(data_encoded, 'Gender', {'Female': 0, 'Male': 1})
    encode_ordinal(data_encoded, 'H_Cal_Consump', {'no': 0, 'yes': 1})
    encode_ordinal(data_encoded, 'Alcohol_Consump', {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
    encode_ordinal(data_encoded, 'Smoking', {'no': 0, 'yes': 1})
    encode_ordinal(data_encoded, 'Food_Between_Meals', {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
    encode_ordinal(data_encoded, 'Fam_Hist', {'no': 0, 'yes': 1})
    encode_ordinal(data_encoded, 'H_Cal_Burn', {'no': 0, 'yes': 1})
    encode_ordinal(data_encoded, 'Transport', {'Bike': 0, 'Walking': 1, 'Public_Transportation': 2, 'Motorbike': 3, 'Automobile': 4})
    encode_ordinal(data_encoded, 'Body_Level', {'Body Level 1': 0, 'Body Level 2': 1, 'Body Level 3': 2, 'Body Level 4': 3})
    #
    return data_encoded

def OneHotEncode(df, label='Body_Level'):
    """
    One hot encoding for categorical columns
    0, 1, 2, 3 for Body_Level
    """
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    categorical_columns.remove(label)
    # one hot for categorical columns
    df_h = pd.get_dummies(df, columns=categorical_columns)
    # label encode Body_Level
    df_h['Body_Level'] = df_h['Body_Level'].map(
        {'Body Level 1': 0, 'Body Level 2': 1, 'Body Level 3': 2, 'Body Level 4': 3})
    #
    return df_h
    
from sklearn.utils import resample
def Resample(df_p):
    df_p1 = df_p[df_p['Body_Level'] == 0]
    df_p2 = df_p[df_p['Body_Level'] == 1]
    df_p3 = df_p[df_p['Body_Level'] == 2]
    df_p4 = df_p[df_p['Body_Level'] == 3]
    #
    max_class_size = max(len(df_p1), len(df_p2), len(df_p3), len(df_p4))
    #
    df_p1 = resample(df_p1, replace=True, n_samples=max_class_size, random_state=0)
    df_p2 = resample(df_p2, replace=True, n_samples=max_class_size, random_state=0)
    df_p3 = resample(df_p3, replace=True, n_samples=max_class_size, random_state=0)
    df_p4 = resample(df_p4, replace=True, n_samples=max_class_size, random_state=0)
    #
    return pd.concat([df_p1, df_p2, df_p3, df_p4])


def SMOTE(X:np.ndarray, y:np.ndarray):
    smote = SMOTE(random_state=0)
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote
    
def Split(df_h, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test
    """
    X = df_h.iloc[:, :-1]
    y = df_h.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test