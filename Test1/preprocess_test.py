import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# joblib
import joblib


def ReadData(pth='./test.csv'):
    df = pd.read_csv(pth)
    return df

# aggregate columns, BMI index, ...
def Aggregate(df, discretize=False):
    weight = df['Weight'].values
    height = df['Height'].values    
    # BMI
    df['BMI'] = weight / (height * height)
    if discretize:
        df['BMI'] = df['BMI'].apply(lambda x: 0 if x < 18.5 else (1 if x < 25 else (2 if x < 30 else 3)))
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
     #
    return data_encoded

def OneHotEncode(df):
    """
    One hot encoding for categorical columns
    0, 1, 2, 3 for Body_Level
    """
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df_h = pd.get_dummies(df, columns=categorical_columns)
    return df_h

def Read_Model(path='./model.pkl'):
    model = joblib.load(path)
    return model

def Decode(y_pred):
    """
    Decode prediction
    0: Body Level 1
    1: Body Level 2
    2: Body Level 3
    3: Body Level 4

    y_pred: list [0, 1, 2, 3, ...]
    """
    return ['Body Level 1' if x == 0 else ('Body Level 2' if x == 1 else ('Body Level 3' if x == 2 else 'Body Level 4')) for x in y_pred]
    