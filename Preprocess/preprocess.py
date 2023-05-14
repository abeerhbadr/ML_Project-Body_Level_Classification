from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import itertools
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC, SMOTEN 
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def preprocess(df, resample_=True):
    df_p = df.copy()
    # categorical
    df_p['Body_Level'] = df_p['Body_Level'].map({
    'Body Level 1': 0, 
    'Body Level 2': 1, 
    'Body Level 3': 2, 
    'Body Level 4': 3})

    # resample
    if resample_:
        df_p1 = df_p[df_p['Body_Level'] == 0]
        df_p2 = df_p[df_p['Body_Level'] == 1]
        df_p3 = df_p[df_p['Body_Level'] == 2]
        df_p4 = df_p[df_p['Body_Level'] == 3]

        max_class_size = max(len(df_p1), len(df_p2), len(df_p3), len(df_p4))

        df_p1 = resample(df_p1, replace=True, n_samples=max_class_size, random_state=0)
        df_p2 = resample(df_p2, replace=True, n_samples=max_class_size, random_state=0)
        df_p3 = resample(df_p3, replace=True, n_samples=max_class_size, random_state=0)
        df_p4 = resample(df_p4, replace=True, n_samples=max_class_size, random_state=0)

        df_p = pd.concat([df_p1, df_p2, df_p3, df_p4])

    # LabelEncoder for categorical columns except Body_Level
    le = LabelEncoder()
    categorical_columns = [col for col in df_p.columns if df_p[col].dtype == 'object']
    for col in categorical_columns:
        df_p[col] = le.fit_transform(df_p[col])

    # scaling
    scaler = StandardScaler()
    df_p.iloc[:, :-1] = scaler.fit_transform(df_p.iloc[:, :-1])

    return df_p
