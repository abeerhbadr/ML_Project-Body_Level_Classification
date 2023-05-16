import pandas as pd
from sklearn.model_selection import train_test_split
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

    y_pred: list [0, 1, 2, 2, 0, 0, 1, 3, ...]
    """
    return ['Body Level '+str(x+1) for x in y_pred]

# - Files to submit: `master_script.py`, `model.pkl`, `requirements.txt`
# if __name__ == '__main__':
def TEST():
    # Read test data
    df_h = ReadData(pth='./test.csv')

    # Preprocess

    # discretize BMI
    DISCRETIZE = False
    # 1. label encode categoricals
    df_h = LabelOrdinalEncode(df_h)
    # 2. Aggregate BMI
    df_h = Aggregate(df_h, discretize=DISCRETIZE)

    # read model
    model = Read_Model(path='./model.pkl')

    # predict
    y_pred = Decode(model.predict(df_h))

    # TODO should we remove header from preds.txt?
    # save prediction to preds.txt
    with open('preds.txt', 'w') as f:
        f.write('Body_Level\n')
        f.write('\n'.join(y_pred))

if __name__ == '__main__':
    TEST()