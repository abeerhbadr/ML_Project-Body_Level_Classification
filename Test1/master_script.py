import preprocess_test as preprocessor_t
# - Files to submit: `master_script.py`, `model.pkl`, `preprocess_test.py`, `requirements.txt`
# if __name__ == '__main__':
def TEST():
    # Read test data
    df_h = preprocessor_t.ReadData(pth='./test.csv')
    
    # Preprocess test data

    # add augmented columns like BMI ...
    AGGREGATE = True

    # discretize BMI
    DISCRETIZE = False

    # one hot encoding for categorical columns
    ONE_HOT = False

    # Preprocess
    df_h = preprocessor_t.LabelOrdinalEncode(df_h)
    if AGGREGATE:
        df_h = preprocessor_t.Aggregate(df_h, discretize=DISCRETIZE)
    if ONE_HOT:
        df_h = preprocessor_t.OneHotEncode(df_h, label='Body_Level')

    # read model
    model = preprocessor_t.Read_Model(path='./model.pkl')

    # predict
    y_pred = model.predict(df_h)

    # decode prediction
    y_pred = preprocessor_t.Decode(y_pred)

    # TODO should we add header to preds.txt?
    # save prediction to preds.txt
    with open('preds.txt', 'w') as f:
        f.write('\n'.join(y_pred))

if __name__ == '__main__':
    TEST()