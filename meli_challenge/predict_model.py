

def predict(df_data):
    df_data = pd.read_csv('data/test.csv')
    print(df_data.head())
    y_pred = predict(df_data)
    print(np.unique(y_pred, return_counts=True))
    df_test = pd.DataFrame({
        'ID': df_data['ID'].astype('int32'),
        'se_fue': y_pred
    })
    df_test.to_csv('data/submission_test.csv',index=False)
    print(df_test.head())