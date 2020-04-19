from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd

def encode_label(df, label_data):
    #lb = LabelBinarizer()
    lb = LabelEncoder()
    lb = lb.fit(df[label_data])
    np_one_hot_labels = lb.transform(df[label_data])
    return np_one_hot_labels, lb
    #return lb

def get_X_Y_train_test(df, x_data, y_data):
    X = df[x_data].values   #.tolist()
    y = df[y_data].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)
    return X_train, X_test, y_train, y_test

def get_df_train_test(df, y_data):
    y = df[y_data].values
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=40, stratify=y)
    return df_train, df_test

if __name__ == '__main__':
    df = pd.read_csv('../../data/train_spanish_cleaned_lemmatized.csv')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=40, stratify=df['category'].values)
    df_train.to_csv('../../data/train_spanish_cleaned_lemmatized_{}.csv'.format('TRAIN'), index=None, header=True)
    df_test.to_csv('../../data/train_spanish_cleaned_lemmatized_{}.csv'.format('TEST'), index=None, header=True)
