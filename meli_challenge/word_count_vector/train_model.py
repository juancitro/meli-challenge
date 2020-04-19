from sklearn.linear_model import LogisticRegression
from meli_challenge.data_process import clean_data, prepare_data
from meli_challenge import evaluate_model
from meli_challenge import models
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
#from nltk.corpus import stopwords
# Downloading the stop words list
#import nltk
#nltk.download('stopwords')


def train_model(file_data, language='es'):
    # clean_data -------------------------------------------------------------
    df = pd.read_csv(file_data)
    df = clean_data.clean_text(df, 'title', lemmatize=True, language=language)

    # prepare_data -----------------------------------------------------------
    #X_train, X_test, y_train, y_test = prepare_data.get_X_Y_train_test(df, 'title', 'category')
    df_train, df_test = prepare_data.get_df_train_test(df, 'category')
    X_train = df_train['title'].values
    #y_train = df_train['category'].values
    X_test = df_test['title'].values
    #y_test = df_test['category'].values
    y_train, label_encoder = prepare_data.encode_label(df_train, 'category')
    y_test = label_encoder.transform(df_test['category'])

    # embedding: Bag of Words (count vector) ----------------------------------
    count_vectorizer = CountVectorizer()
    #count_vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('spanish'))
    # Para prevenir overfitting, es conveniente vectorizar los titulos luego de dividir el dataset
    # en train y test, y realizar la vectorizacion usando los valores del conjunto de entrenamiento
    X_train_emb = count_vectorizer.fit_transform(X_train)

    # train_model ---------------------------------------------------------------
    clf = models.train_random_forest(X_train_emb, y_train)

    # predict test data --------------------------------------------------------
    X_test_emb = count_vectorizer.transform(X_test)
    y_predicted = clf.predict(X_test_emb)

    # evaluate test ------------------------------------------------------------
    accuracy, precision, recall, f1 = evaluate_model.get_metrics(y_test, y_predicted)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    return clf, count_vectorizer, label_encoder

def predict_data(file_data, cls, count_vec, label_encoder):
    df = pd.read_csv(file_data)
    df = clean_data.clean_text(df, 'title')

# predict submission --------------------------------------------------------

if __name__ == '__main__':
    clf, count_vectorizer, label_encoder = train_model('../../data/train_spanish.csv')