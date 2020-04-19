from sklearn.linear_model import LogisticRegression
from meli_challenge.data_process import clean_data, prepare_data
from meli_challenge import evaluate_model
from meli_challenge import models
from meli_challenge import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import joblib
import pickle
#from nltk.corpus import stopwords
# Downloading the stop words list
#import nltk
#nltk.download('stopwords')

def train_model(df_train, df_test, emb='count'):
    print('Prepare data ...')
    X_train = df_train['title'].values
    # y_train = df_train['category'].values
    X_test = df_test['title'].values
    # y_test = df_test['category'].values
    y_train_one_hot, label_encoder = prepare_data.encode_label(df_train, 'category')
    print(y_train_one_hot.shape)
    y_test_one_hot = label_encoder.transform(df_test['category'])
    pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))

    # embedding: Bag of Words (count vector) ----------------------------------
    print("Embedding text...")
    emb_converter = None
    if emb == 'count':
        emb_converter = CountVectorizer()
        # emb_converter = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('spanish'))
    elif emb == 'tfidf':
        emb_converter = TfidfVectorizer()
        # emb_converter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('spanish'))
    # Para prevenir overfitting, es conveniente vectorizar los titulos luego de dividir el dataset
    # en train y test, y realizar la vectorizacion usando los valores del conjunto de entrenamiento
    X_train_emb = emb_converter.fit_transform(X_train)
    pickle.dump(emb_converter, open('{}_features.pkl'.format(emb), 'wb'))

    # train_model ---------------------------------------------------------------
    print('Training model...')
    np_sample_weights = utils.build_sample_weight(df_train, 'category', 'label_quality', 10)
    clf = models.train_random_forest(X_train_emb, y_train_one_hot, np_sample_weights)

    # predict test data --------------------------------------------------------
    X_test_emb = emb_converter.transform(X_test)
    y_predicted = clf.predict(X_test_emb)

    # evaluate test ------------------------------------------------------------
    accuracy, precision, recall, f1 = evaluate_model.get_metrics(y_test_one_hot, y_predicted)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    return clf, emb_converter, label_encoder

def predict_evaluate(file_test, file_classifier, file_label_encoder, file_emb_encoder):
    with open(file_label_encoder, 'rb') as f:
        #label_encoder = pickle.load(f, encoding='latin1')
        #label_encoder = pickle.load(f, encoding='bytes')
        label_encoder = pickle.load(f)
    with open(file_emb_encoder, 'rb') as f:
        # label_encoder = pickle.load(f, encoding='latin1')
        # label_encoder = pickle.load(f, encoding='bytes')
        emb_converter = pickle.load(f)
    df_test = pd.read_csv(file_test)
    df_test = df_test[pd.notnull(df_test['title'])]
    X_test = df_test['title'].values
    y_test_one_hot = label_encoder.transform(df_test['category'])
    clf = joblib.load(file_classifier)
    X_test_emb = emb_converter.transform(X_test)
    y_predicted = clf.predict(X_test_emb)

    # evaluate test ------------------------------------------------------------
    accuracy, precision, recall, f1 = evaluate_model.get_metrics(y_test_one_hot, y_predicted)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


def train_model_from_train_test_files(file_train, file_test, language='es', emb='count'):
    # clean_data -------------------------------------------------------------
    print('Cleanning data...')
    df_train = pd.read_csv(file_train)
    df_train = df_train[pd.notnull(df_train['title'])]
    # df_train = clean_data.clean_text(df_train, 'title', lemmatize=True, language=language)
    df_test = pd.read_csv(file_test)
    df_test = df_test[pd.notnull(df_test['title'])]
    # df_test = clean_data.clean_text(df_test, 'title', lemmatize=True, language=language)

    clf, emb_converter, label_encoder = train_model(df_train, df_test, emb)

    return clf, emb_converter, label_encoder


def train_model_file(file_data, language='es', emb='count'):
    """

    :param file_data:
    :param language:
    :param emb: options --> count, tfidf
    :return:
    """
    # clean_data -------------------------------------------------------------
    print('Cleanning data...')
    df = pd.read_csv(file_data)
    # nan_rows = df_train[df_train['title'].isnull()]
    # print(nan_rows)
    df = df[pd.notnull(df['title'])]
    #df = clean_data.clean_text(df, 'title', lemmatize=True, language=language)

    # prepare_data -----------------------------------------------------------
    print('Split data TRAIN-TEST...')
    #label_encoder = prepare_data.encode_label(df, 'category')
    # X_train, X_test, y_train, y_test = prepare_data.get_X_Y_train_test(df, 'title', 'category')
    df_train, df_test = prepare_data.get_df_train_test(df, 'category')

    clf, emb_converter, label_encoder = train_model(df_train, df_test, emb)

    return clf, emb_converter, label_encoder


def predict_data(file_data, cls, count_vec, label_encoder):
    df = pd.read_csv(file_data)
    df = clean_data.clean_text(df, 'title')

# predict submission --------------------------------------------------------

if __name__ == '__main__':
    #clf, count_vectorizer, label_encoder = train_model('../../data/train_spanish_cleaned_lemmatized.csv')
    #clf, count_vectorizer, label_encoder = train_model_from_train_test_files('../../data/train_spanish_cleaned_lemmatized_TRAIN.csv',
    #                                                                         '../../data/train_spanish_cleaned_lemmatized_TEST.csv')
    predict_evaluate('../../data/train_spanish_cleaned_lemmatized_TEST.csv',
                     '../../outputs/random_forest.joblib',
                     'label_encoder.pkl',
                     'count_features.pkl')

