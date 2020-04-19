import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
# Downloading the stop words list
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from googletrans import Translator
# https://pypi.org/project/googletrans/
# https://py-googletrans.readthedocs.io/en/latest/
#from google.cloud import translate

def save_titles_by_language(df, language):
    df = df[df['language']==language]
    #df = df.drop('language', axis=1)  # axis number, 1 for columns.
    df = df.drop(columns=['language'])
    df.to_csv('../../data/train_{}.csv'.format(language), index=None, header=True)
    return df

def translate_titles(df, text_field, lang_src, lang_dest):
    translator = Translator()
    # If source language is not given, google translate attempts to detect the source language.
    df[text_field] = translator.translate(df[text_field], src=lang_src, dest=lang_dest)  #'es':'spanish'   'pt':'portuguese'
    return df

def clean_text(df, text_field, translate= False, lemmatize=False, language='es'):
    #stemmer = WordNetLemmatizer()
    stemmer = None
    if lemmatize:
        if language=='pt':
            stemmer = SnowballStemmer("portuguese")
        elif language=='es':
            stemmer = SnowballStemmer("spanish")
        else:
            stemmer = SnowballStemmer("spanish")
    documents = []
    for row in range(0,len(df)):
        document = str(df.loc[row][text_field])
        if translate:
            translator = Translator()
            document = str(translator.translate(document, dest='es').text)
        # Replace numbers by nb
        document = re.sub(r'\b\d+(?:\.\d+)?', 'numero', document)
        document = re.sub(r'\d+(?:\.\d+)?\b', ' número ', document)
        # document = re.sub(r'(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b', 'nb')
        # document = re.sub(r'(\s+\-?|^\-?)(\d+|\d*\.\d+)', 'nb', document)
        # Remove all the special characters
        document = re.sub(r'\W', ' ', document)
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        if lemmatize:
            document = document.split()
            #document = [stemmer.lemmatize(word) for word in document]
            document = [stemmer.stem(word) for word in document]
            document = ' '.join(document)
        documents.append(document)
    df[text_field] = documents
    return df


def clean_text_2(df, text_field):
    #df[text_field] = df[text_field].str.replace(r"http\S+", "")
    #df[text_field] = df[text_field].str.replace(r"http", "")
    #df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    #df[text_field] = df[text_field].str.replace(r"@", "at")
    #df[text_field] = df[text_field].str.replace("\r", " ")
    #df[text_field] = df[text_field].str.replace("\n", " ")
    #df[text_field] = df[text_field].str.replace('"', '')
    #df[text_field] = df[text_field].str.replace("'s", "")   # para pronombres posesivos
    df[text_field] = df[text_field].str.lower()
    return df

def lemmatization(df, text_field):
    # https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb
    #nltk.download('punkt')
    #nltk.download('wordnet')
    stemmer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):
        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row][text_field]
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(stemmer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    df[text_field] = lemmatized_text_list
    return df

def remove_stop_words(df, text_field):
    stop_words = list(stopwords.words('spanish'))
    df[text_field] = df[text_field]
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df[text_field] = df[text_field].str.replace(regex_stopword, '')


def show_category_distribution(df):
    #print(df_meli['category'].value_counts())
    print(df.groupby("category").count())

def test_clean_text():
    FILE = '../../data/train_portuguese.csv'
    df_meli = pd.read_csv(FILE)
    print(df_meli.head())
    print(df_meli.tail())
    print(df_meli.describe())
    df_meli = clean_text(df_meli, 'title', translate=False, lemmatize=True,language='pt')
    df_meli.to_csv('../../data/train_portuguese_{}.csv'.format('cleaned_lemmatized'), index=None, header=True)
    print(df_meli.head())
    print(df_meli.tail())

def test_translate():
    FILE = '../../data/train_portuguese.csv'
    df_meli = pd.read_csv(FILE)
    #print(df_meli.head())
    #print(df_meli.tail())
    #print(df_meli.describe())
    #df_meli = translate_titles(df_meli, 'title', 'portuguese', 'spanish')
    df_meli = clean_text(df_meli, 'title')
    #print(df_meli.head())
    #print(df_meli.tail())
    list_to_translate = list(df_meli['title'])
    title = list_to_translate[0]
    print('Original title', title)
    translator = Translator()
    detected_language = translator.detect(title)
    translated_title = translator.translate(title, src='pt', dest='es')
    print('Detected Language', detected_language.lang, detected_language.confidence)
    print('Translated title', translated_title.text)
    translated_titles = list()
    for title in list_to_translate:
        try:
            translated_titles.append(translator.translate(title, src='pt', dest='es'))
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print('Title:', title)
    print(translated_titles)

    # Google
    #translator = translate.Client()
    #print(translator.translate(title, target_language='es', source_language='pt'))


if __name__ == '__main__':
    #FILE = '../../data/train.csv'
    #FILE = '../../data/train_spanish.csv'
    #FILE = '../../data/train_portuguese.csv'
    #df_meli = pd.read_csv(FILE)
    # df_meli = save_titles_by_language(df_meli, 'portuguese')
    # df_meli = save_titles_by_language(df_meli, 'spanish')

    #test_translate()
    #nltk.download('wordnet')
    test_clean_text()

    # document = 'Caña De Pesca 2.50 Tramos?+ (10.3cc856)'
    # print(document)
    # #document = re.sub(r'(\b|\-?|^\-?)(\d+|\d*\.\d+)\b', 'nb', document)
    # document = re.sub(r'\b\d+(?:\.\d+)?', ' número ', document)
    # document = re.sub(r'\d+(?:\.\d+)?\b', ' número ', document)
    # #result = re.sub(r"\d", "", text)
    # #document = document.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    # document = re.sub(r'\W', ' ', document)
    # print(document)
    # #
    #
    # document = re.sub(r'\s+', ' ', document, flags=re.I)
    # print(document)

