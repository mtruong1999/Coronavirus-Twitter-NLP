import sys
import os
import pandas as pd
import pickle
import re
import string
from utils import *

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
# Global Variables
DATA_SOURCE = ""
STOP_WORDS = []

# Some variables for testing
read_nrows = None  # When None, all rows are read.


def text_filter(text):
    text = text.lower()
    # Replace all tagged users from text (e.g. '@mike12') and removes '#' but keeps the words
    text = re.sub(r"#|(@\w+)", "", text)  # e.g. @Tim Hi #hello --> ' Hi hello'

    # Remove links (e.g. any that starts with https, http, www)
    # Tried so many...this was the simplest that actually worked
    text = re.sub(r"https?://\S+|www.\S+", "", text)

    # Remove punctuation, underscores, and other random symbols
    text = re.sub(r"[^\w\s]|_", "", text)  # e.g. 's. Hey. +_=Woo' --> 's Hey Woo'

    # Remove stopwords
    text_list = word_tokenize(text)
    text_list = [word for word in text_list if not word in STOP_WORDS]

    return " ".join(text_list)


def preprocess_data(data):
    """
    Filters out unnecessary text from tweets
    """
    global STOP_WORDS

    # Get 'stopwords'
    try:  # Can someone verify that stopwords successfully downloads on their system from this?
        STOP_WORDS = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        STOP_WORDS = stopwords.words("english")

    # Get word_tokenize nltk resource
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    if DATA_SOURCE == "stanford":
        data[5] = data[5].apply(text_filter)
        # TODO: change sentiments to integers for stanford data
    elif DATA_SOURCE == "kaggle":
        data["OriginalTweet"] = data["OriginalTweet"].apply(text_filter)
        data['Sentiment'] = data['Sentiment'].apply(lambda x: sentiment_to_int(x))


def read_input_data(filepath):
    global DATA_SOURCE
    # Adjust input parameters based on data source - standord dataset does not have a header
    if "stanford" in filepath:
        DATA_SOURCE = "stanford"
        df_data = pd.read_csv(
            filepath, header=None, sep=",", engine="python", nrows=read_nrows,
        )
    elif "kaggle" in filepath:
        DATA_SOURCE = "kaggle"
        df_data = pd.read_csv(filepath, sep=",", engine="python", nrows=read_nrows,)
    else:
        print(
            "Given dataset path not supported. Path must contain one of the strings 'stanford' or 'kaggle'."
        )
        sys.exit(1)
    return df_data


def get_bag_of_words(data, ngram_flag):
    """ Given data, return a bag of words downscaled into "term frequency times inverse document frequency” (tf–idf).
    """
    if int(ngram_flag):
        vectorizer = CountVectorizer()
        X_train_counts = vectorizer.fit_transform(data)
    else:
        vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(5, 5))
        X_train_counts = vectorizer.fit_transform(data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf


def stem(text):
    ps = PorterStemmer()
    stemmed_text = [ps.stem(word) for word in text]
    return " ".join(stemmed_text)


def stemming(data):
    token_data = data.apply(lambda x: word_tokenize(x))
    stem_data = token_data.apply(lambda x: stem(x))
    return stem_data


def lemmatize(text):
    wn = WordNetLemmatizer()
    try:
        lemmatized_text = [wn.lemmatize(word) for word in text]
    except LookupError:
        nltk.download('wordnet')
        lemmatized_text = [wn.lemmatize(word) for word in text]
    return " ".join(lemmatized_text)


def lemmatization(data):
    token_data = data.apply(lambda x: word_tokenize(x))
    lemmatized_data = token_data.apply(lambda x: lemmatize(x))
    return lemmatized_data


if __name__ == "__main__":
    if len(sys.argv) != 5:
        # <Path to Data Directory> - folder where data is located, in each algo
        print("Error: Given " + str(len(sys.argv) - 1) + " arguments but expected 3.")
        print(
            "Usage: python3 src/extract_features.py <Path to Data File> <save to pickle? 0 for no, 1 for yes>"
            " <ngram bag of words flag: 0 for unigram, 1 for ngram>"
            "<normalization type: stem, lemmatize or none>"
        )
        sys.exit(1)

    dataPath = sys.argv[1]
    save_to_pkl = sys.argv[2]
    ngram_flag = sys.argv[3]
    norm_flag = sys.argv[4]

    df_data = read_input_data(dataPath)
    print(df_data)

    preprocess_data(df_data)
    print(df_data)

    if DATA_SOURCE == "stanford":
        train_data = df_data[5]
        y_train = df_data[0]
        ids = df_data[1]
    elif DATA_SOURCE == "kaggle":
        train_data = df_data["OriginalTweet"]
        y_train = df_data['Sentiment']
        ids = df_data['UserName']

    if norm_flag == "stem":
        train_data = stemming(train_data)
    if norm_flag == "lemmatize":
        train_data = lemmatization(train_data)
    print(train_data)

    X_train_tfidf = get_bag_of_words(train_data, ngram_flag)
    print(X_train_tfidf)

    new_data = {'data': X_train_tfidf, 'labels': y_train, 'id': ids}


    if int(save_to_pkl):
        print("Saving data to pickle...")
        if int(ngram_flag):
            pickle.dump(
                # X_train_tfidf,
                new_data,
                open(
                    "../project_data_pickles/"
                    + DATA_SOURCE
                    + "_"
                    + str(read_nrows)
                    + "rows_ngram_tfidf.pkl",
                    "wb",
                ),
            )
        else:
            pickle.dump(
                # X_train_tfidf,
                new_data,
                open(
                    "../project_data_pickles/"
                    + DATA_SOURCE
                    + "_"
                    + str(read_nrows)
                    + "rows_unigramgram_tfidf.pkl",
                    "wb",
                ),
            )
        print("Done")

