import sys
import os
import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Global Variables
DATA_SOURCE = ""
STOP_WORDS = []

# Some variables for testing
read_nrows = 51  # When None, all rows are read.


def text_filter(text):
    text.lower()
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
    elif DATA_SOURCE == "kaggle":
        data["OriginalTweet"] = data["OriginalTweet"].apply(text_filter)


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
    if ngram_flag:
        vectorizer = CountVectorizer()
        X_train_counts = vectorizer.fit_transform(data)
    else:
        vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(5, 5))
        X_train_counts = vectorizer.fit_transform(data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf


if __name__ == "__main__":
    if len(sys.argv) != 3:
        # <Path to Data Directory> - folder where data is located, in each algo
        print("Error: Given " + str(len(sys.argv) - 1) + " arguments but expected 2.")
        print(
            "Usage: python3 src/extract_features.py <ngram bag of words flag: 0 for unigram, 1 for ngram> <Path to Data File>"
        )
        sys.exit(1)

    dataPath = sys.argv[1]
    ngram_flag = sys.argv[2]

    df_data = read_input_data(dataPath)
    print(df_data)
    print(df_data[5][50])

    preprocess_data(df_data)
    print(df_data)
    print(df_data[5][50])

    if DATA_SOURCE == "stanford":
        train_data = df_data[5]
    elif DATA_SOURCE == "kaggle":
        train_data = df_data["OriginalTweet"]

    X_train_tfidf = get_bag_of_words(train_data, ngram_flag)
    print(X_train_tfidf)

