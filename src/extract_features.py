import sys
import os
import pandas as pd
import pickle
import re
import string
from utils import *
import numpy as np

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
    text = re.sub(r"[^\w\s]|_", " ", text)  # e.g. 's. Hey. +_=Woo' --> 's Hey Woo'

    # Uncomment this to remove 'standalone' numbers, e.g. '5 times6' -> ' times6'
    # text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    # Uncomment this to remove ALL numbers instead, e.g. '5 covid19' -> ' covid'
    # text = re.sub("\d+", " ", text)

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
        data["Sentiment"] = data["Sentiment"].apply(lambda x: sentiment_to_int(x))
        # Remove data elements with empty tweets after filtering
        # data = data[data['OriginalTweet'] != ''] # Filtering out blanks like this doesn't carry over to outside of function scope for some reason
        data["OriginalTweet"] = data["OriginalTweet"].apply(
            lambda x: np.nan if not x else x
        )
        data.dropna(subset=["OriginalTweet"], inplace=True)
        data.reset_index(drop=True, inplace=True)


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


def get_bag_of_words(data, ngram_flag, test=False, norm_flag="none"):
    """ Given data, return a bag of words downscaled into "term frequency times inverse document frequency” (tf–idf).
    """
    ngram_range_ = (5, 5) if ngram_flag else (1, 1)
    analyzer_ = "char_wb" if ngram_flag else "word"

    count_filename = DATA_SOURCE + "_" + norm_flag + "_count_vectorizer_"
    count_filename += "ngram.pkl" if ngram_flag else "unigram.pkl"
    count_path = os.path.join("..", "project_data_pickles", count_filename)

    tfidf_filename = DATA_SOURCE + "_" + norm_flag + "_tfidf_transformer_"
    tfidf_filename += "ngram.pkl" if ngram_flag else "unigram.pkl"
    tfidf_path = os.path.join("..", "project_data_pickles", tfidf_filename)

    if not test:
        vectorizer = CountVectorizer(analyzer=analyzer_, ngram_range=ngram_range_)
        X_train_counts = vectorizer.fit_transform(data)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        pickle.dump(vectorizer, open(count_path, "wb"))
        pickle.dump(tfidf_transformer, open(tfidf_path, "wb"))

    else:
        if not os.path.isfile(count_path) or not os.path.isfile(tfidf_path):
            sys.exit(
                "Data prep pickle file missing, must process the {} training set first.".format(
                    DATA_SOURCE
                )
            )

        vectorizer = pickle.load(open(count_path, "rb"))
        X_train_counts = vectorizer.transform(data)

        tfidf_transformer = pickle.load(open(tfidf_path, "rb"))
        X_train_tfidf = tfidf_transformer.transform(X_train_counts)

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
        nltk.download("wordnet")
        lemmatized_text = [wn.lemmatize(word) for word in text]
    return " ".join(lemmatized_text)


def lemmatization(data):
    token_data = data.apply(lambda x: word_tokenize(x))
    lemmatized_data = token_data.apply(lambda x: lemmatize(x))
    return lemmatized_data


if __name__ == "__main__":
    if len(sys.argv) != 5:
        # <Path to Data Directory> - folder where data is located, in each algo
        print("Error: Given " + str(len(sys.argv) - 1) + " arguments but expected 4.")
        print(
            "Usage: python3 src/extract_features.py <Path to Data File> <save to pickle? 0 for no, 1 for yes>"
            " <ngram bag of words flag: 0 for unigram, 1 for ngram>"
            "<normalization type: stem, lemmatize or none>"
        )
        sys.exit(1)

    dataPath = sys.argv[1]
    save_to_pkl = sys.argv[2]
    ngram_flag = int(sys.argv[3])
    norm_flag = sys.argv[4]

    is_test = True if "test" in dataPath or "Test" in dataPath else False

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
        y_train = df_data["Sentiment"]
        ids = df_data["UserName"]

    if norm_flag == "stem":
        train_data = stemming(train_data)
    if norm_flag == "lemmatize":
        train_data = lemmatization(train_data)
    print(train_data)

    X_train_tfidf = get_bag_of_words(
        train_data, ngram_flag, test=is_test, norm_flag=norm_flag
    )
    print(X_train_tfidf)

    new_data = {"data": X_train_tfidf, "labels": y_train, "id": ids}

    if int(save_to_pkl):
        print("Saving data to pickle...")
        test_str = "test" if is_test else "train"
        ngram_str = "ngram" if ngram_flag else "unigram"
        rows_str = "all_" if read_nrows == None else str(read_nrows)

        pickle.dump(
            # X_train_tfidf,
            new_data,
            open(
                "../project_data_pickles/"
                + DATA_SOURCE
                + "_"
                + test_str
                + "_"
                + norm_flag
                + "_"
                + rows_str
                + "rows"
                + "_"
                + ngram_str
                + "_"
                + "tfidf.pkl",
                "wb",
            ),
        )
        print("Done")

