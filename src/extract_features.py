import sys
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Global Variables
DATA_SOURCE = ""

# Some variables for testing
read_nrows = 1000  # When None, all rows are read.


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
        X_train_counts = vectorizer.fit_transform(train_data)
    else:
        vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(5, 5))
        X_train_counts = vectorizer.fit_transform(train_data)

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

    if DATA_SOURCE == "stanford":
        train_data = df_data[5].values
    elif DATA_SOURCE == "kaggle":
        train_data = df_data.OriginalTweet.values

    X_train_tfidf = get_bag_of_words(train_data, ngram_flag)
    print(X_train_tfidf)

