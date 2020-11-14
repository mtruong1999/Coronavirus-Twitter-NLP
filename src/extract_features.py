import sys
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Global Variables
DATA_SOURCE = ""

# Some variables for testing
read_nrows = None  # When None, all rows are read.


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


def get_bag_of_words(data):
    count_vect = CountVectorizer()
    if DATA_SOURCE == "stanford":
        train_data = data[5].values
    elif DATA_SOURCE == "kaggle":
        train_data = data.OriginalTweet.values
    # print(train_data)
    X_train_counts = count_vect.fit_transform(train_data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # <Path to Data Directory> - folder where data is located, in each algo
        print("Error: Given " + str(len(sys.argv) - 1) + " arguments but expected 1.")
        print("Usage: python3 src/extract_features.py <Path to Data File>")
        sys.exit(1)

    dataPath = sys.argv[1]

    df_data = read_input_data(dataPath)
    print(df_data)

    X_train_tfidf = get_bag_of_words(df_data)
    print(X_train_tfidf)
