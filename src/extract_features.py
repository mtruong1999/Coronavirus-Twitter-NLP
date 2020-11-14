import sys
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Some variables for testing
read_nrows = None  # When None, all rows are read.

if __name__ == "__main__":
    if len(sys.argv) != 2:
        # <Path to Data Directory> - folder where data is located, in each algo
        print("Error: Given " + str(len(sys.argv) - 1) + " arguments but expected 1.")
        print("Usage: python3 src/extract_features.py <Path to Data File>")
        sys.exit(1)

    dataDirPath = sys.argv[1]

    # Adjust input parameters based on data source - standord dataset does not have a header
    if "stanford" in dataDirPath:
        df_data = pd.read_csv(
            dataDirPath, header=None, sep=",", engine="python", nrows=read_nrows,
        )
    elif "kaggle" in dataDirPath:
        df_data = pd.read_csv(dataDirPath, sep=",", engine="python", nrows=read_nrows,)
    else:
        print(
            "Given dataset path not supported. Path must contain one of the strings 'stanford' or 'kaggle'."
        )
        sys.exit(1)
    print(df_data)

    count_vect = CountVectorizer()
    if "stanford" in dataDirPath:
        train_data = df_data[5].values
    elif "kaggle" in dataDirPath:
        train_data = df_data.OriginalTweet.values
    # print(train_data)
    X_train_counts = count_vect.fit_transform(train_data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    print(X_train_tfidf)
