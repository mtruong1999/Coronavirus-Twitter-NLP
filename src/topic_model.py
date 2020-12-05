import argparse
import pandas as pd
import pickle
import pyLDAvis
import os
import warnings
import sys
from pyLDAvis import sklearn as sklearn_lda
from sklearn.decomposition import LatentDirichletAllocation as LDA

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Adapted from this guide: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "Path_to_Data_Directory", help="folder where data is located in each algo"
    )
    parser.add_argument(
        "--train_file",
        default="kaggle_1000rows_ngram_tfidf.pkl",
        help="name of train pickle file",
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="name of test pickle file"
    )
    parser.add_argument(
        "--vectorizer_file", type=str, default=None, help="name of vectorizer file"
    )

    args = parser.parse_args()

    dataDirPath = sys.argv[1]

    if not os.path.isdir(dataDirPath):
        print("Error: Directory " " + dataDirPath + " " does not exist.")
        sys.exit(1)

    train_file = args.train_file
    test_file = args.test_file
    vectorizer_file = args.vectorizer_file

    if not os.path.isfile(os.path.join(dataDirPath, train_file)):
        print("Error: Train file " + train_file + " does not exist.")
        sys.exit(1)
    if test_file:
        if not os.path.isfile(os.path.join(dataDirPath, test_file)):
            print("Error: Test file" + test_file + "does not exist.")
            sys.exit(1)
    if not os.path.isfile(os.path.join(dataDirPath, vectorizer_file)):
        print("Error: Vectorizer file " + train_file + " does not exist.")
        sys.exit(1)

    train_pkl = open(os.path.join(dataDirPath, train_file), "rb")
    train_data = pickle.load(train_pkl)
    vectorizer_pkl = open(os.path.join(dataDirPath, vectorizer_file), "rb")
    count_vectorizer = pickle.load(vectorizer_pkl)

    # Tweak the two parameters below
    number_topics = 5
    number_words = 10  # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(train_data["data"])  # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)

    LDAvis_data_filepath = os.path.join("../ldavis/viz" + str(number_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if True:
        LDAvis_prepared = sklearn_lda.prepare(lda, train_data["data"], count_vectorizer)
        with open(LDAvis_data_filepath, "wb") as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, "rb") as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, "../ldavis/viz" + str(number_topics) + ".html")

