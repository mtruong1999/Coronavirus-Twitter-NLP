import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pyLDAvis
import os
import warnings
import seaborn as sns
import sys
from pyLDAvis import sklearn as sklearn_lda
from sklearn.decomposition import LatentDirichletAllocation as LDA
import gensim  # More comprehensive LDA library than sklearn's
import gensim.corpora as corpora
from pprint import pprint

from gensim.models import CoherenceModel  # Compute Coherence Score

warnings.filterwarnings("ignore", category=DeprecationWarning)
sns.set_style("whitegrid")

# Adapted from this guide: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

# Helper function
def plot_N_most_common_words(N, count_data, count_vectorizer):
    import matplotlib.pyplot as plt

    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = zip(words, total_counts)
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title="10 most common words")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette="husl")
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel("words")
    plt.ylabel("counts")
    plt.show()


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]))


def sklearn_LDA(count_vectorizer, data, number_topics, number_words):
    # Visualise the <number_words> most common words
    plot_N_words = False  # toggle plotting on or off - somewhat useful for tuning
    if plot_N_words:
        plot_N_most_common_words(number_words, train_data["data"], count_vectorizer)

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(data)  # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)
    print("=========")

    LDAvis_data_filepath = os.path.join("../ldavis/viz" + str(number_topics))
    compute_new_LDA = True  # toggle on or off computation and saving
    if compute_new_LDA:
        LDAvis_prepared = sklearn_lda.prepare(lda, data, count_vectorizer)
        with open(LDAvis_data_filepath, "wb") as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, "rb") as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, "../ldavis/viz" + str(number_topics) + ".html")


def gensim_LDA(
    lemmatized_data,
    data,
    n_topics=10,
    passes=10,
    chunksize=100,
    per_word_topics=True,
    coherence="c_v",
):
    """Create and output data from a fitted LDA model.

    Keyword arguments:
    n_topics -- (int) The number of requested latent topics to be extracted from the training corpus (default 10)
    passes -- (int) Number of passes through the corpus during training (default 10)
    chunksize -- (int) Number of documents to be used in each training chunk (default 100)
    per_word_topics -- (Bool) If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count) (default True)
    coherence -- ({'u_mass', 'c_v', 'c_uci', 'c_npmi'}) – Coherence measure to be used. Fastest method - ‘u_mass’, ‘c_uci’ also known as c_pmi. (default "c_v")
    """
    data_lemmatized = [i.split(" ") for i in lemmatized_data]

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)  # Create Corpus
    texts = data_lemmatized  # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]  # View

    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=n_topics,
        random_state=42,
        chunksize=100,
        passes=10,
        per_word_topics=True,
    )

    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence="c_v"
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print("\nCoherence Score: ", coherence_lda)


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
    parser.add_argument(
        "--lemmatized_data", type=str, default=None, help="name of lemmatized data file"
    )

    args = parser.parse_args()
    dataDirPath = sys.argv[1]
    train_file = args.train_file
    test_file = args.test_file
    vectorizer_file = args.vectorizer_file
    data_lemmatized_file = args.lemmatized_data

    if not os.path.isdir(dataDirPath):
        print("Error: Directory " " + dataDirPath + " " does not exist.")
        sys.exit(1)
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
    if not os.path.isfile(os.path.join(dataDirPath, data_lemmatized_file)):
        print("Error: lemmatized_data file " + train_file + " does not exist.")
        sys.exit(1)

    train_pkl = open(os.path.join(dataDirPath, train_file), "rb")
    train_data = pickle.load(train_pkl)
    vectorizer_pkl = open(os.path.join(dataDirPath, vectorizer_file), "rb")
    count_vectorizer = pickle.load(vectorizer_pkl)
    data_lemmatized_pkl = open(os.path.join(dataDirPath, data_lemmatized_file), "rb")
    data_lemmatized = pickle.load(data_lemmatized_pkl)

    # Tweak this parameter
    number_topics = 6

    run_sklearn_LDA = False
    if run_sklearn_LDA:
        number_words = 10  # only for visualization
        sklearn_LDA(count_vectorizer, train_data["data"], number_topics, number_words)

    run_gensim_LDA = True
    if run_gensim_LDA:
        gensim_LDA(data_lemmatized, train_data["data"], n_topics=number_topics)

