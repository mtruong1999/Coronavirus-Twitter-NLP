import sys
import os
import pandas as pd
import argparse

from artificial_neural_network import ArtificialNeuralNetwork
from naive_bayes import NaiveBayes

CLASSIFIER_TO_CONSTRUCTOR = {"nb" : NaiveBayes, "ann" : ArtificialNeuralNetwork}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file',
                        default='kaggle_1000rows_ngram_tfidf.pkl',
                        help='name of train pickle file')
    parser.add_argument('--test_file',
                        type=str,
                        default=None,
                        help='name of test pickle file')
    parser.add_argument('--transfer_flag',
                        type=int,
                        default=0,
                        help='flag to use transfer learning')

    args = parser.parse_args()

    dataDirPath = sys.argv[1]
    classifier_type = sys.argv[2]

    if not os.path.isdir(dataDirPath):
        print("Error: Directory \"" + dataDirPath + "\" does not exist.")
        sys.exit(1)

    if classifier_type not in CLASSIFIER_TO_CONSTRUCTOR:
        print("Error: Classifier type must be one of " +
              str(list(CLASSIFIER_TO_CONSTRUCTOR.keys())) + ".")
        sys.exit(1)

    train_file = args.train_file
    test_file = args.test_file
    transfer_flag = args.transfer_flag

    if not os.path.isfile(os.path.join(dataDirPath, train_file)):
        print('Error: Train file ' + train_file + ' does not exist.')
        sys.exit(1)
    if test_file:
        if not os.path.isfile(os.path.join(dataDirPath, test_file)):
            print('Error: Test file' + test_file + 'does not exist.')
            sys.exit(1)

    # Classify data and output results to "classifications.csv"
    # First column is some sort of identifier for the tweets, I think "UserName"?
    # Second column is the class label.
    idSentiments = {}
    classifier = CLASSIFIER_TO_CONSTRUCTOR[classifier_type]()(dataDirPath, idSentiments, train_file, test_file, transfer_flag)
    print(idSentiments)
    
    if idSentiments != {}:
                pd.DataFrame(idSentiments).to_csv("classifications.csv",
                                                   header=["id", "sentiment"], index=False)
