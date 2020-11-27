import sys
import os
import pandas as pd

from artificial_neural_network import ArtificialNeuralNetwork
from naive_bayes import NaiveBayes

CLASSIFIER_TO_CONSTRUCTOR = {"nb" : NaiveBayes, "ann" : ArtificialNeuralNetwork}
# TODO: Maybe create a separate 'extract features' MAIN file that does all of the
# data cleaning/processing etc. and saves the extracted features to a .npy or some
# other file to be loaded in by the algorithms. 
# OR: We can have each algorithm do the data extraction itself. Option 1 might be more
# efficient to avoid having to process the data everytime we run

if __name__ == "__main__":
    if len(sys.argv) != 3:
        # <Classifier Type> - either naive bayes or ANN
        # <Path to Data Directory> - folder where data is located, in each algo
        print("Error: Given " + str(len(sys.argv)) + " arguments but expected 3.")
        print("Usage: python3 src/classify.py <Path to Data Directory> <Classifier Type>")
        sys.exit(1)
    
    dataDirPath = sys.argv[1]
    classifier_type = sys.argv[2]

    # TODO: update arguments for train_file and test_file

    if not os.path.isdir(dataDirPath):
        print("Error: Directory \"" + dataDirPath + "\" does not exist.")
        sys.exit(1)

    if classifier_type not in CLASSIFIER_TO_CONSTRUCTOR:
        print("Error: Classifier type must be one of " +
              str(list(CLASSIFIER_TO_CONSTRUCTOR.keys())) + ".")
        sys.exit(1)

    # Classify data and output results to "classifications.csv"
    # First column is some sort of identifier for the tweets, I think "UserName"?
    # Second column is the class label.
    idSentiments = {}
    classifier = CLASSIFIER_TO_CONSTRUCTOR[classifier_type]()(dataDirPath, idSentiments)
    print(idSentiments)
    
    if idSentiments != {}:
                pd.DataFrame(idSentiments).to_csv("classifications.csv",
                                                   header=["id", "sentiment"], index=False)
