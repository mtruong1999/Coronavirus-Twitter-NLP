import sys
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle


class NaiveBayes(object):
    """
    Trains classifier using the training data in dataDirPath and
    uses the resulting model to classify the test data in dataDirPath.
    The classifications will be stored in idSentiments. However, this
    will be left empty if using validation
    """
    def __init__(self):
        self.model = MultinomialNB()

    def __call__(self, dataDirPath, idSentiments, train_file, test_file):
        model = self.model

        # Get data and labels from pickle file
        pickle_file = open(os.path.join(dataDirPath, train_file), 'rb')
        train = pickle.load(pickle_file)
        X = train['data']
        y = train['labels']

        if not test_file:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = model.score(X_val, y_val)
            print(accuracy)

        else:
            test_pkl_file = open(os.path.join(dataDirPath, test_file), 'rb')
            test = pickle.load(test_pkl_file)
            X_test = test['data']
            model.fit(X, y)
            y_pred = model.predict(X_test)
            idSentiments['id'] = test['id']
            idSentiments['sentiment'] = y_pred
