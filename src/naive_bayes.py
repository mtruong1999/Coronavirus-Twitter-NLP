import sys
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class NaiveBayes:
    """
    Trains classifier using the training data in dataDirPath and
    uses the resulting model to classify the test data in dataDirPath.
    The classifications will be stored in idSentiments. However, this
    will be left empty if using validation
    """
    def __init__(self, dataDirPath, idSentiments):
        model = MultinomialNB()
        # TODO: get train and test data from dataDirPath
        X_train = []
        y_train = []
        X_test = []
        self.train_model(model, X_train, y_train)
        idSentiments = self.predict(model, X_test)


    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)


    def validate(self, model, X_val, y_val):
        pass


    def predict(self, model, X_test):
        return model.predict(X_test)


