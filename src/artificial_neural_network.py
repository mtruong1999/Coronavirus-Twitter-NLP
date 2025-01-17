import os
import sys
import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *

# Code adapted from
# https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-9-neural-networks-with-tfidf-vectors-using-d0b4af6be6d7


class ArtificialNeuralNetwork(object):
    def __init__(self):
        """
        Trains classifier using the training data in dataDirPath and
        uses the resulting model to classify the test data in dataDirPath.
        The classifications will be stored in idSentiments. However, this
        will be left empty if using validation
        """
        print("Using ANN...")

    def __call__(self, dataDirPath, idSentiments, train_file, test_file, transfer_flag):
        # Get data and labels from pickle file
        pickle_file = open(os.path.join(dataDirPath, train_file), "rb")
        train = pickle.load(pickle_file)
        X = train["data"]
        y = train["labels"]

        if transfer_flag:
            y = y.apply(lambda x: covid_to_stanford(x))
        print(y)

        if "stanford" in train_file:
            X, X_val, y, y_val = train_test_split(
                X, y, train_size=41100, random_state=42
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Lets the model scale to the input size
        input_dim = X_train.shape[1]
        print(input_dim)

        model = self.nn(input_dim)
        print(model.summary())

        model_checkpoint_callback = ModelCheckpoint(
            filepath="../saved_models/ann.hdf5", monitor="val_loss"
        )
        history = model.fit_generator(
            generator=self.batch_generator(X_train, y_train, 32),
            epochs=5,
            validation_data=(X_val, y_val),
            steps_per_epoch=X_train.shape[0] / 32,
            callbacks=[model_checkpoint_callback],
        )
        np.save("../saved_models/ann_history.npy", history.history)
        y_pred_prob = model.predict(X_val)
        y_pred = y_pred_prob.argmax(axis=1)
        accuracy = accuracy_score(y_val, y_pred)
        print(accuracy)

        if test_file:
            test_pkl_file = open(os.path.join(dataDirPath, test_file), "rb")
            test = pickle.load(test_pkl_file)
            X_test = test["data"]
            y_test = test["labels"]
            y_pred_prob = model.predict(X_test)
            y_pred = y_pred_prob.argmax(axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            print("accuracy: " + str(accuracy))
            idSentiments["id"] = test["id"]
            idSentiments["sentiment"] = y_pred

    def batch_generator(self, X, y, batch_size):
        samples_per_epoch = X.shape[0]
        number_of_batches = samples_per_epoch / batch_size
        counter = 0
        index = np.arange(np.shape(y)[0])
        np.random.shuffle(index)
        while 1:
            index_batch = index[batch_size * counter : batch_size * (counter + 1)]
            X_batch = X[index_batch, :].toarray()
            y_batch = y[y.index[index_batch]]
            counter += 1
            yield X_batch, y_batch
            if counter > number_of_batches:
                counter = 0

    def nn(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_dim=input_dim))
        model.add(Dense(5, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
