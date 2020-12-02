import os
import sys
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Code adapted from
# https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-9-neural-networks-with-tfidf-vectors-using-d0b4af6be6d7

input_dimension = 35386


class ArtificialNeuralNetwork(object):
    def __init__(self):
        """
        Trains classifier using the training data in dataDirPath and
        uses the resulting model to classify the test data in dataDirPath.
        The classifications will be stored in idSentiments. However, this
        will be left empty if using validation
        """
        # self.ann = self.nn()

    def __call__(self, dataDirPath, idSentiments, train_file, test_file):
        global input_dimension
        # model = self.ann

        # Get data and labels from pickle file
        pickle_file = open(os.path.join(dataDirPath, train_file), "rb")
        train = pickle.load(pickle_file)
        X = train["data"]
        y = train["labels"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # This sets the gobal variable and lets the model scale to the input size
        input_dimension = X_train.shape[1]
        self.ann = self.nn()
        model = self.ann

        model_checkpoint_callback = ModelCheckpoint(
            filepath="../saved_models/ann.hdf5", monitor="val_acc"
        )
        history = model.fit_generator(
            generator=self.batch_generator(X_train, y_train, 32),
            epochs=1,
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
            y_pred_prob = model.predict(X_test)
            y_pred = y_pred_prob.argmax(axis=1)
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

    def nn(self):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_dim=input_dimension))
        model.add(Dense(5, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
