import os
import sys
import argparse
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file',
                        required=True,
                        help='full path to classifications.csv')

    parser.add_argument('--test_file',
                        required=True,
                        help='full path to test file')

    parser.add_argument('--history_file',
                        required=True,
                        help='full path to history.npy')

    parser.add_argument('--results_dir',
                        default='./',
                        help='dir where results file is located')

    parser.add_argument('--test_dir',
                        default='../project_data_pickles',
                        help='dir of test file')

    parser.add_argument('--history_dir',
                        default='../saved_models',
                        help='dir of history file')

    args = parser.parse_args()

    # Create confusion matrix
    results = pd.read_csv(os.path.join(args.results_dir, args.results_file))
    pickle_file = open(os.path.join(args.test_dir, args.test_file), 'rb')
    test_data = pickle.load(pickle_file)

    y_pred = results['sentiment']
    y_true = test_data['labels']

    cm = confusion_matrix(y_true, y_pred)
    sn.heatmap(cm, annot=True)

    plt.title('Confusion Matrix')
    plt.ylabel('True Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.savefig('../results/cm.png')
    plt.close()

    # Model Accuracy and Loss
    history = np.load(os.path.join(args.history_dir, args.history_file), allow_pickle=True).item()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig('../results/accuracy.png')
    plt.close()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('../results/loss.png')
    plt.close()
