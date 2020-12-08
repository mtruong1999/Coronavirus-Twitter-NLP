# Coronavirus-Twitter-NLP
This code classifies a corpus of coronavirus tweets for sentiment

## Setup
1. Installation of [Python 3.7](https://www.python.org/downloads/) or later required

2. To install dependencies, in the `Coronavirus-Twitter-NLP/` directory, run
```
pip install -r requirements.txt
```

## Usage
The Coronavirus Tweets can be downloaded [here](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification).

The General Tweets can be downloaded [here](http://help.sentiment140.com/for-students).

**NOTE:** 
Rename data files with "kaggle" and "stanford" in the filenames for the 
coronavirus and general tweet files, respectively.

To clean, preprocess and extract features from the data set, run:
```
python extract_features.py <Path to Data File> <Save to Pickle> <Ngram> <Normalization Type> 
```
**NOTE:** To save pickle files, create a `project_data_pickles/` in the directory above where this 
repository is saved to.

* `<Save to Pickle>` 0 for no, 1 for yes
* `<Ngram>` 0 for unigram, 1 for 5-gram
* `<Normalization Type>` "stem", "lemmatize", or "none"

To classify tweets, run:
```
python classify.py <Path to Pickle File Dir> <Classifier Type>
    --train_file <Train Data File>
    --test_file <Test Data File>
    --transfer_flag <Transfer Learning? 0 or 1>
```

* `<Classifier Type>` "nb" for naive bayes, "ann" for neural net

Predictions will be saved in "classifications.csv."
The model and the training history of the model will by saved in
"..\saved_models"

To generate the confusion matrix for the predictions and 
train/validation accuracy and loss plots of the model, run:

```
python results.py 
    --results_file <classifications.csv>
    --test_file <Test data file with true labels>
    --history_file <Full Path to history.npy>
    --results_dir <Directory of results file>
    --test_dir <Directory of test file>
    --history_dir <Directory of history file>
```

To build a topic model or run a grid search over LDA parameters, run:
```
python topic_model.py <Path to Data Dir>
    --train_file <Train Data File>
    --vectorizer_file <Count Vectorizer File>
    --lemmatized_data <Limatized Data File>
```
Flags choosing to either run a parameter grid search or build a model from previously discovered optimal parameters can be set in the code's main function. The default behavior is to build a model from previously discovered optimal parameters. If a grid search is performed, "lda_tuning_results_\<X-Y\>.csv" is written containing the results, for X, Y min-topic and max-topic bounds, respectively, for the grid search. If a grid search is not performed, a predefined model is built and an html file is saved to "gensim_viz_symmetric_0.61_15topics.html". This file is an interactive visualization of the topic model and can be opened in a web browser.

To produce a plot of an analysis of the LDA grid search results, run:
```
python analyze_topic_model.py
```
This script assumes there exists the files "../results/lda_tuning_results_7-10.csv" and "../results/lda_tuning_results_10-15.csv". A figure is saved to "../results/tm_gs_results.pdf".
