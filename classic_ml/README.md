# CS-433 Project 2, 2019, Text Classification

## File structure

- `cross_validation.py`: Module containing implementation of hyperparameter validation and classifier training and scoring
- `feature_extraction.py`: Module containing implementations of the procedures required for extracting the different types of features from the processed tweet texts
- `feature_selection.py`: Module containing implementations of the feature normalization and selection procedures
- `plotting.py`: Script containing code to generate the plots
- `preprocessing.py`: Module containing helper functions for performing preprocessing on the tweet text
- `resources.py`: Module containing resources reused in multiple scripts, for efficient importing
- `serialization.py`: Module containing helper functions for (de)serializing and (de)compressing Python objects from/to local storage
- `training.py`: Training script 
    - Performing preprocessing and feature extraction on the training set of tweets
    - Performing feature normalization and training a feature selection model using the training feature vectors
    - Performing cross-validation on the classifiers to fit the hyperparameters and find the best model
- `run.py`: Main script
    - Performing preprocessing and feature extraction on the testing set of tweets
    - Normalization and feature selection using the pre-fitted transformers
    - Generating the baseline submission using the trained best classifier's predictions
 - `files/`: Folder with intermediate datasets, serialized transformers and models
 - `results/`: Folder with final scores, plots and submission file

## Reproducibility

### Prerequisite
1. Python 3.7 is installed

2. That the [data](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files) is downloaded and extracted like :
------------

    ├── data
    │   ├── train_neg.txt
    │   ├── train_pos.txt
    │   ├── test_data.txt

--------

3. That at least these files are present in the `files/` folder (as it is now):

---------------

    ├── files
    │   ├── best_model.gz
    │   ├── test_dataset_reduced.tsv

----------------

**Notes on dependencies**

- [`pandas`]( https://pandas.pydata.org/ ):  Data structures and analysis tools. Used for convient representation of tabular data and results.
- [`seaborn`](https://seaborn.pydata.org/): Python visualization library based on matplotlib. Used for plotting.
- [`nltk`]( http://www.nltk.org/ ): NLP framework. Used for tweet preprocessing and feature extraction tasks.
- [`scikit-learn`]( https://scikit-learn.org/stable/ ): Machine learning framework. Used for the feature selection, cross-validation, training and evaluation of the standard machine learning classifiers.
- [`empath`](https://github.com/Ejhfast/empath-client): Tool for analyzing text across lexical categories. Used for feature extraction.
- [`_pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) and [`compress_pickle`](https://lucianopaz.github.io/compress_pickle/html/): Used for Python object serialization and compression to local storage.

### Run the code

From the root folder of the project

In order to directly generate the classic ML submission again (it is saved at `results/submission.csv`):
```shell
python3 run.py 
```

In order to perform the feature extraction and selection, cross-validation and classifier training on the train set again:
```shell
python3 training.py
```

In order to perform the feature extraction and selection on the test set again, delete the file `test_dataset_reduced.tsv` currently present in the `files/` folder and run again:
```shell
python3 run.py 
```

## Authors (team: Definitely not GRUs)

- Louis Amaudruz
- Andrej Janchevski
- Timoté Vaucher
