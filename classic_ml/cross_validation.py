""" Module containing implementation of hyperparameter validation and classifier training and scoring """

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer

from classic_ml.resources import SEED, NUM_CORES

# List of classifiers, with pre-set default parameters, that will be trained and scored
classifiers_default = {"Naive Bayes": GaussianNB(),
                       "Logistic Regression": LogisticRegression(
                           C=1e-2,
                           solver="sag",
                           tol=1e-5,
                           max_iter=10000,
                           class_weight="balanced",
                           random_state=SEED,
                           n_jobs=NUM_CORES // 2),
                       "SVM": LinearSVC(
                           C=1e-2,
                           penalty="l2",
                           loss="squared_hinge",
                           dual=False,
                           tol=1e-5,
                           class_weight="balanced",
                           random_state=SEED,
                           max_iter=10000),
                       "Neural Network": MLPClassifier(
                           hidden_layer_sizes=(100,),
                           activation="relu",
                           solver="adam",
                           learning_rate="constant",
                           learning_rate_init=0.01,
                           max_iter=1000,
                           tol=1e-5,
                           early_stopping=True,
                           validation_fraction=0.1,
                           random_state=SEED),
                       "Random Forest": RandomForestClassifier(
                           n_estimators=250,
                           criterion="entropy",
                           min_samples_leaf=5,
                           class_weight="balanced",
                           random_state=SEED,
                           n_jobs=NUM_CORES // 2)}

# Define the hyperparameter search space for each classifier, used by the grid search validator
param_grids = {
    "Naive Bayes": {"var_smoothing": np.logspace(-18, 0, 19)},
    "Logistic Regression": {"C": np.logspace(-5, 5, 11),
                            "solver": ["lbfgs", "sag"],
                            "max_iter": [1000, ],
                            "tol": [1e-5, ],
                            "class_weight": ["balanced", ],
                            "random_state": [SEED, ],
                            "n_jobs": [NUM_CORES // 2, ]},
    "SVM": {"C": np.logspace(-5, 5, 11),
            "penalty": ["l2", ],
            "loss": ["squared_hinge", ],
            "dual": [False, ],
            "tol": [1e-5, ],
            "class_weight": ["balanced", ],
            "random_state": [SEED, ],
            "max_iter": [10000, ]},
    "Neural Network": {"hidden_layer_sizes": [(value,) for value in (1, 2, 5, 10, 20, 50, 100, 200, 500)],
                       "activation": ["tanh", "relu"],
                       "solver": ["adam"],
                       "alpha": np.logspace(-5, 5, 11),
                       "learning_rate": ["constant", ],
                       "learning_rate_init": [1e-2, ],
                       "max_iter": [1000, ],
                       "tol": [1e-5, ],
                       "random_state": [SEED, ]},
    "Random Forest": {"n_estimators": [10, 25, 50, 100, 250, 500, 1000],
                      "criterion": ["entropy"],
                      "max_depth": [None, 10],
                      "min_samples_split": [2, ],
                      "min_samples_leaf": [5, ],
                      "n_jobs": [NUM_CORES // 2, ],
                      "random_state": [SEED, ]}}


def get_best_params_for_classifiers(train_set, train_classes):
    """
    Function that implements the complete cross-validation and model training pipeline

    :param train_set: tweet training feature vector values, pandas.DataFrame object
    :param train_classes: tweet training class labels, pandas.Series or numpy.array object

    :returns: dictionary mapping classifier names to their best hyperparameters,
              dictionary mapping classifier names to their models trained using the best hyperparameters,
              dictionary mapping classifier names to the their best mean cross-validation scores
    """

    # Get list of classifier names
    classifier_list = list(classifiers_default.keys())

    # Initialize result dictionaries
    classifiers_best_params = dict.fromkeys(classifier_list)
    classifiers_best_models = dict.fromkeys(classifier_list)
    classifiers_cross_validation_scores = dict.fromkeys(classifier_list)

    # Use 5-fold cross-validation with stratified sampling and random shuffling
    # Set the random seed to the global seed to have the same folds for each run
    validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Score the classifiers on 4 evaluation metrics simultaneously
    scorer = {"Accuracy": make_scorer(accuracy_score, greater_is_better=True),
              "Precision": make_scorer(precision_score, greater_is_better=True),
              "Recall": make_scorer(recall_score, greater_is_better=True),
              "F1": make_scorer(f1_score, greater_is_better=True)}
    evaluation_metrics = list(scorer.keys())

    for classifier_name, classifier_model in classifiers_default.items():
        print("Cross-validating {} classifier...".format(classifier_name))
        classifier_grid_search = GridSearchCV(estimator=classifier_model,
                                              param_grid=param_grids[classifier_name],
                                              scoring=scorer,
                                              cv=validator,
                                              refit="F1",  # After the validation is done, retrain the classifier
                                                           # using the best hyperparameters now on the whole train set
                                                           # The best hyperparameters will be those that maximize
                                                           # the mean F1 score on the validation folds
                                              n_jobs=NUM_CORES // 2,
                                              verbose=1)
        # Run the grid search cross-validation procedure for the classifier
        classifier_grid_search.fit(train_set, train_classes)

        # Retrieve the best hyperparameters and trained models for each classifier
        classifiers_best_params[classifier_name] = classifier_grid_search.best_params_
        classifiers_best_models[classifier_name] = classifier_grid_search.best_estimator_

        # Report the mean test value for each evaluation metric for each classifier trained using the best parameters
        classifier_cross_validation_scores = \
            {metric: classifier_grid_search.cv_results_["mean_test_{}".format(metric)]
             for metric in evaluation_metrics}
        best_model_index = np.argmax(classifier_cross_validation_scores["F1"])
        classifiers_cross_validation_scores[classifier_name] = {metric: scores[best_model_index]
                                                                for metric, scores in
                                                                classifier_cross_validation_scores.items()}
        print("Done with {}!".format(classifier_name))

    return classifiers_best_params, classifiers_best_models, classifiers_cross_validation_scores
