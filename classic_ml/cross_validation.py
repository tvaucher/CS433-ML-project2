import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import make_scorer

from classic_ml.resources import SEED

classifiers_default = {'Naive Bayes': GaussianNB(),
                       'KNN': KNeighborsClassifier(
                           n_neighbors=30,
                           weights="distance",
                           algorithm="auto",
                           metric='euclidean',
                           n_jobs=-1),
                       'Logistic Regression': LogisticRegression(
                           C=1e-2,
                           solver='sag',
                           tol=1e-5,
                           max_iter=10000,
                           class_weight="balanced",
                           random_state=SEED,
                           n_jobs=-1),
                       # TODO: Replace with LinearSVC
                       # 'SVM': SVC(
                       #     C=1e-2,
                       #     kernel='rbf',
                       #     gamma="scale",
                       #     decision_function_shape="ovr",
                       #     class_weight="balanced",
                       #     random_state=SEED),
                       'Neural Network': MLPClassifier(
                           hidden_layer_sizes=(200,),
                           activation="tanh",
                           solver="sgd",
                           learning_rate="invscaling",
                           learning_rate_init=0.01,
                           max_iter=1000,
                           tol=1e-5,
                           early_stopping=True,
                           validation_fraction=0.1,
                           random_state=SEED),
                       'Random Forest': RandomForestClassifier(
                           n_estimators=250,
                           criterion="entropy",
                           min_samples_leaf=5,
                           class_weight="balanced",
                           random_state=SEED,
                           n_jobs=-1)}

param_grids = {
    'Naive Bayes': {'var_smoothing': np.logspace(-18, 0, 19)},
    'KNN': {'n_neighbors': list(range(1, 5)) + list(range(5, 55, 5)),
            'algorithm': ['auto', ],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', cosine],
            'n_jobs': [-1, ]},
    'Logistic Regression': {'C': np.logspace(-5, 5, 11),
                            'solver': ['lbfgs', 'sag'],
                            'max_iter': [1000, ],
                            'tol': [1e-5, ],
                            'class_weight': ['balanced', ],
                            'n_jobs': [-1, ]},
    # TODO: Replace with LinearSVC
    # 'SVM': [{'C': np.logspace(-5, 5, 11),
    #          'kernel': ['poly', ],
    #          'degree': range(1, 15),
    #          'class_weight': ['balanced', ]},
    #         {'C': np.logspace(-5, 5, 11),
    #          'kernel': ['rbf', ],
    #          'gamma': ['scale', ],
    #          'class_weight': ['balanced', ]
    #          }],
    'Neural Network': [{'hidden_layer_sizes': [(value,) for value in (1, 2, 5, 10, 20, 50, 100, 200, 500)],
                        'activation': ['tanh', ],
                        'solver': ['adam'],
                        'alpha': np.logspace(-5, 5, 11),
                        'learning_rate': ['constant', ],
                        'learning_rate_init': [1e-2, ],
                        'max_iter': [1000, ],
                        'tol': [1e-5, ]},
                       {'hidden_layer_sizes': [(value,) for value in (1, 2, 5, 10, 20, 50, 100, 200, 500)],
                        'activation': ['tanh', ],
                        'solver': ['sgd', ],
                        'alpha': np.logspace(-5, 5, 11),
                        'learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'learning_rate_init': [1e-2, ],
                        'max_iter': [1000, ],
                        'tol': [1e-5, ]}],
    'Random Forest': {'n_estimators': [10, 25, 50, 100, 250, 500, 1000],
                      'criterion': ['entropy', 'gini'],
                      'max_depth': [None, 10, 25, 50, 100],
                      'min_samples_split': (2, 3, 5, 10),
                      'min_samples_leaf': (2, 3, 5, 10),
                      'n_jobs': [-1, ]}}


def get_best_params_for_classifiers(train_set, train_classes, use_default_params=False):
    classifier_list = list(classifiers_default.keys())

    classifiers_best_params = dict.fromkeys(classifier_list)
    classifiers_best_models = dict.fromkeys(classifier_list)
    classifiers_cross_validation_scores = dict.fromkeys(classifier_list)

    validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scorer = make_scorer(f1_score, greater_is_better=True)

    if use_default_params:

        for classifier_name, classifier_model in classifiers_default.items():
            print("Cross-validating {} classifier...".format(classifier_name))
            n_jobs_validation = -1 if classifier_name in ["Naive Bayes", "SVM", "Neural Network"] else 1
            classifiers_best_params[classifier_name] = classifier_model.get_params()
            classifier_scores_folds = cross_val_score(estimator=classifier_model,
                                                      X=train_set,
                                                      y=train_classes,
                                                      scoring=scorer,
                                                      cv=validator,
                                                      n_jobs=n_jobs_validation,
                                                      verbose=1)
            classifier_model.fit(train_set, train_classes)
            classifiers_best_models[classifier_name] = classifier_model
            classifiers_cross_validation_scores[classifier_name] = np.mean(classifier_scores_folds)
            print("Done with {}!".format(classifier_name))

    else:

        for classifier_name, classifier_model in classifiers_default.items():
            print("Cross-validating {} classifier...".format(classifier_name))
            n_jobs_validation = -1 if classifier_name in ["Naive Bayes", "SVM", "Neural Network"] else 1
            classifier_grid_search = GridSearchCV(estimator=classifier_model,
                                                  param_grid=param_grids[classifier_name],
                                                  scoring=scorer,
                                                  cv=validator,
                                                  refit=True,
                                                  n_jobs=n_jobs_validation,
                                                  verbose=1)
            classifier_grid_search.fit(train_set, train_classes)
            classifiers_best_params[classifier_name] = classifier_grid_search.best_params_
            classifiers_best_models[classifier_name] = classifier_grid_search.best_estimator_
            classifiers_cross_validation_scores[classifier_name] = classifier_grid_search.best_score_
            print("Done with {}!".format(classifier_name))

    return classifiers_best_params, classifiers_best_models, classifiers_cross_validation_scores