""" Module containing implementations of the feature normalization and selection procedures """

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from classic_ml.resources import SEED

# Define a random forest classifier, to be used as a feature selection model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1)


def get_feature_normalizer(dataset):
    """
    Function to fit a feature scaling model to be reused for both the training and testing features
    Min-max normalization is used to scale the features to the interval [0, 1]

    :param dataset: feature data, pandas.DataFrame or numpy.ndarray object

    :returns: feature scaling model, sklearn.preprocessing.MinMaxScaler object
    """

    scaler = MinMaxScaler()
    scaler.fit(dataset)
    return scaler


def get_feature_selector_correlation_score(dataset, classes, feature_labels, num_features=50):
    """
    Function to fit a feature selection model to be reused for both the training and testing features
    Feature importances are computed as Pearson's chi-squared correlations between the feature and class values

    :param dataset: feature data, pandas.DataFrame or numpy.ndarray object
    :param classes: class labels, pandas.Series or numpy.array object
    :param feature_labels: list of feature names
    :param num_features: number of best descriptors to retain, integer, default value is 50

    :returns: feature selection model, sklearn.feature_selection.SelectKBest object,
              feature scores, dictionary mapping the names of the best features to their importance values
    """

    selector = SelectKBest(chi2, num_features)
    selector.fit(dataset, classes)
    features_scores = {feature_labels[index]: selector.scores_[index]
                       for index in selector.get_support(indices=True)}
    return selector, features_scores


def get_feature_selector_information_gain(dataset, classes, feature_labels, num_features=50):
    """
    Function to fit a feature selection model to be reused for both the training and testing features
    Feature importances are computed as information gain ratios after training a random forest classifier on the data

    :param dataset: feature data, pandas.DataFrame or numpy.ndarray object
    :param classes: class labels, pandas.Series or numpy.array object
    :param feature_labels: list of feature names
    :param num_features: number of best descriptors to retain, integer, default value is 50

    :returns: feature selection model, sklearn.feature_selection.SelectFromModel object,
              feature scores, dictionary mapping the names of the best features to their importance values
    """

    selector = SelectFromModel(random_forest_model, max_features=num_features)
    selector.fit(dataset, classes)
    features_scores = {feature_labels[index]: selector.estimator_.feature_importances_[index]
                       for index in selector.get_support(indices=True)}
    return selector, features_scores
