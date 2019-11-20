import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from classic_ml.resources import SEED

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1)


def normalize_dataset(dataset):
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset)


def select_features_correlation_score(dataset, classes, feature_labels, num_features=30):
    dataset = normalize_dataset(dataset)
    selector = SelectKBest(chi2, num_features)
    dataset_reduced = selector.fit_transform(dataset, classes)
    del dataset
    best_features_scores = {feature_labels[index]: selector.scores_[index]
                            for index in selector.get_support(indices=True)}
    return pd.DataFrame(dataset_reduced, columns=list(best_features_scores.keys())), best_features_scores


def select_features_information_gain(dataset, classes, feature_labels, num_features=30):
    dataset = normalize_dataset(dataset)
    selector = SelectFromModel(random_forest_model, max_features=num_features)
    dataset_reduced = selector.fit_transform(dataset, classes)
    del dataset
    best_features_scores = {feature_labels[index]: selector.estimator_.feature_importances_[index]
                            for index in selector.get_support(indices=True)}
    return pd.DataFrame(dataset_reduced, columns=list(best_features_scores.keys())), best_features_scores
