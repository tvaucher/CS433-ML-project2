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


def get_dataset_normalizer(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    return scaler


def get_feature_selector_correlation_score(dataset, classes, feature_labels, num_features=30):
    selector = SelectKBest(chi2, num_features)
    selector.fit(dataset, classes)
    features_scores = {feature_labels[index]: selector.scores_[index]
                       for index in selector.get_support(indices=True)}
    return selector, features_scores


def get_feature_selector_information_gain(dataset, classes, feature_labels, num_features=30):
    selector = SelectFromModel(random_forest_model, max_features=num_features)
    selector.fit(dataset, classes)
    features_scores = {feature_labels[index]: selector.estimator_.feature_importances_[index]
                       for index in selector.get_support(indices=True)}
    return selector, features_scores
