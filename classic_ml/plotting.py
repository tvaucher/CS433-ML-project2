""" Script containing code to generate the plots """

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from classic_ml.serialization import load_object

# Set the seaborn plotting style
sns.set(style="darkgrid", context="paper")


def plot_cumulative_distribution_for_feature_importances(importances, filepath):
    """
    Function to generate a cumulative distribution plot for the feature importances

    :param importances: feature importance values, pandas.Series object
    :param filepath: local path where to save the plot figure as a image file, string
    """

    # Compute the CDF of the feature importances using cumulative sums of the sorted values
    importances_cdf = importances.sort_values(ascending=False).cumsum().reset_index().drop("index", axis=1)

    # Use the CDF to find what percentage of the total feature importance do the first 40 features have
    importance_40 = importances_cdf.iloc[39][0]

    plt.figure(figsize=(9, 5))
    plt.title("Cumulative distribution of feature importance", fontsize=24)
    sns.lineplot(data=importances_cdf, linewidth=3)
    plt.hlines(y=importance_40, xmin=0, xmax=40, color="red", linewidth=3, linestyle="--")
    plt.vlines(x=40, ymin=0, ymax=importance_40, color="red", linewidth=3, linestyle="--")
    plt.xlabel("Number of features", fontsize=18)
    plt.ylabel("Total feature importance", fontsize=18)
    plt.xticks(list(plt.xticks()[0]) + [40, ], fontsize=14)
    plt.yticks(list(plt.yticks()[0]) + [round(importance_40, 2), ], fontsize=14)
    plt.xlim(-10, len(importances) + 10)
    plt.ylim(0, 1.05)
    plt.savefig(fname=filepath, dpi="figure", format="png")
    plt.show()


def plot_class_distribution_for_features(data, classes, filepath):
    """
    Function to visualize dependence of values of the best features on class

    :param data: tweet feature vector values, pandas.DataFrame object
    :param classes: tweet class values, pandas.Series object
    :param filepath: local path where to save the plot figure as a image file, string
    """

    # Join the tweet feature vectors and classes together
    data = pd.concat([data, classes], axis=1)

    # Calculate the mean value for each feature in samples of each class
    feature_means_by_class = data.groupby("Class").mean().transpose()

    # Calculate the difference between the class means for each feature
    # A positive class difference indicates higher feature values for the positive class and vice versa
    feature_class_differences = feature_means_by_class[1] - feature_means_by_class[-1]
    feature_class_differences = feature_class_differences.sort_values(ascending=False).reset_index()
    feature_class_differences = pd.concat((feature_class_differences.iloc[:15], feature_class_differences[-10:]))
    feature_class_differences.columns = ["Feature", "Class Difference"]

    plt.figure(figsize=(7 * (9 / 5), 7))
    plt.title("Class distribution of best features", fontsize=24)
    sns.barplot(y="Feature",
                x="Class Difference",
                data=feature_class_differences,
                palette=sns.color_palette("RdYlBu", len(feature_class_differences)))
    plt.xlabel("Average difference of feature values across classes", fontsize=18)
    plt.ylabel("Feature", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.06, 0.06)
    plt.tight_layout()
    plt.savefig(fname=filepath, dpi="figure", format="png")
    plt.show()


if __name__ == "__main__":
    # Filepaths of required resources
    feature_selector_filepath = "./files/feature_selector.gz"
    train_dataset_reduced_filepath = "./files/train_dataset_reduced.tsv"
    train_classes_filepath = "./files/train_classes.gz"

    # Load the reduced and normalized training set
    train_dataset_reduced = pd.read_csv(train_dataset_reduced_filepath, sep="\t", header=0, encoding="utf-8")
    # Load the training class labels and convert the vector to a pandas.Series object
    train_classes = pd.Series(load_object(train_classes_filepath), name="Class")

    # Load the feature selector model
    feature_selector = load_object(feature_selector_filepath)
    # Get the complete vector of feature importances from the feature selector and save it as a pandas.Series object
    feature_importances = pd.Series(feature_selector.estimator_.feature_importances_, name="Feature Importance")

    # Generate the first plot and save it locally
    plot_cumulative_distribution_for_feature_importances(feature_importances,
                                                         "./results/importances_cumul_dist.png")
    # Generate the second plot and save it locally
    plot_class_distribution_for_features(train_dataset_reduced,
                                         train_classes,
                                         "./results/features_class_differences.png")
