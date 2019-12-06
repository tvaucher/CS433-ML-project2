import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from classic_ml.serialization import load_object

sns.set(style="darkgrid", context="paper")


def plot_cumulative_distribution_for_feature_importances(importances, filepath):
    importances_cdf = importances.sort_values(ascending=False).cumsum().reset_index().drop("index", axis=1)
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


def plot_class_distribution_for_features(data, filepath):
    feature_means_by_class = data.groupby("Class").mean().transpose()
    feature_sds_by_class = data.groupby("Class").std().transpose()

    feature_class_differences = feature_means_by_class[1] - feature_means_by_class[-1]
    feature_class_differences = feature_class_differences.sort_values(ascending=False).reset_index()
    feature_class_differences.columns = ["Feature", "Class Difference"]

    feature_sds_average = feature_sds_by_class.mean(axis=1)

    plt.figure(figsize=(18, 10))
    plt.title("Class distribution of best features", fontsize=24)
    sns.barplot(y="Feature",
                x="Class Difference",
                data=feature_class_differences,
                palette=sns.color_palette("RdYlBu", len(feature_class_differences)))
    plt.xlabel("Average difference of feature values across classes", fontsize=18)
    plt.ylabel("Feature", fontsize=18)
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(fname=filepath, dpi="figure", format="png", papertype="a4")
    plt.show()


if __name__ == "__main__":
    feature_selector_filepath = "./files/feature_selector.gz"
    train_dataset_reduced_filepath = "./files/train_dataset_reduced.tsv"
    train_classes_filepath = "./files/train_classes.gz"

    train_dataset_reduced = pd.read_csv(train_dataset_reduced_filepath, sep='\t', header=0, encoding='utf-8')
    train_classes = pd.Series(load_object(train_classes_filepath), name="Class")
    feature_selector = load_object(feature_selector_filepath)

    feature_importances = pd.Series(feature_selector.estimator_.feature_importances_, name="Feature Importance")
    plot_cumulative_distribution_for_feature_importances(feature_importances, "./results/importances_cumul_dist.png")

    train_data = pd.concat([train_dataset_reduced, train_classes], axis=1)
    plot_class_distribution_for_features(train_data, "./results/features_class_differences.png")
