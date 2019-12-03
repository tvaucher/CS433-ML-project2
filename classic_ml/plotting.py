import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from classic_ml.serialization import load_object

sns.set(style="darkgrid", context="paper")


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
    train_dataset_reduced_filepath = "./files/train_dataset_reduced.tsv"
    train_classes_filepath = "./files/train_classes.gz"

    train_dataset_reduced = pd.read_csv(train_dataset_reduced_filepath, sep='\t', header=0, encoding='utf-8')
    train_classes = pd.Series(load_object(train_classes_filepath), name="Class")

    train_data = pd.concat([train_dataset_reduced, train_classes], axis=1)

    plot_class_distribution_for_features(train_data, "./results/features_class_differences.png")
