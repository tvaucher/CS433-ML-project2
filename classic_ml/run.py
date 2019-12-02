from os.path import isfile

import csv

import numpy as np
import pandas as pd

from classic_ml.training import read_tweet_file

from classic_ml.serialization import load_object

from classic_ml.preprocessing import preprocess_tweet

from classic_ml.feature_extraction import get_language_style_features
from classic_ml.feature_extraction import get_tf_idf_features
from classic_ml.feature_extraction import get_pos_features
from classic_ml.feature_extraction import get_vader_features
from classic_ml.feature_extraction import get_empath_features

from classic_ml.resources import SEED


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
        csvfile.close()


if __name__ == "__main__":
    np.random.seed(SEED)

    test_tweets, test_ids = read_tweet_file("../data/test_data.txt")
    num_test_tweets = len(test_tweets)

    print("Total number of testing tweets:", num_test_tweets)

    tweet_vocabulary_path = "./files/tweet_vocabulary.gz"
    feature_labels_filepath = "./files/feature_labels.gz"
    feature_normalizer_filepath = "./files/feature_normalizer.gz"
    feature_selector_filepath = "./files/feature_selector.gz"
    best_model_filepath = "./files/best_model.gz"

    test_dataset_filepath = "./files/test_dataset.tsv"
    test_dataset_reduced_filepath = "./files/test_dataset_reduced.tsv"
    submission_filepath = "./files/submission.csv"

    if not isfile(test_dataset_filepath):
        tweets_language_style_features = []
        tweet_documents = []
        tweets_pos_features = []
        tweets_vader_sentiment_features = []
        tweets_empath_sentiment_features = []

        for i, tweet in enumerate(test_tweets):
            if i % 500 == 0:
                print("Processing testing tweet #{}/{}...".format(i, num_test_tweets))

            tweet_tokens, tweet_words = preprocess_tweet(tweet)

            tweets_language_style_features.append(get_language_style_features(tweet_tokens, tweet_words))
            tweet_documents.append(tweet_tokens)
            tweets_pos_features.append(get_pos_features(tweet_words))
            tweets_vader_sentiment_features.append(get_vader_features(tweet))
            tweets_empath_sentiment_features.append(get_empath_features(tweet))

        print("Done!")

        print("Calculating TF-IDF features...")
        tweet_vocabulary = load_object(tweet_vocabulary_path)
        tweets_tf_idf_features = get_tf_idf_features(tweet_documents, tweet_vocabulary)
        print("Done!")

        print("Saving testing features locally...")
        feature_labels = load_object(feature_labels_filepath)

        test_dataset = np.hstack((tweets_language_style_features,
                                  tweets_tf_idf_features,
                                  tweets_pos_features,
                                  tweets_vader_sentiment_features,
                                  tweets_empath_sentiment_features))
        test_dataset = pd.DataFrame(test_dataset, columns=feature_labels)
        test_dataset.to_csv(test_dataset_filepath, sep='\t', header=True, index=False, encoding='utf-8')
        print("Done!")
    else:
        if not isfile(test_dataset_reduced_filepath):
            print("Reading testing features...")
            test_dataset = pd.read_csv(test_dataset_filepath, sep='\t', header=0, encoding='utf-8')
            feature_labels = load_object(feature_labels_filepath)
            print("Done!")

            print("Normalizing testing data using trained normalizer...")
            feature_normalizer = load_object(feature_normalizer_filepath)
            test_dataset = feature_normalizer.transform(test_dataset)
            print("Done!")

            print("Reducing testing data using trained feature selector...")
            feature_selector = load_object(feature_selector_filepath)
            feature_labels_reduced = [feature_labels[index] for index in feature_selector.get_support(indices=True)]
            test_dataset_reduced = pd.DataFrame(feature_selector.transform(test_dataset),
                                                columns=feature_labels_reduced)
            del test_dataset
            test_dataset_reduced.to_csv(test_dataset_reduced_filepath, sep='\t',
                                        header=True, index=False, encoding='utf-8')
            print("Done!")
        else:
            print("Reading reduced testing features...")
            test_dataset_reduced = pd.read_csv(test_dataset_reduced_filepath, sep='\t', header=0, encoding='utf-8')
            print("Done!")

            print("Loading trained classifiers...")
            best_model = load_object(best_model_filepath)
            print("Done!")

            print("Calculating predictions using best classifier...")
            test_predictions = best_model.predict(test_dataset_reduced)
            print("Done!")

            print("Generating submission file...")
            create_csv_submission(test_ids, test_predictions, submission_filepath)
            print("Done!")
