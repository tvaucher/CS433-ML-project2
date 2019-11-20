from os.path import isfile

import re
import numpy as np
import pandas as pd

from collections import OrderedDict

from classic_ml.serialization import save_object, load_object

from classic_ml.preprocessing import preprocess_tweet

from classic_ml.feature_extraction import get_language_style_features
from classic_ml.feature_extraction import reduce_vocabulary, get_tf_idf_features
from classic_ml.feature_extraction import get_pos_features
from classic_ml.feature_extraction import get_vader_features
from classic_ml.feature_extraction import get_empath_features

from classic_ml.resources import SEED, empath_analyzer

from classic_ml.feature_selection import select_features_information_gain


def read_tweet_file(path, id_present=True):
    tweets = []
    tweet_ids = []
    with open(path, 'r', encoding="utf-8") as file:
        for line in file:
            if id_present:
                line_splits = line.split(",")
                tweet_id = int(line_splits[0])
                tweet = ','.join(line_splits[1:])
                tweet_ids.append(tweet_id)
                tweets.append(tweet)
            else:
                tweets.append(line[:-1])
    return tweets, tweet_ids


def remove_duplicate_tweets(tweet_list):
    tweet_set = OrderedDict.fromkeys(tweet_list)
    return list(tweet_set)


if __name__ == "__main__":
    test_tweets, test_ids = read_tweet_file("../data/test_data.txt")
    num_test_tweets = len(test_tweets)

    train_tweets_pos, _ = read_tweet_file("../data/train_pos.txt", id_present=False)
    train_tweets_neg, _ = read_tweet_file("../data/train_neg.txt", id_present=False)
    train_tweets_pos = remove_duplicate_tweets(train_tweets_pos)
    train_tweets_neg = remove_duplicate_tweets(train_tweets_neg)
    num_train_tweets_pos = len(train_tweets_pos)
    num_train_tweets_neg = len(train_tweets_neg)

    train_tweets = train_tweets_pos + train_tweets_neg
    num_train_tweets = num_train_tweets_pos + num_train_tweets_neg
    train_classes = [1, ] * num_train_tweets_pos + [-1, ] * num_train_tweets_neg
    del train_tweets_pos, train_tweets_neg

    print("Total number of testing tweets:", num_test_tweets)
    print("Number of positive training tweets:", num_train_tweets_pos)
    print("Number of negative training tweets:", num_train_tweets_neg)
    print("Total number of training tweets:", num_train_tweets)

    train_dataset_filepath = "./files/train_dataset.tsv"
    train_dataset_reduced_filepath = "./files/train_dataset_reduced.tsv"
    feature_scores_filepath = "./files/feature_scores"

    if not isfile(train_dataset_filepath):
        tweets_language_style_features = []
        tweet_vocabulary = []
        tweet_documents = []
        tweets_pos_features = []
        tweets_vader_sentiment_features = []
        tweets_empath_sentiment_features = []

        for i, tweet in enumerate(train_tweets):
            if i % 5000 == 0:
                print("Processing training tweet #{}...".format(i))

            tweet_tokens, tweet_words = preprocess_tweet(tweet)

            tweets_language_style_features.append(get_language_style_features(tweet_tokens, tweet_words))
            tweet_vocabulary.extend(tweet_tokens)
            tweet_documents.append(tweet_tokens)
            tweets_pos_features.append(get_pos_features(tweet_words))
            tweets_vader_sentiment_features.append(get_vader_features(tweet))
            tweets_empath_sentiment_features.append(get_empath_features(tweet))

        print("Done!")

        print("Calculating TF-IDF features...")
        tweet_vocabulary = reduce_vocabulary(tweet_vocabulary)
        tweets_tf_idf_features = get_tf_idf_features(tweet_documents, tweet_vocabulary)
        print("Done!")

        print("Saving training features locally...")
        feature_labels = []
        feature_labels.extend(["WORD PERCENTAGE", "MEAN WORD LENGTH", "DICTIONARY PERCENTAGE", "UNIQUENESS PERCENTAGE"])
        feature_labels.extend([re.escape(token) + " TF-IDF" for token in tweet_vocabulary])
        feature_labels.extend(["POS NOUNS", "POS VERBS", "POS ADJECTIVES", "POS ADVERBS"])
        feature_labels.extend(["VADER POSITIVE", "VADER NEUTRAL", "VADER NEGATIVE"])
        feature_labels.extend(["EMPATH " + empath_category for empath_category in empath_analyzer.cats.keys()])

        train_dataset = np.hstack((tweets_language_style_features,
                                   tweets_tf_idf_features,
                                   tweets_pos_features,
                                   tweets_vader_sentiment_features,
                                   tweets_empath_sentiment_features))
        train_dataset = pd.DataFrame(train_dataset, columns=feature_labels)
        train_dataset.to_csv(train_dataset_filepath, sep='\t', header=True, index=False, encoding='utf-8')
        print("Done!")
    else:
        if not isfile(train_dataset_reduced_filepath):
            print("Reading training features...")
            train_dataset = pd.read_csv(train_dataset_filepath, sep='\t', header=0, encoding='utf-8')
            feature_labels = train_dataset.columns.values.tolist()
            print("Done!")

            print("Performing feature selection on training features...")
            train_dataset_reduced, feature_scores = \
                select_features_information_gain(train_dataset, train_classes, feature_labels, num_features=100)

            del train_dataset
            train_dataset_reduced.to_csv(train_dataset_reduced_filepath, sep='\t',
                                         header=True, index=False, encoding='utf-8')
            save_object(feature_scores, feature_scores_filepath)
            print("Done!")
        else:
            print("Reading reduced training features...")
            train_dataset_reduced = pd.read_csv(train_dataset_reduced_filepath, sep='\t', header=0, encoding='utf-8')
            feature_scores = load_object(feature_scores_filepath)
            best_features = train_dataset_reduced.columns.tolist()
            print(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
            print("Done!")
