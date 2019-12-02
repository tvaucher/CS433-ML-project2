from os.path import isfile

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

from classic_ml.feature_selection import get_dataset_normalizer, get_feature_selector_information_gain

from classic_ml.cross_validation import get_best_params_for_classifiers


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
    np.random.seed(SEED)

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

    print("Number of positive training tweets:", num_train_tweets_pos)
    print("Number of negative training tweets:", num_train_tweets_neg)
    print("Total number of training tweets:", num_train_tweets)

    tweet_vocabulary_path = "./files/tweet_vocabulary.gz"
    train_dataset_filepath = "./files/train_dataset.tsv"
    feature_labels_filepath = "./files/feature_labels.gz"
    train_dataset_reduced_filepath = "./files/train_dataset_reduced.tsv"
    feature_normalizer_filepath = "./files/feature_normalizer.gz"
    feature_selector_filepath = "./files/feature_selector.gz"
    feature_scores_filepath = "./files/feature_scores.tsv"
    best_model_filepath = "./files/best_model.gz"
    cross_val_results_filepath = "./files/cross_val_results.tsv"

    if not isfile(train_dataset_filepath):
        tweets_language_style_features = []
        tweet_vocabulary = []
        tweet_documents = []
        tweets_pos_features = []
        tweets_vader_sentiment_features = []
        tweets_empath_sentiment_features = []

        for i, tweet in enumerate(train_tweets):
            if i % 5000 == 0:
                print("Processing training tweet #{}/{}...".format(i, num_train_tweets))

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
        save_object(tweet_vocabulary, tweet_vocabulary_path)
        tweets_tf_idf_features = get_tf_idf_features(tweet_documents, tweet_vocabulary)
        print("Done!")

        print("Saving training features locally...")
        feature_labels = []
        feature_labels.extend(["WORD PERCENTAGE", "MEAN WORD LENGTH", "DICTIONARY PERCENTAGE", "UNIQUENESS PERCENTAGE"])
        feature_labels.extend([token + " TF-IDF" for token in tweet_vocabulary])
        feature_labels.extend(["POS NOUNS", "POS VERBS", "POS ADJECTIVES", "POS ADVERBS"])
        feature_labels.extend(["VADER POSITIVE", "VADER NEUTRAL", "VADER NEGATIVE"])
        feature_labels.extend(["EMPATH " + empath_category for empath_category in empath_analyzer.cats.keys()])
        save_object(feature_labels, feature_labels_filepath)

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
            feature_labels = load_object(feature_labels_filepath)
            print("Done!")

            print("Normalizing training data...")
            feature_normalizer = get_dataset_normalizer(train_dataset)
            save_object(feature_normalizer, feature_normalizer_filepath)
            train_dataset = feature_normalizer.transform(train_dataset)
            print("Done!")

            print("Performing feature selection on training features...")
            feature_selector, feature_scores = \
                get_feature_selector_information_gain(train_dataset, train_classes, feature_labels, num_features=40)
            save_object(feature_selector, feature_selector_filepath)
            feature_scores_sorted = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            feature_scores_df = pd.DataFrame(feature_scores_sorted, columns=["Feature", "Importance Score"], index=None)
            feature_scores_df.to_csv(feature_scores_filepath, sep='\t', header=True, index=False, encoding='utf-8')

            train_dataset_reduced = pd.DataFrame(feature_selector.transform(train_dataset),
                                                 columns=feature_scores.keys())
            del train_dataset
            train_dataset_reduced.to_csv(train_dataset_reduced_filepath, sep='\t',
                                         header=True, index=False, encoding='utf-8')
            print("Done!")
        else:
            print("Reading reduced training features...")
            train_dataset_reduced = pd.read_csv(train_dataset_reduced_filepath, sep='\t', header=0, encoding='utf-8')
            print("Done!")

            if not isfile(best_model_filepath):
                print("Cross-validating classifier parameters and training classifiers...")
                best_params, best_models, cross_val_scores = \
                    get_best_params_for_classifiers(train_dataset_reduced, train_classes)
                save_object(best_models["Neural Network"], best_model_filepath)

                cross_val_results = [[classifier, best_params[classifier], cross_val_score]
                                     for classifier, cross_val_score in cross_val_scores.items()]
                cross_val_results_df = pd.DataFrame(cross_val_results,
                                                    columns=["Classifier", "Parameters", "Cross-validation scores"])
                cross_val_results_df.to_csv(cross_val_results_filepath, sep='\t',
                                            header=True, index=False, encoding='utf-8')
                print("Done!")
