from collections import OrderedDict

from classic_ml.preprocessing import preprocess_tweet

from classic_ml.feature_extraction import get_language_style_features
from classic_ml.feature_extraction import reduce_vocabulary, get_tf_idf_features
from classic_ml.feature_extraction import get_vader_features
from classic_ml.feature_extraction import get_empath_features

from classic_ml.resources import SEED, empath_analyzer


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

    print("Total number of testing tweets:", num_test_tweets)
    print("Number of positive training tweets:", num_train_tweets_pos)
    print("Number of negative training tweets:", num_train_tweets_neg)
    print("Total number of training tweets:", num_train_tweets)

    tweet_vocabulary = []
    tweet_documents = []
    tweets_language_style_features = []
    for i, tweet in enumerate(train_tweets[:5000] + train_tweets[-5000:]):
        if i % 10000 == 0:
            print("Pre-processing training tweet #{}".format(i))
        tweet_tokens, tweet_words = preprocess_tweet(tweet)
        if len(tweet_tokens) == 0:
            continue
        tweets_language_style_features.append(get_language_style_features(tweet_tokens, tweet_words))
        tweet_vocabulary.extend(tweet_tokens)
        tweet_documents.append(tweet_tokens)
    print("Done!")

    tweet_vocabulary = reduce_vocabulary(tweet_vocabulary)
    tweets_tf_idf_features = get_tf_idf_features(tweet_documents, tweet_vocabulary)
