import numpy as np
from nltk import FreqDist
from nltk import TextCollection

from classic_ml.resources import dictionary_english, vader_sia, empath_analyzer


def get_language_style_features(tweet_tokens, word_tokens):
    num_tokens = len(tweet_tokens)
    num_words = len(word_tokens)
    word_lengths = [len(word) for word in word_tokens]
    dictionary_words = [word for word in word_tokens if word in dictionary_english]
    num_dictionary_words = len(dictionary_words)
    unique_words = set(word_tokens)
    num_unique_words = len(unique_words)

    features = [num_words / num_tokens,
                np.mean(word_lengths) if num_words > 0 else 0,
                num_dictionary_words / num_words if num_words > 0 else 0,
                num_unique_words / num_words if num_words > 0 else 0]
    return features


def reduce_vocabulary(vocabulary, num_most_common=1000):
    tokens_freq = FreqDist(vocabulary)
    vocabulary_cleaned = tokens_freq.most_common(num_most_common)
    return vocabulary_cleaned


def get_tf_idf_features(documents, vocabulary):
    collection = TextCollection(documents)
    features = []
    for document in documents:
        features.append([collection.tf_idf(word, document) if word in document else 0
                         for word, _ in vocabulary])
    return features


def get_vader_features(tweet):
    vader_scores = vader_sia.polarity_scores(tweet)
    features = [
        vader_scores["pos"],
        vader_scores["neu"],
        vader_scores["neg"]
    ]
    return features


def get_empath_features(tweet):
    empath_analysis = empath_analyzer.analyze(tweet, categories=["positive"], normalize=True)
    return list(empath_analysis.values())
