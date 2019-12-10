"""
Module containing implementations of the procedures required
for extracting the different types of features from the processed tweet texts
"""

import numpy as np
from nltk import FreqDist
from nltk import TextCollection
from nltk import pos_tag

from classic_ml.resources import dictionary_english, vader_sia, empath_analyzer


def get_language_style_features(tweet_tokens, word_tokens):
    """
    Function for calculating simple text statistics using the tweet tokens,
    as a method to quantify language and writing style of the tweet author

    :param tweet_tokens: list of all tokens in the tweet
    :param word_tokens: list of only the English word tokens in the tweet

    :returns: list of 4 computed feature values
    """

    # Calculate what fraction of the tweet tokens are actual words (and not punctuation or special tags)
    num_tokens = len(tweet_tokens)
    num_words = len(word_tokens)
    word_percentage = num_words / num_tokens if num_tokens > 0 else 0

    # Calculate mean word length
    word_lengths = [len(word) for word in word_tokens]
    average_word_length = np.mean(word_lengths) if num_words > 0 else 0

    # Calculate what fraction of the word tokens can be found in the English dictionary
    dictionary_words = [word for word in word_tokens if word in dictionary_english]
    num_dictionary_words = len(dictionary_words)
    dictionary_percentage = num_dictionary_words / num_words if num_words > 0 else 0

    # Calculate what fraction of the word tokens are unique
    unique_words = set(word_tokens)
    num_unique_words = len(unique_words)
    uniqueness_percentage = num_unique_words / num_words if num_words > 0 else 0

    features = [word_percentage,
                average_word_length,
                dictionary_percentage,
                uniqueness_percentage]
    return features


def reduce_vocabulary(vocabulary, num_most_common=1000):
    """
    Helper function to reduce the complete tweet token list to a small vocabulary of only the most frequent tokens

    :param vocabulary: list of all tweet tokens present in the training text, duplicate tokens included
    :param num_most_common: number of most common tokens to preserve, integer, default value is 1000

    :returns: reduced tweet vocabulary, list with :num_most_common unique tokens
    """

    # Calculate the frequencies of occurrence of each tweet token present in the data
    tokens_freq = FreqDist(vocabulary)

    # Keep only the most frequent tokens, discard their frequencies
    vocabulary_cleaned = [token for token, freq in tokens_freq.most_common(num_most_common)]
    return vocabulary_cleaned


def get_tf_idf_features(documents, vocabulary):
    """
    Function for calculating the TF-IDF feature matrix,
    rows represent tweets and columns represent unique tokens in the vocabulary

    :param documents: list of lists of tweet tokens
    :param vocabulary: list of unique tweet tokens

    :returns: computed TF-IDF feature matrix, represented as list of lists of TF-IDF values
    """

    collection = TextCollection(documents)
    features = []
    for document in documents:
        features.append([collection.tf_idf(token, document) if token in document else 0 for token in vocabulary])
    return features


def get_pos_features(word_tokens):
    """
    Function to calculate simple morphological statistics using the POS tags of the word tokens in a tweet

    :param word_tokens: list of only the English word tokens in the tweet

    :returns: list of 4 computed feature values
    """

    num_words = len(word_tokens)
    if num_words == 0:
        return [0, ] * 4

    # Retrieve the Penn Treebank Part-of-Speech tag for each word token
    words_pos_tags = pos_tag(word_tokens)

    # Count the fraction of nouns, verbs, adjectives and adverbs present in the word tokens
    features = {"N": 0, "V": 0, "J": 0, "R": 0}
    for _, tag in words_pos_tags:
        # The first character in the POS tag determines the basic morphological type of the word
        tag_type = tag[0]
        if tag_type in features.keys():
            features[tag_type] += 1 / num_words
    return list(features.values())


def get_vader_features(tweet):
    """
    Function to calculate sentiment features from the raw tweet text using the pre-trained Vader model
    The Vader sentiment intensity analyzer returns 3 tweet polarity scores: positive, neutral and negative

    :param tweet: raw tweet text, string

    :returns: list of 3 computed feature values
    """

    vader_scores = vader_sia.polarity_scores(tweet)
    features = [
        vader_scores["pos"],
        vader_scores["neu"],
        vader_scores["neg"]
    ]
    return features


def get_empath_features(tweet):
    """
    Function to calculate sentiment features from the raw tweet text using the pre-trained Empath model
    The Empath analyzer scores words on 200 pre-defined categories, analyzing the presence of different text topics

    :param tweet: raw tweet text, string

    :returns: list of 200 computed feature values
    """

    empath_analysis = empath_analyzer.analyze(tweet, normalize=True)
    return list(empath_analysis.values())
