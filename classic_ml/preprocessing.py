""" Module containing helper functions for performing preprocessing on the tweet text """

import re
from nltk import pos_tag

from classic_ml.resources import special_tags, positive_emojis, negative_emojis, \
    stopwords_english, dictionary_english, lemmatizer, wordnet


def parse_cropped_ending_of_tweet(tweet):
    """
    Function to parse the cropped endings of tweets using a regular expression
    This is identified by the character sequence "... <url>" at the end of the tweet text
    It is replaced with the special tag <cropped>

    :param tweet: tweet text, string

    :returns: parsed tweet text, string
    """

    return re.sub(r"\S+ \.\.\. <url>$", "<cropped>", tweet)


def parse_hashtags_in_tweet(tweet):
    """
    Function to parse hashtags in tweets using a regular expression
    The hashtag text along with the character "#" is replaced with the special tag <hashtag>

    :param tweet: tweet text, string

    :returns: parsed tweet text, string
    """

    return re.sub(r"#\w+", "<hashtag>", tweet)


def parse_numbers_in_tweet(tweet):
    """
    Function to parse integer and decimal numbers in tweets using a regular expression
    The numbers are replaced with the special tag <number>

    :param tweet: tweet text, string

    :returns: parsed tweet text, string
    """

    return re.sub(r"\d+([.,]\d+)?", " <number> ", tweet)


def parse_time_in_tweet(tweet):
    """
    Function to parse timestamps in 12h and 24h format in tweets using a regular expression
    The timestamps are replaced with the special tag <time>

    :param tweet: tweet text, string

    :returns: parsed tweet text, string
    """

    return re.sub(r"\d?\d:\d\d", "<time>", tweet)


def parse_emojis_in_tweet(tweet):
    """
    Function to parse emojis in tweets using predefined static lists and regular expressions
    All instances of positive emojis are replaced with the special tag <emoji_pos>
    All instances of negative emojis are replaced with the special tag <emoji_neg>

    :param tweet: tweet text, string

    :returns: parsed tweet text, string
    """

    tweet_parsed = tweet
    for emoji in positive_emojis:
        tweet_parsed = re.sub(re.escape(emoji), "<emoji_pos>", tweet_parsed)
    for emoji in negative_emojis:
        tweet_parsed = re.sub(re.escape(emoji), "<emoji_neg>", tweet_parsed)
    return tweet_parsed


def remove_repetitions_in_tokens(tweet_tokens, word_tokens):
    """
    Function to parse repeated characters in tweet tokens using a regular expression

    :param tweet_tokens: list of all tokens in the tweet
    :param word_tokens: list of only the English word tokens in the tweet

    :returns: parsed list of tweet tokens
    """

    tweet_tokens_fixed = []
    for tweet_token in tweet_tokens:
        fixed_token = re.sub(r"(.)\1+", r"\1", tweet_token)
        # Do not fix the tweet token if it is a special tag
        # or if the fixing has damaged the spelling of the English word
        if tweet_token in special_tags or (tweet_token in word_tokens and fixed_token not in dictionary_english):
            tweet_tokens_fixed.append(tweet_token)
        else:
            tweet_tokens_fixed.append(fixed_token)
    return tweet_tokens_fixed


def remove_stopwords_in_tweet(tweet_tokens):
    """
    Function to parse stopwords in tweet tokens using a predefined list

    :param tweet_tokens: list of all tokens in the tweet

    :returns: parsed list of tweet tokens with stopword tokens removed
    """

    return [token for token in tweet_tokens if token not in stopwords_english]


def get_only_word_tokens(tweet_tokens):
    """
    Function to retrieve only tweet tokens with real English words

    :param tweet_tokens: list of all tokens in the tweet

    :returns: parsed list of tweet tokens with only word tokens
    """

    return [token for token in tweet_tokens if re.match(r"^[A-Za-z\']+$", token)]


def get_wordnet_pos(treebank_tag):
    """
    Helper function to convert a POS tag from the Penn Treebank to a WordNet format

    :param treebank_tag: Penn Treebank POS tag, string

    :returns: WordNet POS tag
    """

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_words(words):
    """
    Function to lemmatize word tokens using the WordNet lemmatizer

    :param words: list of word tokens in the tweet

    :returns: list of lemmatized word tokens
    """

    lemmatized_words = []
    # Get the Penn Treebank POS tags for the word tokens
    word_pos_tags = pos_tag(words)
    for word, word_pos_tag in word_pos_tags:
        # Get the WordNet POS tag
        word_pos_tag = get_wordnet_pos(word_pos_tag)
        # Use the WordNet POS tag to lemmatize the word into the correct word form
        lemmatized_words.append(lemmatizer.lemmatize(word, word_pos_tag))
    return lemmatized_words


def preprocess_tweet(tweet):
    """
    Function that performs the whole preprocessing pipeline on a tweet

    :param tweet: tweet text, string

    :returns: list of all tweet tokens, list of only word tokens
    """

    # Parse the tweet text
    tweet_processed = parse_cropped_ending_of_tweet(tweet)
    tweet_processed = parse_hashtags_in_tweet(tweet_processed)
    tweet_processed = parse_time_in_tweet(tweet_processed)
    tweet_processed = parse_emojis_in_tweet(tweet_processed)
    tweet_processed = parse_numbers_in_tweet(tweet_processed)

    # Tokenize the tweet text based on whitespace
    tweet_tokens = tweet_processed.split()
    # Remove stopwords
    tweet_tokens = remove_stopwords_in_tweet(tweet_tokens)
    # Get only proper word tokens
    tweet_word_tokens = get_only_word_tokens(tweet_tokens)
    # Remove repeated characters
    tweet_tokens = remove_repetitions_in_tokens(tweet_tokens, tweet_word_tokens)
    # Lemmatize word tokens
    tweet_word_tokens = lemmatize_words(tweet_word_tokens)

    return tweet_tokens, tweet_word_tokens
