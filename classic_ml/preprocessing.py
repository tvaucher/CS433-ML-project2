import re
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from nltk import pos_tag

from classic_ml.resources import special_tags, positive_emojis, negative_emojis, \
    stopwords_english, dictionary_english, lemmatizer, wordnet


def parse_cropped_ending_of_tweet(tweet):
    return re.sub(r'\S+ \.\.\. <url>$', '<cropped>', tweet)


def parse_hashtags_in_tweet(tweet):
    return re.sub(r"#\w+", "<hashtag>", tweet)


def parse_numbers_in_tweet(tweet):
    return re.sub(r'\d+([.,]\d+)?', ' <number> ', tweet)


def parse_time_in_tweet(tweet):
    return re.sub(r'\d?\d:\d\d', '<time>', tweet)


def parse_emojis_in_tweet(tweet):
    tweet_parsed = tweet
    for emoji in positive_emojis:
        tweet_parsed = re.sub(re.escape(emoji), "<emoji_pos>", tweet_parsed)
    for emoji in negative_emojis:
        tweet_parsed = re.sub(re.escape(emoji), "<emoji_neg>", tweet_parsed)
    return tweet_parsed


def remove_repetitions_in_tokens(tweet_tokens, word_tokens):
    tweet_tokens_fixed = []
    for tweet_token in tweet_tokens:
        fixed_token = re.sub(r"(.)\1+", r"\1", tweet_token)
        if tweet_token in word_tokens and fixed_token not in dictionary_english:
            tweet_tokens_fixed.append(tweet_token)
        else:
            tweet_tokens_fixed.append(fixed_token)
    return tweet_tokens_fixed


def remove_stopwords_in_tweet(tweet_tokens):
    return [token for token in tweet_tokens if token not in stopwords_english]


def get_only_word_tokens(tweet_tokens):
    return [token for token in tweet_tokens if re.match(r"^[A-Za-z\']+$", token)]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_words(words):
    lemmatized_words = []
    word_pos_tags = pos_tag(words)
    for word, word_pos_tag in word_pos_tags:
        word_pos_tag = get_wordnet_pos(word_pos_tag)
        lemmatized_words.append(lemmatizer.lemmatize(word, word_pos_tag))
    return lemmatized_words


def preprocess_tweet(tweet):
    tweet_processed = parse_cropped_ending_of_tweet(tweet)
    tweet_processed = parse_hashtags_in_tweet(tweet_processed)
    tweet_processed = parse_time_in_tweet(tweet_processed)
    tweet_processed = parse_emojis_in_tweet(tweet_processed)
    tweet_processed = parse_numbers_in_tweet(tweet_processed)

    tweet_tokens = tweet_processed.split()
    tweet_tokens = remove_stopwords_in_tweet(tweet_tokens)
    tweet_word_tokens = get_only_word_tokens(tweet_tokens)
    # tweet_tokens = remove_repetitions_in_tokens(tweet_tokens, tweet_word_tokens)
    tweet_word_tokens = lemmatize_words(tweet_word_tokens)

    return tweet_tokens, tweet_word_tokens
