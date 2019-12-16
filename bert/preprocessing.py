'''Module containing the preprocessing methods, if called as standalone, apply preprocessing to the training sets'''
import argparse
import spacy
import en_core_web_sm
from spacy.matcher import Matcher
import re
from tqdm import tqdm

# Set up the tokenizer
nlp = en_core_web_sm.load(disable=['tagger', 'parser', 'ner'])
matcher = Matcher(nlp.vocab)
matcher.add('TAGS', None, [{'TEXT': '<'}, {
            'TEXT': {'REGEX': r'\w+'}}, {'TEXT': '>'}])


def __extend_emojis(emojis):
    ''' Add spaces between emojis characters as both version are present in the data '''
    extended = [' '.join(emoji) for emoji in emojis]
    return [re.escape(emoji) for emoji in emojis + extended]


positive_emojis = __extend_emojis([':)', ":')", ':d', ';)', '^_^',
                                   ';]', ':3', 'x3', 'xxx', 'xx', ':*', 'c:', ':o'])
negative_emojis = __extend_emojis(
    ['<\\3', ':(', ":'(", '-_-', '._.', '- ___ -'])
neutral_emojis = __extend_emojis([':/', ':|', ':||', ':l'])
lolface_emojis = __extend_emojis([':p', 'xp', ';p'])


def parse_emojis(tweet):
    ''' 
    Replace emojis in tweet with tags existing in the pretrained GloVe
    For positive smile: <smile>
    For negative smile: <sadface>
    For neutral smile: <neutralface>
    For tongue out smiles: <lolface>
    '''
    tweet_parsed = tweet
    tweet_parsed = re.sub('|'.join(positive_emojis), '<smile>', tweet_parsed)
    tweet_parsed = re.sub('|'.join(negative_emojis), '<sadface>', tweet_parsed)
    tweet_parsed = re.sub('|'.join(neutral_emojis),
                          '<neutralface>', tweet_parsed)
    tweet_parsed = re.sub('|'.join(lolface_emojis), '<lolface>', tweet_parsed)
    return tweet_parsed


def parse_heart(x):
    ''' Parse <3 to <heart> '''
    return re.sub(r'<\s*3', '<heart>', x)


def parse_cropped_ending_of_tweet(tweet):
    ''' Remove the tags from tweet that are too long: ... <url> '''
    return re.sub(r'\.\.\. <url>$', '', tweet)


def parse_number(x):
    ''' Parse number and dates as GloVe existing <number> tag. If x4... it actually is a repeat '''
    x = re.sub(r'\b[-+]?[.\d]*[\d]+[:,.\d]*(st|nd|rd|th)?\b', '<number>', x)
    return re.sub(r'\bx\d+\b', '<repeat>', x)


def reduce_elong_words(tweet):
    ''' Reduce elongated words according to [Stanford NLP logic](https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb) '''
    return re.sub(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <elong>', tweet)


def reduce_punctuation(tweet):
    ''' Replace !!! to ! <repeat> '''
    for punct in '!?.':
        tweet = re.sub('(\\'+punct+'\ *){2,}', punct + ' <repeat> ', tweet)
    return tweet


def remove_parenthesis_pairs(x):
    ''' Replace (lorem ipsum) by lorem ipsum'''
    return re.sub(r'\((.*?)\)', r'\1', x)


def parse_hashtag(x):
    ''' Split hashtags and add the <hashtag> tag '''
    # TODO: look more into it, can we do more ?
    return re.sub(r'#(\S+)', r'<hashtag> # \1', x)


def preprocess(x):
    ''' Apply the preprocessing pipeline to a given tweet '''
    x = parse_cropped_ending_of_tweet(x)
    x = parse_heart(x)
    x = parse_emojis(x)
    x = parse_number(x)
    x = reduce_punctuation(x)
    x = reduce_elong_words(x)
    x = remove_parenthesis_pairs(x)
    x = parse_hashtag(x)
    x = re.sub(r'["\t]', '', x)
    x = re.sub(r'(\\ <number> ?){2,}', '<number> <repeat> ', x)
    return x


def tokenize(x):
    ''' tokenize the string, return preprocessed list of tokens'''
    parsed = nlp(preprocess(x))
    matches = matcher(parsed)
    indexes = []
    for match_id, start, end in matches:
        if parsed.vocab.strings[match_id] == 'TAGS':
            indexes.append(parsed[start:end])
    with parsed.retokenize() as retokenizer:
        for span in indexes:
            retokenizer.merge(span)
    return [p.text for p in parsed]


def transform(x):
    ''' Remove < > and map some specific GloVe tokens to natural text'''
    to_remove = ['<url>', '<hashtag>', '#']
    to_transform = {'<sadface>': 'sad',
                    '<neutralface>': 'neutral',
                    '<lolface>': 'happy',
                    '<elong>': 'long'}
    for k, v in to_transform.items():
        x = re.sub(k, v, x)
    x = re.sub('|'.join(to_remove), '', x)
    x = re.sub(r'<([^>]+)>', r'\1', x)
    x = re.sub(r'\bn\'t\b', 'not', x)
    return x.strip()