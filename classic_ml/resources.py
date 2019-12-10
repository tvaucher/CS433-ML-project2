""" Module containing resources reused in multiple scripts, for efficient importing """

from os import cpu_count
from nltk import download
from nltk.corpus import words as nltk_dictionary
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import vader
from empath import Empath

# Retrieve number of CPU cores in the machine
NUM_CORES = cpu_count()

# Set random seed globally
SEED = 2019

# Download required NLTK corpora (if not already present)
download("words")
download("stopwords")
download("wordnet")
download("vader_lexicon")

# Retrieve English stopwords
stopwords_english = set(stopwords.words("english"))
# Don't consider "not" as a stopword since it is relevant for sentiment analysis
stopwords_english.remove("not")

# Load the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Retrieve the English dictionary
dictionary_english = set(nltk_dictionary.words())

# Predefine lists of positive and negative emojis commonly found in tweets
positive_emojis = ["<3", ":)", ":')", ":d", ":p", "xp", ";)",
                   ";]", ":3", "xxx", "xx", ":*", "c:", ";p"]
negative_emojis = [r"<\3", ":/", ":(", ":'(", "-_-", ":||", "._.", "- ___ -"]
# Include cases where the emoji characters are split across several tokens
positive_emojis.extend([" ".join(emoji) for emoji in positive_emojis])
negative_emojis.extend([" ".join(emoji) for emoji in negative_emojis])

# Define list of special tags
special_tags = ["<cropped>", "<hashtag>", "<number>", "<time>", "<emoji_pos>", "<emoji_neg>"]

# Load the Vader sentiment model
vader_sia = vader.SentimentIntensityAnalyzer()

# Load the Empath sentiment model
empath_analyzer = Empath()
