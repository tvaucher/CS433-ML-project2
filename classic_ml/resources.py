from nltk import download
from nltk.corpus import words as nltk_dictionary
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import vader
from empath import Empath

SEED = 2019

download("words")
download('stopwords')
download('wordnet')
download('vader_lexicon')

stopwords_english = set(stopwords.words('english'))
stopwords_english.remove("not")

lemmatizer = WordNetLemmatizer()

dictionary_english = set(nltk_dictionary.words())

positive_emojis = ['<3', ':)', ":')", ":d", ":p", "xp", ";)",
                   ";]", ":3", "xxx", "xx", ":*", "c:", ";p"]
negative_emojis = [r"<\3", ":/", ":(", ":'(", "-_-", ":||", "._.", "- ___ -"]
positive_emojis.extend([' '.join(emoji) for emoji in positive_emojis])
negative_emojis.extend([' '.join(emoji) for emoji in negative_emojis])

special_tags = ["<cropped>", "<hashtag>", "<number>", "<time>", "<emoji_pos>", "<emoji_neg>"]

vader_sia = vader.SentimentIntensityAnalyzer()

empath_analyzer = Empath()
