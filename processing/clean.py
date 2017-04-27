import re

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


def replace_nums(question, **kwargs):
    new_question = []
    for word in question:
        new_question += re.sub(r'[0-9]+', ' <num> ', word).split()
    return new_question


def strip_punct(question, **kwargs):
    new_question = []
    for word in question:
        word = re.sub(r"[^A-Za-z0-9,!.]", " ", word)
        new_question += word.split()
    return question


stemmer = SnowballStemmer('english')


def stem(question, **kwargs):
    return [stemmer.stem(word) for word in question]


def lower(question, **kwargs):
    return [word.lower() for word in question]


def fancy_tokenize(question, **kwargs):
    return word_tokenize(' '.join(question))
