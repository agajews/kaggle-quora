from load_data import load_train
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.metrics import edit_distance
import enchant
import re
import pickle
from time import clock

data = load_train()


def replace_nums(question):
    new_question = []
    for word in question:
        new_question.append(re.sub(r'[0-9]+', '<num>', word))
    return new_question


def strip_punct(question):
    question = re.sub(r"[^A-Za-z0-9,!.]", " ", question)
    question = question.replace('-', ' ')
    return question


stemmer = SnowballStemmer('english')


def stem(question):
    return [stemmer.stem(word) for word in question]


spell_dict = enchant.Dict('en_US')


def spellcorrect(question):
    new_question = []
    for word in question:
        if word[0] == '<':
            new_question.append(word)
        elif spell_dict.check(word):
            new_question.append(word)
        else:
            suggestions = spell_dict.suggest(word)
            if suggestions and edit_distance(suggestions[0], word) <= 3:
                new_question.append(suggestions[0])
            else:
                new_question.append('<unk>')
    return new_question


def clean_q(question):
    question = strip_punct(question)
    question = question.lower()
    question = word_tokenize(question)
    question = replace_nums(question)
    # question = spellcorrect(question)
    question = stem(question)
    return question


def unique_tokens(clean_data):
    tokens = set()
    for (_, _, q1, q2, _) in clean_data:
        tokens.update(set(q1))
        tokens.update(set(q2))
    return tokens


clean_data = []
tokens = set()
time = clock()
interval = 1000
for i, (qid1, qid2, q1, q2, duplicate) in enumerate(data):
    if i != 0 and i % interval == 0:
        diff = clock() - time
        time = clock()
        print('{}/{}: {} tokens ({:.2f} mins remaining)'.format(
            i, len(data), len(tokens), (len(data) - i) / interval * diff / 60))
    q1 = clean_q(q1)
    q2 = clean_q(q2)
    tokens.update(set(q1))
    tokens.update(set(q2))
    clean_data.append((qid1, qid2, q1, q2, duplicate))

print('Total tokens: {}'.format(len(tokens)))

with open('train_clean_test.p', 'wb') as f:
    pickle.dump((clean_data, tokens), f)
