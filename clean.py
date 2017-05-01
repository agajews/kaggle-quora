import pickle
import re
from os import path

import numpy as np

from gensim.models import KeyedVectors
from keras.preprocess.sequence import pad_sequences
from keras.preprocess.text import Tokenizer
from load_data import load_test, load_train
from nltk.stem import SnowballStemmer

word2vec = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)


def clean(question):
    question = re.sub(r"[^A-Za-z0-9^,!.']", " ", question)


stemmer = SnowballStemmer('english')


def stem(question):
    return ' '.join([stemmer.stem(word) for word in question.split(' ')])


def process_question(q):
    q = clean(q)
    q = stem(q)
    return q


def load_embeddings(word_index):
    n_tokens = len(word_index) + 1
    print('Found {} tokens'.format(n_tokens))
    embeddings = np.zeros((n_tokens, 300))
    n_missing = 0
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embeddings[i] = word2vec.word_vec(word)
        else:
            n_missing += 1
    print('Missing {}% of embeddings'.format(n_missing / n_tokens * 100))
    return embeddings


def load_clean(maxlen=30):
    if path.exists('data/all_clean.p'):
        with open('data/all_clean.p', 'rb') as f:
            return pickle.load(f)

    train = load_train()[:5000]
    test = load_test()[:5000]

    train_clean = []
    for (qid1, qid2, q1, q2, duplicate) in train:
        q1 = process_question(q1)
        q2 = process_question(q2)
        train_clean.append((qid1, qid2, q1, q2, duplicate))

    test_clean = []
    for (_id, q1, q2) in test:
        q1 = process_question(q1)
        q2 = process_question(q2)
        test_clean.append((_id, q1, q2))

    q1s = [q1 for (_, _, q1, q2, _) in train_clean]
    q2s = [q2 for (_, _, q1, q2, _) in train_clean]
    test_q1s = [q1 for (_, q1, q2) in test_clean]
    test_q2s = [q2 for (_, q1, q2) in test_clean]
    all_qs = q1s + q2s + test_q1s + test_q2s

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_qs)

    sequences_1 = tokenizer.texts_to_sequences(q1s)
    sequences_2 = tokenizer.texts_to_sequences(q2s)
    test_sequences_1 = tokenizer.texts_to_sequences(test_q1s)
    test_sequences_2 = tokenizer.texts_to_sequences(test_q2s)

    x1 = pad_sequences(sequences_1, maxlen=maxlen)
    x2 = pad_sequences(sequences_2, maxlen=maxlen)
    test_x1 = pad_sequences(test_sequences_1, maxlen=maxlen)
    test_x2 = pad_sequences(test_sequences_2, maxlen=maxlen)

    y = np.array([d for (_, _, q1, q2, d) in train_clean])

    split = int(0.1 * len(train_clean))
    val_x1 = x1[-split:]
    x1 = x1[:-split]
    val_x2 = x2[-split:]
    x2 = x2[:-split]
    val_y = y[-split:]
    y = y[:-split]

    embeddings = load_embeddings(tokenizer.word_index)

    with open('data/all_clean.p', 'wb') as f:
        pickle.dump((x1, x2, y, val_x1, val_x2, val_y, test_x1,
                     test_x2, tokenizer.word_index, embeddings), f)