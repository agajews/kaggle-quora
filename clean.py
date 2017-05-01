import pickle
import re
from os import path

import numpy as np

from augment import augmentations
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from load_data import load_test, load_train
from nltk.stem import SnowballStemmer


def clean(question):
    question = re.sub(r"[^A-Za-z0-9^,!.']", " ", question)
    return question


stemmer = SnowballStemmer('english')


def stem(question):
    return ' '.join([stemmer.stem(word) for word in question.split(' ')])


def process_question(q):
    q = clean(q)
    q = stem(q)
    return q


def load_embeddings(word_index):

    word2vec = KeyedVectors.load_word2vec_format(
        'data/GoogleNews-vectors-negative300.bin', binary=True)

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


def load_clean(hyperparams, augment_names):
    fnm = 'data/all_clean_' + '_'.join(augment_names) + '.p'
    if path.exists(fnm):
        with open(fnm, 'rb') as f:
            return pickle.load(f)

    maxlen = hyperparams['maxlen']

    train = load_train()[:50000]
    test = load_test()[:50000]

    print('Found {} train questions'.format(len(train)))
    print('Found {} test questions'.format(len(test)))

    print('Cleaning train data...')
    train_clean = []
    for (qid1, qid2, q1, q2, duplicate) in train:
        q1 = process_question(q1)
        q2 = process_question(q2)
        train_clean.append((qid1, qid2, q1, q2, duplicate))

    print('Cleaning test data...')
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

    qid1s = [qid1 for (qid1, qid2, _, _, _) in train_clean]
    qid2s = [qid2 for (qid1, qid2, _, _, _) in train_clean]

    y = [d for (_, _, q1, q2, d) in train_clean]

    split = int(0.1 * len(train_clean))

    val_q1s = q1s[-split:]
    q1s = q1s[:-split]
    val_q2s = q2s[-split:]
    q2s = q2s[:-split]
    val_y = y[-split:]
    y = y[:-split]
    qid1s = qid1s[:-split]
    qid2s = qid2s[:-split]

    train_data = list(zip(qid1s, qid2s, q1s, q1s, y))

    for augment in [augmentations[n] for n in augment_names]:
        train_data = augment(train_data)

    q1s = [q1 for (qid1, qid2, q1, q2, d) in train_data]
    q2s = [q2 for (qid1, qid2, q1, q2, d) in train_data]
    y = [d for (qid1, qid2, q1, q2, d) in train_data]

    print('Generating input matrices...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_qs)

    sequences_1 = tokenizer.texts_to_sequences(q1s)
    sequences_2 = tokenizer.texts_to_sequences(q2s)
    val_sequences_1 = tokenizer.texts_to_sequences(val_q1s)
    val_sequences_2 = tokenizer.texts_to_sequences(val_q2s)
    test_sequences_1 = tokenizer.texts_to_sequences(test_q1s)
    test_sequences_2 = tokenizer.texts_to_sequences(test_q2s)

    x1 = pad_sequences(sequences_1, maxlen=maxlen)
    x2 = pad_sequences(sequences_2, maxlen=maxlen)
    val_x1 = pad_sequences(val_sequences_1, maxlen=maxlen)
    val_x2 = pad_sequences(val_sequences_2, maxlen=maxlen)
    test_x1 = pad_sequences(test_sequences_1, maxlen=maxlen)
    test_x2 = pad_sequences(test_sequences_2, maxlen=maxlen)

    y = np.array(y)
    val_y = np.array(val_y)

    print('Loading embeddings...')
    embeddings = load_embeddings(tokenizer.word_index)
    print('Done')

    data = (x1, x2, y, val_x1, val_x2, val_y, test_x1,
            test_x2, tokenizer.word_index, embeddings)

    print('Saving in {}'.format(fnm))
    with open(fnm, 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == '__main__':
    load_clean({'maxlen': 30})
