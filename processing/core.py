import pickle
from sys import stdout
from time import clock

import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def load_pickle(fnm):
    with open(fnm, 'rb') as f:
        return pickle.load(f)


def load_train(fnm='data/train.p'):
    return load_pickle(fnm)


def load_test(fnm='data/test.p'):
    return load_pickle(fnm)


def prep_train_mats(maxlen, train_data, val_data, test_data):
    q1s = [' '.join(entry[2]) for entry in train_data]
    q2s = [' '.join(entry[3]) for entry in train_data]
    duplicates = [entry[4] for entry in train_data]

    val_q1s = [' '.join(entry[2]) for entry in val_data]
    val_q2s = [' '.join(entry[3]) for entry in val_data]
    val_duplicates = [entry[4] for entry in val_data]

    test_q1s = [' '.join(entry[1]) for entry in test_data]
    test_q2s = [' '.join(entry[2]) for entry in test_data]

    all_qs = q1s + q2s + val_q1s + val_q2s + test_q1s + test_q2s

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_qs)

    sequences_1 = tokenizer.texts_to_sequences(q1s)
    sequences_2 = tokenizer.texts_to_sequences(q2s)

    val_sequences_1 = tokenizer.texts_to_sequences(val_q1s)
    val_sequences_2 = tokenizer.texts_to_sequences(val_q2s)

    words = tokenizer.word_index

    # words = set(word for q in all_qs for word in q)
    # words = dict(zip(words, range(1, len(words) + 1)))
    #
    # def qs_to_mat(qs):
    #     mat = np.zeros((len(qs), maxlen))
    #     for i, q in enumerate(qs):
    #         q = q[:maxlen]
    #         for j, word in enumerate(q):
    #             if j >= maxlen:
    #                 break
    #             mat[i, maxlen - len(q) + j] = words[word]
    #     return mat
    #
    # x1 = qs_to_mat(q1s)
    # x2 = qs_to_mat(q2s)
    # val_x1 = qs_to_mat(val_q1s)
    # val_x2 = qs_to_mat(val_q2s)

    x1 = pad_sequences(sequences_1, maxlen=maxlen)
    x2 = pad_sequences(sequences_2, maxlen=maxlen)

    val_x1 = pad_sequences(val_sequences_1, maxlen=maxlen)
    val_x2 = pad_sequences(val_sequences_2, maxlen=maxlen)

    y = np.array(duplicates)
    val_y = np.array(val_duplicates)

    return x1, x2, y, val_x1, val_x2, val_y, words


def build_embeddings(words):
    word2vec = KeyedVectors.load_word2vec_format(
        'data/GoogleNews-vectors-negative300.bin', binary=True)
    n_tokens = len(words)
    print('{} unique tokens'.format(n_tokens))
    embeddings = np.zeros((n_tokens + 1, 300))
    n_missing = 0
    for word, i in words.items():
        if word in word2vec.vocab:
            embeddings[i] = word2vec.word_vec(word)
        else:
            n_missing += 1
    print('Missing words: {:0.2f}%'.format(n_missing / n_tokens * 100))
    return embeddings


def load_data(processors, augmentors, hyperparams,
              val_split=0.1, interval=1000):
    maxlen = hyperparams['maxlen']

    train_data = load_train()
    train_data = [(qid1, qid2, q1.split(), q2.split(), duplicate)
                  for (qid1, qid2, q1, q2, duplicate) in train_data]
    print('{} training entries'.format(len(train_data)))

    test_data = load_test()
    test_data = [(_id, q1.split(), q2.split()) for (_id, q1, q2) in test_data]
    print('{} testing entries'.format(len(test_data)))

    def process(questions, extract, update):
        processed = []
        time = clock()
        for i, question in enumerate(questions):
            if (i + 1) % interval == 0:
                diff = clock() - time
                time = clock()
                stdout.write(
                    '\rQuestion {}/{}: ({:.2f} mins remaining)'.format(
                        i + 1, len(questions),
                        (len(questions) - i) / interval * diff / 60))
                stdout.flush()
            q1, q2 = extract(question)
            for processor in processors:
                q1 = processor(q1, **hyperparams)
                q2 = processor(q2, **hyperparams)
            processed.append(update(question, q1, q2))
        print()
        return processed

    print('Processing training data...')
    train_data = process(
        train_data,
        lambda q: (q[2], q[3]),
        lambda q, q1, q2: (q[0], q[1], q1, q2, q[4]))

    print('Processing testing data...')
    test_data = process(
        test_data,
        lambda q: (q[1], q[2]),
        lambda q, q1, q2: (q[0], q1, q2))

    split = int(len(train_data) * val_split)
    data = train_data
    train_data = data[:-split]
    val_data = data[-split:]

    print('Augmenting training data...')
    for augmentor in augmentors:
        train_data = augmentor(train_data, **hyperparams)

    print('Building training matrices...')
    x1, x2, y, val_x1, val_x2, val_y, words = prep_train_mats(
        maxlen, train_data, val_data, test_data)

    print('Preparing word embeddings...')
    embeddings = build_embeddings(words)

    return x1, x2, y, val_x1, val_x2, val_y, embeddings
