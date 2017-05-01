import csv
import pickle
from os import path


def load_train(fnm='data/train.csv', dest_fnm='data/train.p'):
    if path.exists(dest_fnm):
        with open(dest_fnm, 'rb') as f:
            return pickle.load(f)
    data = []
    with open(fnm, 'r') as f:
        f.readline()  # ignore header
        reader = csv.reader(f, delimiter=',')
        for (_id, qid1, qid2, q1, q2, duplicate) in reader:
            data.append((int(qid1), int(qid2), q1, q2, int(duplicate)))
    with open(dest_fnm, 'wb') as f:
        pickle.dump(data, f)


def load_test(fnm='data/test.csv', dest_fnm='data/test.p'):
    if path.exists(dest_fnm):
        with open(dest_fnm, 'rb') as f:
            return pickle.load(f)
    data = []
    with open(fnm, 'r') as f:
        f.readline()  # ignore header
        reader = csv.reader(f, delimiter=',')
        for (id, q1, q2) in reader:
            data.append((id, q1, q2))
    with open(dest_fnm, 'wb') as f:
        pickle.dump(data, f)
