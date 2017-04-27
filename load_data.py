import csv
import pickle


def load_train(fnm, dest_fnm):
    data = []
    with open(fnm, 'r') as f:
        f.readline()  # ignore header
        reader = csv.reader(f, delimiter=',')
        for (_id, qid1, qid2, q1, q2, duplicate) in reader:
            data.append((int(qid1), int(qid2), q1, q2, int(duplicate)))
    with open(dest_fnm, 'wb') as f:
        pickle.dump(data, f)


def load_test(fnm, dest_fnm):
    data = []
    with open(fnm, 'r') as f:
        f.readline()  # ignore header
        reader = csv.reader(f, delimiter=',')
        for (id, q1, q2) in reader:
            data.append((id, q1, q2))
    with open(dest_fnm, 'wb') as f:
        pickle.dump(data, f)
