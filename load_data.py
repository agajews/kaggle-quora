import pickle


def load_train(fnm='train.p'):
    with open(fnm, 'rb') as f:
        return pickle.load(f)


def load_test(fnm='test.p'):
    with open(fnm, 'rb') as f:
        return pickle.load(f)
