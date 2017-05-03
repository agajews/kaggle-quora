import json
import os
import random
from pprint import pprint

import numpy as np

from augment import augmentations
from clean import load_clean
from keras import backend as K
from models import all_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


global_hyperparam_opts = {
    'maxlen': [30]
}


def train(model, model_hyperparams, augment_names, global_hyperparams,
          fnm='results.json'):
    stamp = model + '_2_'
    stamp += ','.join('{}={}'.format(k, v)
                      for k, v in model_hyperparams.items()) + '_'
    stamp += ','.join('{}={}'.format(k, v)
                      for k, v in global_hyperparams.items()) + '_'
    stamp += ','.join(augment_names)
    with open(fnm, 'r') as f:
        results = json.load(f)
    if stamp in results:
        return results['val_loss']
    print('=' * 50)
    print('Model: {}'.format(model))
    print('Model hyperparams:')
    pprint(model_hyperparams)
    print('Global hyperparams:')
    pprint(global_hyperparams)
    print('Augmentations: {}'.format(augment_names))

    print('Loading data...')
    x1, x2, y, val_x1, val_x2, val_y, _, _, _, embeddings = load_clean(
        global_hyperparams, augment_names)

    # x1 = np.vstack([x1, val_x1])
    # x2 = np.vstack([x2, val_x2])
    # y = np.concatenate([y, val_y])
    #
    # perm = np.random.permutation(len(x1))
    # idx_train = perm[:int(len(x1) * 0.9)]
    # idx_val = perm[int(len(x1) * 0.9):]
    #
    # val_x1 = x1[idx_val]
    # x1 = x1[idx_train]
    #
    # val_x2 = x2[idx_val]
    # x2 = x2[idx_train]
    #
    # val_y = y[idx_val]
    # y = y[idx_train]

    x1_c = np.vstack([x1, x2])
    x2_c = np.vstack([x2, x1])
    y_c = np.concatenate([y, y])

    print('Found {} tokens'.format(embeddings.shape[0]))

    print('Training model...')
    # val_loss, misc_data = all_models[model].train(
    #     x1[:5000], x2[:5000], y[:5000], val_x1[:5000], val_x2[:5000],
    #     val_y[:5000], embeddings, stamp, len(results), **model_hyperparams)
    val_loss, misc_data = all_models[model].train(
        x1_c, x2_c, y_c, val_x1, val_x2,
        val_y, embeddings, stamp, len(results), **model_hyperparams)
    print('Val loss: {}'.format(val_loss))
    results[stamp] = {
        'val_loss': val_loss,
        'model_hyperparams': model_hyperparams,
        'global_hyperparams': global_hyperparams,
        'misc': misc_data,
        'model': model
    }
    with open(fnm, 'w') as f:
        json.dump(results, f)
    best_stamp = min(results.keys(), key=lambda k: results[k]['val_loss'])
    best_loss = results[best_stamp]['val_loss']
    best_model = results[best_stamp]['model']
    print('Best so far: {}, {}'.format(best_model, best_loss))
    K.clear_session()
    return val_loss


def train_random(fnm='results.json'):
    global_hyperparams = {}
    for k, opts in global_hyperparam_opts.items():
        global_hyperparams[k] = random.choice(opts)
    model = random.choice(list(all_models.keys()))
    model_hyperparams = {}
    for k, opts in all_models[model].hyperparam_opts.items():
        model_hyperparams[k] = random.choice(opts)
    augment_names = random.sample(
        sorted(list(augmentations.keys())),
        random.randrange(0, len(augmentations)))
    return train(model, model_hyperparams, augment_names,
                 global_hyperparams, fnm)


if __name__ == '__main__':
    # while True:
    #     train_random()
    train('lstm', {
        'activation': 'relu',
        'batch_size': 1024,
        'batchnorm': True,
        'bidirectional': False,
        'dense_depth': 2,
        'dense_size': 128,
        'dropout_p': 0.30,
        'lr': 0.001,
        'lstm_depth': 1,
        'lstm_size': 256,
        'rec_dropout_p': 0.30
    }, [
        'noisify',
        'transitivify',
    ], {'maxlen': 30})
    # train('lstm', {
    #     'activation': 'relu',
    #     'batch_size': 1024,
    #     'batchnorm': True,
    #     'bidirectional': True,
    #     'dense_depth': 1,
    #     'dense_size': 128,
    #     'dropout_p': 0.30,
    #     'lr': 0.002,
    #     'lstm_depth': 1,
    #     'lstm_size': 256,
    #     'rec_dropout_p': 0.30
    # }, [
    #     'transitivify'
    # ], {'maxlen': 30})
