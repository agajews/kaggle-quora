import json
import os
import random
from pprint import pprint

from keras import backend as K

from models import all_models
from processing import (all_augmentors, all_processors, augmentor_order,
                        load_data, processor_order)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


global_hyperparam_opts = {
    'maxlen': [20, 30, 40]
}


def random_fns(fns, p):
    return [f for f in fns if random.random() < p]


def train(model, model_hyperparams, processors, augmentors, global_hyperparams,
          val_split=0.1, fnm='results.json'):
    stamp = '_'.join([model] + processors + augmentors) + '_'
    stamp += ','.join('{}={}'.format(k, v)
                      for k, v in model_hyperparams.items()) + '_'
    stamp += ','.join('{}={}'.format(k, v)
                      for k, v in global_hyperparams.items())
    with open(fnm, 'r') as f:
        results = json.load(f)
    if stamp in results:
        return results['val_loss']
    print('=' * 50)
    print('Model: {}'.format(model))
    print('Model hyperparams:')
    pprint(model_hyperparams)
    print('Processors: {}'.format(processors))
    print('Augmentors: {}'.format(augmentors))
    print('Global hyperparams:')
    pprint(global_hyperparams)

    print('Loading data...')
    x1, x2, y, val_x1, val_x2, val_y, embeddings = load_data(
        [all_processors[p] for p in processors],
        [all_augmentors[a] for a in augmentors],
        global_hyperparams, val_split)

    print('Training model...')
    val_loss, misc_data = all_models[model].train(
        x1, x2, y, val_x1, val_x2,
        val_y, embeddings, stamp, len(results), **model_hyperparams)
    print('Val loss: {}'.format(val_loss))
    results[stamp] = {
        'val_loss': val_loss,
        'model_hyperparams': model_hyperparams,
        'global_hyperparams': global_hyperparams,
        'augmentors': augmentors,
        'processors': processors,
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


def train_random(p=0.5, val_split=0.1, fnm='results.json'):
    global_hyperparams = {}
    for k, opts in global_hyperparam_opts.items():
        global_hyperparams[k] = random.choice(opts)
    model = random.choice(list(all_models.keys()))
    model_hyperparams = {}
    for k, opts in all_models[model].hyperparam_opts.items():
        model_hyperparams[k] = random.choice(opts)
    processors = random_fns(processor_order, p)
    augmentors = random_fns(augmentor_order, p)
    return train(model, model_hyperparams, processors, augmentors,
                 global_hyperparams, val_split, fnm)


if __name__ == '__main__':
    # while True:
    #     train_random()
    train('lstm', {
        'activation': 'relu',
        'batch_size': 1000,
        'batchnorm': True,
        'bidirectional': True,
        'dense_depth': 1,
        'dense_size': 128,
        'dropout_p': 0.3,
        'lr': 0.002,
        'lstm_depth': 1,
        'lstm_size': 256,
        'rec_dropout_p': 0.3
    }, [
        'lower',
        'strip_punct',
        'replace_nums',
        'fancy_tokenize',
        'stem'
    ], ['transitivify'], {'maxlen': 30})
