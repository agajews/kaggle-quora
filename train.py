import json
import os
import random
from pprint import pprint

from clean import load_clean
from keras import backend as K
from models import all_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


global_hyperparam_opts = {
    'maxlen': [30]
}


def train(model, model_hyperparams, global_hyperparams, augment_names,
          fnm='results.json'):
    stamp = model + '_'
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
    print('Global hyperparams:')
    pprint(global_hyperparams)

    print('Loading data...')
    x1, x2, y, val_x1, val_x2, val_y, _, _, _, embeddings = load_clean(
        global_hyperparams, augment_names)

    print('Found {} tokens'.format(embeddings.shape[0]))

    print('Training model...')
    # val_loss, misc_data = all_models[model].train(
    #     x1[:5000], x2[:5000], y[:5000], val_x1[:5000], val_x2[:5000],
    #     val_y[:5000], embeddings, stamp, len(results), **model_hyperparams)
    val_loss, misc_data = all_models[model].train(
        x1, x2, y, val_x1, val_x2,
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


def train_random(p=0.5, val_split=0.1, fnm='results.json'):
    global_hyperparams = {}
    for k, opts in global_hyperparam_opts.items():
        global_hyperparams[k] = random.choice(opts)
    model = random.choice(list(all_models.keys()))
    model_hyperparams = {}
    for k, opts in all_models[model].hyperparam_opts.items():
        model_hyperparams[k] = random.choice(opts)
    return train(model, model_hyperparams,
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
    ], {'maxlen': 30})
