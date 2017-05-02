import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Embedding, Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Nadam

hyperparam_opts = {
    'lstm_size': np.arange(100, 550, 25).tolist(),
    'dense_size': np.arange(100, 550, 25).tolist(),
    'dropout_p': np.around(np.arange(0, 0.8, 0.1), 3).tolist(),
    'rec_dropout_p': np.around(np.arange(0, 0.8, 0.1), 3).tolist(),
    'batch_size': np.arange(100, 1100, 100).tolist(),
    'activation': ['relu', 'tanh', 'elu'],
    'bidirectional': [True, False],
    'lstm_depth': [1, 2],
    'dense_depth': [1, 2, 3],
    'batchnorm': [True, False],
    'lr': [0.0005, 0.001, 0.002, 0.005]
}


def train(x1, x2, y, val_x1, val_x2,
          val_y, embeddings, stamp, exp_num, **hyperparams):

    val_weights = np.ones(len(val_y))
    val_weights *= 0.472001959
    val_weights[val_y == 0] = 1.309028344

    n_tokens = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    maxlen = x1.shape[1]

    lstm_size = hyperparams['lstm_size']
    dense_size = hyperparams['dense_size']
    dropout_p = hyperparams['dropout_p']
    rec_dropout_p = hyperparams['rec_dropout_p']
    batch_size = hyperparams['batch_size']
    activation = hyperparams['activation']
    bidirectional = hyperparams['bidirectional']
    lstm_depth = hyperparams['lstm_depth']
    dense_depth = hyperparams['dense_depth']
    batchnorm = hyperparams['batchnorm']
    lr = hyperparams['lr']

    embedding = Embedding(n_tokens, embedding_size,
                          input_length=maxlen, weights=[embeddings],
                          trainable=False, mask_zero=False)

    lstms = []
    for i in range(lstm_depth):
        return_sequences = i != lstm_depth - 1
        lstm = LSTM(
            lstm_size, dropout=rec_dropout_p,
            recurrent_dropout=rec_dropout_p, return_sequences=return_sequences)
        if bidirectional:
            lstm = Bidirectional(lstm)
        lstms.append(lstm)

    q2_in = Input(shape=[maxlen], dtype='int32')
    q2_vec = embedding(q2_in)
    for lstm in lstms:
        q2_vec = lstm(q2_vec)

    q1_in = Input(shape=[maxlen], dtype='int32')
    q1_vec = embedding(q1_in)
    for lstm in lstms:
        q1_vec = lstm(q1_vec)

    q2_in = Input(shape=[maxlen], dtype='int32')
    q2_vec = embedding(q2_in)
    for lstm in lstms:
        q2_vec = lstm(q2_vec)

    net = concatenate([q1_vec, q2_vec])
    net = Dropout(dropout_p)(net)
    if batchnorm:
        net = BatchNormalization()(net)

    for i in range(dense_depth):
        net = Dense(dense_size, activation=activation)(net)
        net = Dropout(dropout_p)(net)
        if batchnorm:
            net = BatchNormalization()(net)

    y_hat = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[q1_in, q2_in], outputs=y_hat)
    optim = Nadam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    ckpt_path = 'params/exp_{}.ckpt'.format(exp_num)
    ckpt = ModelCheckpoint(
        ckpt_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([x1, x2], y,
                     validation_data=([val_x1, val_x2], val_y, val_weights),
                     epochs=200, batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopping, ckpt],
                     class_weight={0: 1.309028344, 1: 0.472001959})

    best_val = min(hist.history['val_loss'])
    return best_val, {'model_fnm': ckpt_path, 'hist': hist.history}
