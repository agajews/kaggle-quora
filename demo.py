import os
import pickle

import keras.backend as K
from clean import process_question
from keras.layers import LSTM, Dense, Dropout, Embedding, Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('data/all_clean_4_noisify_transitivify.p', 'rb') as f:
    x1, x2, y, val_x1, val_x2, val_y, _, _, words, embeddings = pickle.load(f)

n_tokens = embeddings.shape[0]
embedding_size = embeddings.shape[1]
maxlen = 30
lstm_size = 256
dropout_p = 0.3
rec_dropout_p = 0.3
activation = 'relu'
dense_size = 128


embedding = Embedding(n_tokens, embedding_size,
                      input_length=maxlen, weights=[embeddings],
                      trainable=False, mask_zero=False)

lstm = LSTM(
    lstm_size, dropout=rec_dropout_p,
    recurrent_dropout=rec_dropout_p)

q2_in = Input(shape=[30], dtype='int32')
q2_vec = embedding(q2_in)
q2_vec = lstm(q2_vec)

q1_in = Input(shape=[maxlen], dtype='int32')
q1_vec = embedding(q1_in)
q1_vec = lstm(q1_vec)

q2_in = Input(shape=[maxlen], dtype='int32')
q2_vec = embedding(q2_in)
q2_vec = lstm(q2_vec)

net = concatenate([q1_vec, q2_vec])
net = Dropout(dropout_p)(net)
net = BatchNormalization()(net)

for i in range(2):
    net = Dense(dense_size, activation=activation)(net)
    net = Dropout(dropout_p)(net)
    net = BatchNormalization()(net)

y_hat = Dense(1, activation='sigmoid')(net)

model = Model(inputs=[q1_in, q2_in], outputs=y_hat)
model.load_weights('params/exp_4.ckpt')

tokenizer = Tokenizer()
tokenizer.word_index = words

# q1 = "Do vegetarians help the environment?"
q1 = input('Question 1: ')
q1_clean = process_question(q1)
q1_seqs = tokenizer.texts_to_sequences([q1_clean])
print('q1 => {} => {}'.format(q1_clean, q1_seqs))

# q2 = "Does vegetarianism help climate change?"
q2 = input('Question 2: ')
q2_clean = process_question(q2)
q2_seqs = tokenizer.texts_to_sequences([q2_clean])
print('q2 => {} => {}'.format(q2_clean, q2_seqs))

input('Compute? ')

q1_x = pad_sequences(q1_seqs, maxlen=30)
q2_x = pad_sequences(q2_seqs, maxlen=30)

output = model.predict([q1_x, q2_x])
print(output[0])

K.clear_session()
