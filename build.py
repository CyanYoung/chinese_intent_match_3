import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import keras.backend as K

from nn_arch import dnn, cnn, rnn

from util import map_item


batch_size = 128

path_embed = 'feat/embed.pkl'
path_triple = 'feat/triple_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_triple, 'rb') as f:
    triples = pk.load(f)

funcs = {'dnn': dnn,
         'cnn': cnn,
         'rnn': rnn}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5',
         'dnn_plot': 'model/plot/dnn.png',
         'cnn_plot': 'model/plot/cnn.png',
         'rnn_plot': 'model/plot/rnn.png'}


def triple_loss(margin, delta):
    return K.mean(K.maximum(0.0, delta + margin), axis=-1)


def triple_acc(margin, delta):
    return K.mean(K.cast(K.less_equal(delta + margin, 0.0), K.floatx()), axis=-1)


def compile(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True, name='embed')
    input1 = Input(shape=(seq_len,))
    input2 = Input(shape=(seq_len,))
    input3 = Input(shape=(seq_len,))
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    embed_input3 = embed(input3)
    func = map_item(name, funcs)
    output = func(embed_input1, embed_input2, embed_input3)
    model = Model([input1, input2, input3], output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss=triple_loss, optimizer=Adam(lr=0.001), metrics=[triple_acc])
    return model


def fit(name, epoch, embed_mat, triples, margin):
    anc_sents, pos_sents, neg_sents = triples
    margins = np.ones(len(anc_sents)) * margin
    seq_len = len(anc_sents[0])
    model = compile(name, embed_mat, seq_len)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit([anc_sents, pos_sents, neg_sents], margins, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('dnn', 10, embed_mat, triples, margin=1)
    fit('cnn', 10, embed_mat, triples, margin=1)
    fit('rnn', 10, embed_mat, triples, margin=1)
