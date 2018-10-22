import pickle as pk

import numpy as np

from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.layers import Input, Embedding

from nn_arch import dnn_build, cnn_build, rnn_build

from match import predict

from util import flat_read, map_item


def define_model(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input1 = Input(shape=(seq_len,))
    input2 = Input(shape=(seq_len,))
    input3 = Input(shape=(seq_len,))
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    embed_input3 = embed(input3)
    func = map_item(name, funcs)
    output = func(embed_input1, embed_input2, embed_input3)
    model = Model([input1, input2, input3], output)
    return model


def load_model(name, embed_mat, seq_len):
    model = define_model(name, embed_mat, seq_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


seq_len = 30

path_test = 'data/test.csv'
path_label = 'feat/label_test.pkl'
path_embed = 'feat/embed.pkl'
texts = flat_read(path_test, 'text')
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

path_test_triple = 'data/test_triple.csv'
path_triple = 'feat/triple_test.pkl'
anc_texts = flat_read(path_test_triple, 'anc')
pos_texts = flat_read(path_test_triple, 'pos')
neg_texts = flat_read(path_test_triple, 'neg')
with open(path_triple, 'rb') as f:
    triples = pk.load(f)

funcs = {'dnn': dnn_build,
         'cnn': cnn_build,
         'rnn': rnn_build}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'dnn': load_model('dnn', embed_mat, seq_len),
          'cnn': load_model('cnn', embed_mat, seq_len),
          'rnn': load_model('rnn', embed_mat, seq_len)}


def test_triple(name, triples, margin):
    model = map_item(name, models)
    anc_sents, pos_sents, neg_sents = triples
    deltas = model.predict([anc_sents, pos_sents, neg_sents])
    deltas = np.reshape(deltas, (1, -1))[0]
    preds = deltas + margin < 0.0
    print('\n%s %s %.2f\n' % (name, 'acc:', float(np.mean(preds))))
    for delta, anc, pos, neg, pred in zip(deltas, anc_texts, pos_texts, neg_texts, preds):
        if not pred:
            print('{:.3f} {} | {} | {}'.format(delta, anc, pos, neg))


def test(name, texts, labels, vote):
    preds = list()
    for text in texts:
        preds.append(predict(text, name, vote))
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test_triple('dnn', triples, margin=0.5)
    test_triple('cnn', triples, margin=0.5)
    test_triple('rnn', triples, margin=0.5)
    test('dnn', texts, labels, vote=5)
    test('cnn', texts, labels, vote=5)
    test('rnn', texts, labels, vote=5)
