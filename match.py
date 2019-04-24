import pickle as pk

import numpy as np

from collections import Counter

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from encode import load_encode

from util import map_item


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        core_sents = pk.load(f)
    return core_sents


seq_len = 30

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

paths = {'dnn': 'cache/dnn.pkl',
         'cnn': 'cache/cnn.pkl',
         'rnn': 'cache/rnn.pkl'}

caches = {'dnn': load_cache(map_item('dnn', paths)),
          'cnn': load_cache(map_item('dnn', paths)),
          'rnn': load_cache(map_item('dnn', paths))}

models = {'dnn_encode': load_encode('dnn', embed_mat, seq_len),
          'cnn_encode': load_encode('cnn', embed_mat, seq_len),
          'rnn_encode': load_encode('rnn', embed_mat, seq_len)}


def predict(text, name, vote):
    text = clean(text)
    core_sents, core_labels = map_item(name, caches)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    encode = map_item(name + '_encode', models)
    encode_seq = encode.predict([pad_seq])
    dists = list()
    for core_sent in core_sents:
        dists.append(np.sum(np.square(encode_seq - core_sent)[0]))
    dists = np.array(dists)
    min_dists = sorted(dists)[:vote]
    min_inds = np.argsort(dists)[:vote]
    min_preds = [core_labels[ind] for ind in min_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, dist in zip(min_preds, min_dists):
            formats.append('{} {:.3f}'.format(pred, dist))
        return ', '.join(formats)
    else:
        pairs = Counter(min_preds)
        return pairs.most_common()[0][0]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn', vote=5))
        print('cnn: %s' % predict(text, 'cnn', vote=5))
        print('rnn: %s' % predict(text, 'rnn', vote=5))
