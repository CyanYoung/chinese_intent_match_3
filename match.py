import pickle as pk

import re

import numpy as np

from collections import Counter

from keras.preprocessing.sequence import pad_sequences

from encode import load_model

from util import load_word_re, load_type_re, load_pair, word_replace, map_item


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        core_sents = pk.load(f)
    return core_sents


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
path_label = 'cache/label.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_label, 'rb') as f:
    core_labels = pk.load(f)

paths = {'dnn': 'cache/dnn.pkl',
         'cnn': 'cache/cnn.pkl',
         'rnn': 'cache/rnn.pkl'}

caches = {'dnn': load_cache(map_item('dnn', paths)),
          'cnn': load_cache(map_item('dnn', paths)),
          'rnn': load_cache(map_item('dnn', paths))}

models = {'dnn': load_model('dnn', embed_mat, seq_len),
          'cnn': load_model('cnn', embed_mat, seq_len),
          'rnn': load_model('rnn', embed_mat, seq_len)}


def predict(text, name, vote):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    core_sents = map_item(name, caches)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item(name, models)
    encode_seq = model.predict([pad_seq])
    encode_mat = np.repeat(encode_seq, len(core_sents), axis=0)
    dists = np.sum(np.square(encode_mat - core_sents), axis=-1)
    min_dists = sorted(dists)[:vote]
    min_inds = np.argsort(dists)[:vote]
    min_preds = [core_labels[ind] for ind in min_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, prob in zip(min_preds, min_dists):
            formats.append('{} {:.3f}'.format(pred, prob))
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
