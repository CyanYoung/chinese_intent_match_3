import pickle as pk

import re

import numpy as np

from collections import Counter

from keras.preprocessing.sequence import pad_sequences

from encode import load_model

from util import load_word_re, load_type_re, load_pair, word_replace, flat_read, map_item


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        cache = pk.load(f)
    return cache


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_train = 'data/train.csv'
path_label = 'feat/label_train.pkl'
path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
texts = flat_read(path_train, 'text')
with open(path_label, 'rb') as f:
    labels = pk.load(f)
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

models = {'dnn': load_model('dnn', embed_mat, seq_len),
          'cnn': load_model('cnn', embed_mat, seq_len),
          'rnn': load_model('rnn', embed_mat, seq_len)}


def predict(text, name, vote):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    cache_sents = map_item(name, caches)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item(name, models)
    encode_seq = model.predict([pad_seq])
    encode_mat = np.repeat(encode_seq, len(cache_sents), axis=0)
    dists = np.sqrt(np.sum(np.square(encode_mat - cache_sents), axis=1))
    min_dists = sorted(dists)[:vote]
    min_inds = np.argsort(dists)[:vote]
    min_preds = [labels[ind] for ind in min_inds]
    if __name__ == '__main__':
        min_texts = [texts[ind] for ind in min_inds]
        formats = list()
        for pred, prob, text in zip(min_preds, min_dists, min_texts):
            formats.append('{} {:.3f} {}'.format(pred, prob, text))
        return ', '.join(formats)
    else:
        pairs = Counter(min_preds)
        return pairs.most_common()[0][0]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn', vote=3))
        print('cnn: %s' % predict(text, 'cnn', vote=3))
        print('rnn: %s' % predict(text, 'rnn', vote=3))
