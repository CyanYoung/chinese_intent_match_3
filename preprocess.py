import os

import re

import pandas as pd

from random import shuffle, sample, randint

from util import load_word_re, load_type_re, load_pair, word_replace


def drop(words, bound):
    ind = randint(0, bound)
    words.pop(ind)
    return ''.join(words)


def swap(words, bound):
    ind1, ind2 = randint(0, bound), randint(0, bound)
    words[ind1], words[ind2] = words[ind2], words[ind1]
    return ''.join(words)


def copy(words, bound):
    ind1, ind2 = randint(0, bound), randint(0, bound)
    words.insert(ind1, words[ind2])
    return ''.join(words)


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

aug_rate, pos_rate, neg_rate = 2, 4, 4

funcs = [drop, swap, copy]


def save_triple(path, triples):
    head = 'anc,pos,neg'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, pos, neg in triples:
            f.write(text + ',' + pos + ',' + neg + '\n')


def expand(triples, path_extra_triple):
    extra_triples = list()
    for text, pos, neg in pd.read_csv(path_extra_triple).values:
        extra_triples.append((text, pos, neg))
    shuffle(extra_triples)
    return extra_triples + triples


def make_triple(path_aug_dir, path_train_triple, path_test_triple, path_extra_triple):
    labels = list()
    label_texts = dict()
    files = os.listdir(path_aug_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        labels.append(label)
        label_texts[label] = list()
        with open(os.path.join(path_aug_dir, file), 'r') as f:
            for line in f:
                label_texts[label].append(line.strip())
    triples = list()
    for i in range(len(labels)):
        texts = label_texts[labels[i]]
        res_texts = list()
        for j in range(len(labels)):
            if j != i:
                res_texts.extend(label_texts[labels[j]])
        for text in texts:
            pos_texts = sample(texts, pos_rate)
            neg_texts = sample(res_texts, neg_rate)
            for pos_text, neg_text in zip(pos_texts, neg_texts):
                triples.append((text, pos_text, neg_text))
    shuffle(triples)
    bound = int(len(triples) * 0.9)
    train_triples = expand(triples[:bound], path_extra_triple)
    save_triple(path_train_triple, train_triples)
    save_triple(path_test_triple, triples[bound:])


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def gather(path_aug_dir, path_train, path_test):
    texts, labels = list(), list()
    files = os.listdir(path_aug_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_aug_dir, file), 'r') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts[:bound], labels[:bound])
    save(path_test, texts[bound:], labels[bound:])


def clean(text):
    text = re.sub(stop_word_re, '', text)
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    return word_replace(text, syno_dict)


def augment(text):
    aug_texts = list()
    bound = len(text) - 1
    if bound > 0:
        for func in funcs:
            for _ in range(aug_rate):
                words = list(text)
                aug_texts.append(func(words, bound))
    return aug_texts


def prepare(path_univ_dir, path_aug_dir):
    files = os.listdir(path_univ_dir)
    for file in files:
        text_set = set()
        texts = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = line.strip().lower()
                text = clean(text)
                if text and text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    texts.extend(augment(text))
        with open(os.path.join(path_aug_dir, file), 'w') as f:
            for text in texts:
                f.write(text + '\n')


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_aug_dir = 'data/aug'
    prepare(path_univ_dir, path_aug_dir)
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    gather(path_aug_dir, path_train, path_test)
    path_train_triple = 'data/train_triple.csv'
    path_test_triple = 'data/test_triple.csv'
    path_extra_triple = 'data/extra_triple.csv'
    make_triple(path_aug_dir, path_train_triple, path_test_triple, path_extra_triple)
