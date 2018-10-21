import os

import re

import pandas as pd

from random import shuffle, sample

from util import load_word_re, load_type_re, load_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)


def save_triple(path, triples):
    head = 'anc,pos,neg'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, pos, neg in triples:
            f.write(text + ',' + pos + ',' + neg + '\n')


def extend(triples, path_extra_triple):
    extra_triples = list()
    for text, pos, neg in pd.read_csv(path_extra_triple).values:
        extra_triples.append((text, pos, neg))
    shuffle(extra_triples)
    return extra_triples + triples


def make_triple(path_univ_dir, path_train_triple, path_test_triple, path_extra_triple):
    labels = list()
    label_texts = dict()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        labels.append(label)
        label_texts[label] = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                label_texts[label].append(line.strip())
    neg_fold = 2
    triples = list()
    for i in range(len(labels)):
        texts = label_texts[labels[i]]
        neg_texts = list()
        for j in range(len(labels)):
            if j != i:
                neg_texts.extend(label_texts[labels[j]])
        for j in range(len(texts) - 1):
            for k in range(j + 1, len(texts)):
                sub_texts = sample(neg_texts, neg_fold)
                for neg_text in sub_texts:
                    triples.append((texts[j], texts[k], neg_text))
    shuffle(triples)
    bound = int(len(triples) * 0.9)
    train_triples = extend(triples[:bound], path_extra_triple)
    save_triple(path_train_triple, train_triples)
    save_triple(path_test_triple, triples[bound:])


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def gather(path_univ_dir, path_train, path_test):
    texts = list()
    labels = list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts[:bound], labels[:bound])
    save(path_test, texts[bound:], labels[bound:])


def prepare(path_univ_dir):
    files = os.listdir(path_univ_dir)
    for file in files:
        text_set = set()
        texts = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = re.sub(stop_word_re, '', line.strip())
                for word_type, word_re in word_type_re.items():
                    text = re.sub(word_re, word_type, text)
                text = word_replace(text, homo_dict)
                text = word_replace(text, syno_dict)
                if text not in text_set:
                    text_set.add(text)
                    texts.append(text)
        with open(os.path.join(path_univ_dir, file), 'w') as f:
            for text in texts:
                f.write(text + '\n')


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    prepare(path_univ_dir)
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    gather(path_univ_dir, path_train, path_test)
    path_train_triple = 'data/train_triple.csv'
    path_test_triple = 'data/test_triple.csv'
    path_extra_triple = 'data/extra_triple.csv'
    make_triple(path_univ_dir, path_train_triple, path_test_triple, path_extra_triple)
