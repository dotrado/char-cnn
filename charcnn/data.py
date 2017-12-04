#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
Features, Preprocessing and Datasets, as described in:

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)


"""

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pandas as pd


# available when the project is checked out, not when pip installed.
DATA_LOCAL_PATH = 'data'

# remote path from google cloud storage
DATA_CLOUD_URL = 'https://storage.googleapis.com/char-cnn-datsets'


def preprocess(xtrain, ytrain, xtest, max_len=None):
    """
    Preprocess and featurize the data
    """

    xtrain = [line.lower() for line in xtrain]
    xtest = [line.lower() for line in xtest]
    ytrain = [long(line) for line in ytrain]

    def chars(dataset):
        return reduce(
            lambda x, y: x.union(y),
            (set(line) for line in dataset))

    def onehot(dataset, max_len, vocab_size):
        hot = np.zeros((len(dataset), max_len, vocab_size), dtype=np.bool)
        i = 0
        for line in dataset:
            j = 0
            for char in line:
                if char != 0:
                    hot[i, j, char] = 1.

                j += 1
            i += 1

        return hot

    # get all chars used in train as well as test
    letters = chars(xtrain).union(chars(xtest))

    # determine the maximum text length. in this regime, we are not truncating
    # texts at all. in the paper texts are truncated.
    max_len = max_len or np.max(map(len, xtrain) + map(len, xtest))

    # distinct letters and classes in the dataaset
    vocab = ['�'] + sorted(list(letters))
    classes = sorted(list(set(ytrain)))

    # lookup tables for letters and classes. prepends padding char
    idx_letters = dict(((c, i) for c, i in zip(vocab, range(len(vocab)))))
    idx_classes = dict(((c, i) for c, i in zip(classes, range(len(classes)))))

    # dense integral indices
    xtrain = [[idx_letters[char] for char in list(line)] for line in xtrain]
    xtest = [[idx_letters[char] for char in list(line)] for line in xtest]
    ytrain = [idx_classes[line] for line in ytrain]

    # pad to fixed lengths
    xtrain = pad_sequences(xtrain, max_len)
    xtest = pad_sequences(xtest, max_len)

    xtrain = onehot(xtrain, max_len, len(idx_letters))
    ytrain = to_categorical(ytrain, num_classes=len(classes))
    xtest = onehot(xtest, max_len, len(idx_letters))

    return (
        xtrain,
        ytrain,
        xtest,
        vocab,
        max_len,
        len(classes))


def dbpedia(sample=None, dataset_source=DATA_LOCAL_PATH):
    """
    DBpedia is a crowd-sourced community effort to extract structured
    information from Wikipedia. The DBpedia ontology dataset is constructed by
    picking 14 nonoverlapping classes from DBpedia 2014. From each of these 14
    ontology classes, we randomly choose 40,000 training samples and 5,000
    testing samples. The fields we used for this dataset contain title and
    abstract of each Wikipedia article.
    """

    names = ['label', 'title', 'body']
    df_train = pd.read_csv(dataset_source + '/dbpedia/train.csv.gz', header=None, names=names)
    df_test = pd.read_csv(dataset_source + '/dbpedia/test.csv.gz', header=None, names=names)

    if sample:
        df_train = df_train.sample(frac=sample)
        df_test = df_test.sample(frac=sample)

    xtrain = df_train['body'].values
    ytrain = df_train['label'].values.astype('int32')
    xtest = df_test['body'].values

    return xtrain, ytrain, xtest
