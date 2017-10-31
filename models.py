#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
An implementation of

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)

"""

import numpy as np
import json

from keras.callbacks import TensorBoard
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import keras.utils.np_utils as utils


def char_cnn(n_vocab, max_len, n_classes, weights_path = None):
    "See Zhang and LeCun, 2015"

    model = Sequential()
    model.add(Convolution1D(256, 7, activation='relu', input_shape=(max_len, n_vocab)))
    model.add(MaxPooling1D(3))

    model.add(Convolution1D(256, 7, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Convolution1D(256, 3, activation='relu'))
    model.add(Convolution1D(256, 3, activation='relu'))
    model.add(Convolution1D(256, 3, activation='relu'))
    model.add(Convolution1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def compiled(model):
    "compile with chosen config"

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    return model


def fit(model, xtrain, ytrain, callbacks, batch = 128, epochs = 20, split = 0.1):
    "fit the model"

    return model.fit(xtrain, ytrain,
              batch_size = batch,
              epochs = epochs,
              verbose = 3,
              validation_split = split,
              callbacks = callbacks)


def predict(model, X):
    "predict probability, class for each instance"

    # predict probability of each class for each instance
    all_preds = model.predict(X)

    # for each instance get the index of the class with max probability
    idxs = np.argmax(all_preds, axis = 1)

    # get the values of the highest probability for each instance
    preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]

    return np.array(preds), idxs


def data(xtrain_file, ytrain_file, xtest_file):
    "load and featurize the data"

    def lines(name):
        with open(name) as f:
            return f.read().splitlines()

    def chars(name):
        with open(name) as f:
            return set(f.read())

    def onehot(vals, idx):
        hot = np.zeros((len(vals), len(idx)))
        hot[range(len(vals)), vals] = 1.0
        hot[:, 0] = 0.0 # padding character

        return hot

    xtrain = lines(xtrain_file)
    ytrain = lines(ytrain_file)
    xtest =  lines(xtest_file)

    # vocab on all instances
    letters = reduce(lambda s, t: s.union(t), [chars(fname) for fname in [
        xtrain_file,
        xtest_file]])

    # some stats
    max_len = np.max(map(len, xtrain) + map(len, xtest))
    n_classes = len(set(ytrain))

    # lookup tables for letters and classes. prepends padding char
    vocab = ['�'] + sorted(list(letters))
    idx_letters = dict(((c, i) for c, i in zip(vocab, range(len(vocab)))))
    idx_classes = dict(zip(range(n_classes), range(n_classes)))

    # dense integral indices
    xtrain = [map(idx_letters.get, list(l)) for l in xtrain]
    xtest =  [map(idx_letters.get, list(l)) for l in xtest]
    ytrain = [long(c) for c in ytrain]

    # pad to fixed lengths
    xtrain = pad_sequences(xtrain, max_len)
    xtest  = pad_sequences(xtest, max_len)

    # featurize
    xtrain = np.array([onehot(l, idx_letters) for l in xtrain])
    ytrain = onehot(ytrain, idx_classes)
    xtest =  np.array([onehot(l, idx_letters) for l in xtest])

    return (
        xtrain,
        ytrain,
        xtest,
        vocab,
        max_len,
        n_classes)


def main():
    "learn and predict"

    # read and prepare data
    xtrain, ytrain, xtest, vocab, max_len, n_classes = data(
        'data/xtrain.txt',
        'data/ytrain.txt',
        'data/xtest.txt')

    # compile model
    model = compiled(char_cnn(len(vocab), max_len, n_classes))

    # tensorflow specific, off
    callbacks = []
    if False:
        callbacks.append(TensorBoard(write_images = True))

    # fit model and log out to tensorboard
    history = fit(model, xtrain, ytrain, callbacks)
    model.save_weights('weights.h5')

    # evaluation
    print(history.history)
    with open('metrics.txt', 'w') as f:
        f.write(json.dumps(history.history, indent = 1))

    # prediction
    _, ytest = predict(model, xtest)
    with open('ytest.txt', 'w') as f:
        f.write('\n'.join(map(str, ytest)))

    # test set predictions for inspection
    _, ytrain_predicted = predict(model, xtrain)
    with open('ytrain.predicted.txt', 'w') as f:
        f.write('\n'.join(map(str, ytrain_predicted)))


if __name__ == "__main__":
    main()
