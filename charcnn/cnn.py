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


def char_cnn(n_vocab, max_len, n_classes, weights_path=None):
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
    model.add(Dense(n_classes, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def compiled(model):
    "compile with chosen config"

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def fit(model, xtrain, ytrain, callbacks, batch=128, epochs=5, split=0.1):
    "fit the model"

    return model.fit(xtrain,
                     ytrain,
                     batch_size=batch,
                     epochs=epochs,
                     verbose=3,
                     validation_split=split,
                     callbacks=callbacks)


def predict(model, X):
    "predict probability, class for each instance"

    # predict probability of each class for each instance
    all_preds = model.predict(X)

    # for each instance get the index of the class with max probability
    idxs = np.argmax(all_preds, axis=1)

    # get the values of the highest probability for each instance
    preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]

    return np.array(preds), idxs


def preprocess(xtrain, ytrain, xtest, max_len=None):
    "preprocess and featurize the data"

    xtrain = [line.lower() for line in xtrain]
    xtest = [line.lower() for line in xtest]
    ytrain = [long(line) for line in ytrain]

    def chars(dataset):
        return reduce(
            lambda x, y: x.union(y),
            (set(line) for line in dataset))

    def onehot(chars_list, vocab_size):
        hot = np.zeros((len(chars_list), vocab_size))
        for i, char in enumerate(chars_list):
            if char != 0:
                hot[i, char] = 1.

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

    # onehot
    xtrain = np.array([onehot(line, len(idx_letters)) for line in xtrain])
    ytrain = onehot(ytrain, len(idx_classes))
    xtest = np.array([onehot(line, len(idx_letters)) for line in xtest])

    return (
        xtrain,
        ytrain,
        xtest,
        vocab,
        max_len,
        len(classes))


def main():
    "learn and predict"

    def lines(filename):
        with open(filename) as f:
            return f.read().splitlines()

    # read and prepare data
    xtrain, ytrain, xtest, vocab, max_len, n_classes = preprocess(
        lines('data/test/xtrain.txt'),
        lines('data/test/ytrain.txt'),
        lines('data/test/xtest.txt'))

    # compile model
    model = compiled(char_cnn(len(vocab), max_len, n_classes))

    # tensorflow specific, off
    callbacks = []
    if False:
        callbacks.append(TensorBoard(write_images=True))

    # fit model and log out to tensorboard
    history = fit(model, xtrain, ytrain, callbacks)
    model.save_weights('weights.h5')

    # evaluation
    print(history.history)
    with open('metrics.txt', 'w') as f:
        f.write(json.dumps(history.history, indent=1))

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
