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
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential

import data


def char_cnn(n_vocab, max_len, n_classes, weights_path=None):
    "See Zhang and LeCun, 2015"

    model = Sequential()
    model.add(Conv1D(256, 7, activation='relu', input_shape=(max_len, n_vocab)))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(256, 7, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
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


def main():
    "learn and predict"

    def lines(filename):
        with open(filename) as f:
            return f.read().splitlines()

    # read and prepare data
    xtrain, ytrain, xtest, vocab, max_len, n_classes = data.preprocess(
        lines('data/test/xtrain.txt'),
        lines('data/test/ytrain.txt'),
        lines('data/test/xtest.txt'))

    # compile model
    model = compiled(char_cnn(len(vocab), max_len, n_classes))

    # tensorflow specific, off
    callbacks = []
    if True:
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
