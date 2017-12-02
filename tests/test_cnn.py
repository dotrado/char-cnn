import os

import numpy as np
import pandas as pd

from charcnn import cnn


class TestModel:
    "Keras model"

    def test_constructs_and_compiles_char_cnn(self):
        n_vocab, max_len, n_classes = 83, 453, 12
        model = cnn.compiled(cnn.char_cnn(n_vocab, max_len, n_classes))
        assert model.built, "model not built"

    def xxx_test_training(self):
        xtrain, ytrain, xtest, vocab, max_len, n_classes = cnn.preprocess(
            lines('data/test/xtrain.txt'),
            lines('data/test/ytrain.txt'),
            lines('data/test/xtest.txt'))

        model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, n_classes))
        history = cnn.fit(model, xtrain, ytrain, callbacks)


class TestPipeline:
    "Data and feaures"

    def test_preprocesses_data(self):
        xtrain, ytrain, xtest, vocab, max_len, n_classes = cnn.preprocess(
            lines('data/test/xtrain.txt'),
            lines('data/test/ytrain.txt'),
            lines('data/test/xtest.txt'))

        # 'hello', padded to max_len = 10 with n_vocab = 18.
        got = xtrain[4].astype(np.float)
        want = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])

        assert np.array_equal(got, want)
        assert n_classes == 2
        assert max_len == 10


# Testing utilty functions

def lines(filename):
    with open(filename) as f:
        return f.read().splitlines()
