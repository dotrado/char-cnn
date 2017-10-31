import nose
import numpy as np
import os
import models


class TestModel:
    "Keras model"

    def test_constructs_and_compiles_char_cnn(self):
        n_vocab, max_len, n_classes = 83, 453, 12
        model = models.compiled(models.char_cnn(n_vocab, max_len, n_classes))
        assert model.built, "model not built"

    def test_fits_char_cnn(self):
        xtrain, ytrain, xtest, vocab, max_len, n_classes = models.data(
            'test/xtrain.txt',
            'test/ytrain.txt',
            'test/xtest.txt')

        model = models.compiled(models.char_cnn(len(vocab), max_len, n_classes))
        history = fit(model, xtrain, ytrain)

        assert history.history['acc'] > 0.0


class TestPipeline:
    "Data and feaures"

    def test_loads_data(self):
        xtrain, ytrain, xtest, vocab, max_len, n_classes = models.data(
            'test/xtrain.txt',
            'test/ytrain.txt',
            'test/xtest.txt')

        # 'hello', padded to max_len = 10 with n_vocab = 18.
        got = xtrain[4]
        want = np.array([
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
        ])

        assert np.array_equal(got, want)
