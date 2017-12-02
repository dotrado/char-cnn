# Char-CNN

[![Build Status](https://travis-ci.org/purzelrakete/char-cnn.png?branch=master)](https://travis-ci.org/purzelrakete/char-cnn)

An implementation of

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)

## Installation

```bash
pip install char-cnn
```

## Usage

```python
from charcnn import cnn

xtrain, ytrain, xtest, vocab, max_len, n_classes = cnn.preprocess(
    open('data/test/xtrain.txt').read().splitlines(),
    open('data/test/ytrain.txt').read().splitlines(),
    open('data/test/xtest.txt').read().splitlines())

model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, n_classes))
history = cnn.fit(model, xtrain, ytrain, callbacks)

print(history.history)
```
