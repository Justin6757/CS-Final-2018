#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

import numpy as np


def load_data():
    """
    Load word vectors from GloVe data set (https://nlp.stanford.edu/projects/glove/)
    :return: None
    """
    print(os.listdir('.'))
    os.chdir('../Data')
    print(os.listdir('.'))

    # Load word vectors

    word_list = np.load(
        'Word Vectors/GloVe_words.npy').tolist()  # Load as list; used for getting indices to access word vectors
    word_list = [word.decode('UTF-8') for word in word_list]  # Decode all words in UTF-8 format

    word_vectors = np.load('Word Vectors/GloVe_vectors.npy')  # Load as NumPy array


def tokenize(sentence):
    """
    Remove non-words and split sentence by word
    :param sentence:
    :return: list[str]
    """
    special_chars = re.compile(r'\W+')  # Replace all characters except letters and numbers
    tokens = re.sub(special_chars, ' ', sentence)  # Substitute special characters with spaces
    return [token for token in tokens.split() if token]


if __name__ == '__main__':
    load_data()
