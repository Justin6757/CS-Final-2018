import re
from os import listdir
from os.path import isfile, join

import numpy as np

MAX_SENTENCE_LENGTH = 100
NUM_FILES = 313217  # From metadata.py


def tokenize(sentence):
    """
    Remove non-words and split sentence by word
    :param sentence:
    :return: list[str]
    """
    special_chars = re.compile(r'\W+')  # Replace all characters except letters and numbers
    tokens = re.sub(special_chars, ' ', sentence)  # Substitute special characters with spaces
    return [token for token in tokens.split() if token]


# Load word vectors

word_list = np.load(
    'Word Vectors/GloVe_words.npy').tolist()  # Load as list; used for getting indices to access word vectors
word_list = [word.decode('UTF-8') for word in word_list]  # Decode all words in UTF-8 format

word_vectors = np.load('Word Vectors/GloVe_vectors.npy')  # Load as NumPy array

path = '/Users/MacBook/Documents/LSTM Data/'

negative_files = [f'{path}/Negative/' + f for f in listdir(f'{path}/Negative/')
                  if isfile(join(f'{path}/Negative/', f))]
non_negative_files = [f'{path}/Non-negative/' + f for f in listdir(f'{path}/Non-negative/')
                      if isfile(join(f'{path}/Non-negative/', f))]

# ids = np.zeros((NUM_FILES, MAX_SENTENCE_LENGTH), dtype='int32')

# file_count = 0
# for n in negative_files:
#    with open(n, "r") as f:
#        index = 0
#        line = f.readline()
#        cleaned = tokenize(line)
#        for word in cleaned:
#            try:
#                ids[file_count][index] = word_list.index(word)
#            except ValueError:
#                ids[file_count][index] = 399999 #Vector for unknown words
#            index += 1
#            if index >= MAX_SENTENCE_LENGTH:
#                break
#        file_count += 1

# for nn in non_negative_files:
#    with open(nn, "r") as f:
#        index = 0
#        line = f.readline()
#        cleaned = tokenize(line)
#        for word in cleaned:
#            try:
#                ids[file_count][index] = word_list.index(word)
#            except ValueError:
#                ids[file_count][index] = 399999 #Vector for unknown words
#            index += 1
#            if index >= MAX_SENTENCE_LENGTH:
#                break
#        file_count += 1

# np.save('idsMatrix', ids)
