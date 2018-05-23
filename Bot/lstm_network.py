#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import re
from random import randint

import numpy as np
import tensorflow as tf

from ids import MAX_SENTENCE_LENGTH

BATCH_SIZE = 24
LSTM_UNITS = 64
NUM_CLASSES = 2
ITERATIONS = 100000
NUM_DIMENSIONS = 300

IDS = np.load('idsMatrix.npy')


def load_data(train_model=False):
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

    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
    input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENTENCE_LENGTH])

    data = tf.Variable(tf.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH, NUM_DIMENSIONS]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(word_vectors, input_data)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]))
    bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # sess.restore(sess, tf.train.latest_checkpoint('models'))
    if train_model:
        train(labels, input_data, accuracy, loss, optimizer, sess, saver)


def train(labels, input_data, accuracy, loss, optimizer, sess, saver):
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(10):
        next_batch, next_batch_labels = get_training_batch()
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})

        # Write summary to Tensorboard
        if i % 50 == 0:
            summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
        writer.close()


def tokenize(sentence):
    """
    Remove non-words and split sentence by word
    :param sentence:
    :return: list[str]
    """
    special_chars = re.compile(r'\W+')  # Replace all characters except letters and numbers
    tokens = re.sub(special_chars, ' ', sentence)  # Substitute special characters with spaces
    return [token for token in tokens.split() if token]


def get_training_batch():
    labels = []
    arr = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH])
    for i in range(BATCH_SIZE):
        if i % 2 == 0:
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = IDS[num - 1:num]
    return arr, labels


def get_testing_batch():
    labels = []
    arr = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH])
    for i in range(BATCH_SIZE):
        num = randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = IDS[num - 1:num]
    return arr, labels


if __name__ == '__main__':
    load_data(True)
