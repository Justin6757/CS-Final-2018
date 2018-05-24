#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import time
import os
import re
from random import randint

import numpy as np
import tensorflow as tf

from ids import MAX_SENTENCE_LENGTH, NUM_FILES, NUM_NEGATIVE, NUM_NON_NEGATIVE

BATCH_SIZE = 24
LSTM_UNITS = 64
NUM_CLASSES = 2
ITERATIONS = 100000
NUM_DIMENSIONS = 300


def create(train_model=False):
    return Model(train_model)


class Model:
    def __init__(self, train_model):
        self.IDS = np.load('idsMatrix.npy')

        # Load word vectors

        self.word_list = np.load(
            'Word Vectors/GloVe_words.npy').tolist()  # Load as list; used for getting indices to access word vectors
        self.word_list = [word.decode('UTF-8') for word in self.word_list]  # Decode all words in UTF-8 format

        self.word_vectors = np.load('Word Vectors/GloVe_vectors.npy')  # Load as NumPy array

        os.chdir('../Data')
        print(os.listdir('.'))

        tf.reset_default_graph()

        self.labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
        self.input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENTENCE_LENGTH])

        tf.Variable(tf.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH, NUM_DIMENSIONS]), dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.word_vectors, self.input_data)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]))
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        self.prediction = (tf.matmul(last, weight) + bias)

        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        self.session = tf.InteractiveSession()
        saver = tf.train.Saver()

        if train_model:
            self.train(self.labels, self.input_data, self.accuracy, loss, optimizer, self.session, saver)
        else:
            saver.restore(self.session, tf.train.latest_checkpoint('Models'))

    def to_matrix(self, sentence):
        sentence_matrix = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH], dtype='int32')
        tokenized = tokenize(sentence)
        for index, word in enumerate(tokenized):
            try:
                sentence_matrix[0, index] = self.word_list.index(word)
            except ValueError:
                sentence_matrix[0, index] = 399999  # Vector for unkown words
        return sentence_matrix

    def predict(self, sentence):
        sentence_matrix = self.to_matrix(sentence)
        return self.session.run(self.prediction, {self.input_data: sentence_matrix})[0]

    def train(self, labels, input_data, accuracy, loss, optimizer, sess, saver):
        start = time.time()

        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        merged = tf.summary.merge_all()
        logdir = 'Tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for i in range(1, ITERATIONS + 1):
            print(f'Batch {i}')
            next_batch, next_batch_labels = self.get_training_batch()
            sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})

            # Write summary to Tensorboard
            if i % 50 == 0:
                summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
                writer.add_summary(summary, i)

            # Save the network every 10,000 training iterations
            if i % 10000 == 0:
                save_path = saver.save(sess, 'Models/pretrained_lstm.ckpt', global_step=i)
                print(f'Saved to {save_path}')
            writer.close()

            print(f'Took {(time.time() - start) / 60} minutes')

    def test_model(self):
        iterations = 10
        for i in range(iterations):
            next_batch, next_batch_labels = self.get_testing_batch()
            print('Accuracy for this batch:',
                  (self.session.run(self.accuracy,
                                    {self.input_data: next_batch, self.labels: next_batch_labels})) * 100)

    def get_training_batch(self):
        labels = []
        arr = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH])
        for i in range(BATCH_SIZE):
            if i % 2 == 0:
                num = randint(1, NUM_NEGATIVE)  # Add negative file to train
                labels.append([1, 0])  # Negative label
            else:
                num = randint(NUM_NEGATIVE, NUM_FILES)  # Add non-negative file to train
                labels.append([0, 1])  # Non-negative label
            arr[i] = self.IDS[num - 1]
        return arr, labels

    def get_testing_batch(self):
        labels = []
        arr = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH])
        for i in range(BATCH_SIZE):
            num = randint(NUM_FILES // 2, NUM_FILES // 2 + 2000)
            if num <= NUM_FILES // 2 + 1000:
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            arr[i] = self.IDS[num - 1:num]
        return arr, labels


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
    create(True)
