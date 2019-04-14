# encoding: utf-8
"""
@author: leo 
@version: 1.0
@license: Apache Licence
@file: slu_model.py
@time: 2017/11/19 下午4:59

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import logging
from shutil import copyfile
import json
import time
import os
import h5py
import random

import numpy as np
import keras.layers as layers

from nlp_toolkit.processor.slu_processor import SLUProcessor

from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
from keras.callbacks import TensorBoard, ModelCheckpoint, RemoteMonitor, Callback, LambdaCallback

current_path = os.path.dirname(os.path.abspath(__file__))


base_conf = {
    'arch': 'blstm',
    'out': '',
    'hidden_size': 150,
    'learning_rate': 1e-2,
    'epochs': 10,
    'batch_size': 800,
    'sgdtype': 'adam',
    'decay_rate': 0.0,
    'activation_func': 'tanh',
    'smooth_eps': 1e-8,
    'momentum': 0.1,
    'init_type': 'glorot_uniform',
    'beta1': 0.9,
    'beta2': 0.999,
    'rho': 0.9,
    'dropout': True,
    'dropout_ratio': 0.5,
    'shuffle': True,
    'generator': True
}


class JointSLUModel(object):
    def __init__(self, data_path, prosessor=None, **conf):
        self.data_path = data_path
        if self.data_path is None:
            logging.error('no dataset !!!')
        base_conf.update(conf)
        conf = base_conf

        self.conf = conf
        self.arch = conf['arch']
        self.out = conf.get('out', '')
        if self.out == '':
            self.out = self.arch

        # size of hidden layer of neurons
        self.hidden_size = conf['hidden_size']
        self.learning_rate = conf['learning_rate']

        self.epochs = conf['epochs']
        self.batch_size = conf['batch_size']

        # SGD type: sgd/rmsprop/adagrad/adadelta/adam/adamax
        self.update_f = conf['sgdtype']
        self.decay_rate = conf['decay_rate']

        # activation function for hidden units: sigmoid]
        self.activation = conf['activation_func']

        # epsilon smoothing for rmsprop/adagrad/adadelta/adam/adamax
        self.smooth_eps = conf['smooth_eps']
        self.momentum = conf['momentum']

        # weight initialization function: glorot_uniform/glorot_normal/he_uniform/he_normal
        self.init_type = conf['init_type']
        self.beta1 = conf['beta1']
        self.beta2 = conf['beta2']
        self.rho = conf['rho']
        self.dropout = conf['dropout']
        self.dropout_ratio = conf['dropout_ratio']

        self.shuffle = conf['shuffle']
        self.use_generator = conf['generator']
        # settings form data_set
        if prosessor is None:
            prosessor = SLUProcessor(data_path=self.data_path)
            self.prosessor = prosessor
        else:
            self.prosessor = prosessor

        self.prosessor.read_base_info()
        self.prosessor.read_embeddings_matrix()
        self.embeddings_matrix = prosessor.embeddings_matrix
        self.word2idx = prosessor.words2idx
        self.embedding_size = prosessor.embedding_size
        self.output_vocab_size = len(prosessor.slots2idx)
        self.time_length = prosessor.padding_size
        self.model_base_path = os.path.join(
            prosessor.model_path, 'trained_models', self.out)
        self.model_path = self.model_base_path + '_model.h5'

    def build(self):
        raw_current = Input(shape=(self.time_length,), dtype='int32')
        if len(self.embeddings_matrix) == 0:
            embedding = Embedding(input_dim=len(self.word2idx),
                                  output_dim=self.embedding_size,
                                  input_length=self.time_length)
        else:
            embedding = Embedding(len(self.embeddings_matrix),
                                  self.embedding_size,
                                  weights=[self.embeddings_matrix],
                                  trainable=False)
        current = embedding(raw_current)
        # set optimizer
        if self.update_f == 'sgd':
            opt_func = SGD(lr=self.learning_rate,
                           momentum=self.momentum, decay=self.decay_rate)
        elif self.update_f == 'rmsprop':
            opt_func = RMSprop(lr=self.learning_rate,
                               rho=self.rho, epsilon=self.smooth_eps)
        elif self.update_f == 'adagrad':
            opt_func = Adagrad(lr=self.learning_rate, epsilon=self.smooth_eps)
        elif self.update_f == 'adadelta':
            opt_func = Adadelta(lr=self.learning_rate,
                                rho=self.rho, epsilon=self.smooth_eps)
        elif self.update_f == 'adam':
            opt_func = Adam(lr=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2,
                            epsilon=self.smooth_eps)
        elif self.update_f == 'adamax':
            opt_func = Adamax(lr=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2,
                              epsilon=self.smooth_eps)
        else:
            sys.stderr.write("Invalid optimizer.\n")
            exit()

        # Vallina RNN (LSTM, SimpleRNN, GRU)
        # Bidirectional-RNN (LSTM, SimpleRNN, GRU)
        if self.arch == 'lstm' or self.arch == 'rnn' or self.arch == 'gru' \
                or self.arch == 'blstm' or self.arch == 'brnn' or self.arch == 'bgru':
            if 'rnn' in self.arch:
                forward = SimpleRNN(self.hidden_size, return_sequences=True, activation=self.activation,
                                    kernel_initializer=self.init_type)(current)
                backward = SimpleRNN(self.hidden_size, return_sequences=True, activation=self.activation,
                                     go_backwards=True, kernel_initializer=self.init_type)(current)
            elif 'gru' in self.arch:
                forward = GRU(self.hidden_size, return_sequences=True, init=self.init_type,
                              activation=self.activation)(current)
                backward = GRU(self.hidden_size, return_sequences=True, init=self.init_type,
                               activation=self.activation, go_backwards=True)(current)
            elif 'lstm' in self.arch:
                forward = LSTM(self.hidden_size,
                               return_sequences=True,
                               activation=self.activation,
                               kernel_initializer=self.init_type)(current)

                backward = LSTM(self.hidden_size,
                                return_sequences=True,
                                activation=self.activation,
                                go_backwards=True,
                                kernel_initializer=self.init_type)(current)
            if 'b' in self.arch:
                tagger = layers.concatenate([forward, backward])
            else:
                tagger = forward

            if self.dropout:
                tagger = Dropout(self.dropout_ratio)(tagger)
            prediction = TimeDistributed(
                Dense(self.output_vocab_size, activation='softmax'))(tagger)
            self.model = Model(inputs=raw_current, outputs=prediction)
            self.model.compile(
                loss='categorical_crossentropy', optimizer=opt_func)

        # 2-Stacked Layered RNN (LSTM, SimpleRNN, GRU)
        elif self.arch == '2lstm' or self.arch == '2rnn' or self.arch == '2gru':
            model = Sequential()
            model.add(embedding)
            if self.arch == '2lstm':
                basic_model = LSTM(self.hidden_size, return_sequences=True,
                                   input_shape=(self.time_length,
                                                self.embedding_size),
                                   init=self.init_type,
                                   activation=self.activation)
                stack_model = LSTM(self.hidden_size, return_sequences=True,
                                   input_shape=(self.time_length,
                                                self.hidden_size),
                                   init=self.init_type,
                                   activation=self.activation)
            elif self.arch == '2rnn':
                basic_model = SimpleRNN(self.hidden_size, return_sequences=True,
                                        input_shape=(
                                            self.time_length, self.embedding_size),
                                        init=self.init_type,
                                        activation=self.activation)
                stack_model = SimpleRNN(self.hidden_size, return_sequences=True,
                                        input_shape=(
                                            self.time_length, self.hidden_size),
                                        init=self.init_type,
                                        activation=self.activation)
            else:
                basic_model = GRU(self.hidden_size, return_sequences=True,
                                  input_shape=(self.time_length,
                                               self.embedding_size),
                                  init=self.init_type,
                                  activation=self.activation)
                stack_model = GRU(self.hidden_size, return_sequences=True,
                                  input_shape=(self.time_length,
                                               self.hidden_size),
                                  init=self.init_type,
                                  activation=self.activation)
            model.add(basic_model)
            if self.dropout:
                model.add(Dropout(self.dropout_ratio))
            model.add(stack_model)
            model.add(TimeDistributed(Dense(self.output_vocab_size)))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=opt_func)
            self.model = model
        else:
            sys.stderr.write("Invalid arch.\n")
            exit()

        # save model descriptions
        self.model.summary()

    def load_weight(self, h5file=None):
        if h5file == None:
            h5file = self.model_path
        self.model.load_weights(h5file)

    def visualize(self):
        model_data_path = os.path.join(self.data_path, self.out+'.png')
        try:
            plot_model(self.model, to_file = model_data_path + "_model.png")
        except Exception as e:
            logging.error(str(e))


if __name__ == "__main__":
    print("Hello world")
    m = JointSLUModel(data_path='/Users/leo/Desktop/ailab/ai_nlp_api/app/data/model/2017.10.26')
    m.build()


