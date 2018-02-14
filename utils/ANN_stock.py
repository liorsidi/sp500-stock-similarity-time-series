"""Train and test LSTM classifier"""
import seq2seq
from keras.callbacks import EarlyStopping, History, TensorBoard
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.cross_validation import train_test_split
from tensorflow.python.client import device_lib
from keras.utils import np_utils

import keras
from sklearn.metrics import roc_auc_score


class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class ANN_stock(object):
    """
    an LSTM wrapper of keras lstm model that classify DGA

    """

    def __init__(self, input_shape=10, max_epoch=10, batch_size=16, dim=3, is_reg=True,
                 is_mlp=False,
                 is_seqtoseq=False,
                 is_LSTM=True,
                 output_len=3,
                 activation='relu',
                 dropout_rate=0.2,
                 optimizer='adam',
                 classes=[-1, 0, 1]
                 ):
        """
        a constructor for lstm model for predicting DGA, the default configuration support binary classifications.

        :param max_epoch: number of epochs to train the model
        :param batch_size: the batch size in training
        :param embadding_output_dim: the empadding layers neurons
        :param dropout_rate: the dropout rate before the output layer
        :param activation: the activation function for the output layer
        :param loss: the loss function to optimize - depends on the output layer
        :param optimizer: epends on the output layer
        :param valid_chars: the DGA chars supported by the models - chars that were in the training dataset,
                the input vectors is build with hot-code of chars
        :param max_features: the length of valid_cars (dimention of the input)
        :param max_len: the maximum length of a domain name, the size of the sequence
        :param is_multiclass: is the class is binary class (benign =0, DGA = 1) with score output of probability to DGA
         or is it multicalss where each DGA is class and the output is vector of probabilities
        :param classes: classes naming convention
        :param classes_count: size of the output vector
        :param data_is_ready: is the data to fit is already transformed
        """

        self.input_shape = input_shape

        self.max_epoch = max_epoch
        self.batch_size = batch_size

        self.is_reg = is_reg

        self.optimizer = optimizer

        if is_reg:
            self.loss = 'mean_squared_error'
            self.activation = 'tanh'
            self.classes = None
            self.output_dim = 1
        else:
            self.loss = 'categorical_crossentropy'
            self.activation = 'softmax'
            self.classes = classes
            self.output_dim = len(classes)

        self.output_len = output_len
        self.model = None
        self.is_mlp = is_mlp
        self.is_seqtoseq= is_seqtoseq
        self.is_LSTM= is_LSTM
        self.hidden_dim= dim
        self.dropout_rate= dropout_rate
        self.activation= activation

    def get_params(self):
        return dict(
            is_reg=self.is_reg,
            max_epoch=self.max_epoch,
            batch_size=self.batch_size,
            activation=self.activation,
            loss=self.loss,
            optimizer=self.optimizer,
            classes=self.classes,


        )#TODO

    def __str__(self):
        return str(self.max_epoch) + "_" + str(self.batch_size) + "_" #TODO
    def save(self, path):
        params = self.get_params()
        print params
        pickle.dump(params, open(path + str(self) + '.p', "wb"))
        self.model.save(path + str(self) + '.h5')
        print 'saved lstm model at ' + path
        return path + str(self)

    def build_model(self):
        """Build LSTM model"""
        model = Sequential()

        if self.is_mlp:
            model.add(Dense(self.hidden_dim))
        elif self.is_seqtoseq:
            model.add(LSTM(self.hidden_dim, return_sequences=True, input_shape=self.input_shape))
            model.add(seq2seq.SimpleSeq2Seq(self.output_dim, self.output_len, hidden_dim=self.hidden_dim))
        elif self.is_LSTM:
            model.add(LSTM(self.hidden_dim,return_sequences=True, input_shape=self.input_shape))
            model.add(LSTM(self.output_dim, return_sequences=False))

        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.output_dim, activation=self.activation))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def fit(self, X, y, refit=False, use_history=False, use_tfboard=False):
        # print GPU or CPU
        print 'fitting lstm model...' + str(self)
        print(device_lib.list_local_devices())

        print "input data shape %s" % (X.shape,)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.input_shape = list(X.shape)
        self.input_shape[0] = None
        self.input_shape = tuple(self.input_shape[1:])
        if self.model is None:
            self.model = self.build_model()
        else:
            if not refit:
                return

        self.model.validation_data = (X_valid, y_valid)

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=0, mode='auto')]
        if use_tfboard:
            tensorboard = TensorBoard(log_dir='/home/ise/Desktop/dga_lstm_v2/res/results/tensorBoard/logs',
                                      histogram_freq=0, write_graph=True, write_grads=False,
                                      write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                      embeddings_metadata=None)
            callbacks.append(tensorboard)
        if use_history:
            callbacks.append(Histories())
            return self.model.fit(X_train, y_train,
                                  batch_size=self.batch_size,
                                  epochs=self.max_epoch,
                                  shuffle=True,
                                  #verbose=1,
                                  validation_data=(X_valid, y_valid),
                                  callbacks=callbacks).history

        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.max_epoch,
                       shuffle=True,
                      # verbose=1,
                       validation_data=(X_valid, y_valid),
                       callbacks=callbacks)

    def predict_proba(self, X):
        print 'predicting with ' + str(self) + ' on data shape - %s' % (X.shape,)

        tensorboard = TensorBoard(log_dir='/home/ise/Desktop/dga_lstm_v2/res/results/tensorBoard/logs',
                                  histogram_freq=0, write_graph=True, write_grads=False,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # tensorboard.set_model(self.model)

        return self.model.predict_proba(X, self.batch_size)

    def predict(self, X):
        print 'predicting with ' + str(self) + ' on data shape - %s' % (X.shape,)

        tensorboard = TensorBoard(log_dir='/home/ise/Desktop/dga_lstm_v2/res/results/tensorBoard/logs',
                                  histogram_freq=0, write_graph=True, write_grads=False,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # tensorboard.set_model(self.model)

        return self.model.predict(X, self.batch_size)

    @property
    def __name__(self):
        return "ANN_stock"
