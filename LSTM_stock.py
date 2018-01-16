"""Train and test LSTM classifier"""
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



class LSTM_stock(object):
    """
    an LSTM wrapper of keras lstm model that classify DGA

    """
    def __init__(self, input_shape,  max_epoch=1,batch_size=16, embadding_output_dim = 0, is_reg = True, classes = [-1,0,1], classes_count = 3,
                 deep_lstm_count=1,return_sequences = True, dropout_rate = 0.2):
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
        self.input_shape =input_shape
        self.is_reg = is_reg
        self.embadding_output_dim = embadding_output_dim
        self.batch_size = batch_size

        self.optimizer = 'adam'

        if is_reg:
            self.loss = 'mean_squared_error'
            self.activation = 'tanh'

        else:
            self.loss = 'categorical_crossentropy'
            self.activation = 'softmax'

        self.max_epoch = max_epoch

        self.classes_count = classes_count
        self.classes = classes
        self.model = None

        self.deep_lstm_count = deep_lstm_count
        self.dropout_rate = self.dropout_rate
        self.return_sequences = return_sequences

    def get_params(self):
        return dict(
            is_reg = self.is_reg,
            max_epoch = self.max_epoch,
            batch_size = self.batch_size,
            embadding_output_dim= self.embadding_output_dim,
            activation = self.activation,
            loss = self.loss,
            optimizer = self.optimizer,
            classes = self.classes,
            classes_count = self.classes_count,
            deep_lstm_count = self.deep_lstm_count

        )

    def __str__(self):
        return str(self.max_epoch) + "_" + str(self.batch_size) + "_" + str(self.embadding_output_dim) + "_" +\
            str(self.activation) + "_" + "_" \
                + str(self.classes_count) + "_" + str(self.deep_lstm_count)

    def save(self,path):

        params = self.get_params()
        print params
        pickle.dump(params, open(path + str(self) +'.p', "wb" )  )
        self.model.save(path + str(self) + '.h5')
        print 'saved lstm model at ' + path
        return path + str(self)

    def build_model(self):
        """Build LSTM model"""
        model = Sequential()

        lstm_input_dim = self.input_shape
        if self.embadding_output_dim >0:
            model.add(Embedding(self.max_features, self.embadding_output_dim, input_length=self.max_len))
            lstm_input_dim = self.embadding_output_dim


        # if self.deep_lstm_count > 0:
        #     return_sequences = True

        model.add(LSTM(self.embadding_output_dim,return_sequences=self.return_sequences))
        for layer in range(self.deep_lstm_count):
            return_sequences = True
            if layer == self.deep_lstm_count-1:
                return_sequences = self.return_sequences
            model.add(LSTM(self.embadding_output_dim,return_sequences=return_sequences))

        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.embadding_output_dim, activation=self.deep_activation))

        model.add(Dense(self.classes_count, activation=self.activation))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def fit(self, X, y,refit = False, use_history = False, use_tfboard = False):
        # print GPU or CPU
        print 'fitting lstm model...' + str(self)
        print(device_lib.list_local_devices())

        print "input data shape %s" % (X.shape,)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

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
                           verbose=1,
                           validation_data=(X_valid, y_valid),
                           callbacks = callbacks).history

        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.max_epoch,
                       shuffle=True,
                       verbose=1,
                       validation_data=(X_valid, y_valid),
                       callbacks=callbacks)

    def predict_proba(self, X):
        print 'predicting with ' + str(self) + ' on data shape - %s' % (X.shape,)

        tensorboard = TensorBoard(log_dir='/home/ise/Desktop/dga_lstm_v2/res/results/tensorBoard/logs',
                                    histogram_freq=0, write_graph=True, write_grads=False,
                                    write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None)

        tensorboard.set_model(self.model)

        return self.model.predict_proba(X,self.batch_size)

    def predict(self, X):
        print 'predicting with ' + str(self) + ' on data shape - %s' % (X.shape,)

        tensorboard = TensorBoard(log_dir='/home/ise/Desktop/dga_lstm_v2/res/results/tensorBoard/logs',
                                    histogram_freq=0, write_graph=True, write_grads=False,
                                    write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None)

        tensorboard.set_model(self.model)

        return self.model.predict(X,self.batch_size)
