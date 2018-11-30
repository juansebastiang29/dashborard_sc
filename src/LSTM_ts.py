import pandas as pd
import os
import sys
from datetime import datetime
from datetime import timedelta
import numpy as np
from pandas import read_csv
import math
import h5py

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import callbacks
from sklearn import metrics
from keras.backend import clear_session
from keras.backend import get_session
from keras.backend import set_session
import tensorflow as tf

# Set a random seed for getting reproducible results
from numpy.random import seed
seed(29)
from tensorflow import set_random_seed
set_random_seed(1991)

# arreglo de matriz
def create_sequence(serie):
    num_feat = 1
    serie_s = serie.copy()

    for i in range(num_feat):
        serie = pd.concat([serie, serie_s.shift(-(i + 1))], axis=1)
    serie.dropna(inplace=True)
    return serie


def train_test_reshape(serie, product_name ):
    # dividir la data
    train_size = int(len(serie) * 0.7)
    test_size = len(serie) - train_size
    train, test = serie.iloc[0:train_size, :], serie.iloc[train_size:len(serie), :]
    train_index, test_index = serie.iloc[0:train_size, :].index, \
                              serie.iloc[train_size:len(serie), :].index

    trainX, trainY = train.iloc[:, :-1].values, train.iloc[:, -1].values
    testX, testY = test.iloc[:, :-1].values, test.iloc[:, -1].values

    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return {'serie': serie,
            'X_train': trainX, 'Y_train': trainY,
            'X_test': testX, 'Y_test': testY,
            'train_index': train_index,
            'test_index': test_index,
            'product_name': product_name}


def reset_weights(model):
    session = get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def train_model(data_trainig):


    # sess = get_session()
    # set_session(sess)

    var = tf.Variable(0)
    with tf.Session() as session:
        # configure early stopping
        call_back_ = callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0.00001,
                                             patience=5,
                                             verbose=1,
                                             mode='auto',
                                             baseline=None)

        session.run(tf.global_variables_initializer())
        # model architecture
        model = Sequential()
        model.add(LSTM(32, input_shape=(1, 1),return_sequences=True))
        model.add(LSTM(32))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='adam')
        model.summary()
        history = model.fit(data_trainig['X_train'], data_trainig['Y_train'],
                            validation_data=(data_trainig['X_test'], data_trainig['Y_test']),
                            epochs=200,
                            batch_size=1,
                            verbose=1,
                            callbacks=[call_back_])

        save_model(filepath="../Output/model_%s.h5" % data_trainig['product_name'],
                   model=model,
                   overwrite=True,
                   include_optimizer=False)

        return history

def compute_mae(model,data_trainig):
    trainPredict = model.predict(data_trainig['X_train'])
    testPredict = model.predict(data_trainig['X_test'])
    return metrics.mean_absolute_error(testPredict, data_trainig['Y_test']), testPredict, trainPredict

def forecasting_7_days(model, data_trainig):

    #index
    new_index = pd.date_range(data_trainig['test_index'].tolist()[-1], periods=12).tolist()

    # forecasting 7 days
    forcast_7_days = []
    #predict the first new day
    moving_test_window = [data_trainig['X_test'][-1,:].tolist()]
    moving_test_window = np.array(moving_test_window)

    # forecasting
    for i in range(0, 12):
        prediction_one_step = model.predict(moving_test_window)
        print("This is the prediction", prediction_one_step)
        forcast_7_days.append(prediction_one_step[0,0])
        prediction_one_step = prediction_one_step.reshape(1,1,1)
        moving_test_window = np.concatenate((moving_test_window[:,1:,:],prediction_one_step), axis=1)
    tf.reset_default_graph()
    clear_session()
    print(forcast_7_days)

    return forcast_7_days, new_index