import os
import json
import pickle
import math

try:
  import unzip_requirements
except ImportError:
  pass

import numpy as np
from scipy import signal
from scipy.interpolate import interp2d

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import L1L2

def get_param_from_url(event, param_name):
    """
    Helper function to retrieve query parameters from a Lambda call. Parameters are passed through the
    event object as a dictionary.

    :param event: the event as input in the Lambda function
    :param param_name: the name of the parameter in the query string
    :return: the parameter value
    """
    params = event['queryStringParameters']
    return params[param_name]

def get_data_from_url(event):
    data = event['body']
    data = json.loads(data)
    return np.array(data)

def load_cnn():
    cnn_inputs = Input(shape=(150, 150, 1))

    conv2d_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(cnn_inputs)
    conv2d_2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv2d_1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv2d_2)
    dropout_1 = Dropout(0.25)(max_pool_1)
    conv2d_3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(dropout_1)
    conv2d_4 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv2d_3)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv2d_4)
    dropout_2 = Dropout(0.25)(max_pool_2)
    conv2d_5 = Conv2D(32, kernel_size=(3, 3), activation='relu')(dropout_2)
    conv2d_6 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv2d_5)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv2d_6)
    dropout_3 = Dropout(0.25)(max_pool_3)
    flatten = Flatten()(dropout_3)
    dense = Dense(128, activation='relu')(flatten)
    cnn_outputs = Dense(4, activation='softmax')(dense)

    cnn_model = Model(cnn_inputs, cnn_outputs)

    cnn_model.load_weights('/var/task/models/cnn_weights/cnn_weights')

    return cnn_model

def load_rnn():

    rnn_inputs = Input(shape=(None, 150))

    encoder_1 = LSTM(256,
                    activation='relu',
                    return_sequences=True,
                    kernel_regularizer=L1L2(l1=0.001, l2=0.001))(rnn_inputs)
    encoder_2 = LSTM(256,
                    activation='relu',
                    kernel_regularizer=L1L2(l1=0.001, l2=0.001))(encoder_1)
    rnn_outputs = Dense(4, activation='softmax')(encoder_2)

    rnn_model = Model(rnn_inputs, rnn_outputs)

    rnn_model.load_weights('/var/task/models/rnn_weights/rnn_weights')

    return rnn_model

def trim_silences(Sxx):
    """
    Removes leading and trailing silence from a spectrogram
    Silence is defined as a maximum amplitude less than 1% of
    the overall spectrogram maximum amplitude
    Returns the trimmed spectrogram
    """
    non_silent = list(np.max(Sxx, axis=0) > np.max(Sxx) * 0.01)
    start_id = non_silent.index(True)
    stop_id = len(non_silent) - 1 - non_silent[::-1].index(True)

    return Sxx[:, start_id:stop_id]

def split_spectrogram(Sxx, num_syl):
    """
    Divides the spectrogram of a two sylable word into its component spectrograms
    Returns a list [Spectrogram for sylable 1, Spectrogram for sylable 2]
    """
    half_search_interval = Sxx.shape[1]//(num_syl*4)
    split_points = [0]
    for i in range(1, num_syl):
        center = i * Sxx.shape[1]//num_syl
        next_split_point = center - half_search_interval + np.argmin(Sxx[:, center - half_search_interval:center + half_search_interval].max(axis=0))
        split_points.append(next_split_point)
    split_points.append(-1)
    spectrograms = []
    for i in range(len(split_points) - 1):
        spectrograms.append(trim_silences(Sxx[:, split_points[i]:split_points[i+1]]))
    return spectrograms

def predict(data, num_syl):
    """
    data: a numpy array containing raw audio data
    """

    cnn_model = load_cnn()
    rnn_model = load_rnn()

    spectrograms = []
    shapes = []

    if len(data.shape) > 1:
        data = data.T[0]

    fs = 44000
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=1024, nfft=1024*4, noverlap=900)
    Sxx = Sxx[:150]

    Sxx = trim_silences(Sxx)

    spectrograms = split_spectrogram(Sxx, num_syl)

    cnn_predictions = []
    rnn_predictions = []

    for Sxx in spectrograms:
        Sxx = Sxx / np.max(Sxx)
        Sxx = Sxx.T

        cnn_input = np.expand_dims(resize_spectrogram(Sxx, new_size=(150, 150)), -1)
        cnn_input = np.expand_dims(cnn_input, 0)

        rnn_input = resize_spectrogram(Sxx, new_size=(150, math.floor(Sxx.shape[1]/7.65)))
        rnn_input = np.expand_dims(rnn_input, 0)

        cnn_predictions.append(cnn_model.predict(cnn_input)[0].tolist())
        rnn_predictions.append(rnn_model.predict(rnn_input)[0].tolist())

    return spectrograms, cnn_predictions, rnn_predictions

def resize_spectrogram(Sxx, new_size):
    x = np.linspace(0, Sxx.shape[1]-1, Sxx.shape[1])
    y = np.linspace(0, Sxx.shape[0]-1, Sxx.shape[0])

    f = interp2d(x, y, Sxx, kind='cubic')

    return f(np.linspace(0, Sxx.shape[1]-1, new_size[0]), np.linspace(0, Sxx.shape[0]-1, new_size[1]))

def handler(event, context):
    cnn_model = load_cnn()
    rnn_model = load_rnn()

    num_syl = json.loads(get_param_from_url(event, 'num_syl'))
    data = get_data_from_url(event)

    spectrograms, cnn_predictions, rnn_predictions = predict(data, num_syl)

    body = {"cnn_predictions": json.dumps(cnn_predictions),
            "rnn_predictions": json.dumps(rnn_predictions),
            "spectrograms": json.dumps([Sxx.tolist() for Sxx in spectrograms])}

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
