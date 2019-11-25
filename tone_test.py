import os
import json
import pickle

try:
  import unzip_requirements
except ImportError:
  pass

import numpy as np
from scipy import signal

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def load_model():
    encoder_inputs = Input(shape=(None, 86))
    encoder = LSTM(256, activation='relu', return_state=True)
    encoder_outputs, _, _ = encoder(encoder_inputs)
    dropout_layer = Dropout(0.5)
    dropout_output = dropout_layer(encoder_outputs)
    output_layer = Dense(5, activation='softmax')
    outputs = output_layer(dropout_output)

    model = Model(encoder_inputs, outputs)
    model.load_weights('/var/task/models/seq_weights')

    return model

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

def split_spectrogram(Sxx):
    """
    Divides the spectrogram of a two sylable word into its component spectrograms
    Returns a list [Spectrogram for sylable 1, Spectrogram for sylable 2]
    """
    q1 = Sxx.shape[1]//4
    q3 = 3*q1
    center = np.argmin(Sxx[:, q1:q3].max(axis=0))

    return trim_silences(Sxx[:, :q1+center]), trim_silences(Sxx[:, q1+center:])

def spectrogram(data, num_syl):
    """
    data: a numpy array containing raw audio data
    """

    spectrograms = []
    shapes = []

    if len(data.shape) > 1:
        data = data.T[0]

    fs = 44000
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=1024, nfft=1024*4)
    Sxx = Sxx[:86]

    Sxx = trim_silences(Sxx)

    if num_syl == 1:
        Sxx = Sxx / np.max(Sxx)
        Sxx = np.expand_dims(Sxx.T, 0)
        spectrograms.append(Sxx)
        shapes.append(json.dumps(Sxx.shape))

    elif num_syl == 2:
        Sxx1, Sxx2 = split_spectrogram(Sxx)

        Sxx1 = Sxx1 / np.max(Sxx1)
        Sxx1 = np.expand_dims(Sxx1.T, 0)
        spectrograms.append(Sxx1)
        shapes.append(json.dumps(Sxx1.shape))

        Sxx2 = Sxx2 / np.max(Sxx2)
        Sxx2 = np.expand_dims(Sxx2.T, 0)
        spectrograms.append(Sxx2)
        shapes.append(json.dumps(Sxx2.shape))

    return spectrograms, shapes

def handler(event, context):
    model = load_model()
    with open('/var/task/data.pkl', 'rb') as file:
        data = pickle.load(file)

    spectrograms, shapes = spectrogram(data, 2)

    predictions = []
    for Sxx in spectrograms:
        predictions.append(model.predict(Sxx)[0].tolist())

    return {"shapes": shapes,
            "predictions": predictions}
