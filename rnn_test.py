from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import LSTM, GRU, Input
import numpy as np

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)

print(f'X : {X}')


def lstm1():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)

    print('LSTM 1')

    print(f'o:{o}')
    print(f'h:{h}')
    print(f'c:{c}')


def lstm2():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)

    print('LSTM 2')

    print(f'o:{o}')
    print(f'h:{h}')
    print(f'c:{c}')


def gru1():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)

    print('GRU 1')

    print(f'o:{o}')
    print(f'h:{h}')


def gru2():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)

    print('GRU 2')

    print(f'o:{o}')
    print(f'h:{h}')


lstm1()
