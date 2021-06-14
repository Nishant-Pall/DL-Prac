from keras.models import Model
from keras.layers import LSTM, GRU, Input, Bidirectional
import numpy as np

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)

input_ = Input(shape=(T, D))
rnn = Bidirectional(GRU(M, return_state=True))
x = rnn(input_)

model = Model(inputs=input_, outputs=x)
o1, h1, h2 = model.predict(X)

print(f'o1 : ${o1}')
print(f'h1 : ${h1}')
# print(f'c1 : ${c1}')
print(f'h2 : ${h2}')
# print(f'c2 : ${c2}')
