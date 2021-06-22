import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense, Lambda
from tensorflow.keras.models import Model


def initialize_base_model():

    input = Input(shape=(28, 28,))
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)

    return Model(inputs=input, outputs=x)


base_network = initialize_base_model()

input_a = Input(shape=(28, 28, ))
input_b = Input(shape=(28, 28, ))

vect_output_a = base_network(input_a)
vect_output_b = base_network(input_b)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


# output = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
#     [vect_output_a, vect_output_b])

# model = Model([input_a, input_b], output)

# contrastive_loss is made for siamese networks
# model.compile(loss='contrastive_loss', optimzier='rms')

# model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           epochs=20,
#           batch_size=128,
#           validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y)
#           )
