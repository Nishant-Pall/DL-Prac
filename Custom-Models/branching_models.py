import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.models import Model


layer1 = Dense(32)
layer2_1 = Dense(32)(layer1)
layer2_2 = Dense(32)(layer1)
layer2_3 = Dense(32)(layer1)
layer2_4 = Dense(32)(layer1)

merge = Concatenate([layer2_1, layer2_2, layer2_3, layer2_4])

# lets say num_features is 20
num_features = 20

input_layer = Input(shape=(num_features, ))
first_dense = Dense(units=128, activation='relu')(input_layer)
second_dense = Dense(units=128, activation='relu')(first_dense)

y1_output = Dense(units='1', name='y1_output')(second_dense)
third_dense = Dense(units=128, activation='relu')(second_dense)
y2_output = Dense(units='1', name='y1_output')(third_dense)


model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

# compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'y_1_output': 'mse',
                    'y_2_output': "mae"
                    },
              metircs={'y_1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y_2_output': tf.keras.metrics.RootMeanSquaredError()
                       }
              )
