from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Dense, Flatten


def build_model_with_function():
    # Defining the inputs
    input = Input(shape=(28, 28))

    # Defin the Model
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation="softmax")(x)

    # Define the Output
    model = Model(inputs=input, outputs=predictions)
    return model
