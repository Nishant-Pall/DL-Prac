# USING LOSS FUNCTIONS

# model.compile(loss='mse', optmizers = 'sgd')

# OR

# here you can pass in params that can help with hyperparameter tuning
# from tensorflow.keras.losses import mean_squared_error
# model.compile(loss=mean_squared_error(param=value), optmizer='sgd')

from tensorflow.keras.losses import Loss
import tensorflow as tf


# def my_huber_loss(y_true, y_pred):
#     threshold = 1
#     error = y_true - y_pred
#     is_small_error = tf.abs(error) <= threshold
#     small_error_loss = tf.square(error) / 2
#     big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
# #     		  Boolean to check   value if True     value if False
#     return tf.where(is_small_error, small_error_loss, big_error_loss)


# model.compile(optmizer='sgd', loss=my_huber_loss)

# we use wrapper because the model expects loss to only accept y_true,
# and y_pred. wrapper helps in passing more hyperparameters
def my_huber_loss_with_threshold(threshold):
    def my_huber_loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
        #     		  Boolean to check   value if True     value if False
        return tf.where(is_small_error, small_error_loss, big_error_loss)
    return my_huber_loss


# model.compile(optmizer='sgd', loss=my_huber_loss_with_threshold(threshold=1))

# Inheriting from Loss class
class MyHuberLoss(Loss):
    # define threshold as class value and give it initial value
    threshold = 1
# initializes

    def __init__(self,  threshold):
        super().__init__()
        self.threshold = threshold

# call function is executed when object is instantiated from the class

    def __call__(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * \
            (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)

# model.compile(optmizer='sgd', loss=MyHuberLoss(threshold=1))
