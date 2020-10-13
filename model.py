"""
Script containing the implementation of the U-NET
"""
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv3D, AveragePooling3D, UpSampling3D, concatenate, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import config

init = tf.keras.initializers.he_normal(seed=99)


def swish(x):
    """
    swish activation function
    :param x: input layer
    :return: output layer
    """
    return x * K.sigmoid(x)


def conv_layer(x, filters):
    x = Conv3D(filters, 3, padding='same', kernel_regularizer=l2(5e-5), kernel_initializer=init)(x)
    x = swish(x)
    return x


def down_sampling_layer(x):
    x = AveragePooling3D(pool_size=(1, 2, 2))(x)
    return x


def up_sampling_layer(x):
    x = UpSampling3D((1, 2, 2))(x)
    return x


def U_Net():
    inputs = tf.keras.layers.Input(shape=[config.n_frames, None, None, 3])
    x = inputs
    x = conv_layer(x, 32)
    x = conv_layer(x, 32)

    skips = [x]
    for _ in range(config.n_down_sampling):
        y = down_sampling_layer(skips[-1])
        y = conv_layer(y, 32)
        y = conv_layer(y, 32)
        skips.append(y)

    current_layer_output = skips[-1]
    skips = reversed(skips[:-1])
    concat = tf.keras.layers.Concatenate()

    for skip in skips:
        current_layer_output = up_sampling_layer(current_layer_output)
        current_layer_output = conv_layer(current_layer_output, 32)
        current_layer_output = concat([current_layer_output, skip])

    # This is the last layer of the model
    last = conv_layer(current_layer_output, 32)
    last = conv_layer(last, 32)
    last = Lambda(lambda l: K.mean(l, axis=1))(last)
    last = Conv2D(3, 1)(last)

    model = tf.keras.Model(inputs=inputs, outputs=last)
    return model


if __name__ == '__main__':
    myModel = U_Net()
    print(myModel.summary(line_length=150))
