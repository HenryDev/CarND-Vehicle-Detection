import keras
import numpy
from keras.layers import Convolution2D, LeakyReLU, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

keras.backend.set_image_dim_ordering('th')


def make_model():
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, input_shape=(3, 448, 448), border_mode='same', subsample=(1, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model


def load_weights(model, weight_file):
    data = numpy.fromfile(weight_file, numpy.float32)
    data = data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape:
            kernel_shape, bias_shape = shape
            biases = data[index:index + numpy.prod(bias_shape)].reshape(bias_shape)
            index += numpy.prod(bias_shape)
            kernels = data[index:index + numpy.prod(kernel_shape)].reshape(kernel_shape)
            index += numpy.prod(kernel_shape)
            layer.set_weights([kernels, biases])


def make_better_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 416, 416), border_mode='same', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Convolution2D(64, 1, 1, border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Convolution2D(128, 1, 1, border_mode='same'))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(Convolution2D(256, 1, 1, border_mode='same'))
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(Convolution2D(256, 1, 1, border_mode='same'))
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(Convolution2D(512, 1, 1, border_mode='same'))
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(Convolution2D(512, 1, 1, border_mode='same'))
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(Convolution2D(1000, 1, 1, border_mode='same'))
    model.add(Dense(1000))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model


print(make_better_model().summary())
