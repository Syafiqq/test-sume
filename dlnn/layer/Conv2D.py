from keras import backend as K
from keras.layers import Conv2D as c2D

from dlnn.layer.util.Singleton import Singleton


class Conv2D(c2D):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(Conv2D, self).__init__(filters, **kwargs)

    def build(self, input_shape):
        super(Conv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        import tensorflow as tf
        shape_x = x.shape
        if shape_x[1] > 1 and shape_x[1] != self.filters:
            raise ValueError('Cannot Operate Convolution')

        if shape_x[1] == 1 and shape_x[1] != self.filters:
            x = K.tile(x, (1, self.filters, 1, 1))

        def calls(y):
            xyz = tf.unstack(y)
            xyz[0] += 10
            xyz[1] += 20
            xyz[2] += 30
            return tf.stack(xyz)

        zz = tf.map_fn(calls, x[:])
        return zz

    def compute_output_shape(self, input_shape):
        return super(Conv2D, self).compute_output_shape(input_shape)


class _Filter:
    def filter(self, tensor, window):
        pass

    @staticmethod
    def calculate_padding(window):
        import math
        return int(math.ceil((window - 1.) / 2.))


class AvgFilter(_Filter, metaclass=Singleton):
    def filter(self, tensor, window):
        from dlnn.layer.util import Pad as PadUtil
        import numpy
        pad = _Filter.calculate_padding(window)
        padded = PadUtil.pad_center(tensor, pad)
        padded_shape = padded.shape
        newval = numpy.empty(shape=(1, padded_shape[1] - window + 1, padded_shape[2] - window + 1))
        newval_shape = newval.shape
        for x in (range(newval_shape[2])):
            for y in (range(newval_shape[1])):
                newval[0, y, x] = 1
        return tensor
