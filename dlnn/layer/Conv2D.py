from keras import backend as K
from keras.layers import Conv2D as c2D

from dlnn.layer.util.Singleton import Singleton


class Conv2D(c2D):
    def __init__(self, filters, window, **kwargs):
        self.filters_cls = filters
        self.filters = len(self.filters_cls)
        self.window = window
        super(Conv2D, self).__init__(self.filters, **kwargs)

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
            for k, v in enumerate(self.filters_cls):
                xyz[k] = v.filter(xyz[k], self.window)
            return tf.stack(xyz)

        zz = tf.map_fn(calls, x[:])
        return zz

    def compute_output_shape(self, input_shape):
        return super(Conv2D, self).compute_output_shape(input_shape)


class _Filter:
    def filter(self, tensor, window, filter_fun):
        from dlnn.layer.util import Pad as PadUtil
        pad = _Filter.calculate_padding(window)
        padded = PadUtil.pad_center(tensor, pad)
        padded_shape = padded.shape
        newval = []
        newval_shape = (padded_shape[0] - window + 1, padded_shape[1] - window + 1)
        for y in (range(newval_shape[0])):
            for x in (range(newval_shape[1])):
                newval.append(filter_fun(padded[y:(window + y), x:(window + x)]))
        return K.reshape(newval, newval_shape)

    @staticmethod
    def calculate_padding(window):
        import math
        return int(math.ceil((window - 1.) / 2.))


class AvgFilter(_Filter, metaclass=Singleton):
    def filter(self, tensor, window, **kwargs):
        return super().filter(tensor, window, filter_fun=lambda x: K.mean(x))


class MaxFilter(_Filter, metaclass=Singleton):
    def filter(self, tensor, window, **kwargs):
        return super().filter(tensor, window, filter_fun=lambda x: K.max(x))


class StdDevFilter(_Filter, metaclass=Singleton):
    # cannot use K.std because implementation difference
    # use below function instead
    # @see: https://www.mathworks.com/help/matlab/ref/std.html
    # @see: https://stackoverflow.com/a/43409235
    def reduce_var(self, x, axis=None, keepdims=False):
        import tensorflow as tf
        m = tf.reduce_mean(x, axis=axis, keep_dims=True)
        devs_squared = tf.square(x - m)
        return tf.div(tf.reduce_sum(devs_squared, axis=axis, keep_dims=keepdims),
                      tf.subtract(tf.size(devs_squared, out_type=devs_squared.dtype), 1.0))

    def reduce_std(self, x, axis=None, keepdims=False):
        import tensorflow as tf
        return tf.sqrt(self.reduce_var(x, axis=axis, keepdims=keepdims))

    def filter(self, tensor, window, **kwargs):
        return super().filter(tensor, window, filter_fun=lambda x: self.reduce_std(x))
