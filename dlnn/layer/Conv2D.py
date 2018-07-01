from keras import backend as K
from keras.layers import Conv2D as c2D


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
            return y + 10

        zz = tf.map_fn(calls, x[:])
        return zz

    def compute_output_shape(self, input_shape):
        return super(Conv2D, self).compute_output_shape(input_shape)
