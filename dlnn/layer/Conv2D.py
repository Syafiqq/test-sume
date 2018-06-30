from keras.layers import Conv2D as c2D


class Conv2D(c2D):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(Conv2D, self).__init__(filters, **kwargs)

    def build(self, input_shape):
        super(Conv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return super(Conv2D, self).call(x)

    def compute_output_shape(self, input_shape):
        return super(Conv2D, self).compute_output_shape(input_shape)
