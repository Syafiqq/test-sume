from unittest import TestCase

import numpy
from keras import backend as K


class AvgConvTest(TestCase):
    data = numpy.array([[
        [[0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667]]
    ], [
        [[0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667]]
    ]])

    def test_it_generate_Tensor(self):
        self.assertIsNotNone(self.data)
        tensor = K.variable(self.data)
        # print(K.eval(tensor))

    def test_padding(self):
        import tensorflow as tf
        tensor = K.variable(self.data)
        ntensor = tf.pad(
            tensor=tensor,
            paddings=((1, 1), (1, 1), (1, 1), (1, 1))
        )
        print(K.eval(ntensor))
        print(ntensor.shape)
