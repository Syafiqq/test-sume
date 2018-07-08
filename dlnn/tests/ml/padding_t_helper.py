from unittest import TestCase

import numpy
from keras import backend as K


class AvgConvTest(TestCase):
    data = numpy.array([[
        [[0.1, 0.2, 0.3, 0.4],
         [0.1, 0.2, 0.3, 0.4],
         [0.1, 0.2, 0.3, 0.4],
         [0.1, 0.2, 0.3, 0.4]]
    ], [
        [[0.5, 0.6, 0.7, 0.8],
         [0.5, 0.6, 0.7, 0.8],
         [0.5, 0.6, 0.7, 0.8],
         [0.5, 0.6, 0.7, 0.8]]
    ]])

    def test_it_generate_Tensor(self):
        self.assertIsNotNone(self.data)
        tensor = K.variable(self.data)
        self.assertIsNotNone(tensor)
        # print(K.eval(tensor))

    def test_padding(self):
        import tensorflow as tf
        tensor = K.variable(self.data)
        ntensor = tf.pad(
            tensor=tensor,
            paddings=((1, 1), (1, 1), (1, 1), (1, 1))
        )
        self.assertIsNotNone(ntensor)
        # print(K.eval(ntensor))
        # print(ntensor.shape)

    def test_pad_util(self):
        from dlnn.layer.util import Pad as PadUtil
        tensor = K.variable(self.data[0][0])
        ntensor = PadUtil.pad_center(tensor, 1)
        self.assertIsNotNone(ntensor)
        # print(K.eval(ntensor))
        # print(ntensor.shape)
