from unittest import TestCase

import numpy
from keras import backend as K


class AvgConvTest(TestCase):
    data = numpy.array([
        [[0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667],
         [0.84333333, 0.43666667, 0.86333333, 0.31666667]]
    ])

    def test_it_generate_Tensor(self):
        self.assertIsNotNone(self.data)
        tensor = K.variable(self.data)
        # print(K.eval(tensor))
