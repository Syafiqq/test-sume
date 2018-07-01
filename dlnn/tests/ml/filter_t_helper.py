from unittest import TestCase

import numpy


class FilterTHelper(TestCase):
    data = numpy.array(
        [
            [
                [[0.84333333, 0.43666667, 0.86333333, 0.31666667],
                 [0.84333333, 0.43666667, 0.86333333, 0.31666667],
                 [0.84333333, 0.43666667, 0.86333333, 0.31666667],
                 [0.84333333, 0.43666667, 0.86333333, 0.31666667]]
            ],
            [
                [[0.43666667, 0.86333333, 0.31666667, 0.87666667],
                 [0.43666667, 0.86333333, 0.31666667, 0.87666667],
                 [0.43666667, 0.86333333, 0.31666667, 0.87666667],
                 [0.43666667, 0.86333333, 0.31666667, 0.87666667]]
            ]
        ])

    def test_avg_filter(self):
        from keras import backend as K
        from dlnn.layer.Conv2D import AvgFilter
        tensor = K.variable(self.data[0])
        avg = AvgFilter()
        ntensor = avg.filter(tensor, 3)
        # print(K.eval(ntensor))
        # print(ntensor.shape)
