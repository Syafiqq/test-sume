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
        tensor = K.variable(self.data[0][0])
        fltr = AvgFilter()
        ntensor = fltr.filter(tensor, 3)
        # print(K.eval(ntensor))
        # print(ntensor.shape)

    def test_max_filter(self):
        from keras import backend as K
        from dlnn.layer.Conv2D import MaxFilter
        tensor = K.variable(self.data[0][0])
        fltr = MaxFilter()
        ntensor = fltr.filter(tensor, 3)
        # print(K.eval(ntensor))
        # print(ntensor.shape)

    def test_stddev_filter(self):
        from keras import backend as K
        from dlnn.layer.Conv2D import StdDevFilter
        tensor = K.variable(self.data[0][0])
        fltr = StdDevFilter()
        ntensor = fltr.filter(tensor, 3)
        # print(K.eval(ntensor))
        # print(ntensor.shape)
