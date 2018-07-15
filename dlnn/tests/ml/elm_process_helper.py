from keras.layers import Flatten

from dlnn.tests.ml.testcase import TestCase


def layer_step_9_0():
    return Flatten()


class ElmProcessHelper(TestCase):
    def test_step_0_flatten_data(self):
        from dlnn.tests.ml.repos_helper import corr_step_8_full
        from dlnn.tests.ml.repos_helper import corr_step_9_flatten
        import numpy
        layer = layer_step_9_0()
        x = layer.call(corr_step_8_full)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_9_flatten, rtol=1e-6))
        # print(K.eval(x))
