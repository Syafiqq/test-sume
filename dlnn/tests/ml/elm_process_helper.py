from keras.layers import Flatten

from dlnn.tests.ml.testcase import TestCase


def layer_step_9_0():
    return Flatten()


class ElmProcessHelper(TestCase):
    def test_step_9_flatten_data(self):
        from dlnn.tests.ml.repos_helper import corr_step_8_full
        from dlnn.tests.ml.repos_helper import corr_step_9_flatten
        import numpy
        layer = layer_step_9_0()
        x = layer.call(corr_step_8_full)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_9_flatten, rtol=1e-6))
        # print(K.eval(x))

    def test_step_9_categorical_label(self):
        from dlnn.util import to_categorical
        from dlnn.tests.ml.repos_helper import label_init
        from dlnn.tests.ml.repos_helper import corr_step_9_Y
        import numpy
        result = to_categorical(label_init, numpy.unique(label_init).size).astype(int)
        self.assertTrue(numpy.allclose(result, corr_step_9_Y, rtol=0))
        # print(result)

    def test_unifinv_function(self):
        from scipy import stats
        import numpy
        w = stats.uniform.ppf(numpy.random.rand(5, 12), loc=-.5, scale=(0.5 - -0.5))
        self.assertIsNotNone(w)
        # print(w)
