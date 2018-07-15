from keras import backend as K
from keras.layers import Flatten

from dlnn.tests.ml.testcase import TestCase


def layer_step_9():
    return Flatten()


class ElmProcessHelper(TestCase):
    def test_step_9_flatten_data(self):
        from dlnn.tests.ml.repos_helper import corr_step_8_full
        from dlnn.tests.ml.repos_helper import corr_step_9_flatten
        import numpy
        layer = layer_step_9()
        x = layer.call(corr_step_8_full)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_9_flatten, rtol=1e-6))
        # print(K.eval(x))

    def test_step_9_categorical_label(self):
        from dlnn.util import to_categorical
        from dlnn.tests.ml.repos_helper import label_init
        from dlnn.tests.ml.repos_helper import categorical_label_init
        import numpy
        result = to_categorical(label_init, numpy.unique(label_init).size).astype(int)
        self.assertTrue(numpy.allclose(result, categorical_label_init, rtol=0))
        # print(result)

    def test_unifinv_function(self):
        from scipy import stats
        import numpy
        w = stats.uniform.ppf(numpy.random.rand(5, 12), loc=-.5, scale=(0.5 - -0.5))
        self.assertIsNotNone(w)
        # print(w)

    def test_unifinv_callable_function(self):
        from dlnn.tests.ml.elm_func_test import unifinv_init
        w = unifinv_init((5, 12), dtype=K.tf.float32)
        self.assertIsNotNone(w)
        print(K.eval(w))

    def test_standard_uniform_distribution(self):
        import numpy
        w = numpy.random.uniform(0, 1, (5, 1))
        self.assertIsNotNone(w)
        # print(w)

    def test_contrib_standard_uniform_distribution(self):
        from keras.initializers import RandomUniform
        w = RandomUniform()((5, 1), dtype=K.tf.float32)
        self.assertIsNotNone(w)
        # print(K.eval(w))
