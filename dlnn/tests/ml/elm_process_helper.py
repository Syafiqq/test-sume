from keras import backend as K

from dlnn.tests.ml.testcase import TestCase


def step_10_a_dummy_kernel_init(shape, dtype=None):
    from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy_kernel_init
    return K.variable(corr_step_10_a_dummy_kernel_init)


def step_10_a_dummy_bias_init(shape, dtype=None):
    from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy_bias_init
    return K.variable(corr_step_10_a_dummy_bias_init)


def step_10_b_dummy_kernel_init(shape, dtype=None):
    from dlnn.tests.ml.repos_helper import corr_step_10_b_dummy_kernel_init
    return K.variable(corr_step_10_b_dummy_kernel_init)


def step_10_b_dummy_bias_init(shape, dtype=None):
    from dlnn.tests.ml.repos_helper import corr_step_10_b_dummy_bias_init
    return K.variable(corr_step_10_b_dummy_bias_init)


def step_10_c_dummy_kernel_init(shape, dtype=None):
    from dlnn.tests.ml.repos_helper import corr_step_10_c_dummy_kernel_init
    return K.variable(corr_step_10_c_dummy_kernel_init)


def step_10_c_dummy_bias_init(shape, dtype=None):
    from dlnn.tests.ml.repos_helper import corr_step_10_c_dummy_bias_init
    return K.variable(corr_step_10_c_dummy_bias_init)


class ElmProcessHelper(TestCase):
    def test_step_9_output(self):
        from dlnn.tests.ml.repos_helper import corr_step_8_full
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.elm_func_test import layer_step_9
        import numpy
        i = K.variable(corr_step_8_full)
        layer = layer_step_9()
        layer.build(i.shape)
        x = layer.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_9, rtol=1e-6))
        # print(K.eval(x))

    def test_raw_label_to_categorical_label(self):
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
        # print(K.eval(w))

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

    def test_step_10_a_dummy_non_bias_manual_operation(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy_non_bias
        import numpy
        x = K.dot(K.variable(corr_step_9), step_10_a_dummy_kernel_init(None))
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_a_dummy_non_bias, rtol=1e-6))
        # print(K.eval(x))

    def test_step_10_a_dummy_manual_operation(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy
        import numpy
        x = K.dot(K.variable(corr_step_9), step_10_a_dummy_kernel_init(None)) + step_10_a_dummy_bias_init(None)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_a_dummy, rtol=1e-6))
        # print(K.eval(x))

    def test_step_10_a_dummy_output(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy
        from dlnn.tests.ml.elm_func_test import layer_step_10_a_dummy
        import numpy
        corr_step_9 = K.variable(corr_step_9)
        layer = layer_step_10_a_dummy()
        layer.build(corr_step_9.shape)
        x = layer.call(corr_step_9)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_a_dummy, rtol=1e-6))
        # print(K.eval(x))

    def test_raw_moore_penrose(self):
        from dlnn.tests.ml.repos_helper import corr_step_11_a_dummy, categorical_label_init
        from dlnn.util import MoorePenrose
        x = corr_step_11_a_dummy
        t = categorical_label_init
        r = K.dot(MoorePenrose.pinv2(K.variable(x), 1e-31), K.variable(t))
        self.assertIsNotNone(r)
        # print(K.eval(r))

    def test_step_10_b_dummy_non_bias_manual_operation(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_b_dummy_non_bias
        import numpy
        x = K.dot(K.variable(corr_step_9), step_10_b_dummy_kernel_init(None))
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_b_dummy_non_bias, rtol=1e-6))
        # print(K.eval(x))

    def test_step_10_b_dummy_manual_operation(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_b_dummy
        import numpy
        x = K.dot(K.variable(corr_step_9), step_10_b_dummy_kernel_init(None)) + step_10_b_dummy_bias_init(None)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_b_dummy, rtol=1e-6))
        # print(K.eval(x))

    def test_step_10_b_dummy_output(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_b_dummy
        from dlnn.tests.ml.elm_func_test import layer_step_10_b_dummy
        import numpy
        corr_step_9 = K.variable(corr_step_9)
        layer = layer_step_10_b_dummy()
        layer.build(corr_step_9.shape)
        x = layer.call(corr_step_9)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_b_dummy, rtol=1e-6))
        # print(K.eval(x))

    def test_step_10_c_dummy_non_bias_manual_operation(self):
        from dlnn.tests.ml.repos_helper import corr_step_9
        from dlnn.tests.ml.repos_helper import corr_step_10_c_dummy_non_bias
        import numpy
        x = K.dot(K.variable(corr_step_9), step_10_c_dummy_kernel_init(None))
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_10_c_dummy_non_bias, rtol=1e-6))
        # print(K.eval(x))
