import keras
import numpy
from keras import backend as K
from keras.layers import Flatten, Dense
from scipy import stats

from dlnn.tests.ml.activation_test import layer_step_11_a
from dlnn.tests.ml.cnn_func_test import inputs, step_8
from dlnn.tests.ml.elm_process_helper import step_10_a_dummy_kernel_init, step_10_a_dummy_bias_non_spread_init
from dlnn.tests.ml.repos_helper import normalized, categorical_label_init
from dlnn.tests.ml.testcase import TestCase


def layer_step_9():
    return Flatten()


def layer_step_10_a_dummy():
    return Dense(5, activation=None, use_bias=True, kernel_initializer=step_10_a_dummy_kernel_init,
                 bias_initializer=step_10_a_dummy_bias_non_spread_init, trainable=False)


def layer_step_12_a_dummy():
    return Dense(3, activation=None, use_bias=False, kernel_initializer=keras.initializers.Zeros(), trainable=False)


def unifinv_init(shape, dtype=None):
    return K.variable(stats.uniform.ppf(numpy.random.rand(*shape), loc=-.5, scale=(.5 - -.5)),
                      dtype=dtype)


step_9 = layer_step_9()(step_8)
step_10_a_dummy = layer_step_10_a_dummy()(step_9)
step_11_a_dummy = layer_step_11_a()(step_10_a_dummy)
step_12_a_dummy = layer_step_12_a_dummy()(step_11_a_dummy)


class ElmFuncTest(TestCase):
    def test_input_to_step_9(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_9
        import numpy
        network = Model(inputs=inputs, outputs=step_9)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output, corr_step_9, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_10_a_dummy(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy
        import numpy
        network = Model(inputs=inputs, outputs=step_10_a_dummy)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output, corr_step_10_a_dummy, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_11_a_dummy(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_11_a_dummy
        import numpy
        network = Model(inputs=inputs, outputs=step_11_a_dummy)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output, corr_step_11_a_dummy, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_beta_a_dummy(self):
        from keras import Model
        from dlnn.util import MoorePenrose
        network = Model(inputs=inputs, outputs=step_11_a_dummy)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        beta = K.dot(MoorePenrose.pinv2(output), K.variable(categorical_label_init))
        self.assertIsNotNone(beta)
        # print(K.eval(beta))
        # print(beta.shape)

    def test_check_weights_layer(self):
        from keras import Model
        from dlnn.tests.ml.cnn_func_test import inputs
        network = Model(inputs=inputs, outputs=step_11_a_dummy)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        for i in range(12):
            layer = network.get_layer(index=i).get_weights()
            self.assertIsNotNone(layer)
            # print("step_%d" % (i + 1), layer)
        network.predict(normalized)
        for i in range(12):
            layer = network.get_layer(index=i).get_weights()
            self.assertIsNotNone(layer)
            # print("step_%d" % (i + 1), layer)
