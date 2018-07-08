from unittest import TestCase

from keras import backend as K


class ActivationTest(TestCase):
    def test_sigmoid_activation(self):
        from keras.layers import Activation
        act = Activation('sigmoid')
        x = act.call(K.variable([0.2]))
        self.assertIsNotNone(x)
        # print(K.eval(x))

    def test_activation_1(self):
        from keras.layers import Activation
        act = Activation('sigmoid')
        from dlnn.tests.ml.conv_test import initial_result
        from dlnn.tests.ml.conv_test import ConvTest
        x = act.call(initial_result(ConvTest().corpus()))
        self.assertIsNotNone(x)
        # print(K.eval(x))
