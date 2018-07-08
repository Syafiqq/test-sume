from unittest import TestCase

from keras import backend as K


class ActivationTest(TestCase):
    def test_sigmoid_activation(self):
        from keras.layers import Activation
        act = Activation('sigmoid')
        x = act.call(K.variable([0.2]))
        self.assertIsNotNone(x)
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1(self):
        from keras.layers import Activation
        from dlnn.tests.ml.repos_helper import corr_step_1
        act = Activation('sigmoid')
        x = act.call(corr_step_1)
        self.assertIsNotNone(x)
        # print(K.eval(x))
