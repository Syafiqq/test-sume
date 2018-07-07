from unittest import TestCase

from keras import backend as K


class ActivationTest(TestCase):
    def test_sigmoid_activation(self):
        from keras.layers import Activation
        act = Activation('sigmoid')
        x = act.call(K.variable([0.2]))
        self.assertIsNotNone(x)
        # print(K.eval(x))
