from unittest import TestCase

from keras import backend as K


class ActivationTest(TestCase):
    def test_sigmoid_activation(self):
        from keras.layers import Activation
        act = Activation('sigmoid')
        x = act.call(K.variable([0.2]))
        self.assertIsNotNone(x)
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_1(self):
        from keras.layers import Activation
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_1
        import numpy
        act = Activation('sigmoid')
        x = act.call(corr_step_1_seg_1)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_1, rtol=1e-3))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_2(self):
        from keras.layers import Activation
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_2
        import numpy
        act = Activation('sigmoid')
        x = act.call(corr_step_1_seg_2)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_2, rtol=1e-3))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_3(self):
        from keras.layers import Activation
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_3
        import numpy
        act = Activation('sigmoid')
        x = act.call(corr_step_1_seg_3)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_3, rtol=1e-3))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1(self):
        from keras.layers import Activation
        from dlnn.tests.ml.repos_helper import corr_step_1
        from dlnn.tests.ml.repos_helper import corr_step_2
        import numpy
        act = Activation('sigmoid')
        x = act.call(corr_step_1)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2, rtol=1e-3))
        # print(K.eval(x))
