from unittest import TestCase

from keras import backend as K


def layer_step_2():
    from keras.layers import Activation
    return Activation('sigmoid')


def layer_step_4():
    from keras.layers import Activation
    return Activation('sigmoid')


def layer_step_7():
    from keras.layers import Activation
    return Activation('sigmoid')


class ActivationTest(TestCase):
    def test_sigmoid_activation(self):
        from keras.layers import Activation
        act = Activation('sigmoid')
        x = act.call(K.variable([0.2]))
        self.assertIsNotNone(x)
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_1
        import numpy
        act = layer_step_2()
        x = act.call(corr_step_1_seg_1)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_1, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_2(self):
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_2
        import numpy
        act = layer_step_2()
        x = act.call(corr_step_1_seg_2)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_3
        import numpy
        act = layer_step_2()
        x = act.call(corr_step_1_seg_3)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_3, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_1
        from dlnn.tests.ml.repos_helper import corr_step_2
        import numpy
        act = layer_step_2()
        x = act.call(corr_step_1)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3_seg_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_3_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_4_seg_1
        import numpy
        act = layer_step_4()
        x = act.call(corr_step_3_seg_1)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4_seg_1, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3_seg_2(self):
        from dlnn.tests.ml.repos_helper import corr_step_3_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_4_seg_2
        import numpy
        act = layer_step_4()
        x = act.call(corr_step_3_seg_2)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4_seg_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3_seg_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_3_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_4_seg_3
        import numpy
        act = layer_step_4()
        x = act.call(corr_step_3_seg_3)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4_seg_3, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_3
        from dlnn.tests.ml.repos_helper import corr_step_4
        import numpy
        act = layer_step_4()
        x = act.call(corr_step_3)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6_seg_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_6_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_7_seg_1
        import numpy
        act = layer_step_7()
        x = act.call(corr_step_6_seg_1)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7_seg_1, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6_seg_2(self):
        from dlnn.tests.ml.repos_helper import corr_step_6_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_7_seg_2
        import numpy
        act = layer_step_7()
        x = act.call(corr_step_6_seg_2)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7_seg_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6_seg_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_6_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_7_seg_3
        import numpy
        act = layer_step_7()
        x = act.call(corr_step_6_seg_3)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7_seg_3, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6(self):
        from dlnn.tests.ml.repos_helper import corr_step_6
        from dlnn.tests.ml.repos_helper import corr_step_7
        import numpy
        act = layer_step_7()
        x = act.call(corr_step_6)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7, rtol=1e-6))
        # print(K.eval(x))
