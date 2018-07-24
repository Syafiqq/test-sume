from keras import backend as K

from dlnn.tests.ml.testcase import TestCase


def layer_step_2():
    from keras.layers import Activation
    return Activation('sigmoid')


def layer_step_4():
    from keras.layers import Activation
    return Activation('sigmoid')


def layer_step_7():
    from keras.layers import Activation
    return Activation('sigmoid')


def layer_step_11_a():
    from keras.layers import Activation
    return Activation('sigmoid')


def layer_step_11_b():
    from keras.layers import Activation
    return Activation('sigmoid')


class ActivationTest(TestCase):
    def test_sigmoid_activation(self):
        from keras.layers import Activation
        i = K.variable([0.2])
        act = Activation('sigmoid')
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_1
        import numpy
        i = K.variable(corr_step_1_seg_1)
        act = layer_step_2()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_1, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_2(self):
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_2
        import numpy
        i = K.variable(corr_step_1_seg_2)
        act = layer_step_2()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1_seg_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_1_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_2_seg_3
        import numpy
        i = K.variable(corr_step_1_seg_3)
        act = layer_step_2()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2_seg_3, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_1
        from dlnn.tests.ml.repos_helper import corr_step_2
        import numpy
        i = K.variable(corr_step_1)
        act = layer_step_2()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3_seg_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_3_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_4_seg_1
        import numpy
        i = K.variable(corr_step_3_seg_1)
        act = layer_step_4()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4_seg_1, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3_seg_2(self):
        from dlnn.tests.ml.repos_helper import corr_step_3_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_4_seg_2
        import numpy
        i = K.variable(corr_step_3_seg_2)
        act = layer_step_4()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4_seg_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3_seg_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_3_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_4_seg_3
        import numpy
        i = K.variable(corr_step_3_seg_3)
        act = layer_step_4()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4_seg_3, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_3
        from dlnn.tests.ml.repos_helper import corr_step_4
        import numpy
        i = K.variable(corr_step_3)
        act = layer_step_4()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_4, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6_seg_1(self):
        from dlnn.tests.ml.repos_helper import corr_step_6_seg_1
        from dlnn.tests.ml.repos_helper import corr_step_7_seg_1
        import numpy
        i = K.variable(corr_step_6_seg_1)
        act = layer_step_7()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7_seg_1, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6_seg_2(self):
        from dlnn.tests.ml.repos_helper import corr_step_6_seg_2
        from dlnn.tests.ml.repos_helper import corr_step_7_seg_2
        import numpy
        i = K.variable(corr_step_6_seg_2)
        act = layer_step_7()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7_seg_2, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6_seg_3(self):
        from dlnn.tests.ml.repos_helper import corr_step_6_seg_3
        from dlnn.tests.ml.repos_helper import corr_step_7_seg_3
        import numpy
        i = K.variable(corr_step_6_seg_3)
        act = layer_step_7()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7_seg_3, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_6(self):
        from dlnn.tests.ml.repos_helper import corr_step_6
        from dlnn.tests.ml.repos_helper import corr_step_7
        import numpy
        i = K.variable(corr_step_6)
        act = layer_step_7()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_7, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_10_a_dummy(self):
        from dlnn.tests.ml.repos_helper import corr_step_10_a_dummy
        from dlnn.tests.ml.repos_helper import corr_step_11_a_dummy
        import numpy
        i = K.variable(corr_step_10_a_dummy)
        act = layer_step_11_a()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_11_a_dummy, rtol=1e-6))
        # print(K.eval(x))

    def test_sigmoid_activation_from_corr_step_10_b_dummy(self):
        from dlnn.tests.ml.repos_helper import corr_step_10_b_dummy
        from dlnn.tests.ml.repos_helper import corr_step_11_b_dummy
        import numpy
        i = K.variable(corr_step_10_b_dummy)
        act = layer_step_11_b()
        act.build(i.shape)
        x = act.call(i)
        self.assertIsNotNone(x)
        self.assertTrue(numpy.allclose(K.eval(x), corr_step_11_b_dummy, rtol=1e-6))
        # print(K.eval(x))
