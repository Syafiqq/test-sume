from keras import Input

from dlnn.tests.ml.activation_test import layer_step_2
from dlnn.tests.ml.activation_test import layer_step_4
from dlnn.tests.ml.activation_test import layer_step_7
from dlnn.tests.ml.conv_test import layer_step_1
from dlnn.tests.ml.conv_test import layer_step_3
from dlnn.tests.ml.conv_test import layer_step_6
from dlnn.tests.ml.pooling_test import layer_step_5
from dlnn.tests.ml.pooling_test import layer_step_8
from dlnn.tests.ml.repos_helper import normalized
from dlnn.tests.ml.testcase import TestCase

inputs = Input(shape=normalized.shape[1:])
step_1 = layer_step_1()(inputs)
step_2 = layer_step_2()(step_1)
step_3 = layer_step_3()(step_2)
step_4 = layer_step_4()(step_3)
step_5 = layer_step_5(2)(step_4)
step_6 = layer_step_6()(step_5)
step_7 = layer_step_7()(step_6)
step_8 = layer_step_8(1)(step_7)


class CnnFuncTest(TestCase):
    def test_input_to_step_1(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_1
        import numpy
        network = Model(inputs=inputs, outputs=step_1)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_1, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_2(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_2
        import numpy
        network = Model(inputs=inputs, outputs=step_2)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_2, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_3(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_3
        import numpy
        network = Model(inputs=inputs, outputs=step_3)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_3, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_4(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_4
        import numpy
        network = Model(inputs=inputs, outputs=step_4)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_4, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_5(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_5
        import numpy
        network = Model(inputs=inputs, outputs=step_5)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_5, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_6(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_6
        import numpy
        network = Model(inputs=inputs, outputs=step_6)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_6, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_7(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_7
        import numpy
        network = Model(inputs=inputs, outputs=step_7)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_7, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_8(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_8
        import numpy
        network = Model(inputs=inputs, outputs=step_8)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output[0], corr_step_8, rtol=1e-6))
        # print(output)
        # print(output.shape)

    def test_input_to_step_8_full(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_8_full
        import numpy
        network = Model(inputs=inputs, outputs=step_8)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output, corr_step_8_full, rtol=1e-6))
        # print(output)
        # print(output.shape)
