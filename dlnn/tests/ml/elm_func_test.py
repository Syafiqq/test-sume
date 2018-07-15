from dlnn.tests.ml.cnn_func_test import inputs, step_8
from dlnn.tests.ml.elm_process_helper import layer_step_9_0
from dlnn.tests.ml.repos_helper import normalized
from dlnn.tests.ml.testcase import TestCase

step_9_0 = layer_step_9_0()(step_8)


class ElmFuncTest(TestCase):
    def test_input_to_step_9_0(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import corr_step_9_flatten
        import numpy
        network = Model(inputs=inputs, outputs=step_9_0)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output, corr_step_9_flatten, rtol=1e-6))
        # print(output)
        # print(output.shape)
