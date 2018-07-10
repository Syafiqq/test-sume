from unittest import TestCase


class CnnTest(TestCase):
    def build_helper(self):
        from keras import Sequential
        from dlnn.tests.ml.conv_test import layer_step_1
        from dlnn.tests.ml.activation_test import layer_step_2
        from dlnn.tests.ml.conv_test import layer_step_3

        model = Sequential()
        model.add(layer_step_1())
        model.add(layer_step_2())
        model.add(layer_step_3())
        return model

    def test_input_to_step_1(self):
        from keras import Model
        from dlnn.tests.ml.repos_helper import normalized
        from dlnn.tests.ml.repos_helper import corr_step_1
        import numpy
        model = self.build_helper()
        network = Model(inputs=model.input,
                        outputs=model.get_layer(index=0).output)
        output = network.predict(normalized)
        self.assertIsNotNone(output)
        self.assertTrue(numpy.allclose(output, corr_step_1, rtol=1e-3))
        # print(output)
        # print(output.shape)
