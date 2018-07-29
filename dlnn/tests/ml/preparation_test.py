import numpy as np
from keras import Input, Model
from keras.layers import Lambda, Reshape

from dlnn.tests.ml.repos_helper import corpus, corpus_data, corpus_label
from dlnn.tests.ml.testcase import TestCase


class PreparationTest(TestCase):
    def test_separate_data_and_its_label(self):
        combined = corpus
        _corpus_data, _corpus_label = np.hsplit(combined, [combined.shape[1] - 1])
        self.assertTrue(np.allclose(_corpus_data, corpus_data, rtol=0))
        self.assertTrue(np.allclose(_corpus_label, corpus_label, rtol=0))
        # print(corpus_data, corpus_label)
        self.assertIsNotNone(combined)

    def test_defining_input_tensor(self):
        i = Input(shape=(corpus_data.shape[-1],))
        o = Lambda(lambda x: x)(i)
        network = Model(inputs=i, outputs=o)
        result = network.predict(corpus_data)
        # print(result)
        self.assertIsNotNone(result)

    def test_scaling_value(self):
        i = Input(shape=(corpus_data.shape[-1],))
        o = Lambda(lambda x: x * 1.0 / 300.)(i)
        network = Model(inputs=i, outputs=o)
        result = network.predict(corpus_data)
        # print(result)
        self.assertIsNotNone(result)

    def test_reshape_data(self):
        i = Input(shape=(corpus_data.shape[-1],))
        o = Lambda(lambda x: x * 1.0 / 300.)(i)
        o = Reshape([1, 1, 4])(o)
        network = Model(inputs=i, outputs=o)
        result = network.predict(corpus_data)
        # print(result)
        self.assertIsNotNone(result)
