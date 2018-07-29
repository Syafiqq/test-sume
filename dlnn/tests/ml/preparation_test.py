import numpy as np

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
