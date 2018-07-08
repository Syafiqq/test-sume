from unittest import TestCase

import numpy

normalized = numpy.array([
    [[[0.84333333, 0.43666667, 0.86333333, 0.31666667],
      [0.84333333, 0.43666667, 0.86333333, 0.31666667],
      [0.84333333, 0.43666667, 0.86333333, 0.31666667],
      [0.84333333, 0.43666667, 0.86333333, 0.31666667]]]])

corr_step_1_seg_1 = numpy.array([[0.2844, 0.4763, 0.3593, 0.2622],
                                 [0.4267, 0.7144, 0.5389, 0.3933],
                                 [0.4267, 0.7144, 0.5389, 0.3933],
                                 [0.2844, 0.4763, 0.3593, 0.2622]])
corr_step_1_seg_2 = numpy.array([[0.8433, 0.8633, 0.8633, 0.8633],
                                 [0.8433, 0.8633, 0.8633, 0.8633],
                                 [0.8433, 0.8633, 0.8633, 0.8633],
                                 [0.8433, 0.8633, 0.8633, 0.8633]])
corr_step_1_seg_3 = numpy.array([[0.3667, 0.3957, 0.3375, 0.3661],
                                 [0.3653, 0.2085, 0.2488, 0.3782],
                                 [0.3653, 0.2085, 0.2488, 0.3782],
                                 [0.3667, 0.3957, 0.3375, 0.3661]])
corr_step_1 = numpy.concatenate(([[corr_step_1_seg_1]], [[corr_step_1_seg_2]], [[corr_step_1_seg_3]]), axis=0)


class ReposHelper(TestCase):
    def test_normalize(self):
        self.assertIsNotNone(normalized)
        # print(normalized)

    def test_corr_step_1_seg_1(self):
        self.assertIsNotNone(corr_step_1_seg_1)
        # print(corr_step_1_seg_1)

    def test_corr_step_1_seg_2(self):
        self.assertIsNotNone(corr_step_1_seg_2)
        # print(corr_step_1_seg_2)

    def test_corr_step_1_seg_3(self):
        self.assertIsNotNone(corr_step_1_seg_3)
        # print(corr_step_1_seg_3)

    def test_corr_step_1(self):
        self.assertIsNotNone(corr_step_1)
        # print(corr_step_1)