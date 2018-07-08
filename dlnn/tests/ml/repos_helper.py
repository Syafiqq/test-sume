from unittest import TestCase

import numpy

normalized = numpy.array([
    [[[0.84333333, 0.43666667, 0.86333333, 0.31666667],
      [0.84333333, 0.43666667, 0.86333333, 0.31666667],
      [0.84333333, 0.43666667, 0.86333333, 0.31666667],
      [0.84333333, 0.43666667, 0.86333333, 0.31666667]]]])

corr_step_1_seg_1 = numpy.array([[0.28444444, 0.47629630, 0.35925926, 0.26222222],
                                 [0.42666667, 0.71444444, 0.53888889, 0.39333333],
                                 [0.42666667, 0.71444444, 0.53888889, 0.39333333],
                                 [0.28444444, 0.47629630, 0.35925926, 0.26222222]])
corr_step_1_seg_2 = numpy.array([[0.84333333, 0.86333333, 0.86333333, 0.86333333],
                                 [0.84333333, 0.86333333, 0.86333333, 0.86333333],
                                 [0.84333333, 0.86333333, 0.86333333, 0.86333333],
                                 [0.84333333, 0.86333333, 0.86333333, 0.86333333]])
corr_step_1_seg_3 = numpy.array([[0.36667424, 0.39571821, 0.33745278, 0.36612839],
                                 [0.36525105, 0.20851325, 0.24881943, 0.37823053],
                                 [0.36525105, 0.20851325, 0.24881943, 0.37823053],
                                 [0.36667424, 0.39571821, 0.33745278, 0.36612839]])
corr_step_1 = numpy.concatenate(([[corr_step_1_seg_1]], [[corr_step_1_seg_2]], [[corr_step_1_seg_3]]), axis=1)


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
