import os
from unittest import TestCase

import numpy
import pandas

from sumeq.settings import BASE_DIR


class ReadCSVTest(TestCase):
    corpus_path = os.path.join(BASE_DIR, 'dlnn/resources/databank/datatrainClassify.csv')

    def test_read_csv(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        self.assertIsNotNone(dataframe)
        # print(dataframe)

    def test_get_num_data(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        dim = dataframe.shape
        self.assertEqual(8, dim[0])
        # print(dim)

    def test_get_feature_num_data(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        dim = dataframe.shape
        self.assertEqual(4, dim[1] - 1)
        # print(dim)

    def test_get_feature_data(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        dim = dataframe.shape
        features = dataframe.drop(columns=dim[1] - 1)
        self.assertEqual(4, features.shape[1])
        # print(features)

    def test_convert_feature_to_numpy_array(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        dim = dataframe.shape
        r_feature = dataframe.drop(columns=dim[1] - 1)
        feature = r_feature.values
        f_shape = numpy.shape(feature)
        self.assertEqual((8, 4), f_shape)
        # print(feature)
