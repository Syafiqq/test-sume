import pandas
from django.test import TestCase


class ReadCSVTest(TestCase):
    corpus_path = 'dlnn/resources/databank/datatrainClassify.csv'

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
