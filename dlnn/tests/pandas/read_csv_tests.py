import pandas
from django.test import TestCase


class ReadCSVTest(TestCase):
    corpus_path = 'dlnn/resources/databank/datatrainClassify.csv'

    def test_read_csv(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        self.assertIsNotNone(dataframe)
        # print(dataframe)
