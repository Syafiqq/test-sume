import os
from unittest import TestCase

import numpy
import numpy.matlib

from sumeq.settings import BASE_DIR


class AvgConvTest(TestCase):
    corpus_path = os.path.join(BASE_DIR, 'dlnn/resources/databank/datatrainClassify.csv')

    def corpus(self):
        dataframe = pandas.read_csv(self.corpus_path, header=None)
        dim = dataframe.shape
        r_feature = dataframe.drop(columns=dim[1] - 1)
        feature = r_feature.values.astype(float)
        feature_dim = numpy.shape(feature)
        feature *= 1 / 300.0
        nfeature = numpy.delete(numpy.empty([1, feature_dim[1], feature_dim[1]]), 0, 0)
        for _k, _v in enumerate(feature):
            nfeature = numpy.concatenate((nfeature, [numpy.matlib.repmat(_v, feature_dim[1], 1)]))
        oneliner = numpy.tile(numpy.reshape(feature, (feature_dim[0], 1, feature_dim[1])), (1, feature_dim[1], 1))
        self.assertTrue(numpy.allclose(nfeature, oneliner, rtol=1e-3))
        return oneliner

    def test_corpus(self):
        corpus = self.corpus()
        self.assertIsNotNone(corpus)
        # print(corpus)

    def test_keras_invocation(self):
        import tensorflow as tf
        sess = tf.Session()
        t = tf.constant(self.corpus()[0])
        paddings = tf.constant([[2, 2, ], [2, 2]])
        print(sess.run(tf.pad(t, paddings, "CONSTANT")))
        sess.close()

    def test_padding_using_keras(self):
        import keras
        import tensorflow as tf
        print(keras.backend.eval(tf.pad(self.corpus()[0], ((1, 1), (1, 1)), "CONSTANT")))

    def test_slice(self):
        import keras
        import tensorflow as tf
        x = tf.pad(self.corpus()[0], ((1, 1), (1, 1)), "CONSTANT")
        print(keras.backend.eval(x))
        # print(keras.backend.eval(tf.slice(x, [0, 0], [3, 3])))
        print(keras.backend.eval(x[0:0, 3:3]))
