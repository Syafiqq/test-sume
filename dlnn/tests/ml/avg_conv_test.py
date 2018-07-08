import os
from unittest import TestCase

import numpy
import numpy.matlib
from keras import backend as K

from sumeq.settings import BASE_DIR


def initial_layer():
    from dlnn.layer.Conv2D import Conv2D
    from dlnn.layer.Conv2D import AvgFilter
    from dlnn.layer.Conv2D import MaxFilter
    from dlnn.layer.Conv2D import StdDevFilter
    return Conv2D(
        name='abc',
        filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
        window=3,
        padding='same',
        use_bias=False,
        kernel_size=(4, 4),
        data_format='channels_first',
        input_shape=(1, 4, 4))


def initial_result(x):
    conv = initial_layer()
    result = conv.call(x)
    return result


def initial_result_check():
    return K.variable([[
        [[0.2844, 0.4763, 0.3593, 0.2622],
         [0.4267, 0.7144, 0.5389, 0.3933],
         [0.4267, 0.7144, 0.5389, 0.3933],
         [0.2844, 0.4763, 0.3593, 0.2622]],
        [[0.8433, 0.8633, 0.8633, 0.8633],
         [0.8433, 0.8633, 0.8633, 0.8633],
         [0.8433, 0.8633, 0.8633, 0.8633],
         [0.8433, 0.8633, 0.8633, 0.8633]],
        [[0.3667, 0.3957, 0.3375, 0.3661],
         [0.3653, 0.2085, 0.2488, 0.3782],
         [0.3653, 0.2085, 0.2488, 0.3782],
         [0.3667, 0.3957, 0.3375, 0.3661]],
    ]])


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
        # print(sess.run(tf.pad(t, paddings, "CONSTANT")))
        sess.close()

    def test_padding_using_keras(self):
        pass
        # print(keras.backend.eval(tf.pad(self.corpus()[0], ((1, 1), (1, 1)), "CONSTANT")))

    def test_slice(self):
        import tensorflow as tf
        x = tf.pad(self.corpus()[0], ((1, 1), (1, 1)), "CONSTANT")
        # print(keras.backend.eval(x))
        # print(keras.backend.eval(tf.slice(x, [0, 0], [3, 3])))
        # print(keras.backend.eval(x[0:0, 3:3]))

    def test_raw_conv(self):
        result = initial_result(self.corpus())
        self.assertIsNotNone(result)
        # print(K.eval(result))

    def test_conv(self):
        from keras import Sequential
        from keras import Model
        # from keras.layers import Conv2D

        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)

        model = Sequential()
        model.add(initial_layer())

        layer_name = 'abc'
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.corpus())
        # print(intermediate_output)
        # print(intermediate_output.shape)
