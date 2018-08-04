import keras
import numpy
from keras import Input
from keras import backend as K
from keras.layers import Lambda, Reshape

from dlnn.tests.ml.activation_test import layer_2_activation, layer_4_activation, layer_7_activation, \
    layer_11_a_activation, layer_11_b_activation, layer_11_c_activation
from dlnn.tests.ml.conv_test import layer_1_conv, layer_3_conv, layer_6_conv
from dlnn.tests.ml.elm_func_test import layer_9_flatten, layer_10_a_dense, layer_12_a_dense, layer_10_b_dense, \
    layer_12_b_dense, layer_10_c_dense, layer_12_c_dense, layer_13_concatenate, layer_14_reshape, \
    layer_15_merge_categorical
from dlnn.tests.ml.pooling_test import layer_5_pool, layer_8_pool
from dlnn.tests.ml.repos_helper import corpus_data, label_init, corpus_label
from dlnn.tests.ml.testcase import TestCase
from dlnn.util import to_categorical

inputs = Input(shape=(corpus_data.shape[-1],))
scale = Lambda(lambda x: x * 1.0 / 300.0)(inputs)
reshape = Reshape([1, 1, 4])(scale)
tile = Lambda(lambda x: K.tile(x, (1, 1, 4, 1)))(reshape)
step_1_conv = layer_1_conv()(tile)
step_2_activation = layer_2_activation()(step_1_conv)
step_3_conv = layer_3_conv()(step_2_activation)
step_4_activation = layer_4_activation()(step_3_conv)
step_5_pool = layer_5_pool(2)(step_4_activation)
step_6_conv = layer_6_conv()(step_5_pool)
step_7_activation = layer_7_activation()(step_6_conv)
step_8_pool = layer_8_pool(1)(step_7_activation)
step_9_flatten = layer_9_flatten()(step_8_pool)
step_10_a_dense = layer_10_a_dense()(step_9_flatten)
step_11_a_activation = layer_11_a_activation()(step_10_a_dense)
step_12_a_dense = layer_12_a_dense()(step_11_a_activation)
step_10_b_dense = layer_10_b_dense()(step_9_flatten)
step_11_b_activation = layer_11_b_activation()(step_10_b_dense)
step_12_b_dense = layer_12_b_dense()(step_11_b_activation)
step_10_c_dense = layer_10_c_dense()(step_9_flatten)
step_11_c_activation = layer_11_c_activation()(step_10_c_dense)
step_12_c_dense = layer_12_c_dense()(step_11_c_activation)
step_13_concatenate = layer_13_concatenate()([step_12_a_dense, step_12_b_dense, step_12_c_dense])
step_14_reshape = layer_14_reshape()(step_13_concatenate)
step_15_output = layer_15_merge_categorical()(step_14_reshape)


class DlnnFunctionalTest(TestCase):
    def test_functional_dlnn(self):
        from keras import Model
        from dlnn.util import MoorePenrose
        #
        # Feed Beta
        #
        feed = Model(inputs=inputs, outputs=step_11_a_activation)
        output = feed.predict(corpus_data)
        w_10_a = feed.get_layer(index=13).get_weights()
        w_12_a = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_b_activation)
        output = feed.predict(corpus_data)
        w_10_b = feed.get_layer(index=13).get_weights()
        w_12_b = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_c_activation)
        output = feed.predict(corpus_data)
        w_10_c = feed.get_layer(index=13).get_weights()
        w_12_c = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]

        #
        # Training Model
        #
        network = Model(inputs=inputs, outputs=step_15_output)
        network.compile(optimizer=keras.optimizers.RMSprop(lr=0.0, rho=0.0, epsilon=None, decay=0.0),
                        loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_accuracy, keras.metrics.mape])
        for i in range(25):
            layer = network.get_layer(index=i).get_weights()
            self.assertIsNotNone(layer)
            # print("step_%d" % (i + 1), layer)
        network.get_layer(index=13).set_weights(w_10_a)
        network.get_layer(index=14).set_weights(w_10_b)
        network.get_layer(index=15).set_weights(w_10_c)
        network.get_layer(index=19).set_weights(w_12_a)
        network.get_layer(index=20).set_weights(w_12_b)
        network.get_layer(index=21).set_weights(w_12_c)
        network.fit(corpus_data, to_categorical(label_init, 3), batch_size=corpus_data.shape[0])
        self.assertTrue(numpy.allclose(w_10_a[0], network.get_layer(index=13).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_10_b[0], network.get_layer(index=14).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_10_c[0], network.get_layer(index=15).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_12_a[0], network.get_layer(index=19).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_12_b[0], network.get_layer(index=20).get_weights()[0], rtol=0))
        self.assertTrue(numpy.allclose(w_12_c[0], network.get_layer(index=21).get_weights()[0], rtol=0))
        result = network.predict(corpus_data, batch_size=corpus_data.shape[0])
        # print(result.argmax(axis=1))
        self.assertIsNotNone(result)

    def test_baked_dlnn(self):
        self.assertTrue(True)
        from dlnn.Dlnn import Dlnn
        from dlnn.Dlnn import DLNN_DEFAULT_CONFIG
        train_eval = Dlnn(**DLNN_DEFAULT_CONFIG).train(corpus_data, corpus_label - 1)
        self.assertIsNotNone(train_eval)
        # print(train_eval)
