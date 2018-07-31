from keras import Input, Model
from keras import backend as K
from keras.activations import sigmoid
from keras.initializers import RandomUniform, Zeros
from keras.layers import Lambda, Reshape, Activation, MaxPooling2D, Flatten, Dense, Concatenate
from keras.utils import to_categorical

from dlnn.layer.Conv2D import AvgFilter
from dlnn.layer.Conv2D import Conv2D
from dlnn.layer.Conv2D import MaxFilter
from dlnn.layer.Conv2D import StdDevFilter
from dlnn.layer.MergeCategorical import MergeCategorical
from dlnn.util import MoorePenrose
from dlnn.util.Initializers import Unifinv


class Dlnn(object):
    def __init__(self, scale_min1=0.0, scale_max1=0.0, scale_min2=0.0, scale_max2=0.0):
        self.fully_connected_num = 3
        self.elm_3_dense_1_bias_max = 1.0
        self.elm_3_dense_1_bias_min = 0.0
        self.elm_3_dense_1_kernel_max = 0.5
        self.elm_3_dense_1_kernel_min = -0.5
        self.elm_3_dense_1_units = 4
        self.elm_2_dense_1_bias_max = 1.0
        self.elm_2_dense_1_bias_min = 0.0
        self.elm_2_dense_1_kernel_max = 0.5
        self.elm_2_dense_1_kernel_min = -0.5
        self.elm_2_dense_1_units = 7
        self.category_num = 0
        self.elm_1_dense_1_bias_max = 1.0
        self.elm_1_dense_1_bias_min = 0.0
        self.elm_1_dense_1_kernel_max = 0.5
        self.elm_1_dense_1_kernel_min = -0.5
        self.elm_1_dense_1_units = 5
        self.pool_2_size = 1
        self.pool_1_size = 2
        self.conv_3_window = 3
        self.conv_2_window = 3
        self.conv_1_window = 3
        self.scale_min2 = float(scale_min1)
        self.scale_max2 = float(scale_max1)
        self.scale_max1 = float(scale_min2)
        self.scale_min1 = float(scale_max2)
        self.layer = {}
        self.input_shape = 0

    @staticmethod
    def config(*args, **kwargs):
        return Dlnn()

    def train(self, x, y):
        assert x is not None
        assert y is not None
        self.input_shape = x.shape[-1]
        self.category_num = y.shape[-1]
        self.__build_model()
        self.__train(x, y)
        return self.__evaluate(x, y)

    def __train(self, x, y):
        assert self.layer is not None
        elm_1_beta_net = Model(inputs=self.layer['input'], outputs=self.layer['elm_1_activation_1'])
        elm_1_activation_1_o = elm_1_beta_net.predict(x)
        elm_1_dense_1_w = elm_1_beta_net.get_layer(name='elm_1_dense_1').get_weights()
        elm_1_dense_2_w = [
            K.eval(K.dot(MoorePenrose.pinv3(elm_1_activation_1_o), to_categorical(y, self.category_num)))]
        feed = Model(inputs=inputs, outputs=step_11_b_activation)
        output = feed.predict(corpus_data)
        w_10_b = feed.get_layer(index=13).get_weights()
        w_12_b = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        feed = Model(inputs=inputs, outputs=step_11_c_activation)
        output = feed.predict(corpus_data)
        w_10_c = feed.get_layer(index=13).get_weights()
        w_12_c = [K.eval(K.dot(MoorePenrose.pinv3(output), to_categorical(label_init, 3)))]
        pass

    def __evaluate(self, x, y):
        # TODO : Place Evaluation Process Here
        pass

    def __build_model(self):
        self.layer['input'] = Input(
            shape=(self.input_shape,))
        self.layer['pre_scaling'] = Lambda(
            function=lambda x: (((x - self.scale_min1) / (self.scale_max1 - self.scale_min1)) * (
                    self.scale_max2 - self.scale_min2)) + self.scale_min2,
            name='pre_scaling')(self.layer['input'])
        self.layer['pre_reshaping'] = Reshape(
            target_shape=[1, 1, self.input_shape],
            name='pre_reshaping')(self.layer['pre_scaling'])
        self.layer['pre_tiling'] = Lambda(
            function=lambda x: K.tile(x, (1, 1, self.input_shape, 1)),
            name='pre_tiling')(self.layer['pre_reshaping'])
        self.layer['cnn_conv_1'] = Conv2D(
            filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
            window=self.conv_1_window,
            padding='same',
            use_bias=False,
            kernel_size=(self.category_num, self.category_num),
            data_format='channels_first',
            name='cnn_conv_1')(self.layer['pre_tiling'])
        self.layer['cnn_activation_1'] = Activation(
            activation=sigmoid,
            name='cnn_activation_1')(self.layer['cnn_conv_1'])
        self.layer['cnn_conv_2'] = Conv2D(
            filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
            window=self.conv_2_window,
            padding='same',
            use_bias=False,
            kernel_size=(self.category_num, self.category_num),
            data_format='channels_first',
            name='cnn_conv_2')(self.layer['cnn_activation_1'])
        self.layer['cnn_activation_2'] = Activation(
            activation=sigmoid,
            name='cnn_activation_2')(self.layer['cnn_conv_2'])
        self.layer['cnn_pooling_1'] = MaxPooling2D(
            pool_size=(self.pool_1_size, self.pool_1_size),
            padding='same',
            data_format='channels_first',
            name='cnn_pooling_1')(self.layer['cnn_activation_2'])
        self.layer['cnn_conv_3'] = Conv2D(
            filters=[AvgFilter(), MaxFilter(), StdDevFilter()],
            window=self.conv_3_window,
            padding='same',
            use_bias=False,
            kernel_size=(self.category_num, self.category_num),
            data_format='channels_first',
            name='cnn_conv_3')(self.layer['cnn_pooling_1'])
        self.layer['cnn_activation_3'] = Activation(
            activation=sigmoid,
            name='cnn_activation_3')(self.layer['cnn_conv_3'])
        self.layer['cnn_pooling_2'] = MaxPooling2D(
            pool_size=(self.pool_2_size, self.pool_2_size),
            padding='same',
            data_format='channels_first',
            name='cnn_pooling_2')(self.layer['cnn_activation_3'])
        self.layer['bridge_flatten'] = Flatten(
            name='bridge_flatten')(self.layer['cnn_pooling_2'])
        self.layer['elm_1_dense_1'] = Dense(
            units=self.elm_1_dense_1_units,
            activation=None,
            use_bias=True,
            kernel_initializer=Unifinv(minval=self.elm_1_dense_1_kernel_min, maxval=self.elm_1_dense_1_kernel_max),
            bias_initializer=RandomUniform(minval=self.elm_1_dense_1_bias_min, maxval=self.elm_1_dense_1_bias_max),
            trainable=False,
            name='elm_1_dense_1')(self.layer['bridge_flatten'])
        self.layer['elm_1_activation_1'] = Activation(
            activation=sigmoid,
            name='elm_1_activation_1')(self.layer['elm_1_dense_1'])
        self.layer['elm_1_dense_2'] = Dense(
            units=self.category_num,
            activation=None,
            use_bias=False,
            kernel_initializer=Zeros(),
            trainable=False,
            name='elm_1_dense_2')(self.layer['elm_1_activation_1'])
        self.layer['elm_2_dense_1'] = Dense(
            units=self.elm_2_dense_1_units,
            activation=None,
            use_bias=True,
            kernel_initializer=Unifinv(minval=self.elm_2_dense_1_kernel_min, maxval=self.elm_2_dense_1_kernel_max),
            bias_initializer=RandomUniform(minval=self.elm_2_dense_1_bias_min, maxval=self.elm_2_dense_1_bias_max),
            trainable=False,
            name='elm_2_dense_1')(self.layer['bridge_flatten'])
        self.layer['elm_2_activation_1'] = Activation(
            activation=sigmoid,
            name='elm_2_activation_1')(self.layer['elm_2_dense_1'])
        self.layer['elm_2_dense_2'] = Dense(
            units=self.category_num,
            activation=None,
            use_bias=False,
            kernel_initializer=Zeros(),
            trainable=False,
            name='elm_2_dense_2')(self.layer['elm_2_activation_1'])
        self.layer['elm_3_dense_1'] = Dense(
            units=self.elm_3_dense_1_units,
            activation=None,
            use_bias=True,
            kernel_initializer=Unifinv(minval=self.elm_3_dense_1_kernel_min, maxval=self.elm_3_dense_1_kernel_max),
            bias_initializer=RandomUniform(minval=self.elm_3_dense_1_bias_min, maxval=self.elm_3_dense_1_bias_max),
            trainable=False,
            name='elm_3_dense_1')(self.layer['bridge_flatten'])
        self.layer['elm_3_activation_1'] = Activation(
            activation=sigmoid,
            name='elm_3_activation_1')(self.layer['elm_3_dense_1'])
        self.layer['elm_3_dense_2'] = Dense(
            units=self.category_num,
            activation=None,
            use_bias=False,
            kernel_initializer=Zeros(),
            trainable=False,
            name='elm_3_dense_2')(self.layer['elm_3_activation_1'])
        self.layer['fully_connected_concat'] = Concatenate(
            name='fully_connected_concat')(
            [self.layer['elm_1_dense_2'], self.layer['elm_2_dense_2'], self.layer['elm_3_dense_2']])
        self.layer['fully_connected_reshape'] = Reshape(
            target_shape=(self.fully_connected_num, self.category_num),
            name='fully_connected_reshape')(self.layer['fully_connected_concat'])
        self.layer['fully_connected_merge'] = MergeCategorical(
            categorical_length=self.category_num,
            name='fully_connected_merge')(self.layer['fully_connected_reshape'])
