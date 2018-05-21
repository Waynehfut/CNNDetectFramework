# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       NetRefs on CPLID_Detect
   Description :
   Author :          Wayne
   Date:             2018/5/20
   Create by :       PyCharm
   Check status:     https://waynehfut.github.io
-------------------------------------------------
"""
__author__ = 'Wayne'

from keras.models import *
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from keras.initializers import glorot_uniform
from keras.callbacks import TensorBoard, ModelCheckpoint
from VisCallback import DrawCallback


class ClassfiyNet(object):
    def __init__(self, shape=(864, 1152, 3), classes=2):
        self.shape = shape
        self.classes = classes

    def identity_block(self,X, f, filters, stage, block):
        """
        Implementation of the identity block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters
        X_shortcut = X
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def convolutional_block(self,X, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters
        X_shortcut = X
        X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)
        X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    def resnet_50(self):
        """
        Network definition,you **need** input your own input shape and classes.
        The network is inspired by resnet50, which is proposed by:
        Wu, Songtao, Shenghua Zhong, and Yan Liu. 2017. “Deep Residual Learning for Image Steganalysis.”
        Multimedia Tools and Applications, 1–17. doi:10.1007/s11042-017-4440-4.
        Implementation of the popular ResNet50 the following architecture:
            CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
            -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        :param input_shape: your dataset single slice size, tensor of shape(n_H,n_w,n_C)
        :param classes: your datasets' classes
        :return: resnet-based model
        """
        X_input = Input(self.shape)
        X = ZeroPadding2D((3, 3))(X_input)
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
        X = AveragePooling2D(pool_size=(2, 2))(X)
        X = Flatten()(X)
        X = Dense(self.classes, activation='softmax', name='fc' + str(self.classes),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        model = Model(inputs=X_input, outputs=X, name='resnet_50')
        return model

    def vgg16(self):
        """
        Network definition,you **need** input your own input shape and classes.
        The network is inspired by vggnet, which is proposed by:
        Simonyan, Karen, and Andrew Zisserman. 2015. “Very Deep Convolutional Networks for Large-Scale Image Recognition.”
        International Conference on Learning Representations (ICRL), 1–14. doi:10.1016/j.infsof.2008.09.005.
        Implementation of the popular vgg16 the following architecture:
        :param input_shape: your dataset single slice size, tensor of shape(n_H,n_w,n_C)
        :param classes: your datasets' classes
        :return: vgg16-based model
        """
        X_input = Input(self.shape)
        # Block 1
        X = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(X_input)
        X = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(X)

        # Block 2
        X = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(X)
        X = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(X)

        # Block 3
        X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(X)
        X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(X)
        X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(X)

        # Block 4
        X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(X)
        X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(X)
        X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(X)

        # Block 5
        X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(X)
        X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(X)
        X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(X)

        X = Flatten(name='flatten')(X)
        X = Dense(4096, activation='relu', name='fc1')(X)
        X = Dense(4096, activation='relu', name='fc2')(X)
        X = Dense(self.classes, activation='softmax', name='predictions')(X)
        model = Model(inputs=X_input, outputs=X, name="vgg16")
        return model

    def your_net(self):
        """
        **Implement your network here**
        :return:
        """
        X_input = Input(self.shape)
        # start your network #
        X = ZeroPadding2D((3, 3))(X_input)
        X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), name='max_pool')(X)
        X = Flatten()(X)
        X = Dense(self.classes, activation='softmax', name='predictions')(X_input)
        model = Model(inputs=X_input, outputs=X, name="yournet")
        # end your network #
        return model

    def train_and_test_resnet(self, train_X, train_Y, test_X, test_Y, epochs=1, batch_size=1, network="res",
                              runtime_plot=False):
        """
        train and test resnet performance on your data set,you can switch the net by init the network with {"res","vgg"}
        **Attention: if you implement your network you should afferent `network` parameters with bala(no mean str will be ok)**
        :param X_train: training data
        :param Y_train: training label
        :param X_test: test data
        :param Y_test: test label
        :param epochs: the iteration epoch
        :param batch_size:
        :return:
        """
        if network == "res":
            self.model = self.resnet_50()
        elif network == "vgg":
            self.model = self.vgg16()
        else:
            self.model = self.your_net()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        drawCallback = DrawCallback(runtime_plot=runtime_plot)  # real-time loss curve
        tbCallBack = TensorBoard(log_dir='./Log', histogram_freq=0, write_graph=True,
                                 write_images=True)
        checkpointCallBack = ModelCheckpoint('./Npy_data/waynehfut.hdf5', monitor='loss', verbose=1,
                                             save_best_only=True)
        self.model.fit(x=train_X, y=train_Y, batch_size=batch_size, epochs=epochs,
                       callbacks=[checkpointCallBack,tbCallBack,drawCallback])
        preds = self.model.evaluate(x=test_X, y=test_X)
        return preds

    def val_test(self, inputdata, network="res"):
        """
        validation your model on your img
        :param input_data: img input your should notice that your data size should be same as input_data on model training
        :param network: network type
        :return: predict result
        """
        if network == "res":
            model = self.resnet_50()
        elif network == "vgg":
            model = self.vgg16()
        else:
            model = self.your_net()
        model.load_weights('./Npy_data/waynehfut.hdf5')
        predict_result = model.predict(inputdata, batch_size=1, verbose=1)
        return predict_result
