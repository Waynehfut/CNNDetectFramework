# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       ImageUtils on CPLID_Detect
   Description :     This file implements the image handler which helps you easily import image from your own folder.
                     and also provides some data expand methods and visualized methods to help you locate your concern
                     about the training processing and validation processing.
   Author :          Wayne
   Date:             2018/5/20
   Create by :       PyCharm
   Check status:     https://waynehfut.github.io
-------------------------------------------------
"""
__author__ = 'Wayne'
import numpy as np
import glob
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from LogUtils import LogUtils

logger = LogUtils()


class ImageUtils(object):
    """
    A class used to manage images
    By init the image sources path you can import image and split data-set with specific percentage.
    By init the output path you can output the numpy array to the RGB or Gray image.
    By using the ImageDataGenerator method to create new data to enhance the data sources.
    """

    def __init__(self):
        self.train_img_path = ""  # train img path
        self.train_label_path = ""  # train label path
        self.test_img_path = ""  # test img path
        self.test_label_path = ""  # test label path
        self.log_path = ""  # output log
        self.output_path = ""  # output path
        self.img_path = "Data_test"  # img path
        self.img_type = 'jpg'
        self.generate_dic = ImageDataGenerator(  # generator for data-sets
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    def convert_to_one_hot(self, vector_label, classes):
        """
        Cover [[1,0,1,2]] to [[0,1,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        :param classes: number of your classes
        :return: matrix only contain 1 or 0, ref on the original class.
        """
        vector_label = np.eye(classes)[vector_label.reshape(-1)].T
        return vector_label

    def load_data(self, classes, test_size=0.2):
        """
        Load img from path (and divide into trian-test schema <Optional>),classes are defined by dirs name.
        Firstly, func will load img with classes, and you should notice that your path may not match my schema:
        "img_path/{img_class}/images/{specific_img}.{img_type}"
        Secondly, all image files will sort by filename, stored in img_t with shape of (num,width,height,channels)
        Thirdly, file will divide all file to
        :return: Return numpy array with :
                X_train->(nums,width,height,channels)
                Y_train->(nums,labels)
                X_test->(nums,width,height,channels)
                Y_test->(nums,labels)
        """
        logger.info("***\tloading img\t***")
        class_label = 0
        img_t = None
        img_l = None
        for class_i in classes:
            self.class_i = glob.glob(self.img_path + '/' + class_i + '/images/*.' + self.img_type)
            for item in self.class_i:
                if img_t is None:
                    img_t = np.array(img_to_array(load_img(item)))[np.newaxis]
                else:
                    img_t = np.append(img_t, img_to_array(load_img(item))[np.newaxis], axis=0)
                if img_l is None:
                    img_l = np.array([class_label])
                else:
                    img_l = np.append(img_l, np.array([class_label]), axis=0)
            class_label += 1
        img_l = img_l.reshape((1, img_l.shape[0]))
        img_l = self.convert_to_one_hot(img_l, 2).T

        logger.info("{0},{1}".format(img_t.shape, img_l.shape))
        logger.info("***\tloaded img\t***")
        np.save('Npy_data' + '/imgs.npy', img_t)
        np.save('Npy_data' + '/labels.npy', img_l)
        logger.info("***\tsaved img and labels\t***")
        logger.info("{0} will be set for training set\n{1} will be set for test set\n".format(1 - test_size,
                                                                                              test_size))

        train_X, test_X, train_Y, test_Y = train_test_split(img_t, img_l, test_size=test_size, random_state=0)
        return train_X, test_X, train_Y, test_Y
