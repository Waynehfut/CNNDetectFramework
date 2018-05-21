# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       Main on CPLID_Detect
   Description :
   Author :          Wayne
   Date:             2018/5/21
   Create by :       PyCharm
   Check status:     https://waynehfut.github.io
-------------------------------------------------
"""
__author__ = 'Wayne'
from NetRefs import *
from ImageUtils import *

myImageUtils = ImageUtils()
classes = ['Defective', 'Normal']
train_X, test_X, train_Y, test_Y = myImageUtils.load_data(classes)
logger.info("Train_x size is{0}\n"
            "Train_y size is{1}\n"
            "Test_x size is{2}\n"
            "Test_y size is{3}\n".
            format(train_X.shape,
                   train_Y.shape,
                   test_X.shape,
                   test_Y.shape))
logger.info("*** Start training net****")
myNetworks = ClassfiyNet()
train_X = train_X / 255
test_X = test_X / 255
myNetworks.train_and_test_resnet(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y, network="res",
                                 runtime_plot=True)
