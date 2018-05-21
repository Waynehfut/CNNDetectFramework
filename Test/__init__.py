# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       __init__.py on CPLID_Detect
   Description :
   Author :          Wayne
   Date:             2018/5/21
   Create by :       PyCharm
   Check status:     https://waynehfut.github.io
-------------------------------------------------
"""
__author__ = 'Wayne'
import numpy as np
import h5py
train_dataset = h5py.File('train_signs.h5')

train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

print('11',train_set_y_orig.shape)
train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
print('22',train_set_y_orig)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

yyy = convert_to_one_hot(train_set_y_orig, 6).T
print('33',yyy)