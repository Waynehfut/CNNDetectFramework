# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       LogUtils.py on CPLID_Detect
   Description :
   Author :          Wayne
   Date:             2018/5/21
   Create by :       PyCharm
   Check status:     https://waynehfut.github.io
-------------------------------------------------
"""
__author__ = 'Wayne'
import logging
import time


class LogUtils():
    def __init__(self, isprint=1):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='./Log/log_at_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log',
                            filemode='w')
        self.isprint = isprint

    def info(self, info):
        logging.info(info)
        if self.isprint == 1:
            print(info)

    def warning(self, warning):
        logging.warning(warning)
        if self.isprint == 1:
            print(warning)

    def exception(self, exception):
        logging.exception(exception)
        if self.isprint == 1:
            print(exception)

    def stopLogging(self):
        logging.getLogger().handlers = []
        if self.isprint == 1:
            print("Stop logging")

    def report(self, info):
        logging.info("#####----Report----#####")
        logging.info(info)
        if self.isprint == 1:
            print(info)
        logging.info("#####----Report----#####")
