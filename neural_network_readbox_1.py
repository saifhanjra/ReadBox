# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 12:18:50 2016

@author: engrs
"""

import readbox_loader
import tester_readbox_1
#import sgd_backprop


data= readbox_loader.train_test_data()
test_data, training_data=data

net=tester_readbox_1.Network([24,18,9])
net.SGD(training_data,100,10, 0.1, test_data=test_data)



'''for respectable efficency plöease choose the following paramters'''
#net.SGD(training_data,150, 10, 0.15, test_data=test_data)

