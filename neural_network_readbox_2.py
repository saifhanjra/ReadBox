# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 13:30:50 2016

@author: engrs
"""

import readbox_loader
import tester_readbox_2

data= readbox_loader.train_test_data()
test_data, training_data=data


net = tester_readbox_2.Network([24,18,9], cost=tester_readbox_2.CrossEntropyCost)
net.large_weight_initializer()

net.SGD(training_data, 1000, 10, 0.10, lmbda = 0.000, evaluation_data=test_data, monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=True, monitor_training_cost=True)


