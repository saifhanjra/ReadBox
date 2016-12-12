# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:48:01 2016

@author: engrs
"""
import numpy as np

import random 

def load_data():
    list_of_lists=[]
    inpt_features_vec_list=[]
    output_label_list=[]
    with open('readbox_mod.txt') as f:
         for line in f:
             inner_list = [elt.strip() for elt in line.split(':')]
             list_of_lists.append(inner_list)
             
    for i in xrange(len (list_of_lists)):
        inpt_features=list_of_lists[i][0]
        label=list_of_lists[i][1]
        inpt_features_vec=[elt.strip() for elt in inpt_features.split(',')]
        inpt_features_vec=map(float,inpt_features_vec)
        inpt_features_vec=np.asarray(inpt_features_vec,dtype='float64')
        inpt_features_vec_list.append(inpt_features_vec)
        output_label_list.append(label)
    
    output_label_list=map(int,output_label_list)
    output_label_list=np.asarray(output_label_list)        
        
    training_data_readbox=(inpt_features_vec_list,output_label_list)
    return training_data_readbox
    
    
def load_data_wrapper():
    tr_d=load_data()
    training_inputs = [np.reshape(x, (24, 1)) for x in tr_d[0]]
    
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    return training_data
    
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((9, 1))
    e[j] = 1.0
    return e
    
    
    
def train_test_data():
    trainn_data=[]
#    testt_data=[]
    
    data_test=[]
    label_test=[]
    
    a=load_data_wrapper()
    random.shuffle(a)
    for i in xrange(len(a)):
        if i<10:
            e=a[i][0]
            f=a[i][1]
            g=np.argmax(f)
            data_test.append(e)
            label_test.append(g)
            
#            testt_data.append(a[i])
            
#        if i<800:         
        else:
            trainn_data.append(a[i])
            
            
        testt_data=zip(data_test,label_test)
    
    dataa= testt_data, trainn_data
            
        
            
    return dataa
    
    
    
#def trainig_test_tuple():
#    train_x, train_y, test_x, test_y=[]
#    b=train_test_data()
#    te_data,tr_data=b
#    for i in xrange(len(tr_data)):
#        train_x.append(tr_data[i][1])
#        train_y.append(tr_data[0][i])
#        
#        
#    return train_x, train_y

    
    
    
    
    
    

  
    
    

    
             
        
    
    