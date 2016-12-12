# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:23:19 2016

@author: engrs
""" 
list_of_lists=[]
inpt_features_vec_list=[]
output_label_list=[]
import numpy as np

with open('readbox_mod.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(':')]
        list_of_lists.append(inner_list)
        
      
for i in xrange(len (list_of_lists)):
    inpt_features=list_of_lists[i][0]
    label=list_of_lists[i][1]
    inpt_features_vec=[elt.strip() for elt in inpt_features.split(',')]
    inpt_features_vec=map(float,inpt_features_vec)
    inpt_features_vec=np.asarray(inpt_features_vec,dtype=np.float64)
    inpt_features_vec_list.append(inpt_features_vec)
    output_label_list.append(label)
    

#    
inpt_features_vec_list=np.asarray(inpt_features_vec_list, dtype=np.float64)
#
output_label_list=map(int,output_label_list)
output_label_list=np.asarray(output_label_list)
    
    
    
    
training_data=(inpt_features_vec_list,output_label_list)

tr_d=training_data

a=tr_d[0]
b=tr_d[1]

for i in xrange(len(b)):
    a=np.zeros((9,1))
    j=b[i]
    a[j]=1
    


    
    
#import cPickle
#f=open('books_dna_readbox.pkl','wb')
#cPickle.dump(training_data_readbox,f)
#f.close()
#


            
        
        
        
        
    
