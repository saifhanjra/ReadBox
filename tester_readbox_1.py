# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 12:16:05 2016

@author: engrs
"""

#### Libraries
# Standard library
import random

import matplotlib.pyplot as plt


# Third-party libraries
import numpy as np

class Network():
    

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases=[]
        self.weights=[]
        self.biases_final=[]
        self.weights_final=[]
#        
#        
        for y in sizes[1:]:
            biases_cpy=np.random.randn(y,1)
            self.biases.append(biases_cpy)
        
        
        
        
        for x,y in zip(sizes[:-1], sizes[1:]):
            weights_cpy=np.random.randn(y,x)
            self.weights.append(weights_cpy)
        


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        self.max_epochs=epochs         ### this class variable is storing the max no of epochs
        self.epoch_res=[]             ### i want to store the efficency of my program
        if test_data:
            n_test= len(test_data)
    
    
        n=len(training_data) 
        
        for j in xrange (epochs):   ## for how many times i want to train and test my network 
            random.shuffle(training_data) ## first step in SGD is to shuffle your training data
            mini_batches=[]  ##  I want to convert my whole training data in to mini batches
                        ## I am only interstded in direction of global minima thats why i 
                        ## choose  Stochastic Gradient instead of Gradient descent
                        ## My trainig data contains 50,000 examples
                        ## If i have a neural network with [784,30,200,10]
                        ## i would have to calculate (30*784)+(200*30)+(10*200)=31520 
                        ## 31520 weight updates plus 784+30+200+10=1024 Biase up-da
                        ## Remember we have 50,000 training example so claculating 
                        ## only one update we have to multiply matrix vector 50,000
                        ## times only for singe layer and for forward path we have to do it
                        ## thrice. after this complex algebra caluclation we will move only
                        ## step and one more thing we have 31520 weight updtaes. If i 
                        ## am done with matrix vector multiplication onece then we just use 
                        ## these results for reamining 31519 weights.(Gradient descent is 
                        ## an expensive process )
        
        
            for k in xrange (0,n,mini_batch_size):  ## I have break the whole training examples 
                                                ## into several minibatches
                mini_batch_cpy=training_data[k:k+mini_batch_size]
                mini_batches.append(mini_batch_cpy)
            
        
        
        
            for mini_batch in mini_batches:  ## i will select  the mini_batch one by one 
                                        ## and then i will give the selected mini_batch
                                        ## to function update_mini_batch to calculate the
                                        ## graient.
                self.update_mini_batch(mini_batch,eta)
            
            
            if test_data:
            
                print"Epoch{0}: {1}/{2}".format(j,self.evaluate(test_data),n_test)
                a=self.evaluate(test_data)
                b=10.0
                c=a/b
                print "Efficency during Epoch{0} is : {1}".format(j,c)
                self.epoch_res.append(c)
                if j==self.max_epochs-1:
                
                    epochs_x=[]
                    for i in xrange(self.max_epochs):
                        i=i+1
                        epochs_x.append(i)
                    plt.plot(epochs_x,self.epoch_res, label='learning rate = 3')  #### self.epochs_res=epochs_results
                    plt.xlabel('Epochs')
                    plt.ylabel('Efficency')
                    plt.legend()
                    plt.grid()
                
                    
                    
                    
                 
                
                 
                    
                if j == self.max_epochs-1:
                    
                    for bb in self.biases:
                        self.biases_final_cpy=bb
                        self.biases_final.append(self.biases_final_cpy)
                        
#                    print ('biases =', self.biases_final)
                        
                        
                        
                    for ww in self.weights:
                        self.weights_final_cpy=ww
                        self.weights_final.append(self.weights_final_cpy)
                        
#                    print ('weights =', self.weights_final)
                    
                    self.params=[self.weights_final,self.biases_final]
                        
                        
                        
                
                
            
            else:
                print"Epoch{0} is complete".format(j)
    def my_params(self):
        return self.params
        


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        
        
        
        nabla_b=[]
        nabla_w=[]
        
        for b in self.biases:
            nabla_b_cpy= np.zeros(b.shape)
            nabla_b.append(nabla_b_cpy)
            
        for w in self.weights:
            nabla_w_cpy=np.zeros(w.shape)
            nabla_w.append(nabla_w_cpy)
            
            
            
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w =self.backprop(x,y) ### call the function backprop 
            
            """ back prop function will take the first tuple from mini_batch which in the 
        which is in the form of x,y  x is the single training input and y is the actuall
        output if x is apllied as input rest of explanation about back_prop will be written
        in the function backprop(x,y) here important is delt_nabla_b which is calculated 
        by backprop function i assume this as updated biase matrix with the same dimension
        as net.biases matrix already has and other important thing is delta_nabla_w which is
        updated weight matrix has"""
            
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            """nabla_b is  updated layer by layer for every training input of mini_batch"""
            
            
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            """nabla_w are updated similarly layer by layer for every trainig i/p of 
            mini_batch"""
            
            
            
            
        """once the training i/p in mini_batch is completed now is the time to update
          our weights matrix and biases matrix using gardient descent"""    
          
          
        """" Mini batch gradient Descent Algoritham"""
          
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b) ]
        
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        nabla_b=[]
        nabla_w=[]
        for b in self.biases:
            nabla_b_cpy=np.zeros(b.shape)
            nabla_b.append(nabla_b_cpy)
            
        for w in self.weights:
            nabla_w_cpy=np.zeros(w.shape)
            nabla_w.append(nabla_w_cpy)
            
        """1st step take an input x from mini_batch and run your neural"""
          
        ### network for the very first using randomly intialize your weights 
        ### and biases and after each mini_batch the weights and are going to be 
        ### updated.
          
        activation=x
        activations=[x]
        zs=[]
          
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,activation)+b
            activation=sigmoid_vec(z)
            activations.append(activation)
            zs.append(z)
            
            
         #### after the compltion of this for loop what we get?
        #####  activation value of each neuron in every layer and the activation 
        #### is going to be saved in in list and the velue of z which is dot product of 
        ### weights and activation units of previous layers and also store them in an other 
        ###list
            
        """ 2nd step calculate the error starting from last layer and then propogate this error
           backward till the 1st hidden"""
           
         ### Beautiful thing about propogation,error caclculated with the help of backpropgation
         ### have relation with rate of change of cost function w.r.t   every individual weight and 
         ### every biase present in the netowrk. so this make our life easy at the end of day for
         #### updating weights and biases after every mini_batch using stochastic gardient descent
         ### how this works lets have a look
            
            
            
        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime_vec(zs[-1]) ## error
        nabla_b[-1] =delta  ## rate of change of cost_w.r.t to biases
        nabla_w[-1]=np.dot(delta,activations[-2].transpose()) ###  rate of change of cost_fun
                                                                # w.r.t to weith weights
        """ Above is Last layer error calculation and rate of change of cost_fun w.r.t to biases
        and weights connect in last hidden layer"""
    
        """ Now starting from 2nd last_layer and move backward till i reach 1st hidden layer"""
        
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists
        
        
        for l in xrange(2,self.num_layers):
            ### we need to calculate the errors from 2ndlast
            ### to 2nd layer of so this is reason we have 
            ### start the loo from 2.
            z=zs[-l]
            spv=sigmoid_prime_vec(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*spv
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
            
            
            
        return(nabla_b,nabla_w)  


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        
        test_results=[]
        for x,y in test_data:
         test_results_cpy=(np.argmax(self.feedforward(x)),y)
         test_results.append(test_results_cpy)
         
         
        return sum(int(x == y) for (x, y) in test_results)
         
         
         
         
        
        
#        for x,y in test_results:
#            if int(x==y):
#            return sum(int(x==y))
            

        
        
    def cost_derivative(self,output_activation, y):
        
        """ Return the vector of partial derivatives 
        partial C_X/ partial for the out of activations"""
        
        ### what this function do, most important thing is what kind of cost_function 
        ### we are using here i feel its better to use squared error cost function because I have 
        ###lot data to train  and i can increase the number of layers of nurron or total number of 
        ###in each individual layer so its better to use easiest cost function available
        ## deriavtive of cost function w.r.t to output activation can be repesented as 
        ###(shown below), if the cost function is squraed error
        
        return(output_activation-y)
        
        
        
        
    #    def cost_derivative(self, output_activations, y):
    #        """Return the vector of partial derivatives \partial C_x /
    #        \partial a for the output activations."""
    #        return (output_activations-y) 
    
        
        
                   
    def get_parameters(self):
        return self.weights_final, self.biases_final
        
        

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)
