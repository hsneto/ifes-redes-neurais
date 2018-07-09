# Import libraries:
import numpy as np

### Activation functions: ###
def tanh(x):
    return np.tanh(x)
    
def tanh_ds(x):
    return 1 - np.tanh(x)**2
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def sigmoid_ds(x):
    return sigmoid(x)*(1 - sigmoid(x))

ACTIVATION = {}
ACTIVATION['tanh'] = tanh, tanh_ds
ACTIVATION['sigmoid'] = sigmoid, sigmoid_ds


### Class - Layer: ###
class Layer(object):
    '''Layer structure for neural networks
    
    Attributes:
        yhat (numpy.ndarray): Predicted output from fitting.
        weights (dict of numpy.ndarray): Calculated weights from the fitting for all neurons (starting from index 0).
        
    Args:
        units (int): Number of neurons in this layer.
        rate (float): Learning rate, e.g. eta.
        activation (int): activation functions, e.g. 'tanh' or 'sigmoid'.
    '''
    def __init__(self, units, rate, activation):
        self.units = units
        self.rate = rate
        self.activation = ACTIVATION[activation]
        self.weights = {}  
                    
    def foward(self, X):
        '''Perform forward propagation in this layer based on the example X'''
        self.I = np.array([])
        self.Y = np.array([])
        
        for i in range(self.units):
            try:
                I = self.net_input(X, self.weights[i]) # Calculate with the neuron i weights
                yhat = self.activation[0](I)
        
            except:
                np.random.seed(1)
                self.weights[i] = np.random.uniform(0, 1, X.size + 1)

                I = self.net_input(X, self.weights[i]) # Calculate with the neuron i weights
                yhat = self.activation[0](I)     
                
            try:
                self.I = np.vstack([self.I, I])
                self.yhat = np.vstack([self.yhat, yhat])
                
            except:
                self.I = I
                self.yhat = yhat
              
    def backprop(self, y_prev, y_next=None, weights_next=None, delta_next=None): #delta_k=None p ultima camada y_k = j+1, y_i =j-1
        '''Perform the simplest backpropagation in this layer based on:
        
            y_prev: the output from the previous layer or the network fitting input
            y_next: the expected output (only for the last layer in the neural network)
            weights_next: the calculated weights from the next layer in the neural network\
            (DO NOT use this for the last layer in the neural network)
            delta_next: the calculated delta from the next layer in the neural network\
            (DO NOT use this for the last layer in the neural network)
        '''
        self.delta = np.array([])
        y_prev = np.insert(y_prev, 0, -1) #add bias input
        
        for i in range(self.units):
            try:               
                weights = np.array([])
                
                for j in range(len(weights_next)): #pegar os pesos que saem do neuronio i
                    try:
                        weights = np.vstack([weights, weights_next[j][i+1]])
                        
                    except:
                        weights = weights_next[j][i+1]
                 
                delta = sum(delta_next*weights) * self.activation[1](self.I[i])
                self.weights[i] += self.rate * delta * y_prev
                
            except: #ultima camada
                try:
                    delta = (y_next[i]-self.yhat[i]) * self.activation[1](self.I[i])
                    
                except: #single neuron (usually output layer)
                    delta = (y_next-self.yhat) * self.activation[1](self.I[i])
                
                self.weights[i] += self.rate * delta * y_prev
                
            try:
                self.delta = np.vstack([self.delta, delta])
                
            except:
                self.delta = delta
                                                            
    def net_input(self, X, weights):  
        return np.dot(X, weights[1:]) - weights[0]