import numpy as np
from mlp_layer import Layer

class MultiLayerPerceptron(object):
    '''Neural Network: MultiLayer Perceptron
    '''
    def __init__(self):
        self.n_layers = 0 
        self.layer = {}
        self.weights = {}
    
    def add_layer(self, units, rate, activation):#adicionar error_max
        '''Create your MultiLayer Perceptron model. Add as many layers as you need.
        
        Args:
            units (int): Number of neurons in this layer.
            rate (float): Learning rate, e.g. eta.
            activation (int): activation functions, e.g. 'tanh' or 'sigmoid'.
        '''
        self.layer[self.n_layers] = Layer(units, rate, activation)
        self.n_layers += 1
    
    def fit(self, X, y, epochs, error_max=1e-3):
        '''Train a model with the data input X and the desired output y.

        Args:
            X(numpy.ndarray): Training data (input).
            y(numpy.ndarray): Training data (desired output).
            epochs(int): Number of epochs.
            error_max (float): Maximum acceptable error.
        '''
        epoch = 0
        error = 0
        
        while (epoch<epochs):
            eqm_previous = self._lms(X, y)
            
            for example in range(X.shape[0]):
                self._foward(X[example])
                self._backprop(X[example], y[example])
                
            epoch += 1
            eqm_current = self._lms(X, y)
            error = abs(eqm_current - eqm_previous)
            
            if(error<error_max):   
                break
    
    def _foward(self, X):        
        for i in range(self.n_layers):
            try: #foward com a saida da camada anterior
                self.layer[i].foward(self.layer[i-1].yhat.T)
                
            except: #se nao tiver camada anterior, use os valores de entrada X
                self.layer[i].foward(X)            
                
                
    def _backprop(self, X, y):
        for i in reversed(range(self.n_layers)):
            try: #foward com a saida da camada anterior
                self.layer[i].backprop(y_prev = self.layer[i-1].yhat.T, 
                                       weights_next=self.layer[i+1].weights, 
                                       delta_next=self.layer[i+1].delta)
                
            except: #se nao tiver camada anterior, use os valores de entrada X
                if (i==0): #primeira camada
                    self.layer[i].backprop(y_prev = X, 
                                           weights_next=self.layer[i+1].weights, 
                                           delta_next=self.layer[i+1].delta)
                    
                else: #ultima camada
                    self.layer[i].backprop(y_prev=self.layer[i-1].yhat.T, 
                                           y_next=y)
                    
    def _lms(self, X, y):
        eqm = 0
        p = X.shape[0]
        
        for example in range(p):
            u = self._foward(X[example])
            eqm += (y[example] - self.layer[self.n_layers-1].yhat)**2
        
        eqm /= p
        return eqm   
    
    def predict(self, X):
        '''Predict the outcome to a data using a trained model.
        
        Args:
            X: data from which we want to predict the result.
        '''
        output = np.array([])
        
        for example in X:
            self._foward(example)
            
            try:
                output = np.vstack([output, self.layer[self.n_layers-1].yhat])
            except:
                output = self.layer[self.n_layers-1].yhat
                
        return output
