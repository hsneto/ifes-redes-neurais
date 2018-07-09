import numpy as np

class Adaline(object):
    '''Neural Network: Adaline

    Attributes:
        weight (numpy.ndarray): Calculated weights from the fitting.
        weight_list (numpy.ndarray): Previous calculated weights.
        cost (list): Variation of error in fitting.
        
    Args:
        epochs (int): Number of epochs.
        rate (float): Learning rate, e.g. eta.
        error_max (float): Maximum acceptable error.
    '''
    def __init__(self, epochs = 100, rate = 0.1, error_max = 0.01):
        self.epochs = epochs
        self.rate = rate
        self.error_max = error_max
    
    def fit(self, X, y):
        '''Train a model with the data input X and the desired output y.

        Args:
            X(numpy.ndarray): Training data (input).
            y(numpy.ndarray): Training data (desired output).
        '''
        self.weight = np.random.uniform(0, 1, X.shape[1] + 1) 
        it, self.error_ = 0,0.
        self.cost=[]
        self.weight_list = self.weight

        while it<self.epochs:
            self.error_ = 0
            eqm_previous = self._lms(X, y)

            for i in range(X.shape[0]):
                
                output = self._net_input(X[i])
                update = self.rate*(y[i]-output)
                
                self.weight[0] += update
                self.weight[1:] += update*X[i]
                
            it += 1
            eqm_current = self._lms(X, y)
            self.weight_list = np.vstack([self.weight_list, self.weight])
            self.error_ = abs(eqm_current - eqm_previous)
            self.cost.append(self.error_)

            if(self.error_<self.error_max):      
                break

    def _lms(self, X, y):
        eqm = 0
        p = X.shape[0]
        
        for i in range(p):
            u = self._net_input(X[i])
            eqm += (y[i] - u)**2
        
        eqm /= p
        return eqm      
        
    def _net_input(self, X):  
        return np.dot(X, self.weight[1:]) + self.weight[0]
    
    def predict(self, X_test): 
        '''Predict the outcome to a data using a trained model.
        
        Args:
            X_test: data from which we want to predict the result.
        '''
        return np.where(self._net_input(X_test) >= 0.0, 1, -1)
