import numpy as np

class SOM(object):
  '''
  2-D Self-Organizing Map with Gaussian Neighbourhood function and linearly decreasing learning rate.
  '''
  # To check if the SOM has been trained
  _trained = False
  
  def __init__(self, m, n, dim, n_iterations = 100, alpha=.1, sigma=None, decay_in_time = True):
    '''
    Initializes all necessary components of the TensorFlow Graph

    Args:
      m: map width
      n: map height
      dim: dimension of the input vector
      n_iterations: number of iterations (epochs) - Default value is 100
      alpha: initial value of the learning rate - Default value is 0.1
      sigma: initial value of neighborhood radius - Default value is max(m, n)/2
      decay_in_time: decay the learning rate value through time
    '''
    # Assign input variables
    self._m = m
    self._n = n
    self._dim = dim
    self._n_iterations = int(n_iterations)

    self._alpha = float(alpha)
    if sigma is None:
      self._sigma = max(m,n)/2.0
    else:
      self._sigma = float(sigma)

    self._decay_in_time = True
    if decay_in_time is not True:
      self._decay_in_time = False
      
    ## VARIABLES AND CONSTANT
    # Initialize the vectors of weights for all neurons
    # stored as a matrix Variable of size [m*n, dim]
    self._weights = np.random.randn(m*n, dim)
    
    # Matrix of size [m*n, 2] for SOM grid locations of neurons
    # example: neuron_0 -> position: [0,0]
    #          neuron_1 -> position: [0,1], etc.
    self._vects_locations = np.array(list(self._neuron_locations(m, n)))
    
  def _neuron_locations(self, m, n):
    '''
    Yields one by one the 2-D locations of the individual neurons in the SOM.
    '''
    # Nested iterations over both dimensions 
    # to generate all 2-D locations in the map
    for i in range(m):
      for j in range(n):
        yield np.array([i, j])
        
  def _operations(self, vect_input, iter_input):
    '''
    Compute the operations in the SOM training
    
    Args:
      vect_input: Data to cluster. It should be a 1-D array of dimension [dim] 
      iter_input: Current iterations value
    '''
    # Compute BMU (Best Matching Unit) given a input vector
    # get the index of the neuron which has the smallest euclidean 
    # distance between its weights and the input vector
    bmu_index = np.argmin(np.linalg.norm(
      self._weights - np.repeat(vect_input, self._m*self._n).reshape(vect_input.size,-1).T, 
      axis=1))
    
    # Extract the location of BMU based on the its index
    bmu_loc = self._vects_locations[bmu_index]
    
    # Compute the alpha and sigma values based on iteration number
    _learning_rate_op = 1 - iter_input/self._n_iterations 
    
    if not self._decay_in_time:
        _learning_rate_op = 1
    
    _alpha_op = self._alpha * _learning_rate_op
    _sigma_op = self._sigma * _learning_rate_op
    
    # Compute the learning rate for all neurons based on iteration
    # and location
    bmu_distance = np.linalg.norm(
      self._vects_locations - np.repeat(bmu_loc, self._m*self._n).reshape(bmu_loc.size,-1).T,
      axis=1)
    
    neighbourhood_decay = np.exp(-(bmu_distance/_sigma_op)**2)
    
    learning_rate_op = _alpha_op * neighbourhood_decay
    
    # Update the weights vectors
    learning_rate_multiplier = np.repeat(learning_rate_op, self._dim).reshape(-1,self._dim)
    
    delta_weights = learning_rate_multiplier * (
        np.repeat(vect_input, self._m*self._n).reshape(vect_input.size,-1).T - 
        self._weights)
    
    self._weights += delta_weights
    
  def train(self, input_vects):
    '''
    Trains the SOM
    
    Args:
      input_vects: Data to cluster. It should be a numpy array of dimension [dim] 
    '''
    for iter_no in range(self._n_iterations):
      # Train with each input vector one by one
      for vect in input_vects:
        self._operations(vect, iter_no)
        
    # Store a centroid grid
    centroid_grid = [[] for i in range(self._m)]
    self._weightages = list(self._weights)
    self._locations = list(self._vects_locations)
    
    for index, location in enumerate(self._locations):
      centroid_grid[location[0]].append(self._weightages[index])
    
    self._centroid_grid = centroid_grid
    self._trained = True
    
  def get_centroids(self):
    '''
    Returns a list of 'm' lists, with each inner list containing the 'n' 
    corresponding centroid locations as 1-D NumPy arrays.
    '''
    if not self._trained:
      raise ValueError("SOM not trained yet")
    return self._centroid_grid
  
  def map_vects(self, input_vects):
    '''
    Maps each input vector to the relevant neuron in the SOM grid.
    
    Args:
      input_vects: Data to cluster. It should be a numpy array of dimension [dim] 
    '''
    if not self._trained:
      raise ValueError("SOM not trained yet")
 
    to_return = []
    for vect in input_vects:
      min_index = min([i for i in range(len(self._weightages))],
                      key=lambda x: np.linalg.norm(vect - self._weightages[x]))
      to_return.append(self._locations[min_index])
 
    return to_return