import tensorflow as tf
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
    self._n_iterations = int(n_iterations)

    alpha = float(alpha)
    if sigma is None:
      sigma = max(m,n)/2.0
    else:
      sigma = float(sigma)

    if decay_in_time is not True:
      decay_in_time = False

    #Initialize graph
    self._graph = tf.Graph()
    
    with self._graph.as_default():
      ## VARIABLES AND CONSTANT
      # Initialize the vectors of weights for all neurons
      # stored as a matrix Variable of size [m*n, dim]
      self._weights = tf.Variable(tf.random_normal([m*n, dim]))
      
      # Matrix of size [m*n, 2] for SOM grid locations of neurons
      # example: neuron_0 -> position: [0,0]
      #          neuron_1 -> position: [0,1], etc.
      self._vects_locations = tf.constant(
          np.array(list(self._neuron_locations(m, n))))
      
      ## PLACEHOLDERS
      # The training vector
      self._vect_input = tf.placeholder(tf.float32, [dim])
      # Iteration number (time)
      self._iter_input = tf.placeholder(tf.float32)
      
      ## TRAINING OP
      # Compute BMU (Best Matching Unit) given a input vector
      # get the index of the neuron which has the smallest euclidean 
      # distance between its weights and the input vector
      bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(
          self._weights, tf.reshape(tf.tile(
              self._vect_input, [m*n]), [m*n, dim])), 2), 1)),0)
      
      # Extract the location of BMU based on the its index
      slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
      bmu_loc = tf.reshape(tf.slice(self._vects_locations, slice_input, 
                                    tf.constant(np.array([1, 2]))), [2])
      
      # Compute the alpha and sigma values based on iteration number
      _learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                 self._n_iterations))
      if not decay_in_time:
        _learning_rate_op = 1
        
      _alpha_op = tf.multiply(alpha, _learning_rate_op)
      _sigma_op = tf.multiply(sigma, _learning_rate_op)
      
      # Compute the learning rate for all neurons based on iteration
      # and location
      bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
          self._vects_locations, tf.reshape(tf.tile(
              bmu_loc, [m*n]), [m*n, tf.size(bmu_loc)])), 2), 1)
      
      neighbourhood_decay = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, tf.float32), tf.pow(_sigma_op, 2))))
      
      learning_rate_op = tf.multiply(_alpha_op, neighbourhood_decay)
      
      # Update the weights vectors
      learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])

      delta_weights = tf.multiply(learning_rate_multiplier, tf.subtract(
          tf.reshape(tf.tile(self._vect_input, [m*n]), [m*n, dim]), 
          self._weights)) 
      
      weights_op = tf.add(self._weights, delta_weights)
      
      self._training_op = tf.assign(self._weights, weights_op)
      
      ## INITIALIZE SESSION
      self._sess = tf.Session()
      
      # Initialize variables
      init_op = tf.global_variables_initializer()
      self._sess.run(init_op)
      
  def _neuron_locations(self, m, n):
    '''
    Yields one by one the 2-D locations of the individual neurons in the SOM.
    '''
    # Nested iterations over both dimensions 
    # to generate all 2-D locations in the map
    for i in range(m):
      for j in range(n):
        yield np.array([i, j])
        
  def train(self, input_vects):
    '''
    Trains the SOM
    
    Args:
      input_vects: Data to cluster. It should be a numpy array of dimension [dim] 
    '''
    for iter_no in range(self._n_iterations):
      # Train with each input vector one by one
      for vect in input_vects:
        self._sess.run(self._training_op,
                       feed_dict={self._vect_input: vect,
                                  self._iter_input: iter_no})
        
    # Store a centroid grid
    centroid_grid = [[] for i in range(self._m)]
    self._weightages = list(self._sess.run(self._weights))
    self._locations = list(self._sess.run(self._vects_locations))
    
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
