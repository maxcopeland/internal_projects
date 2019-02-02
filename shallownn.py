"""
Author: Max Copeland

Script to build a simple shallow Neural Network optimized
with gradient descent. See NN_Primer.ipynb for corresponding
demo.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def init_wb(X, y, n_neurons=4):
    """
    Randomly initialize weights for linear function
    to pass to hidden layer in shallow neural network
    (bias term initialized to zero)
    
    input
    -----
    x_size: int
        number of training examples (X.shape[0])
    y_size: int
        output size (Y.shape[0])
    n_neurons: int
        number of nodes within hidden layer
        
    output
    -----
    dict of weight/bias terms
        keys: w1, w2, b1, b2
        """
    np.random.seed(2)
    
    x_size = X.shape[0]
    y_size = y.shape[0]
    
    # Terms fit to first layer
    
    w1 = np.random.randn(n_neurons, x_size) * 1e-2
    b1 = np.zeros((n_neurons, 1))
    
    # Terms to fit to output layer
    w2 = np.random.randn(y_size, n_neurons) * 1e-2
    b2 = np.zeros((y_size, 1))
    
    return {'W1': w1, 'b1':b1, 'W2':w2, 'b2':b2}


def prop_forward(X, params, backward=False):
    """
    Function to run forward propagation in shallow 
    neural network
    
    input
    -----
    X: ndarray 
        data to fit with shape (x_size, n_samples)
    params: dict 
        weights/biases returned from init_wb()
    backward: bool
        if True, run backward propagation
        
    output
    -----
    dict of z, a terms to forward propogate one hidden layer
        z1: input layer fit to linear function
        a1: tanh activation on z1
        z2: a1 activation layer fit to linear function
        a2: sigmoid function fit to a1 as output
        
    """
    
    w1 = params['W1']
    b1 = params['b1']
    w2 = params['W2']
    b2 = params['b2']
    
    # Fit first layer to linear function
    z1 = np.dot(w1, X) + b1
    # Apply activation function to first layer
    a1 = np.tanh(z1)
    # Fit output layer to resulting a1 activation
    z2 = np.dot(w2, a1) + b2
    # Run sigmoid activation on output
    a2 = sigmoid(z2)
    
    return {'Z1':z1, 'A1':a1, 'Z2':z2, 'A2':a2}


def cost_func(a2, y, params):
    """
    Calculate cross-entropy cost 
    
    input
    -----
    a2: ndarray shape (1, samples)
        sigmoid activation of output layer
    y: ndarray shape(1, samples)
        true target
    lin_params: dict
        weight/bias terms
        keys: w1, w2, b1, b2
    
    output
    -----
    float, cross entropy cost 
    """
    
    samples = y.shape[1]
    
    loss = np.add(np.multiply(np.log(a2), y), np.multiply(1-y, np.log(1-a2)))
    cost = -(1/samples) * np.sum(loss)
    
    return cost


def prop_back(X, y, params, forward_prop_dict):
    """
    Compute the backward propogation across
    shallow NN with one hidden layer
    
    input
    -----
    X: ndarray 
        data to fit with shape (x_size, n_samples)
    params: dict 
        weights/biases returned from init_wb()
    forward_prop_dict: dict
        dict with keys 'z1', 'a1', 'z1', 'a2'
        
    output
    -----
    dict of gradients
        keys:
        dw1: w gradient of first layer
        db1: b gradient of first lyaer
        dw2: w gradient of second layer
        db2: b gradient of second layer
    """
    samples = X.shape[1]
    
    w1 = params['W1']
    w2 = params['W2']
    
    a1 = forward_prop_dict['A1']
    a2 = forward_prop_dict['A2']
    
    # Calculate backward propagation from output to input functions
    dz2 = a2 - y
    dw2 = (1/samples) * np.dot(dz2, a1.T)
    db2 = (1/samples) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
    dw1 = (1/samples) * np.dot(dz1, X.T)
    db1 = (1/samples) * np.sum(dz1, axis=1, keepdims=True)
    
    return {'dW1':dw1, 'dW2':dw2, 'db1':db1, 'db2':db2}    


def hypertune(params, backward_prop_dict, eta = 1.2):
    """
    Tunes weight/bias terms using gradient descent for
    *one* iteration
    
    input
    -----
    prop_dict: dict
        contains w1, dw1, w2, dw2, b1, db1, b2, db2
    eta: numeric
        learning rate
        
        
    output
    -----
    dict: updated parameters
    """
    w1 = params['W1']
    b1 = params['b1']
    w2 = params['W2']
    b2 = params['b2']
    
    dw1 = backward_prop_dict['dW1']
    db1 = backward_prop_dict['db1']
    dw2 = backward_prop_dict['dW2']
    db2 = backward_prop_dict['db2']

    w1 = w1 - (eta * dw1)
    b1 = b1 - (eta * db1)
    w2 = w2 - (eta * dw2)
    b2 = b2 - (eta * db2)
    
    return {'W1':w1, 'b1':b1, 'W2':w2, 'b2':b2}


class ShallowNN(BaseEstimator, ClassifierMixin):
    """
    Nerual network with one hidden layer
    """
    def __init__(self, nodes=4, n_iter=1e4, eta=1.2):
        self.nodes = 4
        self.n_iter = int(n_iter)
        self.eta = eta
        self.cost_ = []
        
    def fit(self, X, y=None):
        params = init_wb(X, y, self.nodes)
        
        for i in range(0, self.n_iter):
            forward = prop_forward(X, params)
            cost = cost_func(forward['A2'], y, params)
            backward = prop_back(X, y, params, forward)
            params = hypertune(params, backward, eta=self.eta)
            if i % 1000:
                self.cost_.append(cost)
        self.params_ = params
        self.forward_prop_ = forward
        self.backward_prop_ = backward
        return repr(self)

    def predict(self, X, y=None):
        try:
            getattr(self, "params_")
        except AttributeError:
            raise RuntimeError("Run .fit() on training data before .predict()")
        forward = prop_forward(X, self.params_)
        y_pred = np.array([1 if x>0.5 else 0 for x in np.squeeze(forward['A2'])])
        return y_pred
            
    def score(self, X, y=None):
        y_pred = self.predict(X)
        samples = X.shape[1]
        accuracy = (1 / samples) * (np.dot(y, y_pred.T) + np.dot(1-y, 1-y_pred.T))
        return accuracy
