"""
Script to build simple Logistic Regression classifier.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(x):
    """
    Helper function to compute the sigmoid of a given array
    
    input
    -----
    x: ndarray-like, output vector of linear function
    
    output
    -----
    s: ndarray-like, sigmoid of input"""
    
    sig = 1/(1+np.exp(-x))
    return sig

def init_wb(size, init_val=0):
    """
    Function to initialize weights vector/bias term 
    
    input
    -----
    size: int
        length of weights vector
    kind: int or float
        value to init 
        
    output
    -----
    w, b
        """
    if init_val == 0:
        w = np.zeros((size, 1))
        b = 0
    else:
        w = np.ones((size, 1)) * init_val
        b = init_val
    return w, b

def propagation(X, y, w, b):
    """
    Compute resulting cost function and gradients from
    forward propagation and backward propagation respectively
    
    input
    -----
    X: ndarray-like, training set to fit
    y: vector-like, X's corresponding target
    w: weight vector
    b: bias term
    
    output
    ------
    tuple; dw, db, cost
        dw: numeric, gradient of the weight vector
        db: numeric, gradient of the bias vector
        cost: vector-like, cost of weight, bias terms
    """
    
    samples = X.shape[1]
    
    # Forward Propagation (left to right)
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/samples) * np.sum(y * np.log(A) + (1 - y)*(np.log(1 - A)))
    
    # Backward Propagation (right to left)
    dw = (1/samples) * np.dot(X, (A - y).T)
    db = (1/samples) * np.sum(A - y)
    return {'dw': dw, 'db': db, 'cost':np.squeeze(cost)}

def grad_descent(X, y, w, b, n, eta):
    """
    Function running gradient descent to optimize w and b
    to minimize the cost function

    input
    -----
    X: ndarray-like, training set to fit
    y: vector-like, X's corresponding target
    w: weight vector
    b: bias term
    n: number of iterations to loop
    eta: learning rate
    
    output
    ------
    
    
    """
    costs = []
    
    for i in range(n):
        
        prop = propagation(X, y, w, b)
        dw, db, cost = prop['dw'], prop['db'], prop['cost']
        
        w  = w - (eta * dw)
        b = b - (eta * db)
        
        if i % 100 == 0:
            costs.append(cost)
    
    return {'w':w, 'dw':dw, 'b':b, 'db':db, 'costs':costs}

def prediction(X, w, b):
    """
    Predict binomial classification based on trained w, b
    
    input
    -----
    w: ndarray-like
        weights vector
    b: int or float
        bias term
    X: ndarray-like
    
    output
    -----
    y_pred: np.array
        array of predictions to corresponding X instances
        """
    samples = X.shape[1]
    y_pred = np.zeros((1, samples))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        # Bin probabiliites into actual predictions
        y_prob = A[0, i]
        if y_prob > 0.5:
            y_pred[:, i] = 1
        else:
            y_pred[:, i] = 0
            
    return y_pred


class LogRegClf(BaseEstimator, ClassifierMixin):
    
    def __init__(self, init_params=0, n_iterations=100, eta=0.05):
        self.init_params = init_params
        self.n_iterations = n_iterations
        self.eta = eta
        
    def fit(self, X, y=None):
        self._w, self._b = init_wb(X.shape[0], init_val=self.init_params)
        gd = grad_descent(X, y, self._w, self._b, self.n_iterations, self.eta)
        self.w_ = gd['w']
        self.b_ = gd['b']
        self.dw_ = gd['dw']
        self.db_ = gd['db']
        self.cost_ = gd['costs']
        
    def predict(self, X, y=None):
        try:
            getattr(self, 'w_')
            getattr(self, 'w_')
        except:
            raise RuntimeError('Train classifier using .fit() before running .predict()')
            
        return prediction(X, self.w_, self.b_)
    
    def score(self, X, y_true=None):
        
        return((self.predict(X) == y_true).sum()) / len(y_true)