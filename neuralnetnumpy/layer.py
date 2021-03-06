import numpy as np
from neuralnetnumpy import activation 

class Layer:
    def __init__(self, input_count, neruon_count, activation_func,layer_type):
        self.layer_type = layer_type
        self.activation_func = activation_func
        self.w = np.random.randn(input_count,neruon_count) / np.sqrt(input_count) 
        self.w_delta = np.zeros((input_count,neruon_count))
        self.b = np.ones((neruon_count, 1))
        self.b_delta = np.zeros((neruon_count, 1))
        self.y = np.random.uniform(low = 0, high = 1, size = (neruon_count, 1))
        self.y_activation = np.random.uniform(low = 0, high = 1, size = (neruon_count, 1))
        self.activation_derivative = np.random.uniform(low = 0, high = 1, size = (neruon_count, 1))
  
    def __call__(self,input):
        self.y = np.matrix(np.matmul(input,self.w)+ self.b.T)
        activationmethod = activation.Activation(self.activation_func)
        self.y_activation = np.matrix(activationmethod.getActivation()(self.y))
        self.activation_derivative = activationmethod.activation_derivative
        return self.y_activation
