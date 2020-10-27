import numpy as 

import Layer, Loss, GradientDescent

class NeuralNet:
    def __init__(self, loss_func, optimiser_method, learning_rate, epoch, partition_size, mode, network_arch):
        self.loss_func = loss_func
        self.optimiser_method = optimiser_method
        self.epoch_count = epoch
        self.partition_size = partition_size 
        self.learning_rate = learning_rate
        self.mode = mode
        self.layers = []
        self.loss  = Loss(self.loss_func)
        self.network_arch = network_arch
        if optimiser_method == 'momentum':
            self.momentum = True
            self.optimiser = GradientDescent(True,self.learning_rate,self.beta)
        else:
            self.optimiser = GradientDescent(False,self.learning_rate)
            
    def createNetwork(self):
        network_layers = []
        for index, layer in  enumerate(self.network_arch):
                if layer['layer_type'] != 'input':
                    self.layers.append(Layer(self.network_arch[index-1]['size'],layer['size'], layer['activation'],layer['layer_type']))

    def forwardPass(self, inputs, output):
        layer_out = inputs
        for layer in self.layers:
            layer_out = layer(inputs)
            inputs = layer_out
        if self.mode.lower() == 'train':
            self.loss.getLoss()(layer_out,output)
        elif self.mode.lower() == 'test':
            return layer_out
    
    def backProp(self, inputs,**kwargs):
        upstream_gradient = self.loss.loss_derivative
        for index, layer in enumerate(reversed(self.layers)):
            if layer.layer_type == 'output':
                upstream_gradient =  np.multiply(layer.activation_derivative, upstream_gradient)
                upstream_gradient_w =  np.matmul(self.layers[len(self.layers)-2].y_activation.T, upstream_gradient) 
            if layer.layer_type == 'hidden':
                upstream_gradient =  np.matmul(upstream_gradient, self.layers[len(self.layers) -index].w.T)
                upstream_gradient = np.multiply(upstream_gradient,layer.activation_derivative)
                if (len(self.layers)-index-1) != 0:
                    upstream_gradient_w = np.matmul(self.layers[len(self.layers) -index -2].y_activation.T,upstream_gradient)
                else:
                    upstream_gradient_w = np.matmul(inputs.T,upstream_gradient)
            upstream_gradient_b = np.sum(upstream_gradient,axis=0).T
            self.optimiser(layer, upstream_gradient_w, upstream_gradient_b)

        for layer in self.layers:
            layer.w = layer.w + layer.w_delta
            layer.b = layer.b + layer.b_delta
    
    def train(self, inputs, outputs):
        inputs = np.array_split(inputs, self.partition_size)
        Y =  np.array_split(outputs, self.partition_size)
        for i in range(self.epoch_count):
            for inp_batch, out_batch in zip(inputs, Y):  
                self.forwardPass(inp_batch, out_batch)
                self.backProp(inp_batch)
            if i%500 == 0:
                print('Epoch:{} Loss: {}'.format(i+1,self.loss.loss))
