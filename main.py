import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt 
import matplotlib
import sklearn.linear_model 

from neuralnetnumpy.neuralnet import NeuralNet

def dataScale(X):
    mean_ = np.nanmean(X, axis = 0)
    scale_ = np.nanstd(X, axis = 0)
    print(scale_.shape)
    X -= mean_
    scale_[scale_ == 0.0] = 1.0
    X /= scale_
    return X

def predict(Layers,inputs):
    predictions = np.array([])
    for layer in Layers:
        layer_out = layer(inputs)
        inputs = layer_out
    for pred in layer_out:
        if float(pred)>0.5:
            predictions = np.append(predictions,1)
        else:
            predictions = np.append(predictions,0)
    return predictions


def plot_decision_boundary(pred_func): 
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
    h = 0.01 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10) 
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.gist_stern) 


#********************* Make Moons Data-Set ***************

X, Y = sklearn.datasets.make_moons(200, noise=0.20) 
inputs = np.array(X)

inputs = dataScale(inputs)

outputs = np.array(Y).reshape(200,1)

network_arch = [{'layer_type':'input', 'size':inputs.shape[-1]},
                {'layer_type':'hidden', 'size':6, 'activation':'tanh'},
                {'layer_type':'hidden', 'size':5, 'activation':'tanh'},
                {'layer_type':'hidden', 'size':3, 'activation':'tanh'},
                {'layer_type':'output', 'size':1, 'activation':'sigmoid'}]
loss_func = 'meanSquared'
optimiser_method = 'sgd'
learning_rate = 0.01
epoch = 10000
partition_size = 1
mode = 'train'

neural_net = NeuralNet(loss_func, optimiser_method, learning_rate, epoch, partition_size, mode, network_arch)

neural_net.createNetwork()

neural_net.train(inputs, outputs)

plt.figure(figsize=(16, 32)) 
plt.subplot(5, 2, 1) 
plt.title('DeepNeuralNet') 
plot_decision_boundary(lambda x: predict(neural_net.layers_list, x)) 
plt.show()

clf = sklearn.linear_model.LogisticRegressionCV() 
clf.fit(X, Y) 

plt.figure(figsize=(16, 32)) 
plt.title("Logistic Regression") 
plt.subplot(5, 2, 1) 
plot_decision_boundary(lambda x: clf.predict(x)) 
plt.show()

#************************ Digit Data-Set*********************************

inputs, Y = sklearn.datasets.load_digits( n_class=10, return_X_y=True)

inputs = dataScale(inputs)

outputs = np.zeros((Y.size, Y.max()+1))
outputs[np.arange(Y.size),Y] = 1

network_arch = [{'layer_type':'input', 'size':inputs.shape[-1]},
                {'layer_type':'hidden', 'size':10, 'activation':'leakyRelu'},
                {'layer_type':'output', 'size':outputs.shape[-1], 'activation':'softmax'}]

neural_net = NeuralNet(loss_func, optimiser_method, learning_rate, epoch, partition_size, mode, network_arch)

neural_net.createNetwork()

neural_net.train(inputs, outputs)


neural_net.mode = 'test'
img_index=10
test_out_sample = neural_net.forwardPass(inputs[img_index], 1)

print('predicted',np.argmax(test_out_sample))
print('actual',Y[img_index])

img_index=9
layer1 = neural_net.layers_list[0].w.T[img_index].reshape(8,8)
layer2 = neural_net.layers_list[0].w.T[img_index].reshape(8,8)


plt.gray() 
plt.imshow(layer2,interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 

plt.gray() 
plt.imshow(inputs[img_index].reshape(8,8),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 
