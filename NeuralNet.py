def forwardPass(mode, Layers, inputs, output,loss_func):
    layer_out = inputs
    for layer in Layers:
        layer_out = layer(inputs)
        inputs = layer_out
    if mode.lower() == 'train':
        loss  = Loss(loss_func)
        loss.getLoss()(layer_out,output)
        return Layers, loss
    elif mode.lower() == 'test':
        return layer_out
 
 def backProp(inputs,network_arch, layers, loss, optimiser):
    upstream_gradient = loss.loss_derivative
    for index, layer in enumerate(reversed(layers)):
        if layer.layer_type == 'output':
            upstream_gradient =  np.multiply(layer.activation_derivative, upstream_gradient)
            upstream_gradient_w =  np.matmul(layers[len(layers)-2].y_activation.T, upstream_gradient) 
        if layer.layer_type == 'hidden':
            upstream_gradient =  np.matmul(upstream_gradient, layers[len(layers) -index].w.T)
            upstream_gradient = np.multiply(upstream_gradient,layer.activation_derivative)
            if (len(layers)-index-1) != 0:
                upstream_gradient_w = np.matmul(layers[len(layers) -index -2].y_activation.T,upstream_gradient)
            else:
                upstream_gradient_w = np.matmul(inputs.T,upstream_gradient)
        upstream_gradient_b = np.sum(upstream_gradient,axis=0).T
        optimiser(layer, upstream_gradient_w, upstream_gradient_b)
    
    for layer in layers:
        layer.w = layer.w + layer.w_delta
        layer.b = layer.b + layer.b_delta

def createNetwork(network_arch):
    network_layers = []
    for index, layer in  enumerate(network_arch):
            if layer['layer_type'] != 'input':
                network_layers.append(Layer(network_arch[index-1]['size'],layer['size'], layer['activation'],layer['layer_type']))
    return network_layers

def train(mode, epoch_count, partition_size, inputs, outputs, Layers, optimiser, loss_func):
    avg_loss = 0
    inputs = np.array_split(inputs, partition_size)
    Y =  np.array_split(outputs, partition_size)
    for i in range(epoch_count):
        for inp_batch, out_batch in zip(inputs, Y):            
            Layers, loss = forwardPass(mode, Layers, inp_batch, out_batch,loss_func)
            backProp(inp_batch,network_arch, Layers, loss, optimiser)
        if i%500 == 0:
            print('Epoch:{} Loss: {}'.format(i+1,loss.loss))
    return Layers
    
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
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10) 
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.gist_stern) 

X, Y = sklearn.datasets.make_moons(200, noise=0.20) 
inputs = np.array(X)

outputs = np.array(Y).reshape(200,1)
learning_rate = 0.01
epoch = 10000
optimiser = GradientDescent(False,learning_rate,beta = 0.9)
loss_func = 'meanSquared'
partition_size = 1

network_arch = [{'layer_type':'input', 'size':inputs.shape[-1]},
                {'layer_type':'hidden', 'size':6, 'activation':'tanh'},
                {'layer_type':'hidden', 'size':5, 'activation':'tanh'},
                {'layer_type':'hidden', 'size':3, 'activation':'tanh'},
                {'layer_type':'output', 'size':1, 'activation':'sigmoid'}]

Layers = createNetwork(network_arch)

Layers = train('Train', epoch, partition_size, inputs, outputs, Layers, optimiser,loss_func)

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
h = 0.01 
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 

plt.figure(figsize=(16, 32)) 
plt.subplot(5, 2, 1) 
plt.title('DeepNerualNet') 
plot_decision_boundary(lambda x: predict(Layers, x)) 
plt.show()


import sklearn.linear_model 
clf = sklearn.linear_model.LogisticRegressionCV() 
clf.fit(X, Y) 

plt.figure(figsize=(16, 32)) 
plt.title("Logistic Regression") 
plt.subplot(5, 2, 1) 
plot_decision_boundary(lambda x: clf.predict(x)) 
plt.show()

inputs, Y = sklearn.datasets.load_digits( n_class=10, return_X_y=True)

# outputs = np.zeros((Y.size, Y.max()+1))
# outputs[np.arange(Y.size),Y] = 1

# learning_rate = 0.01
# epoch = 1000
# np.random.seed(1) 

# optimiser = GradientDescent(False,learning_rate,beta =0.9)
# batch_size = 50
# loss_func = 'meanSquared'

# network_arch = [{'layer_type':'input', 'size':inputs.shape[-1]},
#                 {'layer_type':'hidden', 'size':10, 'activation':'leakyRelu'},
#                 {'layer_type':'output', 'size':outputs.shape[-1], 'activation':'softmax'}]

# Layers = createNetwork(network_arch)

# Layers = train('Train', epoch, batch_size, inputs, outputs, Layers, optimiser,loss_func)

# img_index=10
# test_out_sample = forwardPass('test', Layers, inputs[img_index], 1,loss_func)

# print('predicted',np.argmax(test_out_sample))
# print('actual',Y[img_index])

# img_index=9
# layer1 = Layers[0].w.T[img_index].reshape(8,8)
# layer2 = Layers[0].w.T[img_index].reshape(8,8)


# plt.gray() 
# plt.imshow(layer2,interpolation='bilinear', cmap=plt.cm.Greys_r) 
# plt.show() 

# plt.gray() 
# plt.imshow(inputs[img_index].reshape(8,8),interpolation='bilinear', cmap=plt.cm.Greys_r) 
# plt.show() 

