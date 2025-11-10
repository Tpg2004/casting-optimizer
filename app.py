import numpy as np
import matplotlib.pyplot  as plt
from numpy.core.fromnumeric import size
import random
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler

dataset = loadtxt('./data.csv', delimiter=',')
x1 = dataset[:,1:6]
y1 = dataset[:,6]
mean_output=np.mean(y1)
std_output=np.std(y1)
sc=StandardScaler()
x =sc.fit_transform(x1)
y =sc.fit_transform(y1.reshape(len(y1),1))[:,0]



class layer:
    def __init__(self,no_of_input,no_of_neurons,activate):
        self.weights=np.random.rand(no_of_input,no_of_neurons)
        self.activate=activate
        self.bias=np.random.rand(no_of_neurons)
        
    # calulating values of each neutron in layer after activation
    def neuron_values(self,x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self.activation(r)
        return self.last_activation

    # defining our activation function
    def activation(self, r):
        # relu
        if self.activate =="relu":
            return (abs(r)+r)/2
        # tanh
        if self.activate == 'tanh':
            return np.tanh(r)

        # sigmoid

        if self.activate == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        #leaky relu
        if self.activate=="leaky_relu":
            return np.where(r>0,r,r*0.01)


            
    # defining our derivatives of activation function   
    def activate_deriv(self, r):

        if self.activate =="relu":
           return np.where(r>0,1,0.0)

        if self.activate == 'tanh':
            return 1 - r ** 2

        if self.activate == 'sigmoid':
            return r * (1 - r)

        if self.activate == "leaky_relu":
           return np.where(r>0,1,0.01)
            

class neural_network:

    def __init__(self):
        self._layers = []

 # Adds a layer to the neural network.
    def add_layer(self, layer):
        self._layers.append(layer)

# apply forward propogation of neural network and return output
    def forward_propagation(self, X):
        for layer in self._layers:
            X = layer.neuron_values(X)
        return X
    def backpropagation(self, X, y, learning_rate):
    # value after forward propagation 
        y1 = self.forward_propagation(X)

        # Loop over the layers in backward direction and calculate delta for each layer
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - y1
                # The y1 = layer.last_activation in this case
                layer.delta = layer.error * layer.activate_deriv(y1)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.activate_deriv(layer.last_activation)

        # Update the weights

        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers y1 or X itself (for the first hidden layer)
            input = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input.T * learning_rate

    def train(self, X, y, learning_rate, iterations):
        mean_sq_err_s = []
        for i in range(iterations):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)
            if (i%5==0):
                mse = np.mean(np.square(y - nn.forward_propagation(X)))
                mean_sq_err_s.append(mse)
        return  mean_sq_err_s

nn = neural_network()
nn.add_layer(layer(5, 30, 'sigmoid'))
nn.add_layer(layer(30, 30, 'sigmoid'))
nn.add_layer(layer(30, 1, 'leaky_relu'))

errors = nn.train(x, y, 0.6, 2800)
y_pred=nn.forward_propagation(x)

y_pred=list(map(lambda a: a*std_output+mean_output,y_pred))


plt.plot(errors)
plt.title('Changes in MSE')
plt.xlabel('Iteration (every 5th)')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.xlabel("sample-no")
plt.ylabel("defect_percentage")
plt.scatter(x=range(1,37),y= y1)           
plt.scatter(x=range(1,37), y=y_pred)
plt.show()
