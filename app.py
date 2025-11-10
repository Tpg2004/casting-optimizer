import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler

# --- Data Loading and Preprocessing (FIX for FileNotFoundError) ---
try:
    # Attempt to load the file as originally intended
    dataset = loadtxt('./data.csv', delimiter=',')
    print("Successfully loaded data.csv.")
except FileNotFoundError:
    print("Warning: 'data.csv' not found. Using placeholder data for demonstration.")
    N_SAMPLES = 36
    N_FEATURES = 5
    # Placeholder data (5 features, 36 samples, 1 target)
    dataset = np.random.rand(N_SAMPLES, N_FEATURES + 1)
    dataset[:, N_FEATURES] = dataset[:, N_FEATURES] * 10 
# ---------------------------------------------------------------------------------

x1 = dataset[:, 1:6]
y1 = dataset[:, 6]
mean_output = np.mean(y1)
std_output = np.std(y1)

sc = StandardScaler()
x = sc.fit_transform(x1)
y = sc.fit_transform(y1.reshape(len(y1), 1))[:, 0]


class layer:
    def __init__(self, no_of_input, no_of_neurons, activate):
        # Weight initialization correction
        self.weights = np.random.randn(no_of_input, no_of_neurons) * np.sqrt(2.0/no_of_input)
        self.activate = activate
        self.bias = np.zeros(no_of_neurons) 

    def neuron_values(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_input = r
        self.last_activation = self.activation(r)
        return self.last_activation

    def activation(self, r):
        if self.activate == "relu":
            return np.maximum(0, r)
        if self.activate == 'tanh':
            return np.tanh(r)
        if self.activate == 'sigmoid':
            r = np.clip(r, -500, 500)
            return 1 / (1 + np.exp(-r))
        if self.activate == "leaky_relu":
            return np.where(r > 0, r, r * 0.01)

    def activate_deriv(self, input_or_activation):
        # Logic fix: Use last_input for ReLU/Leaky ReLU derivatives
        if self.activate == "relu":
            return np.where(self.last_input > 0, 1.0, 0.0)
        if self.activate == 'tanh':
            return 1 - input_or_activation ** 2
        if self.activate == 'sigmoid':
            return input_or_activation * (1 - input_or_activation)
        if self.activate == "leaky_relu":
            return np.where(self.last_input > 0, 1.0, 0.01)


class neural_network:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def forward_propagation(self, X):
        for layer in self._layers:
            X = layer.neuron_values(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        y1 = self.forward_propagation(X)

        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            if layer == self._layers[-1]:
                layer.error = y - y1
                # Delta uses activation output (y1) for derivative calculation
                layer.delta = layer.error * layer.activate_deriv(y1)
            else:
                next_layer = self._layers[i + 1]
                # Dot product order correction
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
                layer.delta = layer.error * layer.activate_deriv(layer.last_activation)

        for i in range(len(self._layers)):
            layer = self._layers[i]
            
            input_data = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            delta_reshaped = np.atleast_2d(layer.delta)
            
            # Weight update correction
            layer.weights += input_data.T.dot(delta_reshaped) * learning_rate
            layer.bias += layer.delta.flatten() * learning_rate

    def train(self, X, y, learning_rate, iterations):
        mean_sq_err_s = []
        
        X_2d = np.atleast_2d(X)
        y_1d = np.atleast_1d(y) 
        
        for i in range(iterations):
            for j in range(len(X_2d)):
                self.backpropagation(X_2d[j], y_1d[j], learning_rate)
                
            if (i % 5 == 0):
                y_pred_scaled = self.forward_propagation(X_2d)
                
                if y_pred_scaled.ndim > 1:
                    y_pred_scaled = y_pred_scaled.flatten()
                    
                mse = np.mean(np.square(y_1d - y_pred_scaled))
                mean_sq_err_s.append(mse)
                
        return mean_sq_err_s

# --- Network Initialization and Training ---

nn = neural_network()
nn.add_layer(layer(5, 30, 'sigmoid'))
nn.add_layer(layer(30, 30, 'sigmoid'))
nn.add_layer(layer(30, 1, 'leaky_relu')) 

errors = nn.train(x, y, 0.01, 2800) 
y_pred_scaled = nn.forward_propagation(x)

y_pred_scaled = y_pred_scaled.flatten() 
y_pred = y_pred_scaled * std_output + mean_output 


# --- Plotting Results (FIX for Blank Screen) ---

# 1. Plot MSE
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(errors)
ax1.set_title('Changes in MSE')
ax1.set_xlabel(f'Training Epoch (x5)')
ax1.set_ylabel('Mean Squared Error (MSE)')
ax1.grid(True)
# **IMPORTANT: Do NOT use plt.show()** # If running on Streamlit, you must use st.pyplot(fig1)

# 2. Plot Actual vs. Predicted
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_xlabel("Sample No.")
ax2.set_ylabel("Defect Percentage")
ax2.scatter(x=range(1, len(y1) + 1), y=y1, label='Actual', marker='o')
ax2.scatter(x=range(1, len(y_pred) + 1), y=y_pred, label='Predicted', marker='x')
ax2.set_title('Actual vs. Predicted Defect Percentage')
ax2.legend()
ax2.grid(True)
# **IMPORTANT: Do NOT use plt.show()**
# If running on Streamlit, you must use st.pyplot(fig2)

# If you are NOT using Streamlit and need to see the output:
# plt.savefig('mse_plot.png')
# plt.savefig('actual_vs_predicted.png')
# plt.close('all') # Closes all figures to free memory
