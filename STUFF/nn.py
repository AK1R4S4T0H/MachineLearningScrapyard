import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for the input layer
        self.weights.append(np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2 / self.input_dim))
        self.biases.append(np.zeros((1, self.hidden_dim)))
        
        # Initialize weights and biases for the hidden layers
        for i in range(num_hidden_layers-1):
            self.weights.append(np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2 / self.hidden_dim))
            self.biases.append(np.zeros((1, self.hidden_dim)))
            
        # Initialize weights and biases for the output layer
        self.weights.append(np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2 / self.hidden_dim))
        self.biases.append(np.zeros((1, self.output_dim)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def feedforward(self, X):
        # Compute output of the input layer
        self.layer_outputs = [self.relu(np.dot(X, self.weights[0]) + self.biases[0])]
        
        # Compute output of each hidden layer
        for i in range(1, self.num_hidden_layers):
            self.layer_outputs.append(self.relu(np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]))
            
        # Compute output of the output layer
        self.output = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        return self.output
    
    def backpropagation(self, X, y, learning_rate):
        # Compute error of the output layer
        output_error = y - self.output
        output_delta = output_error * self.relu_derivative(self.output)
        
        # Compute errors and deltas for each hidden layer
        hidden_errors = []
        hidden_deltas = []
        for i in range(self.num_hidden_layers-1, 0, -1):
            if i == self.num_hidden_layers-1:
                hidden_error = np.dot(output_delta, self.weights[i].T)
            else:
                hidden_error = np.dot(hidden_deltas[-1], self.weights[i].T)
            hidden_errors.append(hidden_error)
            hidden_delta = hidden_error * self.relu_derivative(self.layer_outputs[i])
            hidden_deltas.append(hidden_delta)
        hidden_deltas.reverse()
        
        # Update weights and biases for each layer
        self.weights[0] += learning_rate * np.dot(X.T, hidden_deltas[0])
        self.biases[0] += learning_rate * np.sum(hidden_deltas[0], axis=0, keepdims=True)
        for i in range(1, self.num_hidden_layers):
            self.weights[i] += learning_rate * np.dot(self.layer_outputs[i-1].T, hidden_deltas[i])
            self.biases[i] += learning_rate * np.sum(hidden_deltas[i], axis=0, keepdims=True)
        self.weights[-1] += learning_rate * np.dot(self.layer_outputs[-1].T, output_delta)
        self.biases[-1] += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

def predict(model, X):
    return model.feedforward(X)
def train(model, X_file, y_file, learning_rate, epochs):
    # Load dataset
    X = np.load(X_file)
    y = np.load(y_file)
    
    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward pass
            model.feedforward(X[i])
            
            # Backward pass
            model.backpropagation(X[i], y[i], learning_rate)


