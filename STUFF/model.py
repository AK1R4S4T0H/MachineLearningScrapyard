#
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.w_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.w_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.w_c = np.random.randn(hidden_size, input_size + hidden_size)
        self.w_o = np.random.randn(hidden_size, input_size + hidden_size)
        
        # Initialize biases
        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        
        # Initialize hidden state and cell state
        self.h = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))
        
    def forward(self, x):
        # Concatenate input and previous hidden state
        input_concat = np.vstack((x, self.h))
        
        # Compute forget gate
        f = sigmoid(np.dot(self.w_f, input_concat) + self.b_f)
        
        # Compute input gate
        i = sigmoid(np.dot(self.w_i, input_concat) + self.b_i)
        
        # Compute candidate cell state
        c_tilde = np.tanh(np.dot(self.w_c, input_concat) + self.b_c)
        
        # Compute cell state
        self.c = f * self.c + i * c_tilde
        
        # Compute output gate
        o = sigmoid(np.dot(self.w_o, input_concat) + self.b_o)
        
        # Compute hidden state
        self.h = o * np.tanh(self.c)
        
        return self.h

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
