import random


x_train = [[1, 2], [2, 3], [3, 1], [4, 3]]
y_train = [0, 0, 1, 1]

learning_rate = 0.1
num_iterations = 100

def sigmoid(z):
    return 1 / (1 + exp(-z))

# Initialize weights randomly
w = [random.random() for _ in range(2)]
b = random.random()

for i in range(num_iterations):
    z = [w[0]*x_train[j][0] + w[1]*x_train[j][1] + b for j in range(4)]
    y_pred = [sigmoid(z[j]) for j in range(4)]
    error = [y_pred[j] - y_train[j] for j in range(4)]
    w[0] -= learning_rate * sum([error[j]*x_train[j][0] for j in range(4)])
    w[1] -= learning_rate * sum([error[j]*x_train[j][1] for j in range(4)])
    b -= learning_rate * sum(error)

# Make predictions on new data
x_test = [[1, 1], [2, 2], [3, 3]]
y_pred = [sigmoid(w[0]*x_test[j][0] + w[1]*x_test[j][1] + b) for j in range(3)]
print(y_pred)
