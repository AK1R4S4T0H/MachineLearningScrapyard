#
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# generate some random data for training and testing
X_train = np.random.rand(1000, 100)
y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.rand(100, 100)
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))