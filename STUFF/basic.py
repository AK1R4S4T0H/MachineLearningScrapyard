# 
import tensorflow as tf
import numpy as np

x_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

y_output = np.array([[0], [1], [1], [0]], dtype=np.float32)

input_layer = tf.keras.layers.Input(shape=(2,))
hidden_layer1 = tf.keras.layers.Dense(4, activation='relu')(input_layer)
hidden_layer2 = tf.keras.layers.Dense(2, activation='relu')(hidden_layer1)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer2)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_input, y_output, epochs=1000, verbose=0)

print(model.predict(x_input))
