# 
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
  layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256, 256, 3)),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(256, (3,3), activation='relu', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(512, (3,3), activation='relu', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(512, (3,3), activation='relu', padding='same'),
  layers.UpSampling2D((2,2)),
  layers.Conv2D(256, (3,3), activation='relu', padding='same'),
  layers.UpSampling2D((2,2)),
  layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  layers.UpSampling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  layers.UpSampling2D((2,2)),
  layers.Conv2D(32, (3,3), activation='relu', padding='same'),
  layers.Conv2D(1, (1,1), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_masks, epochs=10, 
          validation_data=(test_images, test_masks))
