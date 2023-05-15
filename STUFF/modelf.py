#
import tensorflow as tf
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(256, 256, 3))
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
drop4 = layers.Dropout(0.5)(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)


up5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(pool4))
merge5 = layers.concatenate([drop4, up5], axis=3)
conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge5)
conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

up6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
merge6 = layers.concatenate([conv3, up6], axis=3)
conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge6)
conv6 = layers.Conv2D(128, (3, 3), activation='relu
