import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()

train, test = tf.keras.datasets.mnist.load_data()
x_train = train[0]/255.
x_train = x_train[..., None]
y_train = train[1]

onehot = np.zeros((y_train.size, y_train.max()+1))
onehot[np.arange(y_train.size), y_train] = 1

zinp = tf.placeholder(tf.float32, (None, 49))
z = tf.keras.layers.Reshape((7, 7, 1))(zinp)
z = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(z)
z = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(z)
z = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(z)
z = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(z)
z = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(z)

yinp = tf.placeholder(tf.float32, (None, 10))
y = tf.keras.layers.Dense(49, activation='relu')(yinp)
y = tf.keras.layers.Reshape((7, 7, 1))(y)
y = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(y)
y = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(y)
y = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(y)
y = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(y)
y = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(y)

generator_out = tf.keras.layers.Concatenate()([z, y])
generator_out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(generator_out)


dinp = tf.placeholder(tf.float32, (None, 28, 28, 1))
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(dinp)
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(d)
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(d)
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(d)
d = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(d)
d = tf.keras.layers.Flatten()(d)
d = tf.keras.layers.Dense(10, activation='relu')(d)
d = tf.keras.layers.Dense(1, activation='sigmoid')(d)

loss_gen = tf.log(1. - )