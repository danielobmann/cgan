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

zinp = tf.keras.layers.Input((49, ))
z = tf.keras.layers.Reshape((7, 7, 1))(zinp)
z = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(z)
z = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(z)
z = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(z)
z = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(z)
z = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(z)

yinp = tf.keras.layers.Input((10, ))
y = tf.keras.layers.Dense(49, activation='relu')(yinp)
y = tf.keras.layers.Reshape((7, 7, 1))(y)
y = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(y)
y = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(y)
y = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(y)
y = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(y)
y = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(y)

generator_out = tf.keras.layers.Concatenate()([z, y])
generator_out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(generator_out)

generator = tf.keras.Model(inputs=[zinp, yinp], outputs=generator_out)
gvars = generator.trainable_variables

dginp = tf.keras.layers.Input((28, 28, 1))

d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(dginp)
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(d)
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu')(d)
d = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')(d)
d = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(d)
d = tf.keras.layers.Flatten()(d)
d = tf.keras.layers.Dense(10, activation='relu')(d)

dyinp = tf.keras.layers.Input((10, ))
dy = tf.keras.layers.Dense(49, activation='relu')(dyinp)
dy = tf.keras.layers.Dense(128, activation='relu')(dy)
dy = tf.keras.layers.Dense(10, activation='relu')(dy)

discriminator_out = tf.keras.layers.Concatenate()([d, dy])
discriminator_out = tf.keras.layers.Dense(1, activation='sigmoid')(discriminator_out)

discriminator = tf.keras.Model(inputs=[dginp, dyinp], outputs=discriminator_out)
dvars = discriminator.trainable_variables

x_true = tf.placeholder(tf.float32, (None, 28, 28, 1))
label = tf.placeholder(tf.float32, (None, 10))

eps = 1e-7
# Discriminator tries to map x_true to output 1 and generator_out to output 0
# Generator tries to find images such that discriminator thinks they are output 1
loss = tf.reduce_mean(tf.log(discriminator([x_true, label]) + eps) +
                      tf.log(1. - discriminator([generator([zinp, label]), label]) + eps))


lr = 1e-2
opt = tf.train.AdamOptimizer(learning_rate=lr)

train_gen = opt.minimize(loss, var_list=gvars)
train_disc = opt.minimize(-loss, var_list=dvars)

sess.run(tf.global_variables_initializer())

epochs = 20
subepochs = 3
batch_size = 32
N = 60000

for epoch in range(epochs):
    print("### Epoch %d ###" % epoch)
    for subepoch in range(subepochs):
        print("Discriminator subepoch %d" % subepoch)

        # Train discriminator
        LOSS = []
        for batch in range(N//batch_size):
            print("Progress %f " % (batch * batch_size / N), end='\r')
            idx = np.random.choice(N, batch_size)
            x, y = x_train[idx, ...], onehot[idx, ...]
            z = np.random.normal(0, 1, (batch_size, 49))

            fd = {x_true: x,
                  label: y,
                  zinp: z}

            l, _ = sess.run([loss, train_disc], feed_dict=fd)
            LOSS.append(l)
        print("Loss %f" % np.mean(LOSS))

    for subepoch in range(subepochs):
        print("Generator subepoch %d" % subepoch)
        # Train generator
        LOSS = []
        for batch in range(N // batch_size):
            print("Progress %f " % (batch * batch_size / N), end='\r')
            idx = np.random.choice(N, batch_size)
            x, y = x_train[idx, ...], onehot[idx, ...]
            z = np.random.normal(0, 1, (batch_size, 49))
            fd = {zinp: z,
                  label: y,
                  x_true: x}

            l, _ = sess.run([loss, train_gen], feed_dict=fd)
            LOSS.append(l)
        print("Loss %f" % np.mean(LOSS))

    z = np.random.normal(0, 1, (4, 49))
    y = onehot[np.random.choice(N, 4), ...]

    gx = sess.run(generator_out, feed_dict={zinp: z, yinp: y})

    plt.subplot(221)
    plt.imshow(gx[0, ..., 0])
    plt.title(str(np.argmax(y[0, ...])))

    plt.subplot(222)
    plt.imshow(gx[1, ..., 0])
    plt.title(str(np.argmax(y[1, ...])))

    plt.subplot(223)
    plt.imshow(gx[2, ..., 0])
    plt.title(str(np.argmax(y[2, ...])))

    plt.subplot(224)
    plt.imshow(gx[3, ..., 0])
    plt.title(str(np.argmax(y[3, ...])))

    plt.savefig("images/epoch_" + str(epoch) + ".pdf")
    plt.clf()

    print("###############")
