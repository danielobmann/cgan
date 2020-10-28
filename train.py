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


def conv_layer(inp, filters, kernel_size=(3, 3), dropout=0, transpose=False, **kwargs):
    out = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', **kwargs)(inp)
    if transpose:
        out = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, padding='valid', **kwargs)(inp)
    if dropout:
        out = tf.keras.layers.Dropout(rate=dropout)(out)
    out = tf.keras.layers.PReLU(shared_axes=[1, 2])(out)
    return out


# Define the generator
def get_generator(lat_dim=49, class_dim=10, filters=32):
    zinp = tf.keras.layers.Input((lat_dim,))
    m = int(np.sqrt(lat_dim))
    z = tf.keras.layers.Reshape((m, m, 1))(zinp)
    z = conv_layer(z, filters=filters)
    z = conv_layer(z, filters=filters)
    z = conv_layer(z, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    z = conv_layer(z, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    z = conv_layer(z, filters=filters)

    yinp = tf.keras.layers.Input((class_dim, ))
    y = tf.keras.layers.Dense(lat_dim)(yinp)
    y = tf.keras.layers.Reshape((m, m, 1))(y)
    y = conv_layer(y, filters=filters)
    y = conv_layer(y, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    y = conv_layer(y, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    y = conv_layer(y, filters=filters)

    out = tf.keras.layers.Concatenate()([z, y])
    out = conv_layer(out, filters=filters)
    out = conv_layer(out, filters=filters)
    out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(out)

    generator = tf.keras.Model(inputs=[zinp, yinp], outputs=out)
    return [zinp, yinp], out, generator


# Define discriminator
def get_discriminator(img_dim=(28, 28, 1), class_dim=10, filters=32):
    dginp = tf.keras.layers.Input(img_dim)

    d = conv_layer(dginp, filters=filters)
    d = conv_layer(d, filters=filters)
    d = conv_layer(d, filters=filters, strides=(2, 2))
    d = conv_layer(d, filters=filters, strides=(2, 2))
    d = tf.keras.layers.Flatten()(d)
    d = tf.keras.layers.Dense(class_dim, activation='relu')(d)

    dyinp = tf.keras.layers.Input((class_dim,))
    dy = tf.keras.layers.Dense(128, activation='relu')(dyinp)
    dy = tf.keras.layers.Dense(128, activation='relu')(dy)
    dy = tf.keras.layers.Dense(class_dim, activation='relu')(dy)

    out = tf.keras.layers.Concatenate()([d, dy])
    out = tf.keras.layers.Dense(256, activation='relu')(out)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(out)

    discriminator = tf.keras.Model(inputs=[dginp, dyinp], outputs=out)
    return [dginp, dyinp], out, discriminator


ginp, gout, gen = get_generator(filters=32)
gvars = gen.trainable_variables

dinp, dout, disc = get_discriminator(filters=16)
dvars = disc.trainable_variables


# Discriminator tries to map x_true to output 1 and generator_out to output 0
# Generator tries to find images such that discriminator thinks they are output 1
lip = 1e1*sum([(tf.reduce_mean(g**2)-1)**2 for g in tf.gradients(disc(dinp), dinp)])
loss = tf.reduce_mean(disc(dinp) - disc([gen(ginp), ginp[1]]) + lip)

lr = 1e-5
opt = tf.train.AdamOptimizer(learning_rate=lr)

train_gen = opt.minimize(loss, var_list=gvars)
train_disc = opt.minimize(-loss, var_list=dvars)

sess.run(tf.global_variables_initializer())

epochs = 200
batch_size = 32
N = 60000

for epoch in range(epochs):
    LOSSD = []
    LOSSG = []
    print("### Epoch %d ###" % epoch)

    for batch in range(N // batch_size):
        print("Progress %f " % (batch * batch_size / N), end='\r')
        idx = np.random.choice(N, batch_size)
        x, y = x_train[idx, ...], onehot[idx, ...]
        z = np.random.normal(0, 1, (batch_size, 49))

        fd = {dinp[0]: x,
              dinp[1]: y,
              ginp[0]: z,
              ginp[1]: y}

        ld, _ = sess.run([loss, train_disc], feed_dict=fd)
        lg, _ = sess.run([loss, train_gen], feed_dict=fd)
        LOSSD.append(ld)
        LOSSG.append(lg)
    print("Loss %f, %f" % (np.mean(LOSSD), np.mean(LOSSG)))

    z = np.random.normal(0, 1, (4, 49))
    y = onehot[np.random.choice(N, 4), ...]

    gx = sess.run(gout, feed_dict={ginp[0]: z,
                                   ginp[1]: y})

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
