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

latent = 49
img_size = (28, 28, 1)
classes = 10


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
    latent_input = tf.keras.layers.Input((lat_dim,))
    m = int(np.sqrt(lat_dim))
    z = tf.keras.layers.Reshape((m, m, 1))(latent_input)
    z = conv_layer(z, filters=filters)
    z = conv_layer(z, filters=filters)
    z = conv_layer(z, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    z = conv_layer(z, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    z = conv_layer(z, filters=filters)

    label_input = tf.keras.layers.Input((class_dim, ))
    y = tf.keras.layers.Dense(lat_dim)(label_input)
    y = tf.keras.layers.Reshape((m, m, 1))(y)
    y = conv_layer(y, filters=filters)
    y = conv_layer(y, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    y = conv_layer(y, filters=filters, transpose=True, kernel_size=(2, 2), strides=(2, 2))
    y = conv_layer(y, filters=filters)

    out = tf.keras.layers.Concatenate()([z, y])
    out = conv_layer(out, filters=filters)
    out = conv_layer(out, filters=filters)
    out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(out)

    generator = tf.keras.Model(inputs=[latent_input, label_input], outputs=out)
    return generator


# Define discriminator
def get_discriminator(img_dim=(28, 28, 1), class_dim=10, filters=32):
    img_input = tf.keras.layers.Input(img_dim)

    d = conv_layer(img_input, filters=filters)
    d = conv_layer(d, filters=filters)
    d = conv_layer(d, filters=filters, strides=(2, 2))
    d = conv_layer(d, filters=filters, strides=(2, 2))
    d = tf.keras.layers.Flatten()(d)
    d = tf.keras.layers.Dense(class_dim, activation='relu')(d)

    label_input = tf.keras.layers.Input((class_dim,))
    dy = tf.keras.layers.Dense(128, activation='relu')(label_input)
    dy = tf.keras.layers.Dense(128, activation='relu')(dy)
    dy = tf.keras.layers.Dense(class_dim, activation='relu')(dy)

    out = tf.keras.layers.Concatenate()([d, dy])
    out = tf.keras.layers.Dense(256, activation='relu')(out)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(out)

    discriminator = tf.keras.Model(inputs=[img_input, label_input], outputs=out)
    return discriminator


gen = get_generator(filters=32)
gvars = gen.trainable_variables

disc = get_discriminator(filters=16)
dvars = disc.trainable_variables


# Discriminator tries to map x_true to output 1 and generator_out to output 0
# Generator tries to find images such that discriminator thinks they are output 1
LAM = 1e0
z = tf.placeholder(tf.float32, (None, latent))
label = tf.placeholder(tf.float32, (None, classes))
x_true = tf.placeholder(tf.float32, (None, ) + img_size)
line = tf.placeholder(tf.float32, (None, 1, 1, 1))

x_gen = gen([z, label])
x_line = x_true*line + x_gen*(1-line)


lip = LAM*sum([(tf.reduce_mean(g**2)-1)**2 for g in tf.gradients(disc([x_line, label]), [x_line, label])])

loss_disc = tf.reduce_mean(disc([x_gen, label]) - disc([x_true, label]) + lip)
loss_gen = -tf.reduce_mean(disc([x_gen, label]))

lr = 1e-5
opt = tf.train.AdamOptimizer(learning_rate=lr)

train_gen = opt.minimize(loss_gen, var_list=gvars)
train_disc = opt.minimize(loss_disc, var_list=dvars)


def get_disc_batch(batch_size, latent=49, N=60000):
    idx = np.random.choice(N, batch_size)
    lat = np.random.normal(0, 1, (batch_size, latent))
    x_r, lab = x_train[idx, ...], onehot[idx, ...]
    t = np.random.uniform(0, 1, (batch_size, 1, 1, 1))
    return {x_true: x_r, z: lat, label: lab, line: t}


def get_gen_batch(batch_size, latent=49, N=60000):
    idx = np.random.choice(N, batch_size)
    lat = np.random.normal(0, 1, (batch_size, latent))
    lab = onehot[idx, ...]
    return {z: lat, label: lab}


sess.run(tf.global_variables_initializer())

epochs = 200
ncritic = 5
batch_size = 32
N = 60000

for epoch in range(epochs):
    LOSSD = []
    LOSSG = []
    print("### Epoch %d ###" % epoch)

    for batch in range(N // batch_size):
        CRITIC = []
        critic_batches = [get_disc_batch(batch_size=batch_size) for i in range(ncritic)]
        for i in range(ncritic):
            ld, _, lipschitz = sess.run([loss_disc, train_disc, lip], feed_dict=get_disc_batch(batch_size=batch_size))
            CRITIC.append(ld)

        fd = get_gen_batch(batch_size=batch_size)
        lg, _ = sess.run([loss_gen, train_gen], feed_dict=fd)

        LOSSG.append(lg)
        LOSSD.append(np.mean(CRITIC))
        print("Progress: %f, Disc: %f, Gen: %f, Lipschitz: %f" % ((batch*batch_size)/N, np.mean(LOSSD), np.mean(LOSSG), lipschitz), end='\r')

    lat = np.random.normal(0, 1, (4, latent))
    lab = onehot[np.random.choice(N, 4), ...]

    gx = sess.run(gen([z, label]), feed_dict={z: lat, label: lab})

    plt.subplot(221)
    plt.imshow(gx[0, ..., 0])
    plt.title(str(np.argmax(lab[0, ...])))

    plt.subplot(222)
    plt.imshow(gx[1, ..., 0])
    plt.title(str(np.argmax(lab[1, ...])))

    plt.subplot(223)
    plt.imshow(gx[2, ..., 0])
    plt.title(str(np.argmax(lab[2, ...])))

    plt.subplot(224)
    plt.imshow(gx[3, ..., 0])
    plt.title(str(np.argmax(lab[3, ...])))

    plt.savefig("images/epoch_" + str(epoch) + ".pdf")
    plt.clf()

    print("###############")
