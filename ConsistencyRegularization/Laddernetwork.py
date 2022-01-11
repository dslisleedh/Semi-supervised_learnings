import tensorflow as tf


class Encoderlayer(tf.keras.layers.Layer):
    def __init__(self, n_nodes, activation = 'relu'):
        super(Encoderlayer, self).__init__()
        self.n_nodes = n_nodes
        self.activation = activation

        self.W = tf.keras.layers.Dense(self.n_nodes,
                                       activation = 'linear',
                                       use_bias = False
                                       )
        self.BN = tf.keras.layers.BatchNormalization()
        self.gamma = tf.Variable(initial_value=tf.ones(self.n_nodes),
                                 trainable = True if self.activation == 'softmax' else False,
                                 dtype = 'float32'
                                 )
        self.beta = tf.Variable(initial_value = tf.zeros(self.n_nodes),
                                trainable = False if self.activation == 'linear' else True,
                                dtype = 'float32'
                                )
        if self.activation == 'relu':
            self.phi = tf.keras.layers.ReLU()
        elif self.activation == 'softmax':
            self.phi = tf.keras.layers.Softmax()
        else:
            self.phi = tf.keras.layers.Layer()

    def call(self, h, noise_rate):
        z_pre = self.W(h)
        z_tiled = self.BN(z_pre) + tf.random.normal(stddev = noise_rate, shape = tf.shape(z_pre))
        h = self.phi(self.gamma * (z_tiled + self.beta))
        return h, z_tiled


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes, n_labels, last_activation = 'softmax'):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_labels = n_labels
        self.last_activation = last_activation

        self.E = [Encoderlayer(self.n_nodes) for _ in range(self.n_layers)]
        self.C = Encoderlayer(self.n_labels, self.last_activation)

    def call(self, X, noise_rate):
        z_tiled_ = []
        h = tf.random.normal(shape = tf.shape(X), stddev = noise_rate) + X
        for l in self.E:
            h, z_tiled = l(h, noise_rate)
            z_tiled_.append(z_tiled)
        y = self.C(h)
        return y, z_tiled_


class Decoderlayer(tf.keras.layers.Layer):
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

        self.V = tf.keras.layers.Dense(self.n_nodes,
                                       activation = 'relu'
                                       )
        self.BN = tf.keras.layers.BatchNormalization()

    def call(self, X):
        u = self.BN(self.V(X))
        return u

class G_gauss(tf.keras.layers.Layer):
    def __init__(self, n_nodes):
        super(G_gauss, self).__init__()
        self.n_nodes = n_nodes

        self.a1 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a2 = tf.Variable(initial_value=tf.ones(self.n_nodes), trainable=True, dtype='float32')
        self.a3 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a4 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a5 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a6 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a7 = tf.Variable(initial_value=tf.ones(self.n_nodes), trainable=True, dtype='float32')
        self.a8 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a9 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')
        self.a10 = tf.Variable(initial_value=tf.zeros(self.n_nodes), trainable=True, dtype='float32')

    def call(self, z_tilde, u):
        mu = self.a1 * tf.nn.sigmoid(self.a2 * u + self.a3) + self.a4 * u + self.a5
        v = self.a6 * tf.nn.sigmoid(self.a7 * u + self.a8) + self.a9 * u + self.a10
        z_hat = (z_tilde - mu) * v + mu
        return z_hat



class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.D = [
            tf.keras.layers.BatchNormalization()
        ] + [
            Decoderlayer(self.n_nodes) for _ in range(self.n_layers)
        ]
        self.G = [
            G_gauss(self.n_nodes) for _ in range(self.n_layers + 1)
        ]
        self.BN = [
            tf.keras.layers.BatchNormalization() for _ in range(self.n_layers + 1)
        ]

    def call(self, h, z_tilde):
        z_tilde = reversed(z_tilde)
        z_hat_bn_ = []

        u = self.D[0](h)
        z_hat = self.G[0](u)
        z_hat_bn = self.BN[0](z_hat)
        z_hat_bn_.append(z_hat_bn)
        for D, G, BN, z_t in zip(self.D[1:], self.G[1:], self.BN[1:], z_tilde):
            u = D(z_hat)
            z_hat = self.G(u)
            z_hat_bn = BN(z_hat)
            z_hat_bn_.append(z_hat_bn)
        return z_hat_bn_


class Laddernetwork(tf.keras.models.Model):
    def __init__(self, n_layers, n_nodes, n_labels, noise_rate = .2, last_activation = 'softmax', batch_size = 256):
        super(Laddernetwork, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_labels = n_labels
        self.last_activation = last_activation
        self.noise_rate = noise_rate
        self.batch_size = batch_size
        self.layer_lambda = [
            1000.
        ] + [
            10.
        ] + [
            .1 for _ in range(self.n_layers - 2)
        ]

        self.Encoder = Encoder(self.n_layers,
                               self.n_nodes,
                               self.n_labels,
                               last_activation = self.last_activation
                               )
        self.Decoder = Decoder(self.n_layers,
                               self.n_nodes
                               )

    def train_step(self, data):
        X_l, X_u, y = data
        with tf.GradientTape() as tape:
            y_l_c, _ = self.Encoder(X_l, self.noise_rate)

            y_u_c, z_tiled_u_c = self.Encoder(X_u, self.noise_rate)
            _, z_u = self.Encoder(X_u, 0.)

            z_hat_bn = self.Decoder(y_u_c, z_tiled_u_c)
            C_c = tf.reduce_mean(
                tf.losses.binary_crossentropy(y, y_l_c)
            )
            C_d = tf.reduce_sum(
                tf.reduce_sum(tf.square(z_u - z_hat_bn), axis = 0) * self.layer_lambda / self.batch_size / self.n_nodes
            )
            C = C_c + C_d
        grads = tape.gradient(C, self.Encoder.trainable_variables + self.Decoder.trainable)
        self.optimizer.apply_gradients(
            zip(grads, self.Encoder.trainable_variables + self.Decoder.trainable_variables)
        )
        return {'Supervised_loss' : C_c, 'Unsupervised_loss' : C_d}

