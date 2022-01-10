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
        z = self.BN(z_pre) + tf.random.normal(stddev = noise_rate, shape = tf.shape(z_pre))
        h = self.phi(self.gamma * (z + self.beta))
        return h, z_pre


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
        z_pre = []
        h = tf.random.normal(shape = tf.shape(X), stddev = noise_rate) + X
        for l in self.E:
            h, z_pre = l(h, noise_rate)
            z_pre.append(z_pre)
        y = self.C(h)
        return y, z_pre


class Decoderlayer(tf.keras.layers.Layer):
    def __init__(self):


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_nodes):
        super(Decoder, self).__init__()

