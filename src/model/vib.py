from layers import *

class VIB(object):
    def __init__(self, n_in,
            init_mag=9, init_var=0.01, thres=0, featmap_size=None,
            name='vib', reuse=None):
        self.n_in = n_in
        self.thres = thres
        self.featmap_size = featmap_size
        with tf.variable_scope(name, reuse=reuse):
            self.mu = tf.get_variable('mu', shape=[n_in],
                    initializer=tf.initializers.random_normal(1., init_var))
            self.logvar = tf.get_variable('logvar', shape=[n_in],
                    initializer=tf.initializers.random_normal(-init_mag, init_var))

    def kl(self):
        kld = .5*tf.log(1 + tf.square(self.mu)/(tf.exp(self.logvar)+1e-8))
        if self.featmap_size is not None:
            kld *= self.featmap_size[0]*self.featmap_size[1]
        return tf.reduce_sum(kld)

    def get_log_alpha(self):
        return self.logvar - tf.log(tf.square(self.mu)+1e-8)

    def n_active(self):
        log_alpha = self.get_log_alpha()
        return tf.reduce_sum(tf.to_int32(tf.less(log_alpha, self.thres)))

    def mask(self, x, train):
        if train:
            sigma = tf.exp(0.5*self.logvar)
            z = self.mu + sigma*tf.random_normal([tf.shape(x)[0], self.n_in])
        else:
            log_alpha = self.get_log_alpha()
            z = tf.where(tf.less(log_alpha, self.thres), self.mu, tf.zeros_like(self.mu))
            z = tf.expand_dims(z, 0)
        return z

    def mask_ind(self):
        return tf.reshape(tf.where(tf.less(self.log_alpha(), self.thres)), [-1])

    def __call__(self, x, train):
        z = self.mask(x, train)
        z = tf.reshape(z, [-1, self.n_in, 1, 1]) if len(x.shape)==4 else z
        return x*z

    def params(self, trainable=None):
        return [self.mu, self.logvar]

    def mask_ops(self, layer, mask):
        return [tf.assign(layer.mu, tf.gather(self.mu, mask)),
                tf.assign(layer.logvar, tf.gather(self.logvar, mask))]
