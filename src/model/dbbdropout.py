from layers import *
from bbdropout import ber_concrete

class DBBDropout(object):
    def __init__(self, n_in, beta_init=10.0, thres=1e-3, name='dbbd', reuse=None):
        self.n_in = n_in
        self.thres = thres
        with tf.variable_scope(name, reuse=reuse):
            self.bn = BatchNorm(n_in, name='bn',
                    beta_initializer=tf.constant_initializer(beta_init))
            self.beta = self.bn.beta
            self.sigma_uc = tf.get_variable('sigma_uc', [n_in])
            self.sigma = softplus(self.sigma_uc)

    def kl(self, rho=np.sqrt(5)):
        kl = log(rho/self.sigma) + 0.5*(tf.square(self.sigma)+tf.square(self.beta))/rho**2
        return tf.reduce_sum(kl)

    def mask(self, x, train, mask0=None):
        x = global_avg_pool(x) if len(x.shape)==4 else x
        x = tf.stop_gradient(x)
        x = self.bn(x, train, mask=mask0)
        sigma = self.sigma if mask0 is None else tf.gather(self.sigma, mask0)
        x = x + self.sigma*tf.random_normal([self.n_in]) if train else x
        x = tf.clip_by_value(x, 1e-10, 1-1e-10)
        return ber_concrete(1e-1, logit(x)) if train else x

    def __call__(self, x, train, z_in, mask0=None):
        x = x if mask0 is None else tf.gather(x, mask0, axis=1)
        z_in = z_in if mask0 is None else tf.gather(z_in, mask0, axis=1)
        z = z_in * self.mask(x, train, mask0=mask0)
        z = z if train else tf.where(tf.greater(z, self.thres), z, tf.zeros_like(z))
        if not train:
            self.n_active_x = tf.reduce_mean(
                    tf.reduce_sum(tf.to_int32(tf.greater(z, self.thres)), 1))
        z = tf.reshape(z, [-1, self.n_in, 1, 1]) if len(x.shape)==4 else z
        return x*z

    def mask_ind(self, x, z_in):
        z = z_in * self.mask(x, False)
        return tf.reshape(tf.where(tf.greater(z[0], self.thres)), [-1])

    def params(self, trainable=None):
        return self.bn.params(trainable=trainable) + [self.sigma_uc]

    def mask_ops(self, layer, mask):
        return self.bn.mask_ops(layer.bn, mask) \
                + [tf.assign(layer.sigma_uc, tf.gather(self.sigma_uc, mask))]
