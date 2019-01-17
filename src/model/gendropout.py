from layers import *

def ber_concrete(temp, p_logits, n_samples=None):
    shape = tf.shape(p_logits) if n_samples is None else \
            tf.concat([[n_samples], tf.shape(p_logits)], 0)
    u = tf.random_uniform(shape)
    return sigmoid((logit(u) + p_logits)/temp)

class GenDropout(object):
    def __init__(self, n_in, alpha=1e-4, beta=0.5, thres=1e-3,
             name='gendropout', reuse=None):
        self.n_in = n_in
        self.alpha = alpha
        self.beta  = beta
        self.thres = thres
        with tf.variable_scope(name, reuse=reuse):
            self.k_logit = tf.get_variable('k_logit', shape=[n_in],
                        #initializer=tf.random_uniform_initializer(-2.0, 2.0),
                        initializer=tf.constant_initializer(0.5413),
                        trainable=True)
            #self.k_clip = tf.clip_by_value(self.k, 0.0, 1.0)
            self.k = sigmoid(self.k_logit)

    def kl(self):
        kl = (1-self.alpha)*log(self.k) + (1-self.beta)*log(1-self.k)
        kl = kl + tf.lgamma(self.alpha) + tf.lgamma(self.beta)
        kl = kl - tf.lgamma(self.alpha + self.beta)
        return tf.reduce_sum(kl)

    def n_active(self):
        k = self.k
        return tf.reduce_sum(tf.to_int32(tf.greater(k, self.thres)))

    def mask(self, x, train):
        if train:
            p_logits = self.k_logit
            z = ber_concrete(1e-1, p_logits, n_samples=tf.shape(x)[0])
            tf.add_to_collection('z', z)
        else:
            z = self.k
            z = tf.tile(tf.expand_dims(z, 0), [tf.shape(x)[0], 1])
        return z

    def mask_ind(self):
        return tf.reshape(tf.where(tf.greater(self.k, self.thres)), [-1])

    def __call__(self, x, train):
        z = self.mask(x, train)
        z = z if train else tf.where(tf.greater(z, self.thres), z, tf.zeros_like(z))
        z = tf.reshape(z, [-1, self.n_in, 1, 1]) if len(x.shape)==4 else z
        return x*z

    def params(self, trainable=None):
        return [self.k_logit]

    def mask_ops(self, layer, mask):
        return [tf.assign(layer.k_logit, tf.gather(self.k_logit, mask))]
