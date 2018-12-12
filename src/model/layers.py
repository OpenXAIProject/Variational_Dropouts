import tensorflow as tf
import numpy as np

exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1-x)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
tanh = tf.nn.tanh
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
flatten = tf.layers.flatten

class Dense(object):
    def __init__(self, n_in, n_out, name='dense', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            self.W = tf.get_variable('W', shape=[n_in, n_out])
            self.b = tf.get_variable('b', shape=[n_out])

    def __call__(self, x, activation=None, in_mask=None, out_mask=None):
        W = self.W if in_mask is None else \
                tf.gather(self.W, in_mask, axis=0)
        W = W if out_mask is None else \
                tf.gather(W, out_mask, axis=1)
        b = self.b if out_mask is None else tf.gather(self.b, out_mask)
        x = tf.matmul(x, W) + b
        x = x if activation is None else activation(x)
        return x

    def params(self, trainable=None):
        return [self.W, self.b]

    def mask_ops(self, layer, in_mask=None, out_mask=None):
        masked_W = self.W if in_mask is None else \
                tf.gather(self.W, in_mask, axis=0)
        if out_mask is None:
            return [tf.assign(layer.W, masked_W), tf.assign(layer.b, self.b)]
        else:
            masked_W = tf.gather(masked_W, out_mask, axis=1)
            masked_b = tf.gather(self.b, out_mask)
            return [tf.assign(layer.W, masked_W), tf.assign(layer.b, masked_b)]

class Conv(object):
    def __init__(self, n_in, n_out, kernel_size,
            strides=1, padding='VALID', name='conv', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            self.W = tf.get_variable('W',
                    shape=[kernel_size, kernel_size, n_in, n_out])
            self.b = tf.get_variable('b', shape=[n_out])
        self.strides = 1
        self.padding = padding

    def __call__(self, x, activation=None, in_mask=None, out_mask=None):
        W = self.W if in_mask is None else \
                tf.gather(self.W, in_mask, axis=2)
        W = W if out_mask is None else \
                tf.gather(self.b, out_mask, axis=3)
        b = self.b if out_mask is None else \
                tf.gather(self.b, out_mask)
        x = tf.nn.conv2d(x, W,
                strides=[1, 1, self.strides, self.strides],
                padding=self.padding,
                data_format='NCHW')
        x = tf.nn.bias_add(x, b, data_format='NCHW')
        x = x if activation is None else activation(x)
        return x

    def params(self, trainable=None):
        return [self.W, self.b]

    def mask_ops(self, layer, in_mask=None, out_mask=None):
        masked_W = self.W if in_mask is None else \
                tf.gather(self.W, in_mask, axis=2)
        if out_mask is None:
            return [tf.assign(layer.W, masked_W), tf.assign(layer.b, self.b)]
        else:
            masked_W = tf.gather(masked_W, out_mask, axis=3)
            masked_b = tf.gather(self.b, out_mask)
            return [tf.assign(layer.W, masked_W), tf.assign(layer.b, masked_b)]

class BatchNorm(object):
    def __init__(self, n_in, momentum=0.99,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            name='batch_norm', reuse=None):
        self.momentum = momentum
        with tf.variable_scope(name, reuse=reuse):
            self.moving_mean = tf.get_variable('moving_mean', [n_in],
                    initializer=tf.zeros_initializer(), trainable=False)
            self.moving_var = tf.get_variable('moving_var', [n_in],
                    initializer=tf.ones_initializer(), trainable=False)
            self.beta = tf.get_variable('beta', [n_in],
                    initializer=beta_initializer)
            self.gamma = tf.get_variable('gamma', [n_in],
                    initializer=gamma_initializer)

    def __call__(self, x, train, mask=None):
        beta = self.beta if mask is None else tf.gather(self.beta, mask)
        gamma = self.gamma if mask is None else tf.gather(self.gamma, mask)
        moving_mean = self.moving_mean if mask is None \
                else tf.gather(self.moving_mean, mask)
        moving_var = self.moving_var if mask is None \
                else tf.gather(self.moving_var, mask)
        if train:
            if len(x.shape) == 4:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                        gamma, beta, data_format='NCHW')
            else:
                batch_mean, batch_var = tf.nn.moments(x, [0])
                x = tf.nn.batch_normalization(x, batch_mean, batch_var,
                        beta, gamma, 1e-3)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                    moving_mean.assign_sub(
                        (1-self.momentum)*(moving_mean - batch_mean)))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                    moving_var.assign_sub(
                        (1-self.momentum)*(moving_var - batch_var)))
        else:
            if len(x.shape) == 4:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                        gamma, beta,
                        mean=moving_mean, variance=moving_var,
                        is_training=False, data_format='NCHW')
            else:
                x = tf.nn.batch_normalization(x, moving_mean, moving_var,
                        beta, gamma, 1e-3)
        return x

    def params(self, trainable=None):
        params = [self.beta, self.gamma]
        params = params + [self.moving_mean, self.moving_var] \
                if trainable is None else params
        return params

    def mask_ops(self, layer, mask):
        return [tf.assign(layer.moving_mean, tf.gather(self.moving_mean, mask)),
            tf.assign(layer.moving_var, tf.gather(self.moving_var, mask)),
            tf.assign(layer.beta, tf.gather(self.beta, mask)),
            tf.assign(layer.gamma, tf.gather(self.gamma, mask))]

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])
