
# This is extracted from the original implemtation of SBP
# Please refer to the official source code
# https://github.com/necludov/group-sparsity-sbp

from layers import *
from tensorflow.python.ops.distributions import special_math

def phi(x):
    return 0.5*tf.erfc(-x/tf.sqrt(2.0))

def __erfinv(x):
    w = -tf.log((1.0-x)*(1.0+x)-1e-5)
    p_small = 2.81022636e-08*tf.ones_like(x)
    p_small = 3.43273939e-07 + p_small*(w-2.5)
    p_small = -3.5233877e-06 + p_small*(w-2.5)
    p_small = -4.39150654e-06 + p_small*(w-2.5)
    p_small = 0.00021858087 + p_small*(w-2.5)
    p_small = -0.00125372503 + p_small*(w-2.5)
    p_small = -0.00417768164 + p_small*(w-2.5)
    p_small = 0.246640727 + p_small*(w-2.5)
    p_small = 1.50140941 + p_small*(w-2.5)

    p_big = -0.000200214257*tf.ones_like(x)
    p_big = 0.000100950558 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 0.00134934322 + p_big*(tf.sqrt(w) - 3.0)
    p_big = -0.00367342844 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 0.00573950773 + p_big*(tf.sqrt(w) - 3.0)
    p_big = -0.0076224613 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 0.00943887047 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 1.00167406 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 2.83297682 + p_big*(tf.sqrt(w) - 3.0)

    small_mask = tf.cast(tf.less(w, 5.0*tf.ones_like(w)), tf.float32)
    big_mask = tf.cast(tf.greater_equal(w, 5.0*tf.ones_like(w)), tf.float32)
    p = p_small*small_mask + p_big*big_mask
    return p*x

def erfinv(x):
    return special_math.ndtri((x+1.)/2.0)/tf.sqrt(2.)

def erfcx(x):
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """
    K = 3.75
    y = (tf.abs(x)-K) / (tf.abs(x)+K)
    y2 = 2.0*y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01
    result = d/(1.0+2.0*tf.abs(x))
    result = tf.where(tf.is_nan(result), tf.ones_like(result), result)
    result = tf.where(tf.is_inf(result), tf.ones_like(result), result)

    negative_mask = tf.cast(tf.less(x, 0.0), tf.float32)
    positive_mask = tf.cast(tf.greater_equal(x, 0.0), tf.float32)
    negative_result = 2.0*tf.exp(x*x)-result
    negative_result = tf.where(tf.is_nan(negative_result), tf.ones_like(negative_result), negative_result)
    negative_result = tf.where(tf.is_inf(negative_result), tf.ones_like(negative_result), negative_result)
    result = negative_mask * negative_result + positive_mask * result
    return result

def phi_inv(x):
    return tf.sqrt(2.0)*erfinv(2.0*x-1)

class SBP(object):
    def __init__(self, n_in, min_log=-20.0, max_log=0.0, thres=1.0,
            name='sbp', reuse=None):
        self.n_in = n_in
        self.thres = thres
        self.min_log = min_log
        self.max_log = max_log
        self._name = name
        with tf.variable_scope(name, reuse=reuse):
            self.mu = tf.get_variable('mu', shape=[n_in],
                                 initializer=tf.zeros_initializer())
            self.log_sigma = tf.get_variable('log_sigma', shape=[n_in],
                                 initializer=tf.constant_initializer(-5.0))

    def __call__(self, x, train):
        z, snr = self.mask(x, train)
        z = z if train else tf.where(tf.greater(snr, self.thres), z, tf.zeros_like(z))
        z = tf.reshape(z, [-1, self.n_in, 1, 1]) if len(x.shape)==4 else z
        return x*z

    def _mean_truncated_log_normal(self):
        a, b = self.min_log, self.max_log
        mu = tf.clip_by_value(self.mu, -20.0, 5.0)
        log_sigma = tf.clip_by_value(self.log_sigma, -20.0, 5.0)
        sigma = tf.exp(log_sigma)

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        z = phi(beta) - phi(alpha)
        mean = erfcx((sigma-beta)/tf.sqrt(2.0))*tf.exp(b-beta*beta/2)
        mean = mean - erfcx((sigma-alpha)/tf.sqrt(2.0))*tf.exp(a-alpha*alpha/2)
        mean = mean/(2*z)

        return mean

    def _snr_truncated_log_normal(self):
        a, b = self.min_log, self.max_log
        mu = tf.clip_by_value(self.mu, -20.0, 5.0)
        log_sigma = tf.clip_by_value(self.log_sigma, -20.0, 5.0)
        sigma = tf.exp(log_sigma)

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        z = phi(beta) - phi(alpha)
        ratio = erfcx((sigma-beta)/tf.sqrt(2.0))*tf.exp((b-mu)-beta**2/2.0)
        ratio = ratio - erfcx((sigma-alpha)/tf.sqrt(2.0))*tf.exp((a-mu)-alpha**2/2.0)
        denominator = 2*z*erfcx((2.0*sigma-beta)/tf.sqrt(2.0))*tf.exp(2.0*(b-mu)-beta**2/2.0)
        denominator = denominator - 2*z*erfcx((2.0*sigma-alpha)/tf.sqrt(2.0))*tf.exp(2.0*(a-mu)-alpha**2/2.0)
        denominator = denominator - ratio**2
        ratio = ratio/tf.sqrt(denominator)

        return ratio

    def _sample_truncated_normal(self):
        a, b = self.min_log, self.max_log
        mu = tf.clip_by_value(self.mu, -20.0, 5.0)
        log_sigma = tf.clip_by_value(self.log_sigma, -20.0, 5.0)
        sigma = tf.exp(log_sigma)

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        gamma = phi(alpha)+tf.random_uniform(mu.shape)*(phi(beta)-phi(alpha))
        return tf.clip_by_value(phi_inv(tf.clip_by_value(gamma, 1e-5, 1.0-1e-5))*sigma+mu, a, b)

    def kl(self):
        a, b = self.min_log, self.max_log
        mu = tf.clip_by_value(self.mu, -20.0, 5.0)
        log_sigma = tf.clip_by_value(self.log_sigma, -20.0, 5.0)
        sigma = tf.exp(log_sigma)

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        z = phi(beta) - phi(alpha)

        def pdf(x):
            return tf.exp(-x*x/2.0)/tf.sqrt(2.0*np.pi)

        kl = -log_sigma-tf.log(z)-(alpha*pdf(alpha)-beta*pdf(beta))/(2.0*z)
        kl = kl+tf.log(self.max_log-self.min_log)-tf.log(2.0*np.pi*np.e)/2.0
        kl = tf.reduce_sum(kl)
        return kl

    def n_active(self):
        snr = self._snr_truncated_log_normal()
        mask = tf.cast(tf.greater(snr, self.thres), tf.int32)
        n_active = tf.reduce_sum(tf.cast(mask, tf.int32))
        return n_active

    def mask(self, x, train):
        if train:
            z = tf.exp(self._sample_truncated_normal())
            snr = None
        else:
            z = self._mean_truncated_log_normal()
            snr = self._snr_truncated_log_normal()
        return z, snr

    def mask_ind(self):
        snr = self._snr_truncated_log_normal()
        return tf.reshape(tf.where(tf.greater(snr, self.thres)), [-1])

    def params(self, trainable=None):
        return [self.mu, self.log_sigma]

    def mask_ops(self, layer, mask):
        return [tf.assign(layer.mu, tf.gather(self.mu, mask)),
                tf.assign(layer.log_sigma, tf.gather(self.log_sigma, mask))]
