import tensorflow as tf
from misc import softmax_cross_entropy, accuracy

class Net(object):
    def __init__(self):
        self.base = []
        self.bbd = []
        self.dbbd = []
<<<<<<< HEAD
        self.vib = []
=======
        self.sbp = []
        self.gend = []
>>>>>>> 91bb577b992611f6ff2c20c300f4cc7f993f25aa

    def params(self, mode=None, trainable=None):
        params = []
        if mode is None:
            for layer in self.base:
                params += layer.params(trainable=trainable)
            for layer in self.bbd:
                params += layer.params(trainable=trainable)
            for layer in self.dbbd:
                params += layer.params(trainable=trainable)
            for layer in self.sbp:
                params += layer.params(trainable=trainable)
            for layer in self.gend:
                params += layer.params(trainable=trainable)
        else:
            for layer in getattr(self, mode):
                params += layer.params(trainable=trainable)
        return params

    def __call__(self, x, train=True, mode='base'):
        raise NotImplementedError()

    def apply(self, x, train, mode, l, mask_list=None):
        if mode == 'base':
            return x
        elif mode == 'bbd':
            return self.bbd[l](x, train)
        elif mode == 'dbbd':
<<<<<<< HEAD
            z_in = self.bbd[l].mask(x, train)
            if mask_list is not None:
                mask_list.append(self.dbbd[l].mask(x, train))
            return self.dbbd[l](x, train, z_in)
        elif mode == 'vib':
            return self.vib[l](x, train)
=======
            #z = self.bbd[l].mask(x, train)
            #z *= self.dbbd[l].mask(x, train)
            #return self.dbbd[l](x, train, z=z)
            return self.dbbd[l](x, train)
        elif mode == 'sbp':
            return self.sbp[l](x, train)
        elif mode == 'gend':
            return self.gend[l](x, train)
>>>>>>> 91bb577b992611f6ff2c20c300f4cc7f993f25aa
        else:
            raise ValueError('Invalid mode {}'.format(mode))

    def classify(self, x, y, train=True, mode='base'):
        x = self.__call__(x, train=train, mode=mode)
        cent = softmax_cross_entropy(x, y)
        acc = accuracy(x, y)
        return cent, acc

<<<<<<< HEAD
    def kl(self, mode='bbd'):
        kl = [layer.kl() for layer in getattr(self, mode)]
        return tf.add_n(kl)

    def n_active(self, mode='bbd'):
=======
    def reg(self, y, train=True):
        key = 'train_probit' if train else 'test_probit'
        cent = [softmax_cross_entropy(getattr(layer, key), y) \
                for layer in self.dbbd]
        cent = tf.add_n(cent)/float(len(cent))
        return cent

    def kl(self, mode='sbp'):
        kl = [layer.kl() for layer in getattr(self, mode)]
        return tf.add_n(kl)

    def n_active(self, mode='sbp'):
>>>>>>> 91bb577b992611f6ff2c20c300f4cc7f993f25aa
        return [layer.n_active() for layer in getattr(self, mode)]

    def n_active_x(self):
        return [layer.n_active_x for layer in self.dbbd]
