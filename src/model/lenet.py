from net import Net
from layers import *
from bbdropout import BBDropout
from dbbdropout import DBBDropout
from vib import VIB
from sbp import SBP
from gendropout import GenDropout

class LeNetFC(Net):
    def __init__(self, n_units=None, mask=None, thres=1e-3,
            name='lenet_fc', reuse=None):
        super(LeNetFC, self).__init__()
        n_units = [784, 500, 300] if n_units is None else n_units
        self.mask = mask
        with tf.variable_scope(name, reuse=reuse):
            for i in range(3):
                self.base.append(Dense(n_units[i],
                    (10 if i==2 else n_units[i+1]), name='dense'+str(i+1)))
                self.vib.append(VIB(n_units[i], name='vib'+str(i+1)))
                self.bbd.append(BBDropout(n_units[i], name='bbd'+str(i+1)))
                self.dbbd.append(DBBDropout(n_units[i], name='dbbd'+str(i+1)))
                self.sbp.append(SBP(n_units[i], name='sbp'+str(i+1)))
                self.gend.append(GenDropout(n_units[i], name='gend'+str(i+1)))

    def __call__(self, x, train, mode='base'):
        x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
        x = relu(self.base[0](self.apply(x, train, mode, 0)))
        x = relu(self.base[1](self.apply(x, train, mode, 1)))
        x = self.base[2](self.apply(x, train, mode, 2))
        return x

    def measure_dbb_runtime(self, x):
        x = tf.gather(x, self.mask, axis=1)
        masks = []
        for i in range(3):
            z = self.bbd[i].mask(x, False)
            mask0 = self.dbbd[i].mask_ind(x, z)
            masks.append(mask0)
            x = self.dbbd[i](x, False, z, mask0=mask0)
            x = self.base[i](x, in_mask=mask0)
            if i < 2:
                x = relu(x)
        return x, masks

    def build_compressed(self, sess=None, mode='bbd', init=True):
        masks = [layer.mask_ind() for layer in self.bbd]
        sess = tf.Session() if sess is None else sess
        n_units = sess.run([tf.shape(mask)[0] for mask in masks])
        net = LeNetFC(n_units=n_units, mask=tf.constant(sess.run(masks[0])),
                name='compressed_lenet_fc')
        if init:
            init_ops = [param.initializer for param in net.params()]
            sess.run(init_ops)
            mask_ops = []
            for i in range(3):
                mask_ops += self.base[i].mask_ops(net.base[i],
                        in_mask=masks[i], out_mask=(None if i==2 else masks[i+1]))
                mask_ops += self.bbd[i].mask_ops(net.bbd[i], masks[i])
                if mode == 'dbbd':
                    mask_ops += self.dbbd[i].mask_ops(net.dbbd[i], masks[i])
            sess.run(mask_ops)
        return net

class LeNetConv(Net):
    def __init__(self, n_units=None, mask=None,
            name='lenet_conv', reuse=None):
        n_units = [20, 50, 800, 500] if n_units is None else n_units
        self.mask = mask
        super(LeNetConv, self).__init__()
        with tf.variable_scope(name, reuse=reuse):
            self.base.append(Conv(1, n_units[0], 5, name='conv1'))
            self.base.append(Conv(n_units[0], n_units[1], 5, name='conv2'))
            self.base.append(Dense(n_units[2], n_units[3], name='dense3'))
            self.base.append(Dense(n_units[3], 10, name='dense4'))
            for i in range(4):
                self.bbd.append(BBDropout(n_units[i], name='bbd'+str(i+1)))
                self.dbbd.append(DBBDropout(n_units[i], name='dbbd'+str(i+1)))
                self.vib.append(VIB(n_units[i], name='vib'+str(i+1)))
                self.sbp.append(SBP(n_units[i], name='sbp'+str(i+1)))
                self.gend.append(GenDropout(n_units[i], name='gend'+str(i+1)))

    def __call__(self, x, train, mode='base'):
        x = tf.reshape(x, [-1, 1, 28, 28])
        x = pool(relu(self.apply(self.base[0](x), train, mode, 0)))
        x = pool(relu(self.apply(self.base[1](x), train, mode, 1)))
        x = flatten(x)
        x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
        x = relu(self.base[2](self.apply(x, train, mode, 2)))
        x = relu(self.base[3](self.apply(x, train, mode, 3)))
        return x

    def build_compressed(self, sess=None, mode='bbd', init=True):
        masks = [layer.mask_ind() for layer in self.bbd]

        sess = tf.Session() if sess is None else sess
        n_units = sess.run([tf.shape(mask)[0] for mask in masks])

        np_mask2, np_mask3 = sess.run([masks[1], masks[2]])
        np_mask = np.zeros(50, dtype=np.int32)
        np_mask[np_mask2] = 2
        np_mask = np.repeat(np_mask, 16)
        np_mask[np_mask3] += 1
        np_mask = np.where(np_mask[np_mask>1]>2)[0]
        mask = tf.constant(np_mask)

        net = LeNetConv(n_units=n_units, mask=mask, name='compressed_lenet_conv')
        if init:
            init_ops = [param.initializer for param in net.params()]
            sess.run(init_ops)
            mask_ops = []
            mask_ops += self.base[0].mask_ops(net.base[0], out_mask=masks[0])
            mask_ops += self.base[1].mask_ops(net.base[1],
                    in_mask=masks[0], out_mask=masks[1])
            mask_ops += self.base[2].mask_ops(net.base[2],
                    in_mask=masks[2], out_mask=masks[3])
            mask_ops += self.base[3].mask_ops(net.base[3], in_mask=masks[3])
            for i in range(4):
                mask_ops += self.bbd[i].mask_ops(net.bbd[i], masks[i])
                if mode=='dbbd':
                    mask_ops += self.dbbd[i].mask_ops(net.dbbd[i], masks[i])
            sess.run(mask_ops)
        return net

if __name__ == '__main__':
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    x = tf.placeholder(tf.float32, [None, 784])
    net = LeNetConv()
    sess = tf.Session()
    saver = tf.train.Saver(net.get_params('base')+net.get_params('bbd'))
    saver.restore(sess, os.path.join('../results/lenet_conv/bbd/run0', 'model'))

    cnet = net.build_compressed(sess)

    x = tf.placeholder(tf.float32, [None, 784])
    np_x = np.random.rand(1, 784)
    np_y1 = sess.run(net(x, train=False, mode='bbd'), {x:np_x})
    print np_y1
    np_y2 = sess.run(cnet(x, train=False, mode='bbd'), {x:np_x})
    print np_y2
    print np.mean(np_y1-np_y2)
