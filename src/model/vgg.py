from net import Net
from layers import *
from bbdropout import BBDropout
from dbbdropout import DBBDropout
from vib import VIB
from sbp import SBP
from gendropout import GenDropout
from misc import *

class VGG(Net):
    def __init__(self, n_classes, n_units=None, mask=None,
            name='vgg', reuse=None):
        super(VGG, self).__init__()
        n_units = [64, 64, 128, 128, 256, 256, 256,
                512, 512, 512, 512, 512, 512, 512, 512] \
                        if n_units is None else n_units
        self.mask = mask
        self.n_classes = n_classes
        def create_block(l, n_in, n_out):
            self.base.append(Conv(n_in, n_out, 3,
                name='conv'+str(l), padding='SAME'))
            self.base.append(BatchNorm(n_out, name='bn'+str(l)))
            self.bbd.append(BBDropout(n_out, name='bbd'+str(l),
                a_uc_init=2.0))
            self.dbbd.append(DBBDropout(n_out, name='dbbd'+str(l)))
            self.vib.append(VIB(n_out, name='vib'+str(l)))
            self.sbp.append(SBP(n_out, name='sbp'+str(l)))
            self.gend.append(GenDropout(n_out, name='gend'+str(l)))

        with tf.variable_scope(name, reuse=reuse):
            create_block(1, 3, n_units[0])
            for i in range(1, 13):
                create_block(i+1, n_units[i-1], n_units[i])

            self.bbd.append(BBDropout(n_units[13], name='bbd14'))
            self.dbbd.append(DBBDropout(n_units[13], name='dbbd14'))
            self.vib.append(VIB(n_units[13], name='vib14'))
            self.sbp.append(SBP(n_units[13], name='sbp14'))
            self.gend.append(GenDropout(n_units[13], name='gend14'))

            self.base.append(Dense(n_units[13], n_units[14], name='dense14'))
            self.base.append(BatchNorm(n_units[14], name='bn14'))

            self.bbd.append(BBDropout(n_units[14], name='bbd15'))
            self.dbbd.append(DBBDropout(n_units[14], name='dbbd15'))
            self.vib.append(VIB(n_units[14], name='vib15'))
            self.sbp.append(SBP(n_units[14], name='sbp15'))
            self.gend.append(GenDropout(n_units[14], name='gen15'))

            self.base.append(Dense(n_units[14], n_classes, name='dense15'))

    def __call__(self, x, train, mode='base', mask_list=[]):
        def apply_block(x, train, l, mode, p=None):
            conv = self.base[2*l-2]
            bn = self.base[2*l-1]
            x = self.apply(conv(x), train, mode, l-1, mask_list=mask_list)
            if mode == 'sbp':
                x = relu(bn(x, False))
            else:
                x = relu(bn(x, train))
            x = pool(x) if p is None else tf.layers.dropout(x, p, training=train)
            return x

        p_list = [0.3, None,
                0.4, None,
                0.4, 0.4, None,
                0.4, 0.4, None,
                0.4, 0.4, None]
        for l, p in enumerate(p_list):
            x = apply_block(x, train, l+1, mode, p=p)

        x = flatten(x)
        x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
        x = tf.layers.dropout(x, 0.5, training=train) if mode=='base' else x
        x = self.base[2*13](self.apply(x, train, mode, 13, mask_list=mask_list))
        x = relu(self.base[2*13+1](x, train))

        x = tf.layers.dropout(x, 0.5, training=train) if mode=='base' else x
        x = self.base[-1](self.apply(x, train, mode, 14, mask_list=mask_list))
        return x

    def build_compressed(self, sess=None, mode='bbd', init=True):
        if mode == 'sbp':
            masks = [layer.mask_ind() for layer in self.sbp]
        elif mode == 'gend':
            masks = [layer.mask_ind() for layer in self.gend]
        else:
            raise NotImplementedError()

        sess = tf.Session() if sess is None else sess

        np_mask13, np_mask14 = sess.run([masks[12], masks[13]])
        np_mask = np.zeros(512, dtype=np.int32)
        np_mask[np_mask13] = 2
        np_mask[np_mask14] += 1
        np_mask = np.where(np_mask[np_mask>1]>2)[0]
        np_mask_ = np.intersect1d(np_mask13, np_mask14)

        masks[12] = tf.constant(np_mask_)
        masks[13] = tf.constant(np_mask_)

        n_units = sess.run([tf.shape(mask)[0] for mask in masks])
        mask = tf.constant(np_mask)
        net = VGG(self.n_classes, n_units=n_units, mask=mask,
                name='compressed_vgg')

        if init:
            init_ops = [param.initializer for param in net.params()]
            sess.run(init_ops)

            mask_ops = self.base[0].mask_ops(net.base[0], out_mask=masks[0]) \
                    + self.base[1].mask_ops(net.base[1], masks[0]) \
                    + self.bbd[0].mask_ops(net.bbd[0], masks[0])
            if mode=='dbbd':
                mask_ops += self.dbbd[0].mask_ops(net.dbbd[0], masks[0])
            for i in range(1, 13):
                mask_ops += self.base[2*i].mask_ops(net.base[2*i], masks[i-1], masks[i]) \
                        + self.base[2*i+1].mask_ops(net.base[2*i+1], masks[i]) \
                        + self.bbd[i].mask_ops(net.bbd[i], masks[i])
                if mode=='dbbd':
                    mask_ops += self.dbbd[i].mask_ops(net.dbbd[i], masks[i])

            #mask_13 = masks[13] if len(np_mask14) < len(np_mask13) else masks[12]
            mask_ops += [self.base[2*13].mask_ops(net.base[2*13], masks[13], masks[14]),
                self.base[2*13+1].mask_ops(net.base[2*13+1], masks[14]),
                self.sbp[13].mask_ops(net.sbp[13], masks[13]),
                self.gend[13].mask_ops(net.gend[13], masks[13])]
            mask_ops += [self.base[-1].mask_ops(net.base[-1], in_mask=masks[-1]),
                self.sbp[-1].mask_ops(net.sbp[-1], masks[-1]),
                self.gend[-1].mask_ops(net.gend[-1], masks[-1])]
            sess.run(mask_ops)
        return net

if __name__=='__main__':
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    net = VGG(10)
    x = tf.placeholder(tf.float32, [None, 3, 32, 32])
    sess = tf.Session()
    saver = tf.train.Saver(net.get_params('base')+net.get_params('bbd'))
    saver.restore(sess, os.path.join('../results/vgg/cifar10/bbd/run0', 'model'))

    cnet = net.build_compressed(sess)

    np_x = np.random.rand(1, 3, 32, 32)
    np_y1, np_y2 = sess.run([net(x, False, mode='bbd'),
        cnet(x, False, mode='bbd')], {x:np_x})
    print np_y1
    print np_y2
