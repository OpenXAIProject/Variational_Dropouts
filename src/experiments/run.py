from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model.lenet import LeNetFC, LeNetConv
from model.vgg import VGG
from model.misc import *
from utils.logger import Logger
import utils.mnist, utils.cifar10, utils.cifar100
import os
import argparse

np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='lenet_fc')
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--exp_name', type=str, default='trial')
parser.add_argument('--base_dir', type=str, default=None)
parser.add_argument('--bbd_dir', type=str, default=None)
parser.add_argument('--dbbd_dir', type=str, default=None)
parser.add_argument('--sbp_dir', type=str, default=None)
parser.add_argument('--gend_dir', type=str, default=None)
parser.add_argument('--test', action='store_true')
parser.add_argument('--thres', type=float, default=1e-3)
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--init_lr', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--n_iter', type=int, default=0)
args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

net_data = os.path.join(args.net, args.data) \
        if args.net=='vgg' else args.net
base_dir = os.path.join('../results', net_data, 'base')
bbd_dir = os.path.join('../results', net_data, 'bbd', args.exp_name)
dbbd_dir = os.path.join('../results', net_data, 'dbbd', args.exp_name)
vib_dir = os.path.join('../results', net_data, 'vib', args.exp_name)
if args.net == 'lenet_fc' or args.net == 'lenet_conv':
    input_fn = utils.mnist.input_fn
    NUM_TRAIN = utils.mnist.NUM_TRAIN
    NUM_TEST = utils.mnist.NUM_TEST
    n_classes = 10
    net = LeNetFC(thres=args.thres) if args.net == 'lenet_fc' \
            else LeNetConv()
    base_dir = os.path.join('../results', args.net, 'base') \
            if args.base_dir is None else args.base_dir
    bbd_dir = os.path.join('../results', args.net, 'bbd/trial') \
            if args.bbd_dir is None else args.bbd_dir
    dbbd_dir = os.path.join('../results', args.net, 'dbbd/trial') \
            if args.dbbd_dir is None else args.dbbd_dir
    sbp_dir = os.path.join('../results', args.net, 'sbp/iter%d/trial_%e'%(args.n_iter,args.init_lr)) \
            if args.sbp_dir is None else args.sbp_dir
    gend_dir = os.path.join('../results', args.net, 'gend/iter%d/trial_%e'%(args.n_iter,args.init_lr)) \
            if args.gend_dir is None else args.gend_dir

elif args.net == 'vgg':
    if args.data == 'cifar10':
        input_fn = utils.cifar10.input_fn
        NUM_TRAIN = utils.cifar10.NUM_TRAIN
        NUM_TEST = utils.cifar10.NUM_TEST
        n_classes = 10
    elif args.data == 'cifar100':
        input_fn = utils.cifar100.input_fn
        NUM_TRAIN = utils.cifar100.NUM_TRAIN
        NUM_TEST = utils.cifar100.NUM_TEST
        n_classes = 100
    net = VGG(n_classes)

    base_dir = os.path.join('../results', args.net, args.data, 'base') \
            if args.base_dir is None else args.base_dir
    bbd_dir = os.path.join('../results', args.net, args.data, 'bbd/trial') \
            if args.bbd_dir is None else args.bbd_dir
    dbbd_dir = os.path.join('../results', args.net, args.data, 'dbbd/trial') \
            if args.dbbd_dir is None else args.dbbd_dir
    sbp_dir = os.path.join('../results', args.net, '%s/sbp/iter%d/trial_%e'%(\
               args.data,args.n_iter,args.init_lr)) \
            if args.sbp_dir is None else args.sbp_dir
    gend_dir = os.path.join('../results', args.net, '%s/gend/iter%d/trial_%e'%(\
               args.data,args.n_iter,args.init_lr)) \
            if args.gend_dir is None else args.gend_dir

else:
    raise ValueError('Invalid net {}'.format(args.net))

x, y = input_fn(True, args.batch_size)
tx, ty = input_fn(False, args.batch_size)
n_train_batches = NUM_TRAIN // args.batch_size
n_test_batches = NUM_TEST // args.batch_size

def base_train():
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    print ('results saved in {}'.format(base_dir))

    cent, acc = net.classify(x, y)
    tcent, tacc = net.classify(tx, ty, train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)

    global_step = tf.train.get_or_create_global_step()
    if args.net == 'lenet_fc' or args.net == 'lenet_conv':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .7]]
        vals = [1e-3, 1e-4, 1e-5]
    elif args.net == 'vgg':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .8]]
        vals = [1e-3, 1e-4, 1e-5]
    lr = get_staircase_lr(global_step, bdrs, vals)
    loss = cent + 1e-4*l2_loss(base_trn_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(base_vars)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    logfile = open(os.path.join(base_dir, 'train.log'), 'w', 0)
    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc]
    for epoch in range(1, args.n_epochs+1):
        line = 'Epoch {}, lr {:.3e}'.format(epoch, sess.run(lr))
        print(line)
        logfile.write(line+'\n')
        train_logger.clear()
        for it in range(1, n_train_batches+1):
            train_logger.record(sess.run(train_to_run))
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()
        for it in range(1, n_test_batches+1):
            test_logger.record(sess.run(test_to_run))
        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        print()
        logfile.write('\n')

        if epoch%args.save_freq == 0:
            saver.save(sess, os.path.join(base_dir, 'model'))
    logfile.close()
    saver.save(sess, os.path.join(base_dir, 'model'))

def base_test():
    cent, acc = net.classify(tx, ty, train=False)
    base_vars = net.params('base')
    sess = tf.Session()
    tf.train.Saver(base_vars).restore(sess, os.path.join(base_dir, 'model'))
    logger = Logger('cent', 'acc')
    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
    logger.show(header='test')

def bbd_train():
    if not os.path.isdir(bbd_dir):
        os.makedirs(bbd_dir)
    print ('results saved in {}'.format(bbd_dir))

    cent, acc = net.classify(x, y, mode='bbd')
    tcent, tacc = net.classify(tx, ty, mode='bbd', train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)
    bbd_vars = net.params('bbd')

    kl = net.kl(mode='bbd')
    n_active = net.n_active(mode='bbd')
    loss = cent + kl/NUM_TRAIN + 1e-4*l2_loss(base_trn_vars)
    global_step = tf.train.get_or_create_global_step()
    if args.net == 'lenet_fc' or args.net == 'lenet_conv':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .7]]
        vals1 = [args.init_lr*r for r in [1., 0.1, 0.01]]
        vals2 = [0.1*v for v in vals1]
    elif args.net == 'vgg':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .8]]
        vals1 = [args.init_lr*r for r in [1., 0.1, 0.01]]
        vals2 = [0.1*v for v in vals1]
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                var_list=bbd_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                var_list=base_trn_vars)
        train_op = tf.group(train_op1, train_op2)
    saver = tf.train.Saver(bbd_vars+base_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(base_trn_vars).restore(sess, os.path.join(base_dir, 'model'))

    logfile = open(os.path.join(bbd_dir, 'train.log'), 'w', 0)
    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc]
    for epoch in range(1, args.n_epochs+1):
        np_lr1, np_lr2 = sess.run([lr1, lr2])
        line = 'Epoch {}, bbd lr {:.3e}, base lr {:.3e}'.format(epoch, np_lr1, np_lr2)
        print(line)
        logfile.write(line+'\n')
        train_logger.clear()
        for it in range(1, n_train_batches+1):
            train_logger.record(sess.run(train_to_run))
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()
        for it in range(1, n_test_batches+1):
            test_logger.record(sess.run(test_to_run))
        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        np_kl, np_n_active = sess.run([kl, n_active])
        line = 'kl: ' + str(np_kl) + '\n'
        line += 'n_active: ' + str(np_n_active)
        print(line)
        logfile.write(line+'\n')
        print()
        logfile.write('\n')

        if epoch%args.save_freq == 0:
            saver.save(sess, os.path.join(bbd_dir, 'model'))
    logfile.close()
    saver.save(sess, os.path.join(bbd_dir, 'model'))

def bbd_test():
    logfile = open(os.path.join(bbd_dir, 'test.log'), 'w', 0)
    cent, acc = net.classify(tx, ty, mode='bbd', train=False)
    kl = net.kl(mode='bbd')
    n_active = net.n_active(mode='bbd')
    logger = Logger('cent', 'acc')
    sess = tf.Session()
    tf.train.Saver(net.params('base')+net.params('bbd')).restore(
            sess, os.path.join(bbd_dir, 'model'))
    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
    logger.show(header='test', logfile=logfile)
    np_kl, np_n_active = sess.run([kl, n_active])
    line = 'kl: {:.4f}\n'.format(np_kl)
    line += 'n_active: ' + ' '.join(map(str, np_n_active))
    print(line)
    logfile.write(line+'\n')
    logfile.close()

def sbp_train():
    if not os.path.isdir(sbp_dir):
        os.makedirs(sbp_dir)
    print ('results saved in {}'.format(sbp_dir))

    init_lr = args.init_lr
    cent, acc = net.classify(x, y, mode='sbp')
    tcent, tacc = net.classify(tx, ty, mode='sbp', train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)
    sbp_vars = net.params('sbp')
    kl = net.kl(mode='sbp')
    n_active = net.n_active(mode='sbp')
    loss = cent + kl/NUM_TRAIN + 1e-4*l2_loss(base_trn_vars)
    global_step = tf.train.get_or_create_global_step()
    if args.net == 'lenet_fc' or args.net == 'lenet_conv':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .7]]
        vals1 = [init_lr, init_lr*0.1, init_lr*0.01]
        vals2 = [init_lr*0.1, init_lr*0.01, init_lr*0.001]
    elif args.net == 'vgg':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .8]]
        vals1 = [init_lr, init_lr*0.1, init_lr*0.01]
        vals2 = [init_lr*0.1, init_lr*0.01, init_lr*0.001]
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                var_list=sbp_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                var_list=base_trn_vars)
        train_op = tf.group(train_op1, train_op2)
    saver = tf.train.Saver(sbp_vars+base_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(base_vars).restore(sess, os.path.join(base_dir, 'model'))

    logfile = open(os.path.join(sbp_dir, 'train.log'), 'w', 0)
    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc]
    for epoch in range(1, args.n_epochs+1):
        np_lr1, np_lr2 = sess.run([lr1, lr2])
        line = 'Epoch {}, sbp lr {:.3e}, base lr {:.3e}'.format(epoch, np_lr1, np_lr2)
        print(line)
        logfile.write(line+'\n')
        train_logger.clear()
        for it in range(1, n_train_batches+1):
            train_logger.record(sess.run(train_to_run))
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()
        for it in range(1, n_test_batches+1):
            test_logger.record(sess.run(test_to_run))
        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        np_kl, np_n_active = sess.run([kl, n_active])
        line = 'kl: ' + str(np_kl) + '\n'
        line += 'n_active: ' + str(np_n_active)
        print(line)
        logfile.write(line+'\n')
        print()
        logfile.write('\n')

        if epoch%args.save_freq == 0:
            saver.save(sess, os.path.join(sbp_dir, 'model'))
    logfile.close()
    saver.save(sess, os.path.join(sbp_dir, 'model'))

def sbp_test():
    cent, acc = net.classify(tx, ty, mode='sbp', train=False)
    kl = net.kl(mode='sbp')
    n_active = net.n_active(mode='sbp')

    logfile = open(os.path.join(sbp_dir, 'test.log'), 'w', 0)
    logger = Logger('cent', 'acc')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(net.params('base')+net.params('sbp')).restore(
            sess, os.path.join(sbp_dir, 'model'))
    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
    logger.show(header='test', logfile=logfile)
    np_kl, np_n_active = sess.run([kl, n_active])
    line = 'kl: ' + str(np_kl) + '\n'
    line += 'n_active: ' + str(np_n_active)
    print(line)
    logfile.write(line+'\n')

def dbbd_train():
    if not os.path.isdir(dbbd_dir):
        os.makedirs(dbbd_dir)
    print ('results saved in {}'.format(dbbd_dir))

    cent, acc = net.classify(x, y, mode='dbbd')
    tcent, tacc = net.classify(tx, ty, mode='dbbd', train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)
    bbd_vars = net.params('bbd')
    dbbd_trn_vars = net.params('dbbd', trainable=True)
    all_vars = net.params()
    kl = net.kl('dbbd')
    n_active = net.n_active(mode='dbbd')
    n_active_x = net.n_active_x()
    loss = cent + kl/NUM_TRAIN + 1e-4*l2_loss(base_trn_vars)
    global_step = tf.train.get_or_create_global_step()
    if args.net == 'lenet_fc' or args.net == 'lenet_conv':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .7]]
        vals1 = [args.init_lr*r for r in [1., 0.1, 0.01]]
        vals2 = [0.1*v for v in vals1]
    elif args.net == 'vgg':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .8]]
        vals1 = [args.init_lr*r for r in [1., 0.1, 0.01]]
        vals2 = [0.1*v for v in vals1]
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                var_list=dbbd_trn_vars, global_step=global_step)
                #var_list=bbd_vars+dbbd_trn_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                var_list=base_trn_vars)
        train_op = tf.group(train_op1, train_op2)
    saver = tf.train.Saver(all_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #tf.train.Saver(base_trn_vars).restore(sess, os.path.join(base_dir, 'model'))
    tf.train.Saver(base_trn_vars+bbd_vars).restore(sess, os.path.join(bbd_dir, 'model'))

    logfile = open(os.path.join(dbbd_dir, 'train.log'), 'w', 0)
    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc, n_active_x]
    for epoch in range(1, args.n_epochs+1):
        np_lr1, np_lr2 = sess.run([lr1, lr2])
        line = 'Epoch {}, dbbd lr {:.3e}, base lr {:.3e}'.format(
                epoch, np_lr1, np_lr2)
        print(line)
        logfile.write(line+'\n')
        train_logger.clear()
        for it in range(1, n_train_batches+1):
            train_logger.record(sess.run(train_to_run))
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()
        np_n_active_x = 0
        for it in range(1, n_test_batches+1):
            res = sess.run(test_to_run)
            test_logger.record(res[:-1])
            np_n_active_x += np.array(res[-1], dtype=float)
        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        np_kl, np_n_active = sess.run([kl, n_active])
        np_n_active_x = (np_n_active_x/n_test_batches).astype(int)
        line = 'kl: ' + str(np_kl) + '\n'
        line += 'n_active: ' + str(np_n_active) + '\n'
        line += 'n_active_x: ' + str(np_n_active_x.tolist())
        print(line)
        logfile.write(line+'\n')
        print()
        logfile.write('\n')

        if epoch%args.save_freq == 0:
            saver.save(sess, os.path.join(dbbd_dir, 'model'))
    logfile.close()
    saver.save(sess, os.path.join(dbbd_dir, 'model'))

def dbbd_test():
    logfile = open(os.path.join(dbbd_dir, 'test.log'), 'w', 0)
    cent, acc = net.classify(tx, ty, mode='dbbd', train=False)
    kl = net.kl(mode='dbbd')
    n_active = net.n_active(mode='dbbd')
    n_active_x = net.n_active_x(mode='dbbd')
    logger = Logger('cent', 'acc')
    sess = tf.Session()
    tf.train.Saver(net.params()).restore(
            sess, os.path.join(dbbd_dir, 'model'))

    np_n_active_x = 0
    for it in range(1, n_test_batches+1):
        res = sess.run([cent, acc, n_active_x])
        logger.record(res[:-1])
        np_n_active_x += np.array(res[-1], dtype=float)
    logger.show(header='test', logfile=logfile)
    np_kl, np_n_active = sess.run([kl, n_active])
    np_n_active_x = (np_n_active_x/n_test_batches).astype(int).tolist()
    line = 'kl: {:.4f}\n'.format(np_kl)
    line += 'n_active: ' + ' '.join(map(str, np_n_active)) + '\n'
    line += 'n_active_x: ' + ' '.join(map(str, np_n_active_x)) + '\n'
    print(line)
    logfile.write(line+'\n')
    logfile.close()

def vib_train():
    if not os.path.isdir(vib_dir):
        os.makedirs(vib_dir)
    print ('results saved in {}'.format(vib_dir))

    cent, acc = net.classify(x, y, mode='vib')
    tcent, tacc = net.classify(tx, ty, mode='vib', train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)
    vib_vars = net.params('vib')
    kl = net.kl('vib')
    n_active = net.n_active('vib')
    global_step = tf.train.get_or_create_global_step()
    if args.net == 'lenet_fc' or args.net == 'lenet_conv':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .7]]
        vals1 = [args.init_lr*r for r in [1., 0.1, 0.01]]
        vals2 = [0.1*v for v in vals1]
        gamma = 1e-4
    elif args.net == 'vgg':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .8]]
        vals1 = [args.init_lr*r for r in [1., 0.1, 0.01]]
        vals2 = [0.1*v for v in vals1]
        gamma = 1e-5
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)
    gamma = gamma if args.gamma is None else args.gamma
    print ('gamma: {:.3e}'.format(gamma))
    loss = cent + gamma*kl + 1e-4*l2_loss(base_trn_vars)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                var_list=vib_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                var_list=base_trn_vars)
        train_op = tf.group(train_op1, train_op2)
    saver = tf.train.Saver(vib_vars+base_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(base_trn_vars).restore(sess, os.path.join(base_dir, 'model'))

    logfile = open(os.path.join(vib_dir, 'train.log'), 'w', 0)
    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc]

    for epoch in range(1, args.n_epochs+1):
        np_lr1, np_lr2 = sess.run([lr1, lr2])
        line = 'Epoch {}, vib lr {:.3e}, base lr {:.3e}'.format(epoch, np_lr1, np_lr2)
        print(line)
        logfile.write(line+'\n')
        train_logger.clear()
        for it in range(1, n_train_batches+1):
            train_logger.record(sess.run(train_to_run))
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()
        for it in range(1, n_test_batches+1):
            test_logger.record(sess.run(test_to_run))
        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        np_kl, np_n_active = sess.run([kl, n_active])
        line = 'kl: ' + str(np_kl) + '\n'
        line += 'n_active: ' + str(np_n_active)
        print(line)
        logfile.write(line+'\n')
        print()
        logfile.write('\n')

        if epoch%args.save_freq == 0:
            saver.save(sess, os.path.join(vib_dir, 'model'))
    logfile.close()
    saver.save(sess, os.path.join(vib_dir, 'model'))

def vib_test():
    logfile = open(os.path.join(vib_dir, 'test.log'), 'w', 0)
    cent, acc = net.classify(tx, ty, mode='vib', train=False)
    kl = net.kl('vib')
    n_active = net.n_active('vib')
    logger = Logger('cent', 'acc')
    sess = tf.Session()
    tf.train.Saver(net.params('base')+net.params('vib')).restore(
            sess, os.path.join(vib_dir, 'model'))

    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
    logger.show(header='test', logfile=logfile)
    np_kl, np_n_active = sess.run([kl, n_active])

    line = 'kl: ' + str(np_kl) + '\n'
    line += 'n_active: ' + str(np_n_active)
    print(line)
    logfile.write(line+'\n')

def gend_train():
    if not os.path.isdir(gend_dir):
        os.makedirs(gend_dir)
    print ('results saved in {}'.format(gend_dir))

    init_lr = args.init_lr
    cent, acc = net.classify(x, y, mode='gend')
    tcent, tacc = net.classify(tx, ty, mode='gend', train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)
    gend_vars = net.params('gend')
    kl = net.kl(mode='gend')
    n_active = net.n_active(mode='gend')
    loss = cent + kl/NUM_TRAIN + 1e-4*l2_loss(base_trn_vars)
    global_step = tf.train.get_or_create_global_step()
    if args.net == 'lenet_fc' or args.net == 'lenet_conv':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .7]]
        vals1 = [init_lr, init_lr*0.1, init_lr*0.01]
        vals2 = [init_lr*0.1, init_lr*0.01, init_lr*0.001]
    elif args.net == 'vgg':
        bdrs = [int(n_train_batches*args.n_epochs*r) for r in [.5, .8]]
        vals1 = [init_lr, init_lr*0.1, init_lr*0.01]
        vals2 = [init_lr*0.1, init_lr*0.01, init_lr*0.001]
        #vals2 = [1e-3, 1e-4, 1e-5]
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                var_list=gend_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                var_list=base_trn_vars)
        train_op = tf.group(train_op1, train_op2)

    saver = tf.train.Saver(gend_vars+base_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(base_trn_vars).restore(sess, os.path.join(base_dir, 'model'))

    logfile = open(os.path.join(gend_dir, 'train.log'), 'w', 0)

    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc]

    for epoch in range(1, args.n_epochs+1):
        np_lr1, np_lr2 = sess.run([lr1, lr2])
        line = 'Epoch {}, gend lr {:.3e}, base lr {:.3e}'.format(epoch, np_lr1, np_lr2)
        print(line)
        logfile.write(line+'\n')
        train_logger.clear()
        for it in range(1, n_train_batches+1):
            train_logger.record(sess.run(train_to_run))
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()

        for it in range(1, n_test_batches+1):
            test_logger.record(sess.run(test_to_run))
        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        np_kl, np_n_active = sess.run([kl, n_active])
        line = 'kl: ' + str(np_kl) + '\n'
        line += 'n_active: ' + str(np_n_active)
        print(line)
        logfile.write(line+'\n')
        logfile.write('\n')

        if epoch%args.save_freq == 0:
            saver.save(sess, os.path.join(gend_dir, 'model'))
    logfile.close()
    saver.save(sess, os.path.join(gend_dir, 'model'))

def gend_test():
    cent, acc = net.classify(tx, ty, mode='gend', train=False)
    kl = net.kl(mode='gend')
    n_active = net.n_active(mode='gend')

    logfile = open(os.path.join(gend_dir, 'test.log'), 'w', 0)
    logger = Logger('cent', 'acc')
    sess = tf.Session()
    tf.train.Saver(net.params('base')+net.params('gend')).restore(
            sess, os.path.join(gend_dir, 'model'))
    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
    logger.show(header='test', logfile=logfile)
    np_kl, np_n_active = sess.run([kl, n_active])
    line = 'kl: {:.4f}\n'.format(np_kl)
    line += 'n_active: ' + ' '.join(map(str, np_n_active))
    print(line)
    logfile.write(line+'\n')
    logfile.close()

    '''
    print('\nCompressed model')
    cnet = net.build_compressed(sess=sess, mode='gend')
    print ('creating compressed model')
    tf.train.Saver(cnet.params('base')+cnet.params('gend')).save(
            sess, os.path.join(gend_dir, 'cmodel'))
    cent, acc = cnet.classify(tx, ty, mode='gend', train=False)
    logger.clear()
    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
    logger.show(header='test')
    '''
if __name__=='__main__':
    if args.mode == 'base':
        if not args.test:
            base_train()
        else:
            base_test()
    elif args.mode == 'bbd':
        if not args.test:
            bbd_train()
        else:
            bbd_test()
    elif args.mode == 'dbbd':
        if not args.test:
            dbbd_train()
        else:
            dbbd_test()
    elif args.mode == 'vib':
        if not args.test:
            vib_train()
        else:
            vib_test()
    elif args.mode == 'eval_runtime':
        eval_runtime()
    elif args.mode == 'draw':
        print('here')
        draw()
    elif args.mode == 'sbp':
        if not args.test:
            sbp_train()
        else:
            sbp_test()
    elif args.mode == 'gend':
        if not args.test:
            gend_train()
        else:
            gend_test()
    else:
        raise ValueError('Invalid mode {}'.format(args.mode))
