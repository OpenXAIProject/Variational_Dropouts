from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import argparse
import os
from compute_flops_lenet import lenet_dense_flops, lenet_dense_flops_dbb,\
        lenet_conv_flops, lenet_conv_flops_dbb
from compute_flops_vgg import vgg_flops, vgg_flops_dbb

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='lenet_fc')
parser.add_argument('--mode', type=str, default='bbd')
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--header', type=str, default='run')
parser.add_argument('--n_trials', type=int, default=5)
args = parser.parse_args()

net_data = os.path.join(args.net, args.data) \
        if args.net=='vgg' else args.net
save_dir = os.path.join('../results', net_data,
        args.mode)

if args.net == 'lenet_fc':
    base_flops = lenet_dense_flops([784, 500, 300])
    flops_fn = lenet_dense_flops_dbb if args.mode =='dbbd' else lenet_dense_flops
elif args.net == 'lenet_conv':
    base_flops = lenet_conv_flops([20, 50, 800, 500])
    flops_fn = lenet_conv_flops_dbb if args.mode == 'dbbd' else lenet_conv_flops
elif args.net == 'vgg':
    n_classes = 10 if args.data == 'cifar10' else 100
    base_flops = vgg_flops([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512], n_classes)
    if args.mode == 'dbbd':
        flops_fn = lambda x, y: vgg_flops_dbb(x, y, n_classes)
    else:
        flops_fn = lambda x: vgg_flops(x, n_classes)
acc = [0]*args.n_trials
n_active = [0]*args.n_trials
n_active_x = [0]*args.n_trials
for i in range(args.n_trials):
    with open(os.path.join(save_dir, args.header+str(i), 'test.log'), 'r') as f:
        lines = f.readlines()
        acc[i] = float(lines[0].split()[4][:-1])
        n_active[i] = map(int, lines[2].split()[1:])
        if args.mode == 'dbbd':
            n_active_x[i] = map(int, lines[3].split()[1:])
acc = np.array(acc)
n_active_median = np.median(np.array(n_active), 0).astype(int)
if args.mode == 'dbbd':
    n_active_x_median = np.median(np.array(n_active_x), 0).astype(int)
    flops = base_flops/flops_fn(n_active_median, n_active_x_median)
else:
    flops = base_flops/flops_fn(n_active_median)

print ('{:.4f} {:.4f}'.format(np.mean(acc), np.std(acc)))
print ('{:.4f} {:.4f}'.format(np.mean(1-acc), np.std(1-acc)))
print ('{:.4f}'.format(flops))
print (n_active_median)
if args.mode == 'dbbd':
    print (n_active_x_median)
