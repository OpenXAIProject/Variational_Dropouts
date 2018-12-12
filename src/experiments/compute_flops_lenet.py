from __future__ import division
import numpy as np

def conv_flops(input_shape, filters, kernel_size, stride=1, padding=0, relu=True):
    n_ch, height, width = input_shape
    n = kernel_size**2 * n_ch
    flops_per_instance = n + (n-1)

    out_height = (height-kernel_size+2*padding)/stride + 1
    out_width = (width-kernel_size+2*padding)/stride + 1
    total_flops_per_layer = filters * out_height * out_width * flops_per_instance

    # bias_add
    total_flops_per_layer += filters

    if relu:
        total_flops_per_layer += filters * out_height * out_width

    return total_flops_per_layer

def mpool_flops(input_shape, stride=2):
    return np.prod(input_shape)/(stride**2)

def fc_flops(n_in, n_out, relu=True):
    total_flops_per_layer = n_in*n_out + n_out
    if relu:
        total_flops_per_layer += n_out
    return total_flops_per_layer

def lenet_dense_flops(n_active):
    return fc_flops(n_active[0], n_active[1]) \
            + fc_flops(n_active[1], n_active[2]) \
            + fc_flops(n_active[2], 10, relu=False)

# n_active1: BB only
# n_active2: DBB
def lenet_dense_flops_dbb(n_active1, n_active2):
    return 4*n_active1[0] + fc_flops(n_active2[0], n_active1[1]) \
            + 4*n_active1[1] + fc_flops(n_active2[1], n_active1[2]) \
            + 4*n_active1[2] + fc_flops(n_active2[2], 10, relu=False)

def lenet_conv_flops(n_active):
    return conv_flops([1, 28, 28], n_active[0], 5) \
            + mpool_flops([n_active[0], 24, 24]) \
            + conv_flops([n_active[0], 12, 12], n_active[1], 5) \
            + mpool_flops([n_active[1], 8, 8]) \
            + fc_flops(n_active[2], n_active[3]) \
            + fc_flops(n_active[3], 10, relu=False)

def lenet_conv_flops_dbb(n_active1, n_active2):
    return conv_flops([1, 28, 28], n_active1[0], 5) + 4*n_active1[0] \
            + mpool_flops([n_active2[0], 24, 24]) \
            + conv_flops([n_active2[0], 12, 12], n_active1[1], 5) + 4*n_active1[1] \
            + mpool_flops([n_active2[1], 8, 8]) \
            + 4*n_active1[2] + fc_flops(n_active2[2], n_active1[3]) \
            + 4*n_active1[3] + fc_flops(n_active2[3], 10, relu=False)

import sys
if __name__ == '__main__':
    lenet_dense_base = lenet_dense_flops([784, 500, 300])
    lenet_conv_base = lenet_conv_flops([20, 50, 800, 500])

    if len(sys.argv) == 4:
        n_active = [int(x) for x in sys.argv[1:]]
        print lenet_dense_base/lenet_dense_flops(n_active)
    elif len(sys.argv) == 5:
        n_active = [int(x) for x in sys.argv[1:]]
        print lenet_conv_base/lenet_conv_flops(n_active)
    elif len(sys.argv) == 7:
        n_active1 = [int(x) for x in sys.argv[1:4]]
        n_active2 = [int(x) for x in sys.argv[4:]]
        print lenet_dense_base/lenet_dense_flops_dbb(n_active1, n_active2)
    elif len(sys.argv) == 9:
        n_active1 = [int(x) for x in sys.argv[1:5]]
        n_active2 = [int(x) for x in sys.argv[5:]]
        print lenet_conv_base/lenet_conv_flops_dbb(n_active1, n_active2)
