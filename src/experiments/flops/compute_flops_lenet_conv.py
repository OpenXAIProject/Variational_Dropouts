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

nets = [
        ('Original ', [20, 50, 800, 500]),

        ('SSL(2e-3)', [4, 5, 68,  10]),
        ('SSL(1e-3)', [4, 7, 102, 10]),
        ('SSL(5e-4)', [6, 8, 107, 10]),
        ('SSL(1e-4)', [7, 14, 189, 13]),
        ('SSL(5e-5)', [8, 15, 218, 12]),

        ('SVD(5e-2)', [20,  9, 121, 14]),
        ('SVD(2e-2)', [12,  13,  192, 20]),
        ('SVD(1e-2)', [10,  16,  238, 29]),
        ('SVD(5e-3)', [10,  18,  285, 35]),
        ('SVD(1e-3)', [13,  25,  362, 48]),

        ('SBP(2e-2)', [2, 15,  123, 46]),
        ('SBP(1.5e-2)', [10,  18,  112, 33]),
        ('SBP(1e-2)', [10,  18,  128, 38]),
        ('SBP(5e-3)', [11,  20,  143, 46]),
        ('SBP(1e-3)', [14,  23,  204, 52]),

        ('BB(5e-2)', [13,  20,  99,  30]),
        ('BB(2e-2)', [12,  23,  128, 40]),
        ('BB(1e-2)', [13,  25,  156, 54]),
        ('BB(5e-3)', [14,  25,  198, 62]),
        ('BB(1e-3)', [14,  27,  403, 260]),

        ('DBB(5e-2)', [10,  15,  33,  10]),
        ('DBB(2e-2)', [11,  20,  40,  18]),
        ('DBB(1e-2)', [13,  24,  53,  27]),
        ('DBB(5e-3)', [13,  24,  80,  42]),
        ('DBB(1e-3)', [14,  26,  349, 164]),

        ('VIB(5e-2)', [12,  15,  69,  23]),
        ('VIB(2e-2)', [12,  17,  78,  29]),
        ('VIB(1e-2)', [12,  18,  82,  34]),
        ('VIB(5e-3)', [12,  19,  90,  37]),
        ('VIB(1e-3)', [13,  22,  120, 42]),

        ('GD(5e-2)', [14,  25,  249, 70]),
        ('GD(2e-2)', [14,  26,  325, 116]),
        ('GD(1e-2)', [14,  29,  368, 174]),
        ('GD(5e-3)', [14,  32,  378, 268]),
        ('GD(1e-3)', [14,  34,  516, 484])
]

orig_flops = lenet_conv_flops([20, 50, 800, 500])
print(orig_flops)
print("=================================")

for idx, net in enumerate(nets):
    if idx == 0:
        continue
    name, structure = net
    if 'DBB' not in name:
        flops_ = lenet_conv_flops(structure)
    else:
        flops_ = lenet_conv_flops_dbb([13, 25, 156, 54], structure)

    print('{}, {}, {:2.2f}'.format(name, flops_, float(orig_flops)/float(flops_)))

'''
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
'''
