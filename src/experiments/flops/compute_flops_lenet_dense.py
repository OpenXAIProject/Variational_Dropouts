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
        ('Original ', [784, 500, 300]),

        ('SSL(2e-3)', [382, 18, 10]),
        ('SSL(1e-3)', [392, 24, 14]),
        ('SSL(5e-4)', [404, 32, 22]),
        ('SSL(1e-4)', [447, 41, 25]),
        ('SSL(5e-5)', [473, 46, 33]),

        ('SVD(5e-2)', [562, 44, 12]),
        ('SVD(2e-2)', [541, 52, 23]),
        ('SVD(1e-2)', [536, 64, 35]),
        ('SVD(5e-3)', [537, 77, 46]),
        ('SVD(1e-3)', [534, 101, 97]),

        ('SBP(2e-2)', [209, 87, 37]),
        ('SBP(1.5e-2)', [230, 92, 38]),
        ('SBP(1e-2)', [255, 100, 43]),
        ('SBP(5e-3)', [285, 107, 53]),
        ('SBP(1e-3)', [320, 115, 84]),

        ('BB(5e-2)', [227, 66, 37]),
        ('BB(2e-2)', [260, 100, 49]),
        ('BB(1e-2)', [294, 110, 71]),
        ('BB(5e-3)', [332, 117, 90]),
        ('BB(1e-3)', [437, 189, 123]),

        ('DBB(5e-2)', [94, 15, 13]),
        ('DBB(2e-2)', [94, 22, 22]),
        ('DBB(1e-2)', [106, 29, 32]),
        ('DBB(5e-3)', [130, 56, 46]),
        ('DBB(1e-3)', [355, 160, 117]),

        ('VIB(5e-2)', [125, 66,  21]),
        ('VIB(2e-2)', [131, 94,  26]),
        ('VIB(1e-2)', [139, 101, 28]),
        ('VIB(5e-3)', [150, 105, 27]),
        ('VIB(1e-3)', [205, 105, 44]),

        ('GD(5e-2)', [362, 108, 99]),
        ('GD(2e-2)', [440, 118, 123]),
        ('GD(1e-2)', [488, 142, 136]),
        ('GD(5e-3)', [501, 176, 194]),
        ('GD(1e-3)', [506, 216, 273])
]

orig_flops = lenet_dense_flops([784, 500, 300])
print(orig_flops)
print("=================================")

for idx, net in enumerate(nets):
    if idx == 0:
        continue
    name, structure = net
    if 'DBB' not in name:
        flops_ = lenet_dense_flops(structure)
    else:
        flops_ = lenet_dense_flops_dbb([294, 110, 71], structure)

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
