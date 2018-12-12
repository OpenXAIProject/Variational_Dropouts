from compute_flops import conv_flops, fc_flops, mpool_flops
import numpy as np

# takes 3 * 32 * 32,
# conv-bn-relu-dropout: 64 * 32 * 32
# conv-bn-relu: 64 * 32 * 32
# maxpool: 64 * 16 * 16

# conv-bn-relu-dropout: 128 * 16 * 16
# conv-bn-relu: 128 * 16 * 16
# maxpool: 128 * 8 * 8

# conv-bn-relu-dropout: 256 * 8 * 8
# conv-bn-relu-dropout: 256 * 8 * 8
# conv-bn-relu: 256 * 8 * 8
# maxpool: 256 * 4 * 4

# conv-bn-relu-dropout: 512 * 4 * 4
# conv-bn-relu-dropout: 512 * 4 * 4
# conv-bn-relu: 512 * 4 * 4
# maxpool: 512 * 2 * 2

# conv-bn-relu-dropout: 512 * 2 * 2
# conv-bn-relu-dropout: 512 * 2 * 2
# conv-bn-relu: 512 * 2 * 2
# maxpool: 512 * 1 * 1

# dropout: 512
# fc-relu-bn-dropout: 512
# dense: 10

def conv_block_flops(shape, filters, dropout=True):
    # conv + relu
    flops = conv_flops(shape, filters, 3)
    # bn
    flops += filters
    # dropout
    if dropout:
        flops += filters
    return flops

def vgg_flops(n_active, n_classes):
    flops = conv_block_flops([3, 32, 32], n_active[0])
    flops += conv_block_flops([n_active[0], 32, 32], n_active[1], dropout=False)
    flops += mpool_flops([n_active[1], 32, 32])

    flops += conv_block_flops([n_active[1], 16, 16], n_active[2])
    flops += conv_block_flops([n_active[2], 16, 16], n_active[3], dropout=False)
    flops += mpool_flops([n_active[3], 16, 16])

    flops += conv_block_flops([n_active[3], 8, 8], n_active[4])
    flops += conv_block_flops([n_active[4], 8, 8], n_active[5])
    flops += conv_block_flops([n_active[5], 8, 8], n_active[6], dropout=False)
    flops += mpool_flops([n_active[6], 4, 4])

    flops += conv_block_flops([n_active[6], 4, 4], n_active[7])
    flops += conv_block_flops([n_active[7], 4, 4], n_active[8])
    flops += conv_block_flops([n_active[8], 4, 4], n_active[9], dropout=False)
    flops += mpool_flops([n_active[9], 4, 4])

    flops += conv_block_flops([n_active[9], 2, 2], n_active[10])
    flops += conv_block_flops([n_active[10], 2, 2], n_active[11])
    flops += conv_block_flops([n_active[11], 2, 2], n_active[12], dropout=False)
    flops += mpool_flops([n_active[12], 2, 2])

    flops += n_active[13]
    flops += fc_flops(n_active[13], n_active[14]) + n_active[14]
    flops += fc_flops(n_active[14], n_classes, relu=False)
    return flops

def vgg_flops_dbb(n_active1, n_active2, n_classes):
    flops = conv_block_flops([3, 32, 32], n_active1[0]) + 4*n_active1[0]
    flops += conv_block_flops([n_active2[0], 32, 32], n_active1[1], dropout=False) + 4*n_active1[1]
    flops += mpool_flops([n_active2[1], 32, 32])

    flops += conv_block_flops([n_active2[1], 16, 16], n_active1[2]) + 4*n_active1[2]
    flops += conv_block_flops([n_active2[2], 16, 16], n_active1[3], dropout=False) + 4*n_active1[3]
    flops += mpool_flops([n_active2[3], 16, 16])

    flops += conv_block_flops([n_active2[3], 8, 8], n_active1[4]) + 4*n_active1[4]
    flops += conv_block_flops([n_active2[4], 8, 8], n_active1[5]) + 4*n_active1[5]
    flops += conv_block_flops([n_active2[5], 8, 8], n_active1[6], dropout=False) + 4*n_active1[6]
    flops += mpool_flops([n_active2[6], 4, 4])

    flops += conv_block_flops([n_active2[6], 4, 4], n_active1[7]) + 4*n_active1[7]
    flops += conv_block_flops([n_active2[7], 4, 4], n_active1[8]) + 4*n_active1[8]
    flops += conv_block_flops([n_active2[8], 4, 4], n_active1[9], dropout=False) + 4*n_active1[9]
    flops += mpool_flops([n_active2[10], 4, 4])

    flops += conv_block_flops([n_active2[10], 2, 2], n_active1[11]) + 4*n_active1[11]
    flops += conv_block_flops([n_active2[11], 2, 2], n_active1[12]) + 4*n_active1[12]
    flops += conv_block_flops([n_active2[12], 2, 2], n_active1[13]) + 4*n_active1[13]
    flops += mpool_flops([n_active2[13], 2, 2])

    flops += n_active2[13]
    flops += 4*n_active1[13] + fc_flops(n_active2[13], n_active1[14]) + n_active1[14]
    flops += 4*n_active1[14] + fc_flops(n_active2[14], n_classes, relu=False)
    return flops

import sys
if __name__ == '__main__':
    n_classes = int(sys.argv[1])
    base = vgg_flops([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
            n_classes)

    print len(sys.argv)
    if len(sys.argv)==17:
        n_active = [int(x) for x in sys.argv[2:]]
        print base/vgg_flops(n_active, n_classes)
    elif len(sys.argv)==32:
        n_active1 = [int(x) for x in sys.argv[2:17]]
        n_active2 = [int(x) for x in sys.argv[17:]]
        print base/vgg_flops_dbb(n_active1, n_active2, n_classes)
