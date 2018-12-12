import numpy as np
from utils import conv, pool, flatten, dense

def batch_norm(inputs):
    outputs = inputs
    memory = 2*inputs[0]
    return outputs, memory

def vgg(inputs, struct, dbb_biases, dbb=False):

    def _block(inputs, filters, dbb, dbb_bias):
        inputs, mem1 = conv(inputs, filters, 3, strides=1, padding='SAME', dbb=dbb,
                dbb_bias=dbb_bias)
        inputs, mem2 = batch_norm(inputs)
        #print(inputs)
        return inputs, mem1+mem2

    total_mem = 0
    inputs, mem1 = _block(inputs, struct[0], dbb, dbb_biases[0])
    inputs, mem2 = _block(inputs, struct[1], dbb, dbb_biases[1])
    inputs = pool(inputs, 2, strides=2, padding='VALID')

    inputs, mem3 = _block(inputs, struct[2], dbb, dbb_biases[2])
    inputs, mem4 = _block(inputs, struct[3], dbb, dbb_biases[3])
    inputs = pool(inputs, 2, strides=2, padding='VALID')

    inputs, mem5 = _block(inputs, struct[4], dbb, dbb_biases[4])
    inputs, mem6 = _block(inputs, struct[5], dbb, dbb_biases[5])
    inputs, mem7 = _block(inputs, struct[6], dbb, dbb_biases[6])
    inputs = pool(inputs, 2, strides=2, padding='VALID')

    inputs, mem8 = _block(inputs, struct[7], dbb, dbb_biases[7])
    inputs, mem9 = _block(inputs, struct[8], dbb, dbb_biases[8])
    inputs, mem10 = _block(inputs, struct[9], dbb, dbb_biases[9])
    inputs = pool(inputs, 2, strides=2, padding='VALID')

    inputs, mem11 = _block(inputs, struct[10], dbb, dbb_biases[10])
    inputs, mem12 = _block(inputs, struct[11], dbb, dbb_biases[11])
    inputs, mem13 = _block(inputs, struct[12], dbb, dbb_biases[12])
    inputs = pool(inputs, 2, strides=2, padding='VALID')

    inputs, mem_flat = flatten(inputs, struct[13], dbb, dbb_biases[13])
    inputs, mem14 = dense(inputs, struct[14], dbb=dbb, dbb_bias=dbb_biases[14])
    inputs, mem15 = batch_norm(inputs)
    inputs, mem16 = dense(inputs, struct[15], dbb=False)

    total_mem = mem1 + mem2 + mem3 + mem4 + \
                mem5 + mem6 + mem7 + mem8 + \
                mem9 + mem10 + mem11 + mem12 + \
                mem13 + mem14 + mem15 + mem16 + mem_flat

    return inputs, total_mem

nets_cifar10 = [
    ('Original', [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 10]),
    ('SSL', [64, 64, 128, 128, 254, 201, 46, 92, 76, 14, 123, 78, 354, 354, 486, 10]),
    ('SVD', [64, 64, 128, 128, 254, 206, 48, 96, 90, 14, 147, 107, 85, 85, 458, 10]),
    ('SBP', [64, 64, 128, 128, 254, 209, 55, 107, 103, 15, 163, 125, 346, 375, 491, 10]),
    ('BB',  [64, 64, 128, 126, 251, 174, 41, 87, 81, 13, 118, 110, 25, 25, 69, 10]),
    ('DBB', [62, 61, 126, 122, 236, 134, 35, 63, 65, 11, 93, 94, 19, 20, 50, 10])
]
nets_cifar100 = [
    ('Original', [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 10]),
    ('SSL', [64, 64, 128, 128, 255, 255, 135, 235, 162, 25, 300, 238, 512, 512, 512, 10]),
    ('SVD', [64, 64, 128, 128, 255, 255, 135, 234, 171, 26, 332, 257, 301, 311, 512, 10]),
    ('SBP', [64, 64, 128, 128, 255, 255, 135, 218, 173, 26, 345, 270, 512, 512, 512, 10]),
    ('BB',  [64, 64, 128, 128, 255, 255, 134, 213, 152, 25, 269, 256, 47, 47, 188, 10]),
    ('DBB', [63, 63, 126, 127, 246, 235, 129, 167, 125, 22, 218, 227, 36, 36, 105, 10])
]

inputs = [3, 32, 32]
outputs, total_mem = vgg(inputs, nets_cifar10[0][1], nets_cifar10[4][1])

print("=========================================")
print(nets_cifar10[0][0], outputs, total_mem, len(nets_cifar10[0][1]))
orig_mem = total_mem
for idx, net in enumerate(nets_cifar10):
    outputs, total_mem = vgg(inputs, net[1], nets_cifar10[4][1], idx==5)
    print('{}, {}, {}, {:.1f}, {}'.format(net[0], outputs, total_mem,
        float(total_mem)/float(orig_mem)*100, len(net[1])))

outputs, total_mem = vgg(inputs, nets_cifar100[0][1], nets_cifar100[4][1])

print("=========================================")
print(nets_cifar100[0][0], outputs, total_mem, len(nets_cifar100[0][1]))
orig_mem = total_mem
for idx, net in enumerate(nets_cifar100):
    outputs, total_mem = vgg(inputs, net[1], nets_cifar100[4][1], idx==5)
    print('{}, {}, {}, {:.1f}, {}'.format(net[0], outputs, total_mem,
        float(total_mem)/float(orig_mem)*100, len(net[1])))
