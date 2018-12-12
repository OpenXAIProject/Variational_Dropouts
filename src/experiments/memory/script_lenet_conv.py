import numpy as np
from utils import conv, pool, dense, flatten

def lenet_conv(inputs, struct, dbb=False):
    total_mem = 0

    if dbb:
        inputs, mem1 = conv(inputs, struct[0], 5, strides=1, padding='VALID', \
                                dbb=dbb, dbb_bias=13)
        inputs = pool(inputs, 2, strides=2, padding='VALID')

        inputs, mem2 = conv(inputs, struct[1], 5, strides=1, padding='VALID', \
                                dbb=dbb, dbb_bias=25)
        inputs = pool(inputs, 2, strides=2, padding='VALID')

        inputs, mem3 = flatten(inputs, struct[2], dbb=dbb, dbb_bias=156)
        inputs, mem4 = dense(inputs, struct[3], dbb=dbb, dbb_bias=54)
        inputs, mem5 = dense(inputs, 10, dbb=False)

    else:
        inputs, mem1 = conv(inputs, struct[0], 5, strides=1, padding='VALID', dbb=dbb)
        inputs = pool(inputs, 2, strides=2, padding='VALID')

        inputs, mem2 = conv(inputs, struct[1], 5, strides=1, padding='VALID', dbb=dbb)
        inputs = pool(inputs, 2, strides=2, padding='VALID')

        inputs, mem3 = flatten(inputs, struct[2], dbb=dbb)
        inputs, mem4 = dense(inputs, struct[3], dbb=dbb)
        inputs, mem5 = dense(inputs, 10, dbb=False)

    total_mem = mem1 + mem2 + mem3 + mem4 + mem5

    return inputs, total_mem

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

inputs = [3, 28, 28]

outputs, orig_mem = lenet_conv(inputs, nets[0][1])
print("=================================")
print(nets[0][0], outputs, orig_mem)

for idx, net in enumerate(nets):
    if idx == 0:
        continue
    name, structure = net
    outputs, total_mem = lenet_conv(inputs, structure, dbb='DBB' in name)
    print('{}, {}, {}, {:2.2f}'.format(name, outputs, total_mem,
        float(total_mem)/float(orig_mem)*100))
