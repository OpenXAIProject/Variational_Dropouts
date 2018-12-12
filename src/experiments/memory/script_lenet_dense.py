import numpy as np
from utils import conv, pool, dense, flatten

def lenet_dense(inputs, struct, dbb=False):
    total_mem = 0

    if dbb:
        inputs, mem1 = dense(struct[0], struct[1], dbb=dbb, dbb_bias=294)
        inputs, mem2 = dense(inputs, struct[2], dbb=dbb, dbb_bias=110)
        inputs, mem3 = dense(inputs, 10, dbb=dbb, dbb_bias=71)
    else:
        inputs, mem1 = dense(struct[0], struct[1], dbb=dbb, dbb_bias=None)
        inputs, mem2 = dense(inputs, struct[2], dbb=dbb, dbb_bias=None)
        inputs, mem3 = dense(inputs, 10, dbb=dbb, dbb_bias=None)

    total_mem = mem1 + mem2 + mem3

    return inputs, total_mem

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

inputs = [784]

outputs, orig_mem = lenet_dense(inputs, nets[0][1])
print("=================================")
print(nets[0][0], outputs, orig_mem)

for idx, net in enumerate(nets):
    if idx == 0:
        continue
    name, structure = net
    outputs, total_mem = lenet_dense(inputs, structure, dbb='DBB' in name)
    print('{}, {}, {}, {:2.2f}'.format(name, outputs, total_mem,
        float(total_mem)/float(orig_mem)*100))
