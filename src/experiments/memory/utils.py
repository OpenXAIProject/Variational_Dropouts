import numpy as np

ceil = np.ceil

def conv(inputs, filters, kernel_size, strides=1, padding='VALID', \
                use_bias=True, dbb=False, dbb_bias=None):
    C, H, W = inputs
    K = kernel_size
    D = filters

    if padding == 'VALID':
        out_H = ceil(float(H-kernel_size+1) / float(strides))
        out_W = ceil(float(W-kernel_size+1) / float(strides))
        N = out_H*out_W
    elif padding == 'SAME':
        out_H = ceil(float(H) / float(strides))
        out_W = ceil(float(W) / float(strides))
        N = out_H*out_W
    else:
        raise NotImplementedError()

    rows = (K**2)*C
    cols = N
    mem_fmap = rows*cols

    rows = D
    cols = (K**2)*C
    mem_kernel = rows*cols

    memory = mem_fmap + mem_kernel
    memory = memory + D if use_bias else memory
    memory = memory + 2*dbb_bias if dbb else memory
    outputs = [filters, out_H, out_W]

    return outputs, memory

def pool(inputs, pool_size, strides=2, padding='VALID'):
    C, H, W = inputs
    if padding == 'VALID':
        out_H = ceil(float(H-pool_size+1) / float(strides))
        out_W = ceil(float(W-pool_size+1) / float(strides))
        N = out_H*out_W
    elif padding == 'SAME':
        out_H = ceil(float(H) / float(strides))
        out_W = ceil(float(W) / float(strides))
        N = out_H*out_W
    else:
        raise NotImplementedError()

    outputs = [C, out_H, out_W]
    return outputs

def flatten(inputs, dim_out, dbb=False, dbb_bias=None):
    outputs = [dim_out]
    mem3 = dbb_bias*2 if dbb else 0
    return outputs, mem3

def dense(inputs, units, use_bias=True, dbb=False, dbb_bias=None):
    memory = np.prod(inputs)*units
    memory = memory + units if use_bias else memory
    memory = memory + 2*dbb_bias if dbb else memory

    outputs = [units]
    return outputs, memory
