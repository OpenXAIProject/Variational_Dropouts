import numpy as np
import csv
import os
import argparse

def write_textfile(fname, results):
    with open(fname, 'w') as f:
        for result in results:
            f.write(result)

def read_textfile(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def _get_median(result):
    accs = [1.0 - elem['acc'] for elem in result]
    med = np.median(accs)
    std = np.std(accs)
    idx = np.where(accs == med)[0][0]
    n_active = result[idx]['n_active']

    median_result = {'acc': med, 'std': std, 'n_active': n_active}

    return median_result

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='vgg')
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--init_lr', type=float, default=1e-2)
args = parser.parse_args()
net = args.net
data = args.data
mode = args.mode

#lr_list = [5e-2, 2e-2, 1.5e-2, 1e-2, 5e-3, 1e-3]
#lr_list = [5e-2, 2e-2, 1e-2, 5e-3, 1e-3]
lr_list = [2e-3, 1e-3, 5e-4, 1e-4, 5e-5]
#lr_list = [5e-2, 2e-2, 1e-2, 5e-3, 1e-3]

#lr_list = [2e-2, 1e-2, 5e-3, 1e-3]
#lr_list = [1e-3, 5e-4, 1e-4, 5e-5]
#lr_list = [1e-3, 1e-4, 1e-5, 5e-5]
#lr_list = [1e-3, 5e-4, 2e-4, 1e-4]

results = []
for init_lr in lr_list:
    result = []
    for n_iter in [0, 1, 2, 3, 4]:
    #for n_iter in [0, 1, 2]:
    #for n_iter in [5, 6, 7]:
        if net == 'vgg':
            if mode == 'ssl' or mode == 'svd':
                result_dir = os.path.join('../results', net, '%s/%s/iter%d/trial_%g'%(\
                                data, mode, n_iter, init_lr))
            else:
                result_dir = os.path.join('../results', net, '%s/%s/iter%d/trial_%e'%(\
                                data, mode, n_iter, init_lr))
        else:
            if mode == 'ssl' or mode == 'svd':
                result_dir = os.path.join('../results', net, '%s/iter%d/trial_%g'%(\
                                mode, n_iter, init_lr))
            else:
                result_dir = os.path.join('../results', net, '%s/iter%d/trial_%e'%(\
                                mode, n_iter, init_lr))

        contents = read_textfile(os.path.join(result_dir, 'test.log'))
        acc = float(contents[0].split(',')[1][5:])
        n_active = contents[-1].split('[')[1].split(']')[0]
        print(init_lr, n_iter, acc, n_active)
        result_ = {'acc': acc, 'n_active': n_active}
        result.append(result_)
    median_result = _get_median(result)
    line = str(init_lr) + ',' + median_result['n_active'] \
                        + ',' + str(median_result['acc']) + ',' + str(median_result['std']) + '\n'
    results.append(line)
    print(line)

if net == 'vgg':
    write_textfile(os.path.join('../results', net, '%s/%s/%s_%s_%s_summary.csv'%(\
                        data, mode, net, data, mode)), results)
else:
    write_textfile(os.path.join('../results', net, '%s/%s_%s_summary.csv'%(mode, net, mode)), results)
