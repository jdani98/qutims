# -*- coding: utf-8 -*-

import sys
import numpy as np

fname1 = sys.argv[1]
fname2 = sys.argv[2]

print('Start transformation')

def conv_curves(fname):
    iteration = []
    tra_loss  = []
    val_loss  = []
    with open(fname) as f:
        for line in f:
            if line[:3] == 'Ite':
                sectors = line.split(',')
                iteration.append(int(sectors[0].split()[1]))
                tra_loss.append(float(sectors[1].split(':')[1]))
                val_loss.append(float(sectors[2].split(':')[1]))
    return iteration, tra_loss, val_loss

x,y,z = conv_curves(fname1)
outdata = np.column_stack((x,y,z))

np.savetxt(fname2, outdata, fmt=['%6i', '%8.4f', '%8.4f'], header='%6s%8s%8s' %('it','Ltra','Lval'))

