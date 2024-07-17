# -*- coding: utf-8 -*-
"""

Created on 
@author: jdviqueira
"""

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 11})

job_id = input('Insert the slurm JOBID ')

def conv_curves(fname):
    iteration = []
    tra_loss  = []
    val_loss  = []
    with open(fname) as f:
        for line in f:
            if line[:3] == '==>':
                sectors = line.split()
                iteration.append(int(sectors[3]))
                tra_loss.append(float(sectors[4]))
                val_loss.append(float(sectors[5]))
    return iteration, tra_loss, val_loss


x1,y1,z1 = conv_curves('log_'+job_id+'.dat')

y_imin = np.argmin(y1)
y_xmin = x1[y_imin]
y_ymin = y1[y_imin]
print('tra ', y_imin,y_ymin)

z_imin = np.argmin(z1)
z_xmin = x1[z_imin]
z_zmin = z1[z_imin]
print('val ', z_imin,z_zmin)

#plt.figure(1, figsize=(8,5))
plt.plot(x1,y1, linestyle = 'solid', color='b', linewidth=0.5, label='tra (ana.)')
plt.plot(x1,z1, linestyle = 'dashed', color='r', linewidth=0.5, label='val (ana.)')
plt.plot([y_xmin],[y_ymin], 'bo', label='minimum tra')
plt.plot([z_xmin],[z_zmin], 'ro', label='minimum val')
#plt.yscale('log')
plt.grid(which='both')
plt.legend()

plt.savefig('loss_curve_'+job_id+'.png')
