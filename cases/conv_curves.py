# -*- coding: utf-8 -*-
"""
Created on
@author: jdviqueira

Convergence curves for cases (a), (b) and (c) in arXiv:2310.20671. Moving averages are included.
"""

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 11})

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


def mov_avg(x,y,size):
    # moving average with x-axis re-dimension
    xn = np.linspace(x[0],x[-1],len(x)-size)
    yn = np.array([np.mean(y[i:i+size]) for i in range(len(y)-size)])
    return xn,yn

"""
x1,y1,z1 = conv_curves('slurm-4413886_8.out')
x2,y2,z2 = conv_curves('slurm-4413899_8.out')


#plt.figure(1, figsize=(8,5))
plt.plot(x1,y1, label='tra (ana.)')
plt.plot(x1,z1, label='val (ana.)')
plt.plot(x2,y2, label='tra (num.)')
plt.plot(x2,z2, label='val (num.)')
#plt.yscale('log')
plt.grid(which='both')
plt.legend()
"""

plt.close('all')
fig, ax = plt.subplots(1,3, figsize=(2*5.65,2*1.5))


### CASE (a) ######################################################################################
# NUMERICAL
mav_size = 100
x,y,z = conv_curves('dectriang/bash0/log_13192.dat')
xn,yn = mov_avg(x,y,mav_size)
_,zn = mov_avg(x,z,mav_size)
ax[0].plot(xn, yn, linestyle = 'solid', color='b', linewidth=1., label=None)
ax[0].plot(xn, zn, linestyle = 'dashed', color='b', linewidth=1., label=None)

ax[0].plot(x, y, linestyle = 'solid', color='b', alpha=0.05, linewidth=1, label=None)
ax[0].plot(x, z, linestyle = 'dashed', color='b', alpha=0.05, linewidth=1, label=None)


# ANALYTICAL - 1k shots
ids = ['13300', '13304', '13305', '13306', '13406', '13407', '13408', '13409']
for i,id in enumerate(ids):
    
    x,y,z = conv_curves('dectriang/bash1k/log_'+id+'.dat')
    
    ax[0].plot(x, y, linestyle = 'solid', color='r', alpha=0.05, linewidth=0.5, label=None)
    ax[0].plot(x, z, linestyle = 'dashed', color='r', alpha=0.05, linewidth=0.5, label=None)

    mav_size = 50
    xn,yn = mov_avg(x,y,mav_size)
    _,zn = mov_avg(x,z,mav_size)
    ax[0].plot(xn, yn, linestyle = 'solid', color='r', alpha=0.75, linewidth=0.5, label=None)
    ax[0].plot(xn, zn, linestyle = (5, (10, 3)), color='r', alpha=0.75, linewidth=0.5, label=None)


# ANALYTICAL - 10k shots
ids = ['13302', '13307', '13308', '13309', '13410', '13411', '13412', '13413']
for i,id in enumerate(ids):
    
    x,y,z = conv_curves('dectriang/bash10k/log_'+id+'.dat')
    
    ax[0].plot(x, y, linestyle = 'solid', color='tab:orange', alpha=0.05, linewidth=0.5, label=None)
    ax[0].plot(x, z, linestyle = 'dashed', color='tab:orange', alpha=0.05, linewidth=0.5, label=None)

    mav_size = 50
    xn,yn = mov_avg(x,y,mav_size)
    _,zn = mov_avg(x,z,mav_size)
    ax[0].plot(xn, yn, linestyle = 'solid', color='tab:orange', alpha=0.75, linewidth=0.5, label=None)
    ax[0].plot(xn, zn, linestyle = (5, (10, 3)), color='tab:orange', alpha=0.75, linewidth=0.5, label=None)



#ax[0].ticklabel_format(style='plain', useOffset=True)
ax[0].set_yscale('log')
ax[0].set_xlabel('Iterations')
ax[0].grid(which='both')
ax[0].set_ylabel('Loss')
ax[0].set_title('(a)')
ax[0].set_xlim(0,2000)
ax[0].set_yticks([0.01,0.03,0.06,0.1,0.2])
ax[0].set_yticklabels([0.01,0.03,0.06,0.1,0.2])




### CASE (b) ######################################################################################
# NUMERICAL
mav_size = 100
x,y,z = conv_curves('vdp1/bash0/log_13227.dat')
xn,yn = mov_avg(x,y,mav_size)
_,zn = mov_avg(x,z,mav_size)
ax[1].plot(xn, yn, linestyle = 'solid', color='b', linewidth=1., label=None)
ax[1].plot(xn, zn, linestyle = 'dashed', color='b', linewidth=1., label=None)

ax[1].plot(x, y, linestyle = 'solid', color='b', alpha=0.05, linewidth=1, label=None)
ax[1].plot(x, z, linestyle = 'dashed', color='b', alpha=0.05, linewidth=1, label=None)


# ANALYTICAL - 1k shots
ids = ['13312', '13313', '13314', '13315', '13414', '13415', '13416', '13417']
for i,id in enumerate(ids):
    
    x,y,z = conv_curves('vdp1/bash1k/log_'+id+'.dat')
    
    ax[1].plot(x, y, linestyle = 'solid', color='r', alpha=0.05, linewidth=0.5, label=None)
    ax[1].plot(x, z, linestyle = 'dashed', color='r', alpha=0.05, linewidth=0.5, label=None)

    mav_size = 50
    xn,yn = mov_avg(x,y,mav_size)
    _,zn = mov_avg(x,z,mav_size)
    ax[1].plot(xn, yn, linestyle = 'solid', color='r', alpha=0.5, linewidth=0.5, label=None)
    ax[1].plot(xn, zn, linestyle = (5, (10, 3)), color='r', alpha=0.5, linewidth=0.5, label=None)


# ANALYTICAL - 10k shots
ids = ['13316', '13317', '13318', '13319', '13418', '13419', '13420', '13421']
for i,id in enumerate(ids):
    
    x,y,z = conv_curves('vdp1/bash10k/log_'+id+'.dat')
    
    ax[1].plot(x, y, linestyle = 'solid', color='tab:orange', alpha=0.05, linewidth=0.5, label=None)
    ax[1].plot(x, z, linestyle = 'dashed', color='tab:orange', alpha=0.05, linewidth=0.5, label=None)

    mav_size = 50
    xn,yn = mov_avg(x,y,mav_size)
    _,zn = mov_avg(x,z,mav_size)
    ax[1].plot(xn, yn, linestyle = 'solid', color='tab:orange', alpha=0.5, linewidth=0.5, label=None)
    ax[1].plot(xn, zn, linestyle = (5, (10, 3)), color='tab:orange', alpha=0.5, linewidth=0.5, label=None)



#ax[1].ticklabel_format(style='plain', useOffset=True)
ax[1].set_yscale('log')
ax[1].set_xlabel('Iterations')
ax[1].grid(which='both')
#ax[1].set_ylabel('Loss')
##ax[1].set_yticks([0.04,0.06,0.1,0.2,0.3])
##ax[1].set_yticklabels([0.04,0.06,0.1,0.2,0.3])
ax[1].set_yticks([0.07,0.1,0.12,0.2,0.3,0.4])
ax[1].set_yticklabels([0.07,0.1,0.12,0.2,0.3,0.4])
ax[1].set_xlim(0,2000)
ax[1].set_title('(b)')



### CASE (c) ######################################################################################
# NUMERICAL
mav_size = 100
x,y,z = conv_curves('vdp2/bash0/log_13244.dat')
#x = x[:-mav_size]
xn,yn = mov_avg(x,y,mav_size)
_,zn = mov_avg(x,z,mav_size)
ax[2].plot(xn, yn, linestyle = 'solid', color='b', linewidth=1., label='tra (num.)')
ax[2].plot(xn, zn, linestyle = 'dashed', color='b', linewidth=1., label='val (num.)')

ax[2].plot(x, y, linestyle = 'solid', color='b', alpha=0.05, linewidth=1, label=None)
ax[2].plot(x, z, linestyle = 'dashed', color='b', alpha=0.05, linewidth=1, label=None)


# ANALYTICAL - 1k shots
ids = ['13320', '13321', '13322', '13323', '13422', '13424', '13425', '13426']
for i,id in enumerate(ids):
    if i == len(ids)-1:
        label_tra = 'tra ($10^3$)'; label_val = 'val ($10^3$)'
    else:
        label_tra = None ; label_val = None
    
    x,y,z = conv_curves('vdp2/bash1k/log_'+id+'.dat')
    
    ax[2].plot(x, y, linestyle = 'solid', color='r', alpha=0.05, linewidth=0.5, label=None)
    ax[2].plot(x, z, linestyle = 'dashed', color='r', alpha=0.05, linewidth=0.5, label=None)

    mav_size = 50
    xn,yn = mov_avg(x,y,mav_size)
    _,zn = mov_avg(x,z,mav_size)
    ax[2].plot(xn, yn, linestyle = 'solid', color='r', alpha=0.75, linewidth=0.5, label=label_tra)
    ax[2].plot(xn, zn, linestyle = (5, (10, 3)), color='r', alpha=0.75, linewidth=0.5, label=label_val)


# ANALYTICAL - 10k shots
ids = ['13324', '13325', '13326', '13327', '13427', '13428', '13430', '13432']
for i,id in enumerate(ids):
    if i == len(ids)-1:
        label_tra = 'tra ($10^4$)'; label_val = 'val ($10^4$)'
    else:
        label_tra = None ; label_val = None
    
    x,y,z = conv_curves('vdp2/bash10k/log_'+id+'.dat')
    
    ax[2].plot(x, y, linestyle = 'solid', color='tab:orange', alpha=0.05, linewidth=0.5, label=None)
    ax[2].plot(x, z, linestyle = 'dashed', color='tab:orange', alpha=0.05, linewidth=0.5, label=None)

    mav_size = 50
    xn,yn = mov_avg(x,y,mav_size)
    _,zn = mov_avg(x,z,mav_size)
    ax[2].plot(xn, yn, linestyle = 'solid', color='tab:orange', alpha=0.75, linewidth=0.5, label=label_tra)
    ax[2].plot(xn, zn, linestyle = (5, (10, 3)), color='tab:orange', alpha=0.75, linewidth=0.5, label=label_val)


#ax[2].ticklabel_format(style='plain', useOffset=True)
ax[2].set_yscale('log')
ax[2].set_xlabel('Iterations')
ax[2].grid(which='both')
ax[2].set_xlim(0,2000)
#ax[2].set_ylabel('Loss')
ax[2].set_yticks([0.04,0.06,0.1,0.2,0.3,0.4])
ax[2].set_yticklabels([0.04,0.06,0.1,0.2,0.3,0.4])
ax[2].set_title('(c)')

ax[2].legend(loc='lower right', bbox_to_anchor=(1.51, 0.25), ncol=1,fontsize=10)


plt.savefig('article_optim_curves_log.pdf', bbox_inches='tight')
