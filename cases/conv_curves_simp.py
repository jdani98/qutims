# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Convergence curves for cases (a), (b) and (c) in arXiv:2310.20671, averaged over N processes when sampling noise is added. Moving averages are included.
"""
# CONVERGENCE CURVES SIMPLIFIED

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
mav_size = 50
x,y,z = conv_curves('dectriang/bash0/log_13192.dat')
xn,yn = mov_avg(x,y,mav_size)
_,zn = mov_avg(x,z,mav_size)
ax[0].plot(xn, yn, linestyle = 'solid', color='b', linewidth=1., label=None)
ax[0].plot(xn, zn, linestyle = 'dashed', color='b', linewidth=1., label=None)

###ax[0].plot(x, y, linestyle = 'solid', color='b', alpha=0.05, linewidth=1, label=None)
###ax[0].plot(x, z, linestyle = 'dashed', color='b', alpha=0.05, linewidth=1, label=None)


# ANALYTICAL - 0 shots
x,y,z = conv_curves('dectriang/bash0k/log_13626.dat')

mav_size = 50
xn,yn = mov_avg(x,y,mav_size)
_ ,zn = mov_avg(x,z,mav_size)

ax[0].plot(xn, yn, linestyle = 'solid', color='limegreen', linewidth=0.5, label=None)
ax[0].plot(xn, zn, linestyle = (5, (10, 3)), color='limegreen', linewidth=0.5, label=None)


# ANALYTICAL - 1k shots
ids = ['13300', '13304', '13305', '13306', '13406', '13407', '13408', '13409']

ys = [];  zs = []
yns = []; zns = []
for i,id in enumerate(ids):
    x,y,z = conv_curves('dectriang/bash1k/log_'+id+'.dat')
    ys.append(y)
    zs.append(z)

ys = np.array(ys)
zs = np.array(zs)


ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

mav_size = 50
xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
_ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)



ax[0].fill_between(xn, ymeann-ystdn, ymeann+ystdn, facecolor='r', alpha=0.2, label=None)
ax[0].plot(xn, ymeann, linestyle = 'solid', color='r', linewidth=0.5, label=None)

ax[0].fill_between(xn, zmeann-zstdn, zmeann+zstdn, facecolor='r', alpha=0.2, label=None)
ax[0].plot(xn, zmeann, linestyle = (5, (10, 3)), color='r', linewidth=0.5, label=None)



# ANALYTICAL - 10k shots
ids = ['13302', '13307', '13308', '13309', '13410', '13411', '13412', '13413']

ys = [];  zs = []
yns = []; zns = []
for i,id in enumerate(ids):
    x,y,z = conv_curves('dectriang/bash10k/log_'+id+'.dat')
    ys.append(y)
    zs.append(z)

ys = np.array(ys)
zs = np.array(zs)


ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

mav_size = 50
xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
_ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)



ax[0].fill_between(xn, ymeann-ystdn, ymeann+ystdn, facecolor='tab:orange', alpha=0.2, label=None)
ax[0].plot(xn, ymeann, linestyle = 'solid', color='tab:orange', linewidth=0.5, label=None)

ax[0].fill_between(xn, zmeann-zstdn, zmeann+zstdn, facecolor='tab:orange', alpha=0.2, label=None)
ax[0].plot(xn, zmeann, linestyle = (5, (10, 3)), color='tab:orange', linewidth=0.5, label=None)



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
mav_size = 50
x,y,z = conv_curves('vdp1/bash0/log_13227.dat')
xn,yn = mov_avg(x,y,mav_size)
_,zn = mov_avg(x,z,mav_size)
ax[1].plot(xn, yn, linestyle = 'solid', color='b', linewidth=1., label=None)
ax[1].plot(xn, zn, linestyle = 'dashed', color='b', linewidth=1., label=None)

###ax[1].plot(x, y, linestyle = 'solid', color='b', alpha=0.05, linewidth=1, label=None)
###ax[1].plot(x, z, linestyle = 'dashed', color='b', alpha=0.05, linewidth=1, label=None)


# ANALYTICAL - 0 shots
x,y,z = conv_curves('vdp1/bash0k/log_13627.dat')

mav_size = 50
xn,yn = mov_avg(x,y,mav_size)
_ ,zn = mov_avg(x,z,mav_size)

ax[1].plot(xn, yn, linestyle = 'solid', color='limegreen', linewidth=0.5, label=None)
ax[1].plot(xn, zn, linestyle = (5, (10, 3)), color='limegreen', linewidth=0.5, label=None)


# ANALYTICAL - 1k shots
ids = ['13312', '13313', '13314', '13315', '13414', '13415', '13416', '13417']

ys = [];  zs = []
yns = []; zns = []
for i,id in enumerate(ids):
    x,y,z = conv_curves('vdp1/bash1k/log_'+id+'.dat')
    ys.append(y)
    zs.append(z)

ys = np.array(ys)
zs = np.array(zs)


ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

mav_size = 50
xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
_ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)



ax[1].fill_between(xn, ymeann-ystdn, ymeann+ystdn, facecolor='r', alpha=0.2, label=None)
ax[1].plot(xn, ymeann, linestyle = 'solid', color='r', linewidth=0.5, label=None)

ax[1].fill_between(xn, zmeann-zstdn, zmeann+zstdn, facecolor='r', alpha=0.2, label=None)
ax[1].plot(xn, zmeann, linestyle = (5, (10, 3)), color='r', linewidth=0.5, label=None)


# ANALYTICAL - 10k shots
ids = ['13316', '13317', '13318', '13319', '13418', '13419', '13420', '13421']

ys = [];  zs = []
yns = []; zns = []
for i,id in enumerate(ids):
    x,y,z = conv_curves('vdp1/bash10k/log_'+id+'.dat')
    ys.append(y)
    zs.append(z)

ys = np.array(ys)
zs = np.array(zs)


ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

mav_size = 50
xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
_ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)



ax[1].fill_between(xn, ymeann-ystdn, ymeann+ystdn, facecolor='tab:orange', alpha=0.2, label=None)
ax[1].plot(xn, ymeann, linestyle = 'solid', color='tab:orange', linewidth=0.5, label=None)

ax[1].fill_between(xn, zmeann-zstdn, zmeann+zstdn, facecolor='tab:orange', alpha=0.2, label=None)
ax[1].plot(xn, zmeann, linestyle = (5, (10, 3)), color='tab:orange', linewidth=0.5, label=None)



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
mav_size = 50
x,y,z = conv_curves('vdp2/bash0/log_13244.dat')
#x = x[:-mav_size]
xn,yn = mov_avg(x,y,mav_size)
_,zn = mov_avg(x,z,mav_size)
ax[2].plot(xn, yn, linestyle = 'solid', color='b', linewidth=1., label='tra (num.)')
ax[2].plot(xn, zn, linestyle = 'dashed', color='b', linewidth=1., label='val (num.)')

###ax[2].plot(x, y, linestyle = 'solid', color='b', alpha=0.05, linewidth=1, label=None)
###ax[2].plot(x, z, linestyle = 'dashed', color='b', alpha=0.05, linewidth=1, label=None)


# ANALYTICAL - 0 shots
x,y,z = conv_curves('vdp2/bash0k/log_13629.dat')

mav_size = 50
xn,yn = mov_avg(x,y,mav_size)
_ ,zn = mov_avg(x,z,mav_size)

ax[2].plot(xn, yn, linestyle = 'solid', color='limegreen', linewidth=0.5, label='tra (an.)')
ax[2].plot(xn, zn, linestyle = (5, (10, 3)), color='limegreen', linewidth=0.5, label='val (an.)')


# ANALYTICAL - 1k shots
ids = ['13320', '13321', '13322', '13323', '13422', '13424', '13425', '13426']

ys = [];  zs = []
yns = []; zns = []
for i,id in enumerate(ids):
    x,y,z = conv_curves('vdp2/bash1k/log_'+id+'.dat')
    ys.append(y)
    zs.append(z)

ys = np.array(ys)
zs = np.array(zs)


ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

mav_size = 50
xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
_ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)



ax[2].fill_between(xn, ymeann-ystdn, ymeann+ystdn, facecolor='r', alpha=0.2, label=None)
ax[2].plot(xn, ymeann, linestyle = 'solid', color='r', linewidth=0.5, label='tra ($10^3$)')

ax[2].fill_between(xn, zmeann-zstdn, zmeann+zstdn, facecolor='r', alpha=0.2, label=None)
ax[2].plot(xn, zmeann, linestyle = (5, (10, 3)), color='r', linewidth=0.5, label='val ($10^3$)')


# ANALYTICAL - 10k shots
ids = ['13324', '13325', '13326', '13327', '13427', '13428', '13430', '13432']

ys = [];  zs = []
yns = []; zns = []
for i,id in enumerate(ids):
    x,y,z = conv_curves('vdp2/bash10k/log_'+id+'.dat')
    ys.append(y)
    zs.append(z)

ys = np.array(ys)
zs = np.array(zs)


ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

mav_size = 50
xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
_ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)



ax[2].fill_between(xn, ymeann-ystdn, ymeann+ystdn, facecolor='tab:orange', alpha=0.2, label=None)
ax[2].plot(xn, ymeann, linestyle = 'solid', color='tab:orange', linewidth=0.5, label='tra ($10^4$)')

ax[2].fill_between(xn, zmeann-zstdn, zmeann+zstdn, facecolor='tab:orange', alpha=0.2, label=None)
ax[2].plot(xn, zmeann, linestyle = (5, (10, 3)), color='tab:orange', linewidth=0.5, label='val ($10^4$)')


#ax[2].ticklabel_format(style='plain', useOffset=True)
ax[2].set_yscale('log')
ax[2].set_xlabel('Iterations')
ax[2].grid(which='both')
ax[2].set_xlim(0,2000)
#ax[2].set_ylabel('Loss')
ax[2].set_yticks([0.04,0.06,0.1,0.2,0.3,0.4])
ax[2].set_yticklabels([0.04,0.06,0.1,0.2,0.3,0.4])
ax[2].set_title('(c)')

ax[2].legend(loc='lower right', bbox_to_anchor=(1.51, 0.15), ncol=1,fontsize=10)


plt.savefig('article_optim_curves_log_simp.pdf', bbox_inches='tight')
