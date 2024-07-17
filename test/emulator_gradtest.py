# -*- coding: utf-8 -*-


#from qurecnets.models import CZme3, CZladder2p1, encodeP3, encodeP2
#from qurecnets.loss_fs import mse
#from qurecnets.readout import expectZ
#from qurecnets.emc import emulator

from qurecnets.emc import EMCZ2

import numpy as np
from time import time



client = None

"""
### DASK !!! #######
# To initialize DASK, first launch the interactive session for this script, and then submit the DASK cluster
from dask import delayed
#from dask_jobqueue import SLURMCluster
from distributed import Client, wait
from time import time

from dask_cluster import create_dask_client

print('modules imported')

#info = "./dask_cluster_ft3/scheduler_info.json"
info = "./scheduler_info.json"
client = create_dask_client(info)

print(client)
print(type(client))
#####################
"""



nT = 2; nE = 1; nM = 1; nL = 1; nx = 0
np.random.seed(0)
param0 = np.random.random(2*(nE*nx + (nE+nM)*nL) + 3*nE)
xin = np.random.random(nT*nE).reshape(nT,nE)
yin = np.random.random(nT)

#class qrnn(emulator, mse, expectZ, CZladder2p1, encodeP2):
#    pass

QRNN = EMCZ2(nT,nE,nM,nL,nx, shots=0, rseed = None)
print(QRNN.nE)

yout = QRNN.evaluate(param0,xin)
print(yout)



#################
print('Gradients for circuit evaluation')
t0 = time()
grad_fd = QRNN.grad_fd(param0,xin, client=client)
t1 = time()
print(grad_fd)
print(t1-t0)

t0 = time()
grad_psr = QRNN.grad_psr(param0, xin, client=client)
t1 = time()
print(grad_psr)
print(t1-t0)

print(grad_psr-grad_fd)
print(np.all(grad_psr-grad_fd < 1.e-5))


#"""
#################
print('Gradients for Loss function')
paramBL = np.concatenate(([0.3],param0))
t0 = time()
grad_fd = QRNN.grad_BL_fd(paramBL, xin,yin,3, client=client)
t1 = time()
print(grad_fd)
print(t1-t0)

t0 = time()
grad_psr = QRNN.grad_BL_psr(paramBL, xin,yin,3, client=client)
t1 = time()
print(grad_psr)
print(t1-t0)

print(grad_psr-grad_fd)
print(np.all(grad_psr-grad_fd < 1.e-5))
#"""