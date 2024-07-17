# -*- coding: utf-8 -*-

import numpy as np

#from qurecnets.models import CZme3, CZladder2p1, encodeP3, encodeP2
#from qurecnets.loss_fs import mse
#from qurecnets.readout import expectZ
#from qurecnets.emc import emulator

from qurecnets.emc import EMCZ2


### DASK !!! #######
# To initialize DASK, first launch the interactive session for this script, and then submit the DASK cluster
from dask import delayed
#from dask_jobqueue import SLURMCluster
from distributed import Client, wait
from time import time

from dask_cluster import create_dask_client

print('modules imported')

info = "./scheduler_info.json"
client = create_dask_client(info)

print(client)
print(type(client))
#####################



nT = 10; nE = 2; nM = 1; nL = 1; nx = 1
param0 = np.random.random(2*(nE*nx + (nE+nM)*nL) + 3*nE)
xin = np.random.random(nT*nE).reshape(nT,nE)
yin = np.random.random(nT)

#class qrnn(emulator, mse, expectZ, CZladder2p1, encodeP2):
#    pass

QRNN = EMCZ2(nT,nE,nM,nL,nx)
print(QRNN.nE)

yout = QRNN.evaluate(param0,xin)
print(yout)

partial1 = QRNN.psr1(param0,xin,10)
print(partial1)


paramBL = np.concatenate(([0.],param0))
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
