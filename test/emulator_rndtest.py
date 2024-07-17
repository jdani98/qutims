# -*- coding: utf-8 -*-

import numpy as np

#from qurecnets.models import CZme3, CZladder2p1, encodeP3, encodeP2
#from qurecnets.loss_fs import mse
#from qurecnets.readout import expectZ
#from qurecnets.emc import emulator

from qurecnets.emc import EMCZ2

from time import time



client = None

#"""
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
#"""



nT = 3; nE = 1; nM = 1; nL = 1; nx = 1
np.random.seed(0)
param0 = np.random.random(2*(nE*nx + (nE+nM)*nL) + 3*nE)
xin = np.random.random(nT*nE).reshape(nT,nE)
yin = np.random.random(nT)


nshots = 2000
QRNN = EMCZ2(nT,nE,nM,nL,nx, shots=nshots, rseed = 60)
print(QRNN.nE)

print('TEST RANDOMNESS OF evaluate METHOD')
print('Evaluate twice to see no randomness with 0 shots')
yout = QRNN.evaluate(param0,xin,shots=0)
print(yout)

yout = QRNN.evaluate(param0,xin,shots=0)
print(yout)


print('Evaluate many times to check theoretical standard deviation')
youts = []

n_experiments = 1000
for i in range(n_experiments):
    youti = QRNN.evaluate(param0,xin) #shots=nshots
    youts.append(list(youti))

youts_mean = np.mean(youts, axis=0)
print(youts_mean)

youts_std = np.std(youts, axis=0)
print(youts_std)

youts_std_theo = np.sqrt((1-yout**2)/nshots)
print(youts_std_theo)


print('Evaluate twice to see randomness with same class seed')
yout = QRNN.evaluate(param0,xin,shots=100)
print(yout)

yout = QRNN.evaluate(param0,xin,shots=100)
print(yout)


print('Evaluate twice to see no randomness if new seeds are intercalated')
np.random.seed(4)
yout = QRNN.evaluate(param0,xin,shots=100)
print(yout)

np.random.seed(4)
yout = QRNN.evaluate(param0,xin,shots=100)
print(yout)



print('TEST RANDOMNESS OF grad METHODS')
print('Gradients for circuit evaluation')
# no expected randomness neither exploding gradients:
grad_fd = QRNN.grad_fd(param0,xin, client=client)
print(grad_fd)
grad_fd = QRNN.grad_fd(param0,xin, client=client)
print(grad_fd)

# expected randomness but no exploding gradients:
grad_psr = QRNN.grad_psr(param0, xin, client=client)
print(grad_psr)
grad_psr = QRNN.grad_psr(param0, xin, client=client)
print(grad_psr)


#"""
#################
print('Gradients for Loss function')
paramBL = np.concatenate(([0.3],param0))
# no expected randomness neither exploding gradients:
grad_fd = QRNN.grad_BL_fd(paramBL, xin,yin,3, client=client)
print(grad_fd)
grad_fd = QRNN.grad_BL_fd(paramBL, xin,yin,3, client=client)
print(grad_fd)

# expected randomness but no exploding gradients:
grad_psr = QRNN.grad_BL_psr(paramBL, xin,yin,3, client=client)
print(grad_psr)
grad_psr = QRNN.grad_BL_psr(paramBL, xin,yin,3, client=client)
print(grad_psr)
#"""

# Initial fixed seed is not warranted when using DASK with several workers. It is warranted when no DASK client is used.
# Thus, reproducibility may not be satisfied when using DASK.