# -*- coding: utf-8 -*-

import numpy as np

#from qurecnets.models import CZme3, CZladder2p1, encodeP3, encodeP2
#from qurecnets.loss_fs import mse
#from qurecnets.readout import expectZ
#from qurecnets.emc import emulator

from qurecnets.emc import EMCZ2

from time import time



nT = 4; nE = 2; nM = 1; nL = 1; nx = 1
param0 = np.random.random(2*(nE*nx + (nE+nM)*nL) + 3*nE)
xin = np.random.random(nT*nE).reshape(nT,nE)
yin = np.random.random(nT)

#class qrnn(emulator, mse, expectZ, CZladder2p1, encodeP2):
#    pass

QRNN = EMCZ2(nT,nE,nM,nL,nx)
print(QRNN.nE)

yout = QRNN.evaluate(param0,xin)
print(yout)




t0 = time()
hess_fd = QRNN.hess_fd(param0,xin)
t1 = time()
print(hess_fd)
print(t1-t0)

t0 = time()
hess_psr = QRNN.hess_psr(param0, xin, shots=0)
t1 = time()
print(hess_psr)
print(t1-t0)

print(hess_psr-hess_fd)