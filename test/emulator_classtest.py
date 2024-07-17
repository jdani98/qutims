# -*- coding: utf-8 -*-

import numpy as np

from qurecnets.models import CZme3, CZladder2p1, encodeP3, encodeP2
from qurecnets.loss_fs import mse
from qurecnets.readout import expectZ
from qurecnets.emc import emulator

np.random.seed(1)

nT = 20; nE = 2; nM = 3; nL = 5; nx = 2
param0 = np.random.random(3*(nE*nx + (nE+nM)*nL) + 3*nE)
print(param0)
xin = np.random.random(nT*nE).reshape(nT,nE)
yin = np.random.random(nT)

class qrnn(emulator, encodeP3, CZme3, expectZ, mse):
    pass

QRNN = qrnn(nT,nE,nM,nL,nx,0,None)

# Check attributes
print(QRNN.nT)
print(QRNN.nE)
print(QRNN.nM)
print(QRNN.nL)
print(QRNN.nx)
print(QRNN.NE)
print(QRNN.NM)
print(QRNN.shots)

# Check inherited methods
encoding = QRNN.encode(xin[5],param0)
print(encoding)

ansatz = QRNN.evolve(param0)
print(ansatz)

invented_probs = np.random.random(nT*2**nE).reshape(nT,2**nE)
expectation = QRNN.readout(invented_probs)


# Check main methods
print(QRNN.check_params(param0)) # check number of parameters
#yout = QRNN.evaluate(param0,xin, shots=1024)
yout = QRNN.evaluate(param0,xin) # check if shots=None takes attribute from __init__
print(yout)

partial1 = QRNN.psr1(param0,xin,2) # analytical partial derivative
print(partial1)

paramBL = np.concatenate(([1.],param0))
loss_partial1 = QRNN.deriv_BL_fd(paramBL, xin, yin, 3, 5, eps=1.e-5) # numerical partial derivative of Loss_fn
print(loss_partial1)

loss_partial1 = QRNN.deriv_BL_psr(paramBL, xin, yin, 3, 5) # analytical partial derivative of Loss_fn
print(loss_partial1)

grad_fd = QRNN.grad_BL_fd(paramBL, xin,yin,3)
print(grad_fd)

grad_psr = QRNN.grad_BL_psr(paramBL, xin,yin,3)
print(grad_psr)