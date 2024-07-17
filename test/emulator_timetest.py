# -*- coding: utf-8 -*-

import timeit


setup_code = """
import numpy as np

from qurecnets.models import CZme3, CZladder2p1, encodeP3, encodeP2
from qurecnets.loss_fs import mse
from qurecnets.readout import expectZ
from qurecnets.emc import emulator

np.random.seed(1)

nT = 100; nE = 8; nM = 2; nL = 1; nx = 1
param0 = np.random.random(3*(nE*nx + (nE+nM)*nL) + 3*nE)
#print(param0)
xin = np.random.random(nT*nE).reshape(nT,nE)
yin = np.random.random(nT)

class qrnn(emulator, encodeP3, CZme3, expectZ, mse):
    pass
    
QRNN = qrnn(nT,nE,nM,nL,nx,0)
"""

test_code = """
QRNN.evaluate(param0,xin)
"""

repetitions = 10
evaluate_time = timeit.timeit(setup=setup_code, stmt=test_code, number=repetitions)/repetitions
print(evaluate_time)