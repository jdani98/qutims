# -*- coding: utf-8 -*-
"""
 Title: EMC model evaluation test
 Description: This script is to test that emc.dm.emc.evaluate works.
 
Created on Tue Oct  3 18:49:43 2023
@author: jdviqueira
"""

import sys
sys.path.append('..')

import numpy as np
from time import time
from qurecnets.emc.dm import emc

#np.random.seed(0)


nT = 20; nE = 2; nM = 2; nL = 2; nx = 1

x = 2*np.pi*np.random.rand(nT*nE).reshape(nT,nE)
theta = 2*np.pi*np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE)


network = emc(nT,nE,nM,nL,nx)

print(network)
print(network.nT)
print(network.nE)
print(network.nM)
print(network.nL)
print(network.nx)
print(network.NE)
print(network.NM)

t0 = time()
evalu = network.evaluate(x, theta, savegrad=False)
t1 = time()
evalu = network.evaluate(x, theta, savegrad=True)
t2 = time()
evaluZ = network.evaluate_Z()

print(evalu)
print(evaluZ)
print(t1-t0, ' s')
print(t2-t1, ' s')