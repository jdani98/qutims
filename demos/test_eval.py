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
from random import random
from time import time
from qurecnets.emc.dm import emc

nT = 20; nE = 2; nM = 2; nL = 2; nx = 1

x = np.array([[2*np.pi*random() for i in range(nE)] for j in range(nT)])
theta = np.array([2*np.pi*random() for i in range(3*(nx*nE + (nE+nM)*nL + 
                                                     nE))])


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

print(evalu)
print(t1-t0, ' s')
print(t2-t1, ' s')