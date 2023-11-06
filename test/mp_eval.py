# -*- coding: utf-8 -*-
"""
 Title: EMC model evaluation test in MP module
 Description: This script is to test that emc.mp.evaluate works.
 
Created on Mon Nov  6 11:41:00 2023
@author: jdviqueira
"""

import sys
sys.path.append('..')

import numpy as np
from time import time
from qurecnets.emc.mp import evaluate

#np.random.seed(0)

nT = 20; nE = 2; nM = 2; nL = 2; nx = 1

x = 2*np.pi*np.random.rand(nT*nE).reshape(nT,nE)
theta = 2*np.pi*np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE)

param = {tuple(range(nT)): theta}

t0 = time()
evalu = evaluate((nT,nE,nM,nL,nx,param,x))
t1 = time()

print(evalu)
print(t1-t0, ' s')