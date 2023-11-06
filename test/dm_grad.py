# -*- coding: utf-8 -*-
"""
 Title: EMC model PSR gradient test
 Description: This script is to test that emc.dm.emc accurately computes the
 analytical gradient with respect to the 2-point finite difference method.

Created on Tue Oct  3 19:17:25 2023
@author: jdviqueira
"""

import numpy as np
from time import time

import sys
sys.path.append('../')
from qurecnets.emc.dm import emc

#np.random.seed(0)


nT = 20; nE = 2; nM = 3; nL = 2; nx = 1

x = 2*np.pi*np.random.rand(nT*nE).reshape(nT,nE)
theta = 2*np.pi*np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE)


network = emc(nT,nE,nM,nL,nx)



### NUMERICAL PARTIAL DERIVATIVES
print('NUMERICAL PARTIAL DERIVATIVES')
def tpfd(i,eps=1.e-8):
    # two-point finite difference derivative with respect to theta_i
    theta_ps = np.copy(theta); theta_ps[i] += eps
    theta_ms = np.copy(theta); theta_ms[i] -= eps
    network.evaluate(x,theta_ps, savegrad=False)
    evalu_ps = network.evaluate_Z()
    network.evaluate(x,theta_ms, savegrad=False)
    evalu_ms = network.evaluate_Z()
    deriv = (evalu_ps-evalu_ms)/(2*eps)
    return deriv

grad_num = []
t0 = time()
for i in range(len(theta)):
    grad_num.append(tpfd(i,eps=1.e-8))
grad_num = np.array(grad_num)
t1 = time()
print(grad_num)
print(t1-t0, ' s')



### ANALYTICAL PARTIAL DERIVATIVES
print('\nANALYTICAL PARTIAL DERIVATIVES')
t2 = time()
solu = network.evaluate(x,theta, savegrad=True)
network.grad_psr(theta)
grad_ana = network.grad_psr_Z()
t3 = time()
print(grad_ana)
print(t3-t2, ' s')




### COMPARISON
diff = np.round(grad_ana-grad_num,6)

print('\n Any error? ', diff.any())