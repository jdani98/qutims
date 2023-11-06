# -*- coding: utf-8 -*-
"""
 Title: EMC model PSR Hessian test
 Description: This script is to test that emc.dm.emc accurately computes the
 analytical Hessian with respect to the 2-point finite difference method.

Created on Wed Oct  4 13:09:33 2023
@author: jdviqueira
"""

import numpy as np
from random import random
from time import time

import sys
sys.path.append('../')
from qurecnets.emc.dm import emc


nT = 10; nE = 2; nM = 3; nL = 2; nx = 1

x = np.array([[2*np.pi*random() for i in range(nE)] for j in range(nT)])
theta = np.array([2*np.pi*random() for i in range(3*(nx*nE + (nE+nM)*nL + 
                                                     nE))])


network = emc(nT,nE,nM,nL,nx)



### NUMERICAL 2nd-ORDER PARTIAL DERIVATIVES
print('NUMERICAL 2nd-ORDER PARTIAL DERIVATIVES')
def tpfd2(i,j,eps=1.e-4):
    # two-point finite difference 2nd-order derivative w.r.t. theta_i, theta_j
    theta_ps_ps = np.copy(theta); theta_ps_ps[i] += eps; theta_ps_ps[j] += eps
    theta_ps_ms = np.copy(theta); theta_ps_ms[i] += eps; theta_ps_ms[j] -= eps
    theta_ms_ps = np.copy(theta); theta_ms_ps[i] -= eps; theta_ms_ps[j] += eps
    theta_ms_ms = np.copy(theta); theta_ms_ms[i] -= eps; theta_ms_ms[j] -= eps
    network.evaluate(x,theta_ps_ps)
    solu_ps_ps = network.evaluate_Z()
    network.evaluate(x,theta_ps_ms)
    solu_ps_ms = network.evaluate_Z()
    network.evaluate(x,theta_ms_ps)
    solu_ms_ps = network.evaluate_Z()
    network.evaluate(x,theta_ms_ms)
    solu_ms_ms = network.evaluate_Z()
    deriv2 = (1./(4*eps**2))*(solu_ps_ps-solu_ps_ms-solu_ms_ps+solu_ms_ms)
    return deriv2

hess_num = np.zeros((len(theta),len(theta),nT))
t0 = time()
for i in range(len(theta)):
    for j in range(len(theta)):
        hess_num[i,j] = tpfd2(i,j, eps=1.e-4)
t1 = time()
print(hess_num)
print(t1-t0, ' s')



### ANALYTICAL 2nd-ORDER PARTIAL DERIVATIVES
print('\nANALYTICAL 2nd-ORDER PARTIAL DERIVATIVES')
t2 = time()
solu = network.evaluate(x,theta, savegrad=True)
network.hess_psr(theta)
hess_ana = network.hess_psr_Z()
t3 = time()
print(hess_ana)
print(t3-t2, ' s')



### COMPARISON
diff = np.round(hess_ana-hess_num,6)

print('\n Any error? ', diff.any())