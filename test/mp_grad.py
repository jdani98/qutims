# -*- coding: utf-8 -*-
"""
 Title: EMC model PSR gradient test in MP module
 Description: This script is to test that emc.mp accurately computes the
 analytical derivative with respect to the 2-point finite difference method.
 Then, we test its performance inside a multiprocessing task.

Created on Mon Nov  6 12:01:00 2023
@author: jdviqueira
"""

import numpy as np
from time import time

import sys
sys.path.append('../')
from qurecnets.emc.mp import evaluate, derivative1
from multiprocessing import Pool, cpu_count

#np.random.seed(0)


nT = 20; nE = 2; nM = 3; nL = 2; nx = 1

x = 2*np.pi*np.random.rand(nT*nE).reshape(nT,nE)
theta = 2*np.pi*np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE)


### NUMERICAL PARTIAL DERIVATIVES
print('NUMERICAL PARTIAL DERIVATIVES')
def tpfd(i,eps=1.e-8):
    # two-point finite difference derivative with respect to theta_i
    theta_ps = np.copy(theta); theta_ps[i] += eps
    theta_ms = np.copy(theta); theta_ms[i] -= eps
    param_ps = {tuple(range(nT)): theta_ps}
    evalu_ps = evaluate((nT,nE,nM,nL,nx,param_ps,x))
    param_ms = {tuple(range(nT)): theta_ms}
    evalu_ms = evaluate((nT,nE,nM,nL,nx,param_ms,x))
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



### ANALYTICAL PARTIAL DERIVATIVES IN SEQUENTIAL
print('\nANALYTICAL PARTIAL DERIVATIVES IN SEQUENTIAL')
grad_seq = []
t2 = time()
for i in range(len(theta)):
    grad_seq.append(derivative1((nT,nE,nM,nL,nx,theta,x, i)))
grad_seq = np.array(grad_seq)
t3 = time()
print(grad_seq)
print(t3-t2, ' s')



### ANALYTICAL PARTIAL DERIVATIVES IN PARALLEL
print('\nANALYTICAL PARTIAL DERIVATIVES IN PARALLEL')
grad_par = []
t4 = time()
if __name__ == '__main__':
    with Pool(len(theta)) as p:
        grad_par.append(p.map(derivative1, [(nT,nE,nM,nL,nx,theta,x,i) 
                                          for i in range(len(theta))]))
grad_par = np.array(grad_par)
t5 = time()
print(grad_par)
print(t5-t4, ' s in ', cpu_count(), ' virtual cores')



### COMPARISONS
diff1 = np.round(grad_seq-grad_num,6)
diff2 = np.round(grad_par-grad_num,6)

print('\n Any error? ', diff1.any(), diff2.any())