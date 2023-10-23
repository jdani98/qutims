# -*- coding: utf-8 -*-
"""
 Title: MultiProcessing module test
 Description: This script is to test the emc.mp module.

Created on Mon Oct 23 11:04:50 2023
@author: jdviqueira
"""

import numpy as np
import random
from time import time

import sys
sys.path.append('../')
from qurecnets.emc.dm import emc
from qurecnets.emc.mp import evaluate, derivative1
from scipy.optimize import minimize

#from multiprocessing import Pool


#nT = 20; nE = 2; nM = 3; nL = 2; nx = 1

### LOAD DATA #################################################################
fname = 'dataset_dectriang_a0b100_del12_1000p.dat'    # !!! filename - case (a)
#fname = 'dataset_vdp1_a0b100_del15_mu2_1000p.dat'    # !!! filename - case (b)
#fname = 'dataset_vdp2_a0b100_del05_del118_mu02_mu11_1000p.dat' # !!! filename 
#- case (c)


nE = 1; nM = 2; nL = 2; nx = 3              # !!! network hyperparameters
###############################################################################
data = np.loadtxt('data/'+fname)

### DATA PREPARATION  #########################################################
# Case (a):
x_data = np.array([[data[i,1]] for i in range(len(data[:,1]))]) # 2D-array
y_data = np.array([item for item in data[:len(x_data),2]])    # 1D-array

# Case (b):
#x_data = np.array([[data[i,1],data[i,1]] for i in range(len(data[:,1]))]) # 2D
# Note. Here we re-upload the same series in 2 input qubits.
#y_data = np.array([item for item in data[:len(x_data),2]])    # 1D-array

# Case (c):
#x_data = np.array([[data[i,1],data[i,2]] for i in range(len(data[:,1]))]) # 2D
#y_data = np.array([item for item in data[:len(x_data),3]])    # 1D-array


nT = 20 # size of predicting window
Nwout = 5  # size of window to predict
nsamples = int(len(data[:,1])/nT)

sequences = x_data.reshape(nsamples,nT,1)  # for case (a)
#sequences = x_data.reshape(nsamples,nT,1) # for cases (b) and (c)
targets =   np.array([item[-Nwout:] for item in y_data.reshape(nsamples,nT)])


# Divide data into training and validation set, and test set
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in the training
# + validation set

random.seed(0)
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) 
# samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] 
# samples for training

# TRAINING DATA
train_sequences = np.array([sequences[i] for i in tr_samples])
train_targets   = np.array([targets[i] for i in tr_samples])

# VALIDATION DATA
val_sequences   = np.array([sequences[i] for i in val_samples])
val_targets     = np.array([targets[i] for i in val_samples])
###############################################################################


param0 = np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE+1); param0[0] = 0.

def cost_function(params):
    L = 0
    npoints = 0

    ttheta = {tuple(range(nT)): params[1:]}

    for isample in range(len(train_sequences)):
        evalu = evaluate((nT,nE,nM,nL,nx,ttheta,train_sequences[isample]))
        yb = np.array([params[0]]*len(evalu)) + evalu
        L += sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:],train_targets[isample])])
        npoints += Nwout

    L = np.sqrt(L/npoints)

    return L


def cost_val(params):
    L = 0
    npoints = 0

    ttheta = {tuple(range(nT)): params[1:]}

    for isample in range(len(val_sequences)):
        evalu = evaluate((nT,nE,nM,nL,nx,ttheta,val_sequences[isample]))
        yb = np.array([params[0]]*len(evalu)) + evalu
        L += sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:],val_targets[isample])])
        npoints += Nwout

    L = np.sqrt(L/npoints)

    return L



def grad(params):
    grad = np.zeros(len(params))
    ttheta = {tuple(range(nT)): params[1:]}
    npoints = 0

    for isample in range(len(train_sequences)):
        gradis = []

        evalu = evaluate((nT,nE,nM,nL,nx,ttheta,train_sequences[isample]))
        yb = np.array([params[0]]*len(evalu)) + evalu
        
        from multiprocessing import Pool

        if __name__ == '__main__':
            with Pool(len(params)-1) as p:
                gradis.append(p.map(derivative1, [(nT,nE,nM,nL,nx,params[1:],train_sequences[isample],i) for i in range(len(params)-1)]))
        grad1 = [sum([(ybi-yi)*partiali for (ybi,yi,partiali) in zip(yb[-Nwout:],train_targets[isample], partial[-Nwout:])]) for partial in gradis[0]]
        grad1_0 = sum([(ybi-yi) for (ybi,yi) in zip(yb[-Nwout:],train_targets[isample])])
        #print(grad1_0)
        #print(grad1)
        grad1 = np.array([grad1_0] + grad1)
        #print(grad1)
        grad += grad1
        #print(isample)
        npoints += Nwout
    grad = np.array(grad)
    L = cost_function(params)
    grad = (1./L)*(1./npoints)*grad
    return grad


def my_callback(params):
    global iteration
    iteration += 1
    train_loss = cost_function(params)
    val_loss = cost_val(params)
    print('Iteration %6i, Training loss: %8.6f, Validation loss: %8.6f' %(iteration,train_loss,val_loss))



t0 = time()
### OPTIMISATION  #############################################################

# Initial parameters:
param0 = np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE+1); param0[0] = 0.
#param0 = np.loadtxt() # case (a)
#param0 = np.loadtxt() # case (b)
#param0 = np.loadtxt() # case (c)



iteration = 0
sol = minimize(cost_function,param0,jac=grad, method='L-BFGS-B', options={'maxiter':1,'maxfun':500000, 'gtol':1e-3}, callback=my_callback)
t1 = time()




print('NETWORK HYPERPARAMETERS')
print('nE=%i  nM=%i  nL=%i  nx=%i' %(nE,nM,nL,nx))
print('\nDATA PREPARATION')
print('Validation samples: ', val_samples)
print('Training samples: ', tr_samples)
print('\nOPTIMISATION')
print('Initial parameters:\n', param0)

print(sol)

print('Time(s): ', t1-t0)
