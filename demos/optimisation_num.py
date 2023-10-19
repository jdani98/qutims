# -*- coding: utf-8 -*-
"""
 Title: EMC model Training
 Description: This script uses the EMC emulator and the optimisation class
 to perform training and validation of a dataset.
 It uses numerical gradients.

Created on Wed Oct 11 16:08:08 2023
@author: jdviqueira
"""

import numpy as np
import random
from scipy.optimize import minimize

import sys
sys.path.append('..')
import qurecnets.emc.dm as dm

from time import time

# comment or uncomment lines as necessary

### LOAD DATA #################################################################
fname = 'dataset_dectriang_a0b100_del12_1000p.dat'    # !!! filename - case (a)
#fname = 'dataset_vdp1_a0b100_del15_mu2_1000p.dat'    # !!! filename - case (b)
#fname = 'dataset_vdp2_a0b100_del05_del118_mu02_mu11_1000p.dat' # !!! filename 
#- case (c)


nE = 1; nM = 2; nL = 2; nx = 3              # !!! network hyperparameters
print('NETWORK HYPERPARAMETERS')
print('nE=%i  nM=%i  nL=%i  nx=%i' %(nE,nM,nL,nx))  
###############################################################################
data = np.loadtxt('data/'+fname)

### DATA PREPARATION  #########################################################
print('\nDATA PREPARATION')
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
N = 5  # size of window to predict
nsamples = int(len(data[:,1])/nT)

sequences = x_data.reshape(nsamples,nT,1)  # for case (a)
#sequences = x_data.reshape(nsamples,nT,1) # for cases (b) and (c)
targets =   np.array([item[-N:] for item in y_data.reshape(nsamples,nT)])


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
print('Validation samples: ', val_samples)
print('Training samples: ', tr_samples)

# TRAINING DATA
train_sequences = np.array([sequences[i] for i in tr_samples])
train_targets   = np.array([targets[i] for i in tr_samples])

# VALIDATION DATA
val_sequences   = np.array([sequences[i] for i in val_samples])
val_targets     = np.array([targets[i] for i in val_samples])
###############################################################################



t0 = time()
### OPTIMISATION  #############################################################
print('\nOPTIMISATION')

# Initial parameters:
param0 = np.random.rand(3*nx*nE+3*(nE+nM)*nL+3*nE+1); param0[0] = 0.
#param0 = np.loadtxt() # case (a)
#param0 = np.loadtxt() # case (b)
#param0 = np.loadtxt() # case (c)
print('Initial parameters:\n', param0)

# Create the optimiser object and then, search the solution
# !!! This process may last several minutes or hours
optimiser = dm.emc_optimization(nT, nE, nM, nL, nx)
optimiser.load_x(train_sequences,val_sequences)
optimiser.load_y(train_targets,val_targets)
solution = optimiser.optimize(param0, method='L-BFGS-B', 
                    options={'maxiter':1000,'maxfun':500000, 'gtol':1e-4})

print(solution)