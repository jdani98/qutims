# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira
"""

#### LOAD MODULES #######################################################################
import numpy as np
import random

import sys
sys.path.append('../../..') # append sys path to main directory

from qurecnets.models import CZladder2p1, encodeP2
from qurecnets.loss_fs import mse
from qurecnets.readout import expectZ
from qurecnets.emc import emulator


#import pennylane as qml
from pennylane import numpy as npp
from pennylane.optimize import AdamOptimizer #, GradientDescentOptimizer, AdagradOptimizer

from time import time, sleep
import csv


job_id_init = sys.argv[1]
job_id = sys.argv[2]


client = None
#"""
####### DASK ###############################
from dask import delayed
from distributed import Client, wait
from dask_cluster import create_dask_client

print('modules imported')

info = "./scheduler_info_"+job_id+".json"
client = create_dask_client(info)

print(client)
############################################
#"""



#### DATA PREPARATION  ##################################################################
fname = '../data_dectriang_ab_100_1000p.dat' # INPUT
data = np.loadtxt(fname)

# QRNN size and layers
nE = 1; nM = 2; nL = 3; nx = 3

x_data = np.array([[data[i,1]] for i in range(len(data[:,1]))]) # 2D-array!
y_data = np.array([item for item in data[:len(x_data),2]])    # 1D-array!

nT = 20 # size of predicting window
Nwout = 5  # size of window to predict
nsamples = int(len(data[:,1])/nT)

sequences = x_data.reshape(nsamples,nT,1)
targets =   np.array([item[-Nwout:] for item in y_data.reshape(nsamples,nT)])


# Divide data into training and validation set, and test set
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in the training + validation set

random.seed(2)
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) # samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # samples for training

# TRAINING DATA
train_sequences = np.array([sequences[i] for i in tr_samples])
train_targets   = np.array([targets[i] for i in tr_samples])

# VALIDATION DATA
val_sequences   = np.array([sequences[i] for i in val_samples])
val_targets     = np.array([targets[i] for i in val_samples])
#####################################################################

fname_init = '../bash0/param0_'+str(job_id_init)+'.dat'
param0 = np.loadtxt(fname_init)

#rnd_seed = 13192#int(job_id) # !!!
#np.random.seed(rnd_seed)
#param0 = np.concatenate(([0.],2.*np.pi*np.random.random(2*(nE*nx + nL*(nE+nM))+1*nE)))
#np.random.seed(None)

np.savetxt('param0_'+str(job_id_init)+'_'+str(job_id)+'.dat', param0)

#print('seed for init ', rnd_seed)
print('len param0 ', len(param0))
print('nT=%i; nE=%i; nM=%i; nL=%i; nx=%i' %(nT,nE,nM,nL,nx))



##### DEFINE CLASS and COST FUNCTIONS ########
class train_qrnn(emulator, encodeP2, CZladder2p1, expectZ, mse):
    """Class for QRNN training. Inherits classes for emulation. Same constructor.
    Adds scores for training and validation datasets: MSE of all involved points (last Nwout points of all train/val windows).
    """

    def loss_train(self, params,x,y,shots=None):
        tr_losses = []
        for i in range(Ntr):
            evalu = self.evaluate(params[1:],x[i], shots=shots)
            yb = np.array([params[0]]*len(evalu)) + evalu
            tr_losses += [(1./Nwout)*sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:], y[i][-Nwout:])])]
        Etr = np.mean(tr_losses)
        return Etr

    def validate(self, params,x,y, shots=None):
        val_losses = []
        for i in range(Nval):
            evalu = self.evaluate(params[1:],x[i], shots=shots)
            yb = np.array([params[0]]*len(evalu)) + evalu
            val_losses += [(1./Nwout)*sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:], y[i][-Nwout:])])]
        Eval = np.mean(val_losses)
        return Eval
####################################



#### OPTIMIZATION  ############################################################
logfile = 'log_'+job_id+'.dat'

nshots = 10000
LR = 0.001 # Learning Rate
epochs = 2000
shuffle = True
opt = AdamOptimizer(stepsize=LR)

with open(logfile, 'a') as f:
    f.write('seed for init %s' %job_id)
    f.write('\n')
    f.write('nshots %i' %nshots)
    f.write('\n')
    f.write('len param0 %i' %len(param0))
    f.write('\n')
    f.write('nT=%i; nE=%i; nM=%i; nL=%i; nx=%i' %(nT,nE,nM,nL,nx))
    f.write('\n')
    f.write('adam config: LR=%f, shuffle=%s' %(LR,str(shuffle)))
    f.write('\n')


if shuffle:
    # for reproducibility, the shuffle must be the same as in the origin optimisation
    shuffle_repeat = np.loadtxt('../bash0/.shuffle_indices_'+str(job_id_init)+'.csv', dtype=np.int32, delimiter=',')


param0 = npp.array(param0, requires_grad=True)
params = param0

best_Eval = 50. # high value
Ntr = len(train_sequences); Nval = len(val_sequences)
print('Begin optimization')

checkpoints = [499,999,1499,1999]

### CREATE TRAINING QRNN OBJECT
trqrnn = train_qrnn(nT,nE,nM,nL,nx, shots=nshots)

for it in range(epochs):
    
    if shuffle:
        #indices = np.random.permutation(Ntr)
        indices = shuffle_repeat[it]
        
    else:
        indices = list(range(Ntr))

    X_shuffled = train_sequences[indices]
    y_shuffled = train_targets[indices]

    #t0 = time()
    for xsample, ysample in zip(X_shuffled, y_shuffled):
        xsample = npp.array(xsample, requires_grad=False)
        ysample = npp.array(ysample, requires_grad=False)

        gradient = trqrnn.grad_BL_psr(params, xsample, ysample, Nwout=Nwout, client=client)
        params = npp.array(opt.apply_grad(gradient, params))
    #t1 = time()
    #print(t1-t0)


    Etr = trqrnn.loss_train(params, train_sequences, train_targets)
    Eval = trqrnn.validate(params, val_sequences, val_targets)


    with open(logfile, 'a') as f:
        f.write('==> Loss (RMSE)    %4i  %10.6f %10.6f' %(it, np.sqrt(Etr),np.sqrt(Eval)))
        f.write('\n')
    #print('==> Loss (RMSE)    %4i  %10.6f %10.6f' %(it, np.sqrt(Etr),np.sqrt(Eval)))

    if Eval < best_Eval:
        np.savetxt('param_best_'+str(job_id_init)+'_'+str(job_id)+'.dat', params)
        best_Eval = Eval
        best_gradient = trqrnn.grad_BL_psr(params, xsample, ysample, Nwout, client=client)
        np.savetxt('.grad_best_'+str(job_id_init)+'_'+str(job_id)+'.dat', best_gradient)

    
    if it in checkpoints:
        np.savetxt('.param_'+str(it)+'_'+str(job_id_init)+'_'+str(job_id)+'.dat', params)
        gradient = trqrnn.grad_BL_psr(params, xsample, ysample, Nwout, client=client)
        np.savetxt('.grad_'+str(it)+'_'+str(job_id_init)+'_'+str(job_id)+'.dat', best_gradient)

client.shutdown()
#print('Loss (RMSE): ' , np.sqrt(Etr), np.sqrt(Eval))
#print(params)