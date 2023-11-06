# -*- coding: utf-8 -*-
"""
 Title: Series visualisation
 Description: Script to generate plots of datasets and predictions.

Created on Thu Oct 19 13:33:00 2023
@author: jdviqueira
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import qurecnets.emc.dm as dm
import random


def rmse(a,b):
    return np.sqrt((1./len(a))*sum([(ai-bi)**2 for ai,bi in zip(a,b)]))


#fname_x = 'dataset_dectriang_a0b100_del12_1000p.dat' # !!! DATA INPUT FILENAME
#fname_p = 'par_dectriang_a0b100_del12_1000p_optnum-1.dat' # !!! PARAMETERS
#fname_x = 'dataset_vdp1_a0b100_del15_mu2_1000p.dat' # (b)
#fname_p = 'par_vdp1_a0b100_del15_mu2_1000p_optnum-8.dat' # (b)
fname_x = 'dataset_vdp2_a0b100_del05_del118_mu02_mu11_1000p.dat' # (c)
fname_p = 'par_vdp2_a0b100_del05_del118_mu02_mu11_1000p_optnum-8.dat' # (c)


data = np.loadtxt('data/'+fname_x)
#x_data = np.array([[data[i,1]] for i in range(len(data[:,1]))]) # input data
#x_data = np.array([[data[i,1],data[i,1]] for i in range(len(data[:,1]))]) #(b)
x_data = np.array([[data[i,1],data[i,2]] for i in range(len(data[:,1]))]) #(c)


## PREPARING DATA #############################################################
# Data preprocessing
nT = 20 # Prediction window size
N = 5  # Size of window to predict
nsamples = int(len(data[:,1])/nT)

times = data[:,0].reshape(nsamples,nT)
#sequences = x_data.reshape(nsamples,nT,1) # (a)
sequences = x_data.reshape(nsamples,nT,2) # (b) and (c)
#tarseq = data[:,2].reshape(nsamples,nT) # (a) and (b)
tarseq = data[:,3].reshape(nsamples,nT) # (c)
#targets = np.array([item[-N:] for item in data[:,2].reshape(nsamples,nT)]) #
targets = np.array([item[-N:] for item in data[:,3].reshape(nsamples,nT)])# (c)
tartimes =  np.array([item[-N:] for item in times])


# Divide data into training + validation set and test set
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in training 
# + validation set

random.seed(0)
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) #
# samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # 
#samples for training

# TRAINING DATA
train_times     = np.array([times[i] for i in tr_samples])
train_tartimes  = np.array([tartimes[i] for i in tr_samples])
train_tarseq    = np.array([tarseq[i] for i in tr_samples])
train_sequences = np.array([sequences[i] for i in tr_samples])
train_targets   = np.array([targets[i] for i in tr_samples])

# VALIDATION DATA
val_times       = np.array([times[i] for i in val_samples])
val_tartimes    = np.array([tartimes[i] for i in val_samples])
val_tarseq      = np.array([tarseq[i] for i in val_samples])
val_sequences   = np.array([sequences[i] for i in val_samples])
val_targets     = np.array([targets[i] for i in val_samples])

# TEST DATA
test_times      = times[trval_nsamples:]
test_tartimes   = tartimes[trval_nsamples:]
test_tarseq     = tarseq[trval_nsamples:]
test_sequences  = sequences[trval_nsamples:]
test_targets    = targets[trval_nsamples:]

# TEST DATA FILLING ALL POINTS
trval_npoints = trval_nsamples * nT
npoints = nT * nsamples
ts_npoints = npoints - trval_npoints
filltest_times  = np.array([data[:,0][trval_npoints+N*i:trval_npoints+N*(i+1)] 
                            for i in range(ts_npoints//N)])
filltest_sequences = np.array([x_data[trval_npoints-nT+N+N*i:
                    trval_npoints-nT+N+N*i+nT] for i in range(ts_npoints//N)])
#filltest_targets = np.array([data[:,2][trval_npoints+N*i:trval_npoints+N*i+N] 
#                             for i in range(ts_npoints//N)])
filltest_targets = np.array([data[:,3][trval_npoints+N*i:trval_npoints+N*i+N] 
                             for i in range(ts_npoints//N)]) # case (c)
###############################################################################


params = np.loadtxt('data/'+fname_p)[:,1]

#nE = 1; nM = 2; nL = 2; nx = 3 # (a)
#nE = 2; nM = 3; nL = 2; nx = 1 # (b)
nE = 2; nM = 3; nL = 5; nx = 3 # (c)


model = dm.emc(nT,nE,nM,nL,nx)


# TRAINING PREDICTIONS
train_outputs = []
for train_sample in train_sequences:
    model.evaluate(train_sample,params[1:])
    evalu = model.evaluate_Z()
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    train_outputs += [ypredi[-N:]]
train_outputs = np.array(train_outputs)#print(x_in.shape)
train_rmse = rmse(train_outputs.flatten(),train_targets.flatten())
print('Training RMSE: ', train_rmse)


# VALIDATION PREDICTIONS
val_outputs = []
for val_sample in val_sequences:
    model.evaluate(val_sample,params[1:])
    evalu = model.evaluate_Z()
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    val_outputs += [ypredi[-N:]]
val_outputs = np.array(val_outputs)
val_rmse = rmse(val_outputs.flatten(),val_targets.flatten())
print('Validation RMSE: ', val_rmse)


# TEST PREDICTIONS
test_outputs = []
for test_sample in test_sequences:
    model.evaluate(test_sample,params[1:])
    evalu = model.evaluate_Z()
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    test_outputs += [ypredi[-N:]]
test_outputs = np.array(test_outputs)
test_rmse = rmse(test_outputs.flatten(),test_targets.flatten())
print('Test RMSE: ', test_rmse)


# FILL TEST PREDICTIONS
filltest_outputs = []
for filltest_sample in filltest_sequences:
    model.evaluate(filltest_sample,params[1:])
    evalu = model.evaluate_Z()
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    filltest_outputs += [ypredi[-N:]]
filltest_outputs = np.array(filltest_outputs)
filltest_rmse = rmse(filltest_outputs.flatten(),filltest_targets.flatten())
print('Complete test RMSE: ', filltest_rmse)



plt.close('all')
plt.figure(1,figsize=(20,5))
for x,y in zip(train_times,train_tarseq):
    plt.plot(x,y,'-', color='tab:orange')
for x,y in zip(val_times,val_tarseq):
    plt.plot(x,y,'-', color='tab:blue')
for x,y in zip(test_times,test_tarseq):
    plt.plot(x,y,'-', color='tab:green')

for x,y in zip(train_tartimes,train_outputs):
    plt.plot(x, y, '.', color='red')
for x,y in zip(val_tartimes,val_outputs):
    plt.plot(x, y, '.', color='blue')
for x,y in zip(test_tartimes,test_outputs):
    plt.plot(x, y, '.', color='green')
for x,y in zip(filltest_times,filltest_outputs):
    plt.plot(x, y, '*', color='pink')


plt.savefig('data/'+fname_p[:-4]+'.pdf')
print('Figure saved at data/%s.pdf' %fname_p[:-4])
