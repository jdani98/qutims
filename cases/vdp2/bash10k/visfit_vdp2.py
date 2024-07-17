# -*- coding: utf-8 -*-
"""
Created on

@author: jdviqueira
"""

import sys
sys.path.append('../../..')

import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import csv

from qurecnets.emc import EMCZ2


def rmse(a,b):
    return np.sqrt((1./len(a))*sum([(ai-bi)**2 for ai,bi in zip(a,b)]))

origin = input('Insert JOBID origin ')
job_id = input('Insert JOBID ')

fname_x = '../data_vdp_mu_1_3_del_5_16_ab_100_1000p.dat' # INPUT
fname_p = 'param_best_'+origin+'_'+job_id+'.dat' #PARAMETERS

data = np.loadtxt(fname_x)


nE = 2; nM = 3; nL = 5; nx = 3

nshots = 0


x_data = np.array([[data[i,1],data[i,2]] for i in range(len(data[:,1]))]) 


## PREPARING DATA #############################################################
# Data preprocessing
Npoints = len(x_data)
nT = 20 # Prediction window size
Nwout = 5  # Size of window to predict
nsamples = int(len(data[:,1])/nT)

times = data[:,0].reshape(nsamples,nT)
sequences = x_data.reshape(nsamples,nT,2)

tarseq = data[:,3].reshape(nsamples,nT)
#tarseq = data[:,2].reshape(nsamples,nT)
targets =   np.array([item[-Nwout:] for item in data[:,3].reshape(nsamples,nT)])
#targets =   np.array([item[-Nwout:] for item in data[:,2].reshape(nsamples,nT)])
tartimes =  np.array([item[-Nwout:] for item in times])


# Divide data into training + validation set and test set
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in training + validation set

random.seed(0) # !!! must be the same as in training
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) # samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # samples for training

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
filltest_times  = np.array([data[:,0][trval_npoints+Nwout*i:trval_npoints+Nwout*(i+1)] for i in range(ts_npoints//Nwout)])
filltest_sequences = np.array([x_data[trval_npoints-nT+Nwout+Nwout*i:trval_npoints-nT+Nwout+Nwout*i+nT] for i in range(ts_npoints//Nwout)])
filltest_targets = np.array([data[:,3][trval_npoints+Nwout*i:trval_npoints+Nwout*i+Nwout] for i in range(ts_npoints//Nwout)])
#filltest_targets = np.array([data[:,2][trval_npoints+Nwout*i:trval_npoints+Nwout*i+Nwout] for i in range(ts_npoints//Nwout)])
###############################################################################


params = np.loadtxt(fname_p)
Nparam = len(params)

qrnn = EMCZ2(nT,nE,nM,nL,nx, shots=nshots)

# TRAINING PREDICTIONS
train_outputs = []
for train_sample in train_sequences:
    evalu = qrnn.evaluate(params[1:], train_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    train_outputs += [ypredi[-Nwout:]]
train_outputs = np.array(train_outputs)
train_rmse = rmse(train_outputs.flatten(),train_targets.flatten())
print('Training RMSE: ', train_rmse)


# VALIDATION PREDICTIONS
val_outputs = []
for val_sample in val_sequences:
    evalu = qrnn.evaluate(params[1:], val_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    val_outputs += [ypredi[-Nwout:]]
val_outputs = np.array(val_outputs)
val_rmse = rmse(val_outputs.flatten(),val_targets.flatten())
print('Validation RMSE: ', val_rmse)


# TEST PREDICTIONS
test_outputs = []
for test_sample in test_sequences:
    evalu = qrnn.evaluate(params[1:], test_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    test_outputs += [ypredi[-Nwout:]]
test_outputs = np.array(test_outputs)
test_rmse = rmse(test_outputs.flatten(),test_targets.flatten())
print('Test RMSE: ', test_rmse)


# FILL TEST PREDICTIONS
filltest_outputs = []
for filltest_sample in filltest_sequences:
    evalu = qrnn.evaluate(params[1:], filltest_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    filltest_outputs += [ypredi[-Nwout:]]
filltest_outputs = np.array(filltest_outputs)
filltest_rmse = rmse(filltest_outputs.flatten(),filltest_targets.flatten())
print('Complete test RMSE: ', filltest_rmse)


command = "seff %s" %job_id[-5:]
result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
resultstr = result.stdout
info = {}
for line in resultstr.splitlines():
    try:
        key, value = line.split(": ")
        info[key] = value
    except:
        pass

with open('log_%s.dat' %job_id) as file:
    itcount = 0
    for i,line in enumerate(file.readlines()):
        if line[:3] == '==>':
            itcount += 1
        if i == 1:
            nshots = int(line.split(' ')[1])
        if i == 4:
            items = line.split(':')[1].split(',')
            LR = float(items[0].split('=')[1])
            shuffle = items[1].split('=')[1][:-1]


with open('optimisations_info_vdp2.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow([None,info['Job ID'], int(info['Nodes'])*int(info['Cores per node']), info['Memory Utilized'], info['CPU Utilized'], info['Job Wall-clock time'],
                     Npoints, nT,Nwout,nE,nM,nL,nx,Nparam,'Adam',itcount,None,None,'False',shuffle,LR,nshots,
                     train_rmse,val_rmse,test_rmse,filltest_rmse])



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

plt.savefig(fname_p.split('/')[-1][:-4]+'_'+str(nshots//1000)+'k.png')
