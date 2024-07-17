# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Plots of results for datasets (a), (b) and (c) arXiv:2310.20671.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from qurecnets.emc import EMCZ2
import random

plt.rcParams.update({'font.size': 11})


def generate_subplot(axis,titulo,seed):
    #data, x_data, nT, nsamples, times, sequences, tarseq, targets, tartimes = dataTuple
    #nE, nM, nL, nx = config
    global data, x_data, nT, nsamples, times, sequences, tarseq, targets, tartimes
    global nE, nM, nL, nx
    global tarcolumn

    # Divide data into training + validation set and test set
    TRVAL = 80
    trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
    VAL = 20
    val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in training + validation set

    random.seed(seed)
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
    filltest_times  = np.array([data[:,0][trval_npoints+N*i:trval_npoints+N*(i+1)] for i in range(ts_npoints//N)])
    filltest_sequences = np.array([x_data[trval_npoints-nT+N+N*i:trval_npoints-nT+N+N*i+nT] for i in range(ts_npoints//N)])
    filltest_targets = np.array([data[:,tarcolumn][trval_npoints+N*i:trval_npoints+N*i+N] for i in range(ts_npoints//N)])
    ###############################################################################


    params = np.loadtxt(fname_p)

    qrnn = EMCZ2(nT,nE,nM,nL,nx)


    # TRAINING PREDICTIONS
    train_outputs = []
    #print(x_in.shape)
    for train_sample in train_sequences:
        evalu = qrnn.evaluate(params[1:],train_sample)
        ypredi = np.array([params[0]]*len(evalu)) + evalu
        train_outputs += [ypredi[-N:]]
    train_outputs = np.array(train_outputs)


    # VALIDATION PREDICTIONS
    val_outputs = []
    for val_sample in val_sequences:
        evalu = qrnn.evaluate(params[1:],val_sample)
        ypredi = np.array([params[0]]*len(evalu)) + evalu
        val_outputs += [ypredi[-N:]]
    val_outputs = np.array(val_outputs)


    # TEST PREDICTIONS
    test_outputs = []
    for test_sample in test_sequences:
        evalu = qrnn.evaluate(params[1:],test_sample)
        ypredi = np.array([params[0]]*len(evalu)) + evalu
        test_outputs += [ypredi[-N:]]
    test_outputs = np.array(test_outputs)


    # FILL TEST PREDICTIONS
    filltest_outputs = []
    for filltest_sample in filltest_sequences:
        evalu = qrnn.evaluate(params[1:],filltest_sample)
        ypredi = np.array([params[0]]*len(evalu)) + evalu
        filltest_outputs += [ypredi[-N:]]
    filltest_outputs = np.array(filltest_outputs)

    
    for x,y in zip(train_times,train_sequences):
        axis.plot(x,y, color='tab:orange', linestyle='dashed', linewidth=0.5, alpha=0.75,  label=r'$x_{(t)}$ (tra)')
    for x,y in zip(val_times,val_sequences):
        axis.plot(x,y, color='tab:blue', linestyle='dashed',  linewidth=0.5, alpha=0.75,  label=r'$x_{(t)}$ (val)')
    for x,y in zip(test_times,test_sequences):
        axis.plot(x,y, color='tab:green', linestyle='dashed', linewidth=0.5, alpha=0.75,   label=r'$x_{(t)}$ (tes)')


    for x,y in zip(train_times,train_tarseq):
        axis.plot(x,y,'-', color='tab:orange', label=r'$y_{(t)}$ (tra)')
    for x,y in zip(val_times,val_tarseq):
        axis.plot(x,y,'-', color='tab:blue', label=r'$y_{(t)}$ (val)')
    for x,y in zip(test_times,test_tarseq):
        axis.plot(x,y,'-', color='tab:green', label=r'$y_{(t)}$ (tes)')

    for x,y in zip(train_tartimes,train_outputs):
        axis.plot(x, y, '.', color='red', markersize=2, label=r'$\overline{y}_{(t)}$ (tra)')
    for x,y in zip(val_tartimes,val_outputs):
        axis.plot(x, y, '.', color='blue', markersize=2, label=r'$\overline{y}_{(t)}$ (val)')
    for x,y in zip(filltest_times,filltest_outputs):
        axis.plot(x, y, '.', color='goldenrod', markersize=1, label=r'$\overline{y}_{(t)}$ (fte)')
    for x,y in zip(test_tartimes,test_outputs):
        axis.plot(x, y, '.', color='darkgoldenrod', markersize=2, label=r'$\overline{y}_{(t)}$ (tes)')

    axis.set_xticks(np.arange(0,100,10))
    axis.set_xlim(0,100)
    axis.set_yticks(np.arange(-0.75,0.75+0.1,0.25))
    axis.set_ylim(-0.75,0.75)
    axis.grid(color='gainsboro')
    #axis.tick_params(labelsize=14)
    #axis.set_xticklabels(['' for i in np.arange(0,51,5)])
    axis.set_xlabel('$t$', loc='right')
    axis.set_title(titulo)



plt.close('all')
fig, ax = plt.subplots(3, figsize=(2*5.8,2*3))

### FIRST PLOT ################################################################
fname_x = 'dectriang/data_dectriang_ab_100_1000p.dat' # INPUT
fname_p = 'dectriang/bash0/param_best_13192.dat' #PARAMETERS

data = np.loadtxt(fname_x)
x_data = np.array([[data[i,1]] for i in range(len(data[:,1]))])


## PREPARING DATA
# Data preprocessing
nT = 20 # Prediction window size
N = 5  # Size of window to predict
nsamples = int(len(data[:,1])/nT)

times = data[:,0].reshape(nsamples,nT)
sequences = x_data.reshape(nsamples,nT,1)
tarseq = data[:,2].reshape(nsamples,nT)
targets =   np.array([item[-N:] for item in data[:,2].reshape(nsamples,nT)])
tartimes =  np.array([item[-N:] for item in times])

tarcolumn = 2

nE = 1; nM = 2; nL = 3; nx = 3
generate_subplot(ax[0], '(a)', 2)



### SECOND PLOT ################################################################
fname_x = 'vdp1/data_vdp_mu_2_del_15_ab_100_1000p.dat' # INPUT
fname_p = 'vdp1/bash0/param_best_13227.dat' #PARAMETERS

data = np.loadtxt(fname_x)
x_data = np.array([[data[i,1],data[i,1]] for i in range(len(data[:,1]))])


## PREPARING DATA
# Data preprocessing
nT = 20 # Prediction window size
N = 5  # Size of window to predict
nsamples = int(len(data[:,1])/nT)

times = data[:,0].reshape(nsamples,nT)
sequences = x_data.reshape(nsamples,nT,2)
tarseq = data[:,2].reshape(nsamples,nT)
targets =   np.array([item[-N:] for item in data[:,2].reshape(nsamples,nT)])
tartimes =  np.array([item[-N:] for item in times])

tarcolumn = 2

nE = 2; nM = 2; nL = 4; nx = 1
generate_subplot(ax[1], '(b)', 1)



### THRID PLOT ################################################################
fname_x = 'vdp2/data_vdp_mu_1_3_del_5_16_ab_100_1000p.dat' # INPUT
fname_p = 'vdp2/bash0/param_best_13244.dat' #PARAMETERS

data = np.loadtxt(fname_x)
x_data = np.array([[data[i,1],data[i,2]] for i in range(len(data[:,1]))])


## PREPARING DATA
# Data preprocessing
nT = 20 # Prediction window size
N = 5  # Size of window to predict
nsamples = int(len(data[:,1])/nT)

times = data[:,0].reshape(nsamples,nT)
sequences = x_data.reshape(nsamples,nT,2)
tarseq = data[:,3].reshape(nsamples,nT)
targets =   np.array([item[-N:] for item in data[:,3].reshape(nsamples,nT)])
tartimes =  np.array([item[-N:] for item in times])

tarcolumn = 3

nE = 2; nM = 3; nL = 5; nx = 3
generate_subplot(ax[2], '(c)', 0)


strokes_labels = ax[2].get_legend_handles_labels()

strokes = []
labels = []
for stroke, label in zip(*strokes_labels):
    if label not in labels:
        strokes.append(stroke)
        labels.append(label)

#print(strokes,labels)

fig.legend(strokes, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=10)
plt.subplots_adjust(hspace=0.5)



# SAVE !
plt.savefig('article_plots.pdf')

"""
plt.close('all')
fig, ax = plt.subplots(3,1, figsize=(12.5,10))


# PLOT N.1: dimmed triangular wave

tr_data = np.loadtxt('posterplot1_tr.dat')
ts_data = np.loadtxt('posterplot1_ts.dat')
pr_data = np.loadtxt('posterplot1_pr.dat')

t_tr = tr_data[:,0]; x_tr0 = tr_data[:,1]; y_tr = tr_data[:,2]
t_ts = ts_data[:,0]; x_ts0 = ts_data[:,1]; y_ts = ts_data[:,2]
t    = pr_data[:,0]; x_pr0 = pr_data[:,1]; y_pr = pr_data[:,2]


half_bkg_tr = patches.Rectangle((0,-2), t_tr[-1], 4, color='mistyrose', alpha=0.5)
half_bkg_ts = patches.Rectangle((t_tr[-1],-2), t_ts[-1], 4, color='honeydew', alpha=0.5)
#ax[0].vlines(t_tr[-1],-2,2, linestyle='-', color='darkseagreen', linewidth=0.5)
ax[0].add_patch(half_bkg_tr)
ax[0].add_patch(half_bkg_ts)
ax[0].plot(t_tr, x_tr0, '-', alpha = 0.3, color='tab:blue', label='Train IN $x_{(t)}$')
ax[0].plot(t_ts, x_ts0, '-', alpha = 0.3, color='darkcyan', label='Test IN $x_{(t)}$')
ax[0].plot(t_tr, y_tr, '-s', markersize=5, color='tab:blue', label='Train OUT set $y_{(t)}$')
ax[0].plot(t_ts, y_ts, '--D', markersize=5, color='darkcyan', label='Test OUT $y_{(t)}$')
ax[0].plot(t, y_pr,'--.', color='orangered', label = 'Estimation $\overline{y_{(t)}}$')
ax[0].set_xticks(np.arange(0,50+1,5))
ax[0].set_xlim(0,50)
ax[0].set_yticks(np.arange(-0.75,1.0,0.25))
ax[0].set_ylim(-1.0,0.75)
ax[0].grid(color='gainsboro')
ax[0].tick_params(labelsize=14)
ax[0].set_xticklabels(['' for i in np.arange(0,51,5)])
ax[0].legend(fontsize=10, loc='lower right', ncol=3, labelspacing=0.0)




# PLOT N.2: uni-variate Van der Pol oscillator

tr_data = np.loadtxt('posterplot2_tr.dat')
ts_data = np.loadtxt('posterplot2_ts.dat')
pr_data = np.loadtxt('posterplot2_pr.dat')

t_tr = tr_data[:,0]; x_tr0 = tr_data[:,1]; y_tr = tr_data[:,2]
t_ts = ts_data[:,0]; x_ts0 = ts_data[:,1]; y_ts = ts_data[:,2]
t    = pr_data[:,0]; x_pr0 = pr_data[:,1]; y_pr = pr_data[:,2]


half_bkg_tr = patches.Rectangle((0,-2), t_tr[-1], 4, color='mistyrose', alpha=0.5)
half_bkg_ts = patches.Rectangle((t_tr[-1],-2), t_ts[-1], 4, color='honeydew', alpha=0.5)
#ax[1].vlines(t_tr[-1],-2,2, linestyle='-', color='darkseagreen', linewidth=0.5)
ax[1].add_patch(half_bkg_tr)
ax[1].add_patch(half_bkg_ts)
ax[1].plot(t_tr, x_tr0, '-', alpha = 0.3, color='tab:blue', label='Train IN $x_{(t)}$')
ax[1].plot(t_ts, x_ts0, '-', alpha = 0.3, color='darkcyan', label='Test IN $x_{(t)}$')
ax[1].plot(t_tr, y_tr, '-s', markersize=5, color='tab:blue', label='Train OUT set $y_{(t)}$')
ax[1].plot(t_ts, y_ts, '--D', markersize=5, color='darkcyan', label='Test OUT $y_{(t)}$')
ax[1].plot(t, y_pr,'--.', color='orangered', label = 'Estimation $\overline{y_{(t)}}$')
ax[1].set_xticks(np.arange(0,51,5))
ax[1].set_xlim(0,50)
ax[1].set_yticks(np.arange(-0.75,1.0,0.25))
ax[1].set_ylim(-1.0,0.75)
ax[1].grid(color='gainsboro')
ax[1].tick_params(labelsize=14)
ax[1].set_xticklabels(['' for i in np.arange(0,51,5)])
ax[1].legend(fontsize=10, loc='lower right', ncol=3, labelspacing=0.0)




# PLOT N.3: bi-variate Van der Pol oscillator

tr_data = np.loadtxt('posterplot3_tr.dat')
ts_data = np.loadtxt('posterplot3_ts.dat')
pr_data = np.loadtxt('posterplot3_pr.dat')

t_tr = tr_data[:,0]; x_tr0 = tr_data[:,1]; x_tr1 = tr_data[:,2]; y_tr = tr_data[:,3]
t_ts = ts_data[:,0]; x_ts0 = ts_data[:,1]; x_ts1 = ts_data[:,2]; y_ts = ts_data[:,3]
t    = pr_data[:,0]; x_pr0 = pr_data[:,1]; x_pr1 = pr_data[:,2]; y_pr = pr_data[:,3]


half_bkg_tr = patches.Rectangle((0,-2), t_tr[-1], 4, color='mistyrose', alpha=0.5)
half_bkg_ts = patches.Rectangle((t_tr[-1],-2), t_ts[-1], 4, color='honeydew', alpha=0.5)
#ax[2].vlines(t_tr[-1],-2,2, linestyle='-', color='darkseagreen', linewidth=0.5)
ax[2].add_patch(half_bkg_tr)
ax[2].add_patch(half_bkg_ts)
ax[2].plot(t_tr, x_tr0, '-', alpha = 0.3, color='tab:blue', label='Train IN $x_{(t)}^0$')
ax[2].plot(t_ts, x_ts0, '-', alpha = 0.3, color='darkcyan', label='Test IN $x_{(t)}^0$')
ax[2].plot(t_tr, x_tr1, '-', alpha = 0.3, color='goldenrod', label='Train IN $x_{(t)}^1$')
ax[2].plot(t_ts, x_ts1, '-', alpha = 0.3, color='green', label='Test IN $x_{(t)}^1$')
ax[2].plot(t_tr, y_tr, '-s', markersize=5, color='tab:blue', label='Train OUT set $y_{(t)}$')
ax[2].plot(t_ts, y_ts, '--D', markersize=5, color='darkcyan', label='Test OUT $y_{(t)}$')
ax[2].plot(t, y_pr,'--.', color='orangered', label = 'Estimation $\overline{y_{(t)}}$')
ax[2].set_xticks(np.arange(0,51,5))
ax[2].set_xlim(0,50)
ax[2].set_yticks(np.arange(-0.75,1.0,0.25))
ax[2].set_ylim(-1.0,0.75)
ax[2].grid(color='gainsboro')
ax[2].tick_params(labelsize=14)
#ax[2].set_xticklabels(np.array([[str(2*i),''] for i in np.arange(0,25,5)]).flatten())
ax[2].legend(fontsize=9, loc='lower right', ncol=4, labelspacing=0.0)
ax[2].set_xlabel('$t$', fontsize=16, loc='center')

plt.savefig('poster_plots_all.png', bbox_inches='tight', dpi=300)
"""
