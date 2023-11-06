# -*- coding: utf-8 -*-
"""
 Title: EM3CX model Density Matrix emulation with MultiProcessing
 Description: this is a python module that contains functions to emulate with
 Density Matrix methods (arXiv:2310.20671) the Quantum Recurrent Neural Network
 model called "EM3CX model", which consists of a singular-structured EMC cir-
 cuit to enhance multiprocessing.
 This MultiProcessing is used when computing gradients.
 It should be used only on HPC platforms.

Created on Mon Oct 23 10:22:49 2023
@author: jdviqueira
"""

import numpy as np
from scipy.optimize import minimize

import sys
sys.path.append('../..')
from qurecnets.qualgebra import Rx,Ry,Rz

from time import time

from multiprocessing import Pool
#if __name__ == '__main__':
#    print('control')


def evaluate(inputs):
    """
    This function evaluates from beginning to end the EMCZ3 model full circuit.
    The output is the Z-expectation value for every time step.
    The parameters can be different every time step.
    This function tries to minimize the flux of information between workers and
    scheduler in parallelized process.

    Parameters
    ----------
    A tuple with the following content.
    nT : INT
        Number of time steps.
    nE : INT
        Number of register E qubits.
    nM : INT
        Number of register M qubits.
    nL : INT
        Number of evolution ansatz layers.
    nx : INT
        Number of data re-uploadings.
    theta : dictionary
        Trainable parameters. Must include the parameters for all the circuit 
        blocks with the following structure:
            * key: tuple
                Positions (blocks) of the corresponding array parameters.
            * value: array
                Parameters for the circuit block(s) indicated in the keys.
    x : array
        Inputs dataset. Shape (nT,nE)

    Returns
    -------
    evaluation : array

    """
    
    nT,nE,nM,nL,nx,theta,x = inputs
    NE = 2**nE; NM = 2**nM
    
    def Uop_bZ(sub_thetas):
        # 3-parameter operator in Z-basis, to apply in register "A"
        Uo = np.dot(Rx(sub_thetas[2]),np.dot(Rz(sub_thetas[1]),
                                             Rx(sub_thetas[0])))
        return Uo
    
    def encoding(xt,theta):
        # Ansatz-V-generator. Description: see dm.py.
        sA = [1.]
        for j,xj in enumerate(xt):
            encgat = Ry(xj)
            Uin = np.copy(encgat)
            for repi in range(nx):
                Uin = np.dot(encgat,np.dot(Uop_bZ(theta[3*nx*j+
                                                3*repi:3*nx*j+3*repi+3]),Uin))
            sAj = [Uin[0][0],Uin[1][0]]
            sA = np.kron(sA,sAj)
        rhoA = np.outer(sA,sA.conjugate())
        return rhoA
    
    def emcz3_ansatz(theta):
        # Ansatz-W-generator. Description: see dm.py.
        sign0 = [+1,-1]
        signB = [+1,-1]
        for i in range(nM-1):
            signB = np.kron(signB,sign0)
        signA = [+1,-1]
        for i in range(nE-1):
            signA = np.kron(signA,sign0)
        idenB = np.array([+1 for i in range(NM)])
        signsB = [idenB,signB]
        boolsA = np.array([int((1-signi)/2) for signi in signA])
        csign = np.array([])
        for it in boolsA:
            csign = np.concatenate((csign,signsB[it]))
        # // Create operator to apply over all qubits as first layer
        Uab  = np.array([[1]])
        for i in range(nE):
            Uab = np.kron(Uab,Uop_bZ(theta[3*nx*nE+3*i:3*nx*nE+3*i+3]))
        for i in range(nE,nE+nM):
            Uab = np.kron(Uab,Uop_bZ(theta[3*nx*nE+3*i:3*nx*nE+3*i+3]))
        Uabcx = []
        for csigni,row in zip(csign,Uab):
            Uabcx += [csigni*row]
        Ut = np.array(Uabcx)
        # // Loop to apply full operator as many times as nlayers. Parameters 
        # are different for each
        for li in range(1,nL):
            Uab  = np.array([[1]])
            for i in range(nE):
                Uab = np.kron(Uab,Uop_bZ(theta[3*nx*nE+3*(nE+nM)*li+
                                              3*i:3*nx*nE+3*(nE+nM)*li+3*i+3]))
            for i in range(nE,nE+nM):
                Uab = np.kron(Uab,Uop_bZ(theta[3*nx*nE+3*(nE+nM)*li+
                                              3*i:3*nx*nE+3*(nE+nM)*li+3*i+3]))
            Uabcx = []
            for csigni,row in zip(csign,Uab):
                Uabcx += [csigni*row]
            Ut = np.dot(Uabcx, Ut)
        # // Applying 3-parameter rotations over regA before measurement
        InB = np.diag([1. for i in range(NM)])
        Ua  = np.array([[1]])
        for i in range(nE):
            Ua = np.kron(Ua,Uop_bZ(theta[3*nx*nE+3*(nE+nM)*nL+3*i:3*nx*nE+
                                         3*(nE+nM)*nL+3*i+3]))
        Uf  = np.kron(Ua,InB)
        Ut  = np.dot(Uf,Ut)
        return np.matrix(Ut)
    
    
    Uts = {}
    for key,value in theta.items():
        Uts[key] = emcz3_ansatz(value)
    
    keys = Uts.keys()
    
    probs = []
    rhoB = np.zeros([NM,NM]); rhoB[0,0] = 1.
    
    
    for ib in range(len(x)):
        key = list(filter(lambda x: ib in x, keys))[0]
        Ut = Uts[key]
        th = theta[key]
        
        rhoA = encoding(x[ib], th)
        
        rhoAB = np.kron(rhoA,rhoB)    
        
        rhoB = np.dot(Ut[0:NM,:], np.dot(rhoAB,Ut[0:NM,:].H))
        probs += [[np.trace(rhoB).real]]
        for i in range(1,NE):
            rhoBi = np.dot(Ut[i*NM:(i+1)*NM,:], np.dot(rhoAB,Ut[i*NM:(i+1)*NM,:].H))
            probs[ib] += [np.trace(rhoBi).real]
            rhoB += rhoBi
    
    distributions = np.array(probs)
    
    flips = [1,-1]
    for i in range(nE-1):
        flips = np.kron(flips, [1,-1])
        
    evaluation = np.array([np.dot(flips,item) for item in distributions])
    
    return evaluation



def derivative1(inputs):
    """
    Analytical 1st-order derivative with PSR.

    Parameters
    ----------
    A tuple with the following content.
    nT : INT
        Number of time steps.
    nE : INT
        Number of register E qubits.
    nM : INT
        Number of register M qubits.
    nL : INT
        Number of evolution ansatz layers.
    nx : INT
        Number of data re-uploadings.
    theta : array
        Trainable parameters.
    x : array
        Inputs dataset. Shape (nT,nE)
    ish : INT
        Position of paramater to derive.

    Returns
    -------
    deriv : array

    """
    
    nT,nE,nM,nL,nx,theta,x, ish = inputs
    
    theta_ps = np.copy(theta); theta_ps[ish] += np.pi/2.
    theta_ms = np.copy(theta); theta_ms[ish] -= np.pi/2.
    
    deriv = np.zeros(nT)
    for r in range(nT):
        ttheta_ps = {tuple(list(range(r))+list(range(r+1,nT))) : theta,
                  (r,) : theta_ps}
        deriv += evaluate((nT,nE,nM,nL,nx,ttheta_ps,x))
        ttheta_ms = {tuple(list(range(r))+list(range(r+1,nT))) : theta,
                  (r,) : theta_ms}
        deriv -= evaluate((nT,nE,nM,nL,nx,ttheta_ms,x))
    deriv = 0.5*deriv
    
    return deriv



def em3cx_optimization(tr_seq,val_seq,tr_tar,val_tar,param0,hyperparams,options):
    """
    Routine for EM3CX model training with analytical gradients and parallelization.
    
    Parameters
    ----------
    tr_seq : array
        Training input data. shape=(ntr_samp,nT,nE)
    val_seq : array
        Validation input data. shape=(nval_samp,nT,nE)
    tr_tar : array
        Training output data. shape=(ntr_samp,nT,nE)
    val_tar : array
        Validation output data. shape=(nval_samp,nT,nE)
    hyperparams : tuple
        Hyperparameters for network configuration - (nT, nE, nM, nL, nx, Nwout)
        Where:
            * nT number of input points per window,
            * nE number of reg. E qubits,
            * nM number of reg. M qubits,
            * nL number of ansatz layers,
            * nx number of re-uploading parameters,
            * Nwout number of points to predict per window (last ones)
    options : dict
        Scipy.minimize options. See 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        for information.

    Returns
    -------
    sol : object
        Solution of the optimization, from the scipy.minimize.
    """
    
    nT, nE, nM, nL, nx, Nwout = hyperparams
    
    global iteration
    iteration = 0
    
    def cost_function(params):
        L = 0
        npoints = 0
        ttheta = {tuple(range(nT)): params[1:]}
        for isample in range(len(tr_seq)):
            evalu = evaluate((nT,nE,nM,nL,nx,ttheta,tr_seq[isample]))
            yb = np.array([params[0]]*len(evalu)) + evalu
            L += sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:],tr_tar[isample])])
            npoints += Nwout
        L = np.sqrt(L/npoints)
        return L


    def cost_val(params):
        L = 0
        npoints = 0
        ttheta = {tuple(range(nT)): params[1:]}
        for isample in range(len(val_seq)):
            evalu = evaluate((nT,nE,nM,nL,nx,ttheta,val_seq[isample]))
            yb = np.array([params[0]]*len(evalu)) + evalu
            L += sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:],val_tar[isample])])
            npoints += Nwout
        L = np.sqrt(L/npoints)
        return L



    def grad(params):
        grad = np.zeros(len(params))
        ttheta = {tuple(range(nT)): params[1:]}
        npoints = 0
        for isample in range(len(tr_seq)):
            gradis = []
            evalu = evaluate((nT,nE,nM,nL,nx,ttheta,tr_seq[isample]))
            yb = np.array([params[0]]*len(evalu)) + evalu
            #from multiprocessing import Pool
            #if __name__ == '__main__':
            with Pool(len(params)-1) as p:
                gradis.append(p.map(derivative1, [(nT,nE,nM,nL,nx,params[1:],tr_seq[isample],i) for i in range(len(params)-1)]))
            grad1 = [sum([(ybi-yi)*partiali for (ybi,yi,partiali) in zip(yb[-Nwout:],tr_tar[isample], partial[-Nwout:])]) for partial in gradis[0]]
            grad1_0 = sum([(ybi-yi) for (ybi,yi) in zip(yb[-Nwout:],tr_tar[isample])])
            grad1 = np.array([grad1_0] + grad1)
            grad += grad1
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
        

    sol = minimize(cost_function,param0,jac=grad, method='L-BFGS-B', options=options, callback=my_callback)
    
    return sol   