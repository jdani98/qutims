# -*- coding: utf-8 -*-
"""
 Title: EM3CX model Density Matrix emulation with MultiProcessing
 Description: this is a python module that contains functions to emulate with
 Density Matrix methods (arXiv:XXXX.XXXXX) the Quantum Recurrent Neural Network
 model called "EM3CX model", which consists of a singular-structured EMC cir-
 cuit to enhance multiprocessing.
 This MultiProcessing is used when computing gradients.

Created on Mon Oct 23 10:22:49 2023
@author: jdviqueira
"""

import numpy as np
from scipy.optimize import minimize

import sys
sys.path.append('../..')
from qurecnets.qualgebra import Rx,Ry,Rz

from time import time

#from multiprocessing import Pool
#if __name__ == '__main__':
#    print('control')


def evaluate(inputs):
    """
    This function evaluates from beginning to end the EMCZ3 model full circuit.
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




def optimize(input_data,output_data,param0,hyperparams,optimizer_options):
    
    nT, nE, nM, nL, nx, Nwout = hyperparams
    
    train_sequences, val_sequences = input_data
    train_targets,   val_targets   = output_data
    
    
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
                print('control')
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
        
    
    sol = minimize(cost_function,param0,jac=grad, method='L-BFGS-B', options={'maxiter':1000,'maxfun':500000, 'gtol':1e-3}, callback=my_callback)
    
    return sol
    
    
    
    

class em3cx_optimization:
    
    def __init__(self,nT,nE,nM,nL=1,nx=1):
        """
        Common attributes for every QRNN.

        Parameters
        ----------
        nT : int
            Number of time frames = number of intermediate measurements.
        nE : int
            Number of exchange-register qubits.
        nM : int
            Number of memory-register qubits.
        nL : int
            Number of ansatz layers.
        nx : int
            Number of data re-uploadings.

        Returns
        -------
        None.

        """
        
        self.nT = nT
        self.nE = nE
        self.nM = nM
        self.nL = nL
        self.nx = nx
    
    
    def load_x(self,tr_data,val_data):
        """
        Load input data.

        Parameters
        ----------
        tr_data : array
            Training input data. shape=(ntr_samp,nT,nE)
        val_data : array
            Validation input data. shape=(nval_samp,nT,nE)

        Returns
        -------
        None.

        """
        
        self.x_tr = tr_data
        self.ntr_samp = len(tr_data)
        self.x_val = val_data
        self.nval_samp = len(val_data)
        self.iteration = 0
    
    
    
    def load_y(self,tr_data,val_data):
        """
        Load output data (labels).

        Parameters
        ----------
        tr_data : array
            Training output data. shape=(ntr_samp,nT,nE)
        val_data : array
            Validation output data. shape=(nval_samp,nT,nE)

        Returns
        -------
        None.

        """
        
        assert self.ntr_samp==len(tr_data)
        assert self.nval_samp==len(val_data)
        self.y_tr = tr_data
        self.y_val = val_data
        self.Nwout = tr_data.shape[1]
        # we choose the length of the targets as the length of output window to
        # compute the loss function
        
    
    
    def cost_function(self,params):
        L = 0
        npoints = 0
    
        ttheta = {tuple(range(self.nT)): params[1:]}
    
        for isample in range(self.ntr_samp):
            evalu = evaluate((self.nT,self.nE,self.nM,self.nL,self.nx,ttheta,
                              self.x_tr[isample]))
            yb = np.array([params[0]]*len(evalu)) + evalu
            L += sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-self.Nwout:],
                                            self.y_tr[isample][-self.Nwout:])])
            npoints += self.Nwout
    
        L = np.sqrt(L/npoints)
    
        return L
    
    
    def cost_val(self,params):
        L = 0
        npoints = 0
    
        ttheta = {tuple(range(self.nT)): params[1:]}
    
        for isample in range(self.nval_samp):
            evalu = evaluate((self.nT,self.nE,self.nM,self.nL,self.nx,
                              ttheta,self.x_val[isample]))
            yb = np.array([params[0]]*len(evalu)) + evalu
            L += sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-self.Nwout:],
                                        self.y_val[isample][-self.Nwout:])])
            npoints += self.Nwout
    
        L = np.sqrt(L/npoints)
    
        return L
    
    
    
    def grad(self,params):
        grad = np.zeros(len(params))
        ttheta = {tuple(range(self.nT)): params[1:]}
        npoints = 0
    
        for isample in range(self.ntr_samp):
            gradis = []
            
            #print(self.x_tr[isample])
            evalu = evaluate((self.nT,self.nE,self.nM,self.nL,self.nx,
                              ttheta,self.x_tr[isample]))
            yb = np.array([params[0]]*len(evalu)) + evalu
            
            from multiprocessing import Pool
    
            #if __name__ == '__main__':
            print('control')
            with Pool(len(params)-1) as p:
                gradis.append(p.map(derivative1, [(self.nT,self.nE,self.nM,
                self.nL,self.nx,params[1:],self.x_tr[isample],i) 
                                        for i in range(len(params)-1)]))
            print(gradis[0])
            grad1 = [sum([(ybi-yi)*partiali for (ybi,yi,partiali) in 
                          zip(yb[-self.Nwout:],self.y_tr[isample],
                             partial[-self.Nwout:])]) for partial in gradis[0]]
            grad1_0 = sum([(ybi-yi) for (ybi,yi) in zip(yb[-self.Nwout:],
                                                    self.y_tr[isample])])
            grad1 = np.array([grad1_0] + grad1)
            grad += grad1
            npoints += self.Nwout
        grad = np.array(grad)
        L = self.cost_function(params)
        grad = (1./L)*(1./npoints)*grad
        return grad
    
    
    def my_callback(self,params):
        global iteration
        iteration += 1
        train_loss = self.cost_function(params)
        val_loss = self.cost_val(params)
        print('Iteration %6i, Training loss: %8.6f, Validation loss: %8.6f' 
              %(iteration,train_loss,val_loss))
    
    
    