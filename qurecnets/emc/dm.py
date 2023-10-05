# -*- coding: utf-8 -*-
"""
 Title: EMC model Density Matrix emulation
 Description: this is a python module that contains a class to emulate with
 Density Matrix methods (arXiv:XXXX.XXXXX) the Quantum Recurrent Neural Network
 model called "EMC model".

Created on Tue Oct  3 16:41:03 2023
@author: jdviqueira
"""

import numpy as np

import sys
sys.path.append('../..')
from qurecnets.qualgebra import Rx,Ry,Rz


class emc:
    """
    Class to define the EMC multivariate model circuit with Density Matrix emu-
    lation methods.
    
    The Exchange-Memory Controlled gates model is a QRNN with a free-shape an-
    satz that must satisfy 3 conditions: (i) must encode input data; (ii) must 
    entangle registers E and M, and (iii) must be trainable.
    
    See emc_circ_graph.txt to visualize the general circuit structure.
    
    
    The main attributes of this class build the ansatz for the EMCZ3 model, a
    multilayer ansatz consisting of: (i) encoding part -layers of Ry gates pa-
    rameterised by input data, alternated by series of parameterised Rx-Rz-Rx
    gate- and (ii) evolution part -a column of Rx-Rz-Rx gates over each qubit
    plus a group of CZs connecting the exchange (E) register with each of the
    nB qubits in the memory (M) register-. Finally, a column of Rx-Rz-Rx gates
    over the E register.
    
    See emcz3_circ_graph.txt to visualize the EMCZ3 circuit structure.
    
    
    To change this ansatz, you may create a new class which inherits this one
    and add a new method.
    """
    
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
        
        self.NE = 2**nE
        self.NM = 2**nM
        
        # defining change of sign for computing Z expectation value
        flips = [1,-1]
        for i in range(self.nE-1):
            flips = np.kron(flips, [1,-1])
        self.flips = flips
    
    
    
    def emcz3_ansatz(self,theta):
        """
        Generate ansatz EVOLUTION&ENTANGLEMENT part W(theta) for EMCZ3 model.

        Parameters
        ----------
        theta : array
            Trainable parameters for ansatz.

        Returns
        -------
        Ut : matrix
            Ansatz.

        """
        
        assert len(theta) == 3*(self.nE*self.nx+self.nL*(self.nE+self.nM)+
                                self.nE)
        
        def Uop_bZ(sub_thetas):
            # 3-parameter operator, to apply in register E
            Uo = np.dot(Rx(sub_thetas[2]),np.dot(Rz(sub_thetas[1]),
                                                 Rx(sub_thetas[0])))
            return Uo
        
        # /1/ Create array of signs to apply the effect of a X^{\otimes n_M}
        # over register M:
        # More significative qubits are the upper ones
        
        sign0 = [+1,-1]
        signB = [+1,-1]
        for i in range(self.nM-1):
            signB = np.kron(signB,sign0)
        signA = [+1,-1]
        for i in range(self.nE-1):
            signA = np.kron(signA,sign0)
        idenB = np.array([+1 for i in range(self.NM)])
        signsB = [idenB,signB]
        boolsA = np.array([int((1-signi)/2) for signi in signA])
        csign = np.array([])
        for it in boolsA:
            csign = np.concatenate((csign,signsB[it]))
        # /1/
        
        # /2/ Create operator to apply over all qubits as first layer
        Uab  = np.array([[1]])
        for i in range(self.nE):
            Uab = np.kron(Uab,Uop_bZ(theta[3*self.nx*self.nE+3*i:
                                                3*self.nx*self.nE+3*i+3]))
        for i in range(self.nE,self.nE+self.nM):
            Uab = np.kron(Uab,Uop_bZ(theta[3*self.nx*self.nE+3*i:
                                                3*self.nx*self.nE+3*i+3]))
        Uabcx = []
        for csigni,row in zip(csign,Uab):
            Uabcx += [csigni*row]
        Ut = np.array(Uabcx)
        # /2/
        
        # /3/ Loop to apply full operator as many times as nlayers. Parameters
        # are different for each layer
        for li in range(1,self.nL):
            Uab  = np.array([[1]])
            for i in range(self.nE):
                Uab = np.kron(Uab,Uop_bZ(theta[3*self.nx*self.nE+
                3*(self.nE+self.nM)*li+3*i:3*self.nx*self.nE+
                3*(self.nE+self.nM)*li+3*i+3]))
            for i in range(self.nE,self.nE+self.nM):
                Uab = np.kron(Uab,Uop_bZ(theta[3*self.nx*self.nE+
                3*(self.nE+self.nM)*li+3*i:3*self.nx*self.nE+
                3*(self.nE+self.nM)*li+3*i+3]))
            Uabcx = []
            for csigni,row in zip(csign,Uab):
                Uabcx += [csigni*row]
            Ut = np.dot(Uabcx, Ut)
        # /3/
        
        # /4/ Applying 3-parameter rotations over regA before measurement
        InB = np.diag([1. for i in range(self.NM)])
        Ua  = np.array([[1]])
        for i in range(self.nE):
            Ua = np.kron(Ua,Uop_bZ(theta[3*self.nx*self.nE+3*(self.nE+
            self.nM)*self.nL+3*i:3*self.nx*self.nE+3*(self.nE+self.nM)*self.nL+
            3*i+3]))
        Uf  = np.kron(Ua,InB)
        # /4/
        
        Ut  = np.dot(Uf,Ut)
        
        return np.matrix(Ut)
    
    
    
    def encoding(self,xt,theta):
        """
        Generate ANSATZ ENCODING part V(xt,theta)

        Parameters
        ----------
        xt : array
            x_(t) data.
        theta : array
            Trainable parameters.

        Returns
        -------
        rhoA : matrix
            Register E density matrix after qubit re-starts and encoding.

        """
        assert len(theta) == 3*(self.nE*self.nx+self.nL*(self.nE+self.nM)+
                                self.nE)
        assert len(xt) == self.nE
        
        def Uop_bZ(sub_thetas):
            # 3-parameter operator, to apply in register E
            Uo = np.dot(Rx(sub_thetas[2]),np.dot(Rz(sub_thetas[1]),
                                                 Rx(sub_thetas[0])))
            return Uo
        
        sA = [1.]
        for j,xj in enumerate(xt):
            encgat = Ry(xj)
            Uin = np.copy(encgat)
            for repi in range(self.nx):
                Uin = np.dot(encgat,np.dot(Uop_bZ(theta[3*self.nx*j+
                3*repi:3*self.nx*j+3*repi+3]),Uin))
            sAj = [Uin[0][0],Uin[1][0]]
            sA = np.kron(sA,sAj)
        
        rhoA = np.outer(sA,sA.conjugate())
        
        return rhoA
    
    
    
    def evaluate(self,x,theta, savegrad=False):
        """
        Emulate the full circuit with the ansatz and return its outputs.
        Designed to evaluate the circuit inside an optimization process.
        
        Parameters
        ----------
        x : array
            Input data.
            x = [[x^0_(0),x^1_(0),...], [x^0_(1),x^1_(1),...], ...]
        theta : array
            Trainable ansatz parameters.
        savegrad : bool
            If True, 2 ansatze with \pm \pi/2 shift are computed and saved in
            memory as a new attribute.

        Returns
        -------
        y : array
            Outputs of the circuit, made up of distribution probabilities for
            every time-step.
            y = [[y^(0..00)_(0), y^(0..01)_(0), ..., y^(1..11)_(0)],
                 [y^(0..00)_(1), y^(0..01)_(1), ..., y^(1..11)_(1)],
                         ...
                 [y^(0..00)_(T-1), y^(0..01)_(T-1), ..., y^(1..11)_(T-1)]]

        """
        
        self.Ut = self.emcz3_ansatz(theta)
        self.x = x
        self.th = theta
        
        if savegrad:
            Ut_ps = {}
            Ut_ms = {}
            for i in range(len(theta)):
                theta_ps = np.copy(theta); theta_ps[i] += np.pi/2.
                theta_ms = np.copy(theta); theta_ms[i] -= np.pi/2.
                Ut_ps[i] = self.emcz3_ansatz(theta_ps)
                Ut_ms[i] = self.emcz3_ansatz(theta_ms)
            self.Ut_ps = Ut_ps
            self.Ut_ms = Ut_ms
        
        probs = []
        
        rhoB = np.zeros([self.NM,self.NM]); rhoB[0,0] = 1.
        
        for ib in range(len(x)):
            rhoA = self.encoding(x[ib], theta)
            rhoAB = np.kron(rhoA,rhoB)
            rhoB = np.dot(self.Ut[0:self.NM,:], np.dot(rhoAB,
                                                       self.Ut[0:self.NM,:].H))
            probs += [[np.trace(rhoB).real]]
            for i in range(1,self.NE):
                rhoBi = np.dot(self.Ut[i*self.NM:(i+1)*self.NM,:],
                np.dot(rhoAB,self.Ut[i*self.NM:(i+1)*self.NM,:].H))
                probs[ib] += [np.trace(rhoBi).real]
                rhoB += rhoBi
        
        self.evaluation = np.array(probs)
        
        return np.array(probs)
    
    
    
    def psr1(self,theta):
        """
        Evaluate the outputs at every time t by shifting the parameter
        theta_i at circuit block t_k.

        Parameters
        ----------
        theta : array
            Trainable parameters.

        Returns
        -------
        J_ki : dictionary
            Outputs after shift of theta_i at time t_k. J_ki contains keys
            (k,i,t) indicating:
                * k -> shifted block
                * i -> shifted parameter
                * t -> array of distribution probabilities (output) at time t

        """
        
        def subroutine1(xin,rhoB,theta_sh):
            # shift in encoding
            rhoA = self.encoding(xin, theta_sh)
            rhoAB = np.kron(rhoA,rhoB)
            rhoB = np.dot(self.Ut[0:self.NM,:], np.dot(rhoAB,
                                                       self.Ut[0:self.NM,:].H))
            probs = [np.trace(rhoB).real]
            for i in range(1,self.NE):
                rhoBi = np.dot(self.Ut[i*self.NM:(i+1)*self.NM,:],
                np.dot(rhoAB,self.Ut[i*self.NM:(i+1)*self.NM,:].H))
                probs += [np.trace(rhoBi).real]
                rhoB += rhoBi
            return rhoB,np.array(probs)
        
        def subroutine2(xin,rhoB,Ush):
            # shift in evolution. Recycle ansatz.
            rhoA = self.encoding(xin, theta)
            rhoAB = np.kron(rhoA,rhoB)
            rhoB = np.dot(Ush[0:self.NM,:], np.dot(rhoAB,Ush[0:self.NM,:].H))
            probs = [np.trace(rhoB).real]
            for i in range(1,self.NE):
                rhoBi = np.dot(Ush[i*self.NM:(i+1)*self.NM,:],
                np.dot(rhoAB,Ush[i*self.NM:(i+1)*self.NM,:].H))
                probs += [np.trace(rhoBi).real]
                rhoB += rhoBi
            return rhoB,np.array(probs)
        
        
        J_ki = {}
        for ksh in range(self.nT):

            rhoB = np.zeros([self.NM,self.NM]); rhoB[0,0] = 1.
            for ib in range(ksh):
                rhoB, probsib = subroutine1(self.x[ib],rhoB,theta)
            for ish in range(3*(self.nx*self.nE)):
                theta_ps = np.copy(theta); theta_ps[ish] += np.pi/2.
                rhoB_ps, prob_ps = subroutine1(self.x[ksh],rhoB,theta_ps)
                theta_ms = np.copy(theta); theta_ms[ish] -= np.pi/2.
                rhoB_ms, prob_ms = subroutine1(self.x[ksh],rhoB,theta_ms)
                J_ki[(ksh,ish,ksh)] = 0.5*(prob_ps - prob_ms)
                for ib in range(ksh+1,self.nT):
                    rhoB_ps, prob_ps = subroutine1(self.x[ib],rhoB_ps,theta)
                    rhoB_ms, prob_ms = subroutine1(self.x[ib],rhoB_ms,theta)
                    J_ki[(ksh,ish,ib)] = 0.5*(prob_ps - prob_ms)
            
            for ish in range(3*(self.nx*self.nE), len(theta)):
                rhoB_ps, prob_ps = subroutine2(self.x[ksh],rhoB,
                                               self.Ut_ps[ish])
                rhoB_ms, prob_ms = subroutine2(self.x[ksh],rhoB,
                                               self.Ut_ms[ish])
                J_ki[(ksh,ish,ksh)] = 0.5*(prob_ps - prob_ms)
                for ib in range(ksh+1,self.nT):
                    rhoB_ps, prob_ps = subroutine2(self.x[ib],rhoB_ps, self.Ut)
                    rhoB_ms, prob_ms = subroutine2(self.x[ib],rhoB_ms, self.Ut)
                    J_ki[(ksh,ish,ib)] = 0.5*(prob_ps - prob_ms)
                    
        return J_ki
    
    
    
    def grad_psr(self,theta):
        """
        Evaluate the analytical gradients with the Parameter Shift Rule, by
        summing psr1 items over k.

        Parameters
        ----------
        theta : array
            Trainable parameters.

        Returns
        -------
        grad : array
            First-order derivative of every output with respect to the para-
            meters. Its shape is (Nth,nT,NE).

        """
        
        J_ki = self.psr1(theta)
        
        grad = np.zeros([len(theta),self.nT,self.NE])
        for key,value in J_ki.items():
            k,i,t = key
            grad[i,t] += value
            
        self.gradient = grad
        return grad
    
    
    
    def evaluate_w2shift_k_neq_l(self,theta,i,j,k,l,sh1,sh2):
        """
        Evaluate circuit with two shifts at different blocks.
        Defined to compute the Hessian.

        Parameters
        ----------
        theta : array
            Parameters.
        i : INT
            Position for shift1.
        j : INT
            Position for shift2.
        k : INT
            Time for shift1.
        l : INT
            Time for shift2.
        sh1 : INT
            Value of shift1.
        sh2 : INT
            Value of shift2.

        Returns
        -------
        probs_t : array
            Outputs at every time after shifts at block t_k, parameter theta_i
            and block t_l, parameter theta_j.

        """
        
        assert j<=i
        assert l!=k
        
        if l<k:
            lsh = l; jsh = j; shift2 = sh2
            ksh = k; ish = i; shift1 = sh1
        
        else:
            lsh = k; jsh = i; shift2 = sh1
            ksh = l; ish = j; shift1 = sh2
        
        lsh = min([l,k])
        ksh = max([l,k])
        
        jsh = list([j,i])[np.argmin([l,k])]
        ish = list([j,i])[np.argmax([l,k])]
        
        shift2 = list([sh2,sh1])[np.argmin([l,k])]
        shift1 = list([sh2,sh1])[np.argmax([l,k])]
        
        barrier = 3*self.nx*self.nE
        
        def subroutine(rhoB,rhoA,Uans):
            # shift in evolution. Recycle ansatz.
            rhoAB = np.kron(rhoA,rhoB)
            rhoB = np.dot(Uans[0:self.NM,:], np.dot(rhoAB,Uans[0:self.NM,:].H))
            probs = [np.trace(rhoB).real]
            for i in range(1,self.NE):
                rhoBi = np.dot(Uans[i*self.NM:(i+1)*self.NM,:], np.dot(rhoAB,
                Uans[i*self.NM:(i+1)*self.NM,:].H))
                probs += [np.trace(rhoBi).real]
                rhoB += rhoBi
            return rhoB, probs
        
        rhoB = np.zeros([self.NM,self.NM]); rhoB[0,0] = 1.
        probs_t = []
        for ib in range(lsh):
            Uen = self.encoding(self.x[ib], theta)
            rhoB, probs = subroutine(rhoB,Uen,self.Ut)
            probs_t += [probs]
        # shift2:
        theta_sh = np.copy(theta); theta_sh[jsh] += shift2
        if jsh<barrier:
            Uen = self.encoding(self.x[lsh], theta_sh)
            Uans = self.Ut
        else:
            Uen = self.encoding(self.x[lsh], theta)
            Uans = self.emcz3_ansatz(theta_sh)
        rhoB, probs = subroutine(rhoB,Uen,Uans)
        probs_t += [probs]
        #::
        for ib in range(lsh+1,ksh):
            Uen = self.encoding(self.x[ib], theta)
            rhoB, probs = subroutine(rhoB,Uen,self.Ut)
            probs_t += [probs]
        # shift1:
        theta_sh = np.copy(theta); theta_sh[ish] += shift1
        if ish<barrier:
            Uen = self.encoding(self.x[ksh], theta_sh)
            Uans = self.Ut
        else:
            Uen = self.encoding(self.x[ksh], theta)
            Uans = self.emcz3_ansatz(theta_sh)
        rhoB, probs = subroutine(rhoB,Uen,Uans)
        probs_t += [probs]
        #::
        for ib in range(ksh+1,self.nT):
            Uen = self.encoding(self.x[ib], theta)
            rhoB, probs = subroutine(rhoB,Uen,self.Ut)
            probs_t += [probs]
        
        return np.array(probs_t)
    
    
    
    def evaluate_w2shift_k_eq_l(self,theta,i,j,k,l,sh1,sh2):
        """
        Evaluate circuit with two shifts at the same block.
        Defined to compute the Hessian.

        Parameters
        ----------
        theta : TYPE
            Parameters.
        i : TYPE
            Position for shift1.
        j : TYPE
            Position for shift2.
        k : TYPE
            Time for shift1.
        l : TYPE
            Time for shift2.
        sh1 : TYPE
            Value of shift1.
        sh2 : TYPE
            Value of shift2.

        Returns
        -------
        probs_t : array
            Outputs at every time after shifts at block t_k, parameter theta_i
            and block t_l, parameter theta_j.

        """
        
        assert j<=i
        assert l==k
        
        lsh = l; jsh = j; shift2 = sh2
        ksh = k; ish = i; shift1 = sh1
        
        def subroutine(rhoB,rhoA,Uans):
            # shift in evolution. Recycle ansatz.
            rhoAB = np.kron(rhoA,rhoB)
            rhoB = np.dot(Uans[0:self.NM,:], np.dot(rhoAB,Uans[0:self.NM,:].H))
            probs = [np.trace(rhoB).real]
            for i in range(1,self.NE):
                rhoBi = np.dot(Uans[i*self.NM:(i+1)*self.NM,:], np.dot(rhoAB,
                Uans[i*self.NM:(i+1)*self.NM,:].H))
                probs += [np.trace(rhoBi).real]
                rhoB += rhoBi
            return rhoB,probs
        
        rhoB = np.zeros([self.NM,self.NM]); rhoB[0,0] = 1.
        probs_t = []
        for ib in range(lsh):
            Uen = self.encoding(self.x[ib], theta)
            rhoB, probs = subroutine(rhoB,Uen,self.Ut)
            probs_t += [probs]
        # 2shift:
        theta_2sh = np.copy(theta)
        theta_2sh[jsh] += shift2
        theta_2sh[ish] += shift1
        Uen = self.encoding(self.x[l], theta_2sh)
        Uans = self.emcz3_ansatz(theta_2sh)
        rhoB, probs = subroutine(rhoB,Uen,Uans)
        probs_t += [probs]
        #::
        for ib in range(l+1,self.nT):
            Uen = self.encoding(self.x[ib], theta)
            rhoB, probs = subroutine(rhoB,Uen,self.Ut)
            probs_t += [probs]
        
        return np.array(probs_t)
    
    
    
    def psr2(self,theta):
        """
        Evaluate the outputs at every time t by shifting the parameter
        theta_i at circuit block t_k, and theta_j at t_l.
        Defined to later compute the 2nd order Parameter Shift Rule.

        Parameters
        ----------
        theta : array
            Trainable parameters.

        Returns
        -------
        O_ps_ps : dictionary
            Outputs after two positive shifts.
        O_ps_ms : dictionary
            Outputs after positive shift at theta_i,t_k 
            and negative at theta_j, t_l.
        O_ms_ps : dictionary
            Outputs after negative shift at theta_i,t_k 
            and positive at theta_j, t_l.
        O_ms_ms : dictionary
            Outputs after two negative shifts.

        Keys' meaning:
            (k,i,l,j) indicating:
                * k -> shifted block at theta_i
                * i -> shifted theta_i
                * l -> shifted block at theta_j
                * j -> shifted theta_j

        """

        O_ps_ps = {}; O_ps_ms = {}; O_ms_ps = {}; O_ms_ms = {}
        ps = +np.pi/2.; ms = -np.pi/2.
        
        ## For i!=j:
        for ish in range(len(theta)):
            for jsh in range(ish):
                for ksh in range(self.nT):
                    for lsh in range(self.nT):
                        if ksh!=lsh: ## for k!=l
                            O_ps_ps[ksh,ish,
                            lsh,jsh] = self.evaluate_w2shift_k_neq_l(theta,ish,
                            jsh,ksh,lsh,ps,ps)
                            O_ps_ms[ksh,ish,
                            lsh,jsh] = self.evaluate_w2shift_k_neq_l(theta,ish,
                            jsh,ksh,lsh,ps,ms)
                            O_ms_ps[ksh,ish,
                            lsh,jsh] = self.evaluate_w2shift_k_neq_l(theta,ish,
                            jsh,ksh,lsh,ms,ps)
                            O_ms_ms[ksh,ish,
                            lsh,jsh] = self.evaluate_w2shift_k_neq_l(theta,ish,
                            jsh,ksh,lsh,ms,ms)
                        else: ## for k=l
                            O_ps_ps[ksh,ish,
                            ksh,jsh] = self.evaluate_w2shift_k_eq_l(theta,ish,
                            jsh,ksh,ksh,ps,ps)
                            O_ps_ms[ksh,ish,
                            ksh,jsh] = self.evaluate_w2shift_k_eq_l(theta,ish,
                            jsh,ksh,ksh,ps,ms)
                            O_ms_ps[ksh,ish,
                            ksh,jsh] = self.evaluate_w2shift_k_eq_l(theta,ish,
                            jsh,ksh,ksh,ms,ps)
                            O_ms_ms[ksh,ish,
                            ksh,jsh] = self.evaluate_w2shift_k_eq_l(theta,ish,
                            jsh,ksh,ksh,ms,ms)
        
        ## For i=j:
        for ish in range(len(theta)):
            for ksh in range(self.nT):
                for lsh in range(ksh): ## for k!=l, symmetry in t
                    O_ps_ps_ikl = self.evaluate_w2shift_k_neq_l(theta,ish,ish,
                                                                ksh,lsh,ps,ps)
                    O_ps_ps[ksh,ish,lsh,ish] = O_ps_ps_ikl
                    O_ps_ps[lsh,ish,ksh,ish] = O_ps_ps_ikl
                    O_ps_ms_ikl = self.evaluate_w2shift_k_neq_l(theta,ish,ish,
                                                                ksh,lsh,ps,ms)
                    O_ps_ms[ksh,ish,lsh,ish] = O_ps_ms_ikl
                    O_ps_ms[lsh,ish,ksh,ish] = O_ps_ms_ikl
                    O_ms_ps_ikl = self.evaluate_w2shift_k_neq_l(theta,ish,ish,
                                                                ksh,lsh,ms,ps)
                    O_ms_ps[ksh,ish,lsh,ish] = O_ms_ps_ikl
                    O_ms_ps[lsh,ish,ksh,ish] = O_ms_ps_ikl
                    O_ms_ms_ikl = self.evaluate_w2shift_k_neq_l(theta,ish,ish,
                                                                ksh,lsh,ms,ms)
                    O_ms_ms[ksh,ish,lsh,ish] = O_ms_ms_ikl
                    O_ms_ms[lsh,ish,ksh,ish] = O_ms_ms_ikl
                
                lsh = ksh        
                O_ps_ps[ksh,ish,ksh,ish] = self.evaluate_w2shift_k_eq_l(theta,
                                                        ish,ish,ksh,ksh,ps,ps)
                O_ps_ms[ksh,ish,ksh,ish] = self.evaluation
                O_ms_ps[ksh,ish,ksh,ish] = self.evaluation
                O_ms_ms[ksh,ish,ksh,ish] = self.evaluate_w2shift_k_eq_l(theta,
                                                        ish,ish,ksh,ksh,ms,ms)
        
        return O_ps_ps, O_ps_ms, O_ms_ps, O_ms_ms
    
    
    
    def hess_psr(self,theta):
        """
        Evaluate the analytical Hessian with the Parameter Shift Rule.
        Only a triangular submatrix is computed, because of the symmetry.
    
        Parameters
        ----------
        theta : array
            Trainable parameters.
    
        Returns
        -------
        hess : dictionary
            Analytical Hessian, where the keys indicate the first and second
            partial derivative.
    
        """
        #print('Call to grad_psr')
        
        O_ps_ps, O_ps_ms, O_ms_ps, O_ms_ms = self.psr2(theta)
        
        hess = {}
        for i in range(len(theta)):
            for j in range(i+1):
                for t in range(self.nT):
                    hess[i,j] = 0
        
        for key,value in O_ps_ps.items():
            k,i,l,j = key
            hess[i,j] += 0.25*value

        for key,value in O_ms_ms.items():
            k,i,l,j = key
            hess[i,j] += 0.25*value
            
        for key,value in O_ps_ms.items():
            k,i,l,j = key
            hess[i,j] -= 0.25*value
        
        for key,value in O_ms_ps.items():
            k,i,l,j = key
            hess[i,j] -= 0.25*value

        self.hessian = hess
        return hess
    
    
    
    def evaluate_Z(self):
        """
        Evaluate the output expectation Z value, from the array of probabili-
        ties.
        Necessary previous evaluation with method evaluate.

        Returns
        -------
        Array of expectation Z value for every time step.

        """
        return np.array([np.dot(self.flips,item) for item in self.evaluation])
    
    
    
    def grad_psr_Z(self):
        """
        Evaluate the analytical gradients of the Z expectation values.
        Necessary previous evaluation with method grad_psr.

        Returns
        -------
        Array of expectation Z value 1st derivative  for every time step.
        Shape is (Ntheta,nT)

        """
        return np.array([[np.dot(self.flips,item1) for item1 in item2] for 
                         item2 in self.gradient])
    
    
    
    def hess_psr_Z(self):
        """
        Evaluate the analytical Hessians of the Z expectation values.
        Necessary previous evaluation with method hess_psr.

        Returns
        -------
        Array of expectation Z value 1st derivative  for every time step.
        Shape is (Ntheta,Ntheta,nT)

        """
        hessZ = np.zeros((len(self.th),len(self.th),self.nT))
        for key,value in self.hessian.items():
            i = key[0]; j = key[1]
            expZs = [np.dot(self.flips,item) for item in value]
            hessZ[i,j] = expZs
            if i!=j: hessZ[j,i] = expZs
        return hessZ