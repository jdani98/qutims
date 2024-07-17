# -*- coding: utf-8 -*-

"""
 Title: Operator models for encoding and evolution.
 Description: Operator model classes for encoding and evolution. Includes ansatz with linear and multiqubit entanglement, and encoding, both with 2- and 3-parameter rotation gate groups.

Created 06/06/2024
@author: jdviqueira
"""

import numpy as np
from qurecnets.qualgebra import Rx,Ry,Rz


class base:
    def __init__(self,nT,nE,nM,nL,nx, shots=0, rseed=None):
        self.nT = nT
        self.nE = nE
        self.nM = nM
        self.nL = nL
        self.nx = nx

        self.NE = 2**nE
        self.NM = 2**nM

        self.shots = shots
        np.random.seed(rseed)




class CZladder2p1(base):
    def evolve(self,theta):
        """Multilayer Hardware-efficient ansatz. Rotation gates are in 2-parameters groups, |Rx|-|Rz|.
        A layer consists of rotations and a ladder of controlled-Z (CZ) gates. A final layer of Rx is applied before reg. E measurement.
        This class must be inherited by or with emc.emulator class.

        Args:
            thetas (numpy.array or list): trainable parameters. Order: parameters fill the pairs |Rx|-|Rz| in order, first the Rx and then the Rz. Then, the qubits from above.
                                          Finally, they fill the layers in order. First 'encode_Nparams' are not used here but are reserved for encoding.

        Returns:
            numpy.array: 4-rank tensor operator for evolution of the quantum state.
        """

        def U2(sub_thetas):
            # 2-parameter rotation operator
            Uo = Rz(sub_thetas[1])@Rx(sub_thetas[0])
            return Uo
        
        preindex = self.encode_Nparams() # number of parameters used in encoding

        base = [[1,1],[1,-1]]
        csign = [1,1]
        for k in range(self.nE+self.nM-1):
            csign = [base[i%2][j]*item for i,item in enumerate(csign) for j in range(2)]
        csign = np.array(csign)

        
        # // Create operator to apply over all qubits as first layer
        Uab  = np.array([[1]])
        for i in range(self.nE):
            Uab = np.kron(Uab,U2(theta[preindex+2*i:preindex+2*i+2])) # acts on reg. A
        for i in range(self.nE,self.nE+self.nM):
            Uab = np.kron(Uab,U2(theta[preindex+2*i:preindex+2*i+2])) # acts on reg. B
        Uabcx = []
        for csigni,row in zip(csign,Uab):
            Uabcx += [csigni*row]
        Ut = np.array(Uabcx)
        
        
        # // Loop to apply full operator as many times as nlayers. Parameters are different for each
        # layer
        for li in range(1,self.nL):
            Uab  = np.array([[1]])
            for i in range(self.nE):
                Uab = np.kron(Uab,U2(theta[preindex+2*(self.nE+self.nM)*li+2*i:preindex+2*(self.nE+self.nM)*li+2*i+2])) # acts on reg. A
            for i in range(self.nE,self.nE+self.nM):
                Uab = np.kron(Uab,U2(theta[preindex+2*(self.nE+self.nM)*li+2*i:preindex+2*(self.nE+self.nM)*li+2*i+2])) # acts on reg. B
            Uabcx = []
            for csigni,row in zip(csign,Uab):
                Uabcx += [csigni*row]
            Ut = np.dot(Uabcx, Ut)
        # //
        
        
        # // Applying Rx rotations over regA before measurement
        InB = np.diag([1. for i in range(self.NM)]) # identity operator for regB
        Ua  = np.array([[1]])
        findex = preindex+2*(self.nE+self.nM)*self.nL
        for i in range(self.nE):
            Ua = np.kron(Ua,Rx(theta[findex+i])) # A operator in {|0>,|1>} basis, i.e. final rotation over regA
        Uf  = np.kron(Ua,InB) # Final operator in {|0>,|1>} basis,i.e. final 3-rotation over regA
        # //
        
        Ut  = np.dot(Uf,Ut) # !!! MASTER OPERATOR
        
        return np.array([[Ut[self.NM*i:self.NM*(i+1),self.NM*j:self.NM*(j+1)] for i in range(self.NE)] for j in range(self.NE)]) # U divided!



    def evolve_Nparams(self):
        """Number of parameters used in this ansatz.

        Returns:
            int
        """
        return 2*(self.nE+self.nM)*self.nL + self.nE




class CZme3(base):
    def evolve(self, theta):
        """Multilayer high-entanglement (Multi-Entangled) ansatz. Rotation gates are 3-parameters groups |Rx|-|Rz|-|Rx|.
        A layer consists of rotations and controlled-Z (CZ) gates, entangling every qubit in reg. E with every qubit in reg. M.
        A final layer of rotations is applied on reg. E qubits before measurement.
        This class must be inherited by or with emc.emulator class.

        Args:
            thetas (numpy.array or list): trainable parameters. Order: parameters fill the groups |Rx|-|Rz|-|Rx| in order, first the Rx, then the Rz and then the second Rx. Then, the qubits from above.
                                          Finally, they fill the layers in order. First 'encode_Nparams' are not used here but are reserved for encoding.

        Returns:
            numpy.array: 4-rank tensor operator for evolution of the quantum state.
        """

        def U3(sub_thetas):
            # 3-parameter rotation operator
            Uo = Rx(sub_thetas[2])@Rz(sub_thetas[1])@Rx(sub_thetas[0])
            return Uo
        
        preindex = self.encode_Nparams() # number of parameters used in encoding
        # Ansatz-W-generator. Description: se dm.py.
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
        # // Create operator to apply over all qubits as first layer
        Uab  = np.array([[1]])
        for i in range(self.nE):
            Uab = np.kron(Uab,U3(theta[preindex+3*i:preindex+3*i+3]))
        for i in range(self.nE,self.nE+self.nM):
            Uab = np.kron(Uab,U3(theta[preindex+3*i:preindex+3*i+3]))
        Uabcx = []
        for csigni,row in zip(csign,Uab):
            Uabcx += [csigni*row]
        Ut = np.array(Uabcx)
        # // Loop to apply full operator as many times as nlayers. Parameters are different for each
        for li in range(1,self.nL):
            Uab  = np.array([[1]])
            for i in range(self.nE):
                Uab = np.kron(Uab,U3(theta[preindex+3*(self.nE+self.nM)*li+3*i:preindex+3*(self.nE+self.nM)*li+3*i+3]))
            for i in range(self.nE,self.nE+self.nM):
                Uab = np.kron(Uab,U3(theta[preindex+3*(self.nE+self.nM)*li+3*i:preindex+3*(self.nE+self.nM)*li+3*i+3]))
            Uabcx = []
            for csigni,row in zip(csign,Uab):
                Uabcx += [csigni*row]
            Ut = np.dot(Uabcx, Ut)
        # // Applying 3-parameter rotations over regA before measurement
        InB = np.diag([1. for i in range(self.NM)])
        Ua  = np.array([[1]])
        findex = preindex + 3*(self.nE+self.nM)*self.nL
        for i in range(self.nE):
            Ua = np.kron(Ua,U3(theta[findex+3*i:findex+3*i+3]))
        Uf  = np.kron(Ua,InB)
        Ut  = np.dot(Uf,Ut)
        return np.array([[Ut[self.NM*i:self.NM*(i+1),self.NM*j:self.NM*(j+1)] for i in range(self.NE)] for j in range(self.NE)]) # U divided!
    


    def evolve_Nparams(self):
        """Number of parameters used in this ansatz.

        Returns:
            int
        """
        return 3*(self.nE+self.nM)*self.nL + 3*self.nE



class encodeP2(base):
    def encode(self,xt,theta):
        """Encoding ansatz V(xt). Each variable is encoded in a qubit, without entanglement.
        Rotations are 2-parameters groups |Rx|-|Rz|.
        This class must be inherited by or with emc.emulator class.

        Args:
            xt (numpy.array): x_(t) data. 2-rank tensor with shape (nT,nE)
            theta (array): Trainable parameters (all). Order: parameters fill the groups in order, and then by columns acting on different qubits. Finally, layers of data re-uploading.

        Returns
            sA (numpy.array): Encoded statevector.

        """

        def U2(sub_thetas):
            # 2-parameter rotation operator
            Uo = Rz(sub_thetas[1])@Rx(sub_thetas[0])
            return Uo
        
        sA = [1.]
        for j,xj in enumerate(xt):
            encgat = Ry(xj) # encoding gate
            Uin = np.copy(encgat)
            for repi in range(self.nx):
                Uin = encgat@U2(theta[2*self.nx*j+2*repi:2*self.nx*j+2*repi+2])@Uin
            sAj = [Uin[0][0],Uin[1][0]] # initial state of qubit j - regA
            sA = np.kron(sA,sAj)
        
        return sA
    
    
    def encode_Nparams(self):
        """Number of parameters used in this encoding.

        Returns:
            int
        """
        return 2*self.nE*self.nx



class encodeP3(base):
    def encode(self,xt,theta):
        """Encoding ansatz V(xt), with data re-uploading. Each variable is encoded in a qubit, without entanglement.
        Rotations are 3-parameters groups |Rx|-|Rz|-|Rx|.
        This class must be inherited by or with emc.emulator class.

        Args:
            xt (numpy.array): x_(t) data. 2-rank tensor with shape (nT,nE)
            theta (array): Trainable parameters (all). Order: parameters fill the groups in order, and then by columns acting on different qubits. Finally, layers of data re-uploading.

        Returns
            sA (numpy.array): Encoded statevector.

        """

        def U3(sub_thetas):
            # 3-parameter rotation operator
            Uo = Rx(sub_thetas[2])@Rz(sub_thetas[1])@Rx(sub_thetas[0])
            return Uo
        
        sA = [1.]
        for j,xj in enumerate(xt):
            encgat = Ry(xj) # encoding gate
            Uin = np.copy(encgat)
            for repi in range(self.nx):
                Uin = encgat@U3(theta[3*self.nx*j+3*repi:3*self.nx*j+3*repi+3])@Uin
            sAj = [Uin[0][0],Uin[1][0]] # initial state of qubit j - regA
            sA = np.kron(sA,sAj)
        
        return sA
    
    
    def encode_Nparams(self):
        """Number of parameters used in this encoding.

        Returns:
            int
        """
        return 3*self.nE*self.nx