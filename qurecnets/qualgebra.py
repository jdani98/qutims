# -*- coding: utf-8 -*-
"""
 Title: QUantum ALGEBRA
 Description: this is a python module which contains several common quantum states, gates to
 build quantum gate-based circuits and operations over quantum matrices. Fully based in NumPy.

Created on 17/02/2023
@author: jdviqueira

Copyright 2024 José Daniel Viqueira
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
"""

import numpy as np


# Quantum states as numpy column vectors
def zero():
    # zero state |0>
    return np.array([[1.],[0.]])

def one():
    # one state |1>
    return np.array([[0.],[1.]])

def plus():
    # plus state |+> = (|0>+|1>)/√2
    return (1./np.sqrt(2.))*np.array([[1.],[1.]])

def minus():
    # minus state |-> = (|0>-|1>)/√2
    return (1./np.sqrt(2.))*np.array([[1.],[-1.]])


# Quantum gates as numpy matrices
def H():
    # Hadamard gate
    m = (1./np.sqrt(2.))*np.array([[1,1],[1,-1]])
    return m

def X():
    # NOT 1-qubit gate
    m = np.array([[0.,1.],[1.,0.]])
    return m

def Rx(th):
    # Rotation around X-axis of Bloch sphere for 1 qubit
    m = np.array([[np.cos(th/2.),-1.j*np.sin(th/2)],[-1.j*np.sin(th/2),np.cos(th/2.)]])
    return m

def Ry(th):
    # Rotation around Y-axis of Bloch sphere for 1 qubit
    m = np.array([[np.cos(th/2.),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2.)]])
    return m

def Rz(th):
    # Rotation around Z-axis of Bloch sphere for 1 qubit
    m = np.array([[np.cos(th/2.)-1.j*np.sin(th/2.),0],[0,np.cos(th/2.)+1.j*np.sin(th/2.)]])
    return m

def U(th,ph,la):
    # General U3 1-qubit gate
    m = np.array([[np.cos(th/2.),                             -(np.cos(la)+1.j*np.sin(la))*np.sin(th/2.)],
                  [(np.cos(ph)+1.j*np.sin(ph))*np.sin(th/2.), (np.cos(la+ph)+1.j*np.sin(la+ph))*np.cos(th/2.)]])
    return m

def I():
    # Identity gate
    return np.array([1.,0.],[0.,1.])


# Operations over quantum matrices
def hermitian(m):
    #defining hermitian conjugate if an operator
    return m.conj().T