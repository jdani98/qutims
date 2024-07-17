# -*- coding: utf-8 -*-

"""
 Title: Loss functions.
 Description: This module contains the class to compute MSE loss function and its derivative.
Created 06/06/2024
@author: jdviqueira
"""

#import sys
#sys.path.append('/mnt/netapp1/Store_CESGA/home/cesga/jdviqueira/myfiles/qutims_dev')
import numpy as np


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


class mse(base):
    """Class for Mean Squared Error Loss function.
    """
    def loss(self, yb,y,Nwout):
        """Loss function. Mean Squared Error of prediction and targets.

        Args:
            yb (numpy.array): Output series evaluated from QRNN. shape (nT,)
            y (numpy.array): Target series. shape (nT,)
            Nwout (int): Number of points to predict. Loss is computed with yb[-Nwout:]-y[-Nwout:]

        Returns:
            float: MSE Loss function
        """
        return (1./Nwout)*np.sum([(ybi-yi)**2 for ybi,yi in zip(yb[-Nwout:],y[-Nwout:])])



    def loss_deriv(self, yb,y,partials,Nwout):
        """Loss function derivative. Mean Squared Error of prediction and targets.

        Args:
            yb (numpy.array): Output series evaluated from QRNN. shape (nT,)
            y (numpy.array): Target series. shape (nT,)
            partials (numpy.array): Series of partial derivatives of expectation values every time step.
            Nwout (int): Number of points to predict. Loss derivative is computed with yb[-Nwout:]-y[-Nwout:]

        Returns:
            float: MSE Loss function derivative
        """
        return (1./Nwout)*np.sum(2.*(yb-y)[-Nwout:]*partials[-Nwout:])