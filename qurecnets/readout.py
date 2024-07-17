# -*- coding: utf-8 -*-

"""
 Title: Measurement observables of QRNN.
 Description: This module contains the class to compute the Z-expectation values with and without sampling noise, after having computed the probabilities of measurement in the QRNN circuit.

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



class expectZ(base):
    """Class to evaluate Z-expectation value each time step after QRNN emulator methods emulates and returns the probabilities of measurement.
    """

    def readout(self, probs, shots=None, rseed=None):
        """Compute Z-expectation value.

        Args:
            probs (numpy.array): Probabilities of measurement every time step after QRNN emulation. shape (nT,NE)
            shots (int, optional): shots (int, optional): Number of shots for simulated sampling noise. Defaults to None.
                                   If None, the number is defined in the constructor; else, the argument value is considered.
            rseed (None or int, optional): seed for sampling noise. Defaults to None.
                                           If None, the number is defined in the constructor; else, the argument value is considered.
        Returns:
            numpy.array: Z-expectation value for time in nT time steps.
        """
        if shots is None:
            shots = self.shots
        if rseed is not None:
            np.random.seed(rseed)

        flips = [1,-1]
        for i in range(self.nE-1):
            flips = np.kron(flips, [1,-1])
        flips = np.array(flips)
        expZ = np.array([flips@item for item in probs])

        if not shots:
            return expZ
        if shots:
            sZ = np.sqrt((1.-expZ*expZ)/shots)
            return np.random.normal(expZ,sZ)