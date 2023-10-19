# -*- coding: utf-8 -*-
"""
 Title: Data generation - DECreasing TRIANGular
 Description: Script to generate 'decreasing triangular' signals.
 Input series is the original series.
 Output series is the same series, a few points in the future.

Created on Thu Oct 19 10:49:44 2023
@author: jdviqueira
"""

import numpy as np


def ftf2(t,period,cte):
    """
    Triangular function of integer time frames.

    Parameters
    ----------
    t : integer
        Time variable.
    period : integer
        Period of the triangular function.
    cte : float
        Scaling factor.

    Returns
    -------
    float
        Value of triangular wave in the specified instant t.

    """
    c = 0.5
    semiperiod = period/2
    slope = 1/semiperiod
    tp = t%period
    ph = tp//semiperiod
    tsp = tp%semiperiod
    y = (-c + slope*tsp)*(-1)**ph
    return 2*cte*y


npoints = 1000 # !!! introduce number of points 
a, b = 0, 100  # !!! introduce physical time limits

delay1 = 12 # !!! introduce delay of time between input and output
a1 = a; b1 = b+delay1
delay_points1 = int(npoints/(b-a)*delay1)

t = np.linspace(a, b, npoints)
t1 = np.linspace(a1, b1, npoints + delay_points1)

lamda = 0.02 # !!! decreasing coefficient
P = 5        # !!! period
amp = 0.75   # !!! amplitude
sol = np.exp(-lamda*t1)*ftf2(t1,P,amp)

signalI = sol[:-delay_points1]
signalO = sol[delay_points1:]

tag = ''
if len(tag)>0: tag = '('+tag+')'
name = 'dataset_dectriang_a%ib%i_del%i_%ip%s.dat' %(a,b,delay1,npoints,tag)
np.savetxt('data/'+name, np.vstack((t,signalI,signalO)).T)
print('Decreasing triangular series was saved at data/%s.' %name)