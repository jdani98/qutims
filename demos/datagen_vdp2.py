# -*- coding: utf-8 -*-
"""
 Title: Data generation - Van Der Pol 2 inputs
 Description: Script to generate 'Van der Pol 2 inputs' signals.
 Input series are the original series.
 Output series is a linear combination of both inputs, delaye a few points in
 the future.

Created on Thu Oct 19 11:21:50 2023
@author: jdviqueira
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

def vdp(t, z):
    x, y = z
    return [y, mu*(1 - x**2)*y - x + A*np.sin(omega*t)]


npoints = 1000 # !!! introduce number of points 
a, b = 0, 100  # !!! introduce physical time limits

delay1 = 5 # !!! introduce delay of time between input 1 and output
a1 = a; b1 = b+delay1
delay_points1 = int(npoints/(b-a)*delay1)

delay2 = 18 # !!! introduce delay of time between input 2 and output
a2 = a; b2 = b+delay2
delay_points2 = int(npoints/(b-a)*delay2)

t = np.linspace(a, b, npoints)
t1 = np.linspace(a1, b1, npoints + delay_points1)
t2 = np.linspace(a2, b2, npoints + delay_points2)

# NON-FORCED VAN DER POL OSCILLATOR (signalI0)
mu0 = 2; mu=mu0     # !!! Van der Pol coefficient
A = 0.0             # perturbation amplitude
omega = 2*np.pi/5.  # angular frequency of the perturbation
sol1 = solve_ivp(vdp, [a1, b1], [1, 0], t_eval=t1).y[0]

# NON-FORCED VAN DER POL OSCILLATOR (signalI1)
mu1 = 1; mu=mu1     # !!! Van der Pol coefficient
A = 0.0             # perturbation amplitude
omega = 2*np.pi/5.  # angular frequency of the perturbation
sol2 = solve_ivp(vdp, [a2, b2], [1, 0], t_eval=t2).y[0]


# SIGNALS SUM (signalO)
c1 = 1.0    # !!! weight of signalI0
c2 = 0.1    # !!! weight of signalI1
sol3 = c1*sol1[delay_points1:] + c2*sol2[delay_points2:]


signalI0 = 0.25*sol1[:-delay_points1]
signalI1 = 0.25*sol2[:-delay_points2]
signalO  = 0.25*sol3


tag = ''
if len(tag)>0: tag = '('+tag+')'
name = ('dataset_vdp2_a%ib%i_del0%i_del1%i_mu0%i_mu1%i_%ip%s.dat' 
        %(a,b,delay1,delay2,mu0,mu1,npoints,tag))
np.savetxt('data/'+name, np.vstack((t,signalI0,signalI1,signalO)).T)
print('Van der Pol 2 input series was saved at data/%s.' %name)