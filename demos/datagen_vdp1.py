# -*- coding: utf-8 -*-
"""
 Title: Data generation - Van Der Pol 1 input
 Description: Script to generate 'Van der Pol 1 input' signals.
 Input series is the original series.
 Output series is the same series, a few points in the future.

Created on Thu Oct 19 11:12:27 2023
@author: jdviqueira
"""

import numpy as np
from scipy.integrate import solve_ivp

def vdp(t, z):
    x, y = z
    return [y, mu*(1 - x**2)*y - x + A*np.sin(omega*t)]


npoints = 1000 # !!! introduce number of points 
a, b = 0, 100  # !!! introduce physical time limits

delay1 = 15 # !!! introduce delay of time between input and output
a1 = a; b1 = b+delay1
delay_points1 = int(npoints/(b-a)*delay1)


t = np.linspace(a, b, npoints)
t1 = np.linspace(a1, b1, npoints + delay_points1)

# FORCED VAN DER POL OSCILLATOR (signalI)
mu = 2              # !!! Van der Pol coefficient
A = 1.0             # !!! perturbation amplitude
omega = 2*np.pi/3.  # !!! angular frequency of the perturbation
sol1 = solve_ivp(vdp, [a1, b1], [1, 0], t_eval=t1).y[0]


# FORWARDED SIGNAL (signalO)
sol3 = sol1[delay_points1:]


signalI = 0.25*sol1[:-delay_points1]
signalO  = 0.25*sol3


tag = ''
if len(tag)>0: tag = '('+tag+')'
name = 'dataset_vdp1_a%ib%i_del%i_mu%i_%ip%s.dat' %(a,b,delay1,mu,npoints,tag)
np.savetxt('data/'+name, np.vstack((t,signalI,signalO)).T)
print('Van der Pol 1 input series was saved at data/%s.' %name)