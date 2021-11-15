import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random

print('imported')


######## PARAMETERS ########

#We focus here on simulating the stimulus of the rats

###Fixed parameters
N = 40 # average number of pulses (n_pulses/s)
q = 1 # unit of evidence, magnitude of one click without adaptation (psi = 1)
# an ideal observer adds one q at every right pulse and subtracts one q at every left pulse
dt = 0.02 # time step for simulations (in s)
dx = 0.25*q # bin size for space discretization (values of vector x)
isi = 0 #the minimum interval time between 2 stimuli
sim_params = {
            'q': q,
            'N': N,
            'dt': dt,
            'dx': dx,
            }

###Other parameters

dur = np.random.uniform(0.1, 1.2) #a trial has a duration in [0, 1.2] (in sec)
gamma = 0.5 #the difficulty of the trial (the more gamma is low, the more difficult it is)

######## STIMULI ########

def rates_of_clicks(gamma):
    '''
    MC 04/11/21
    Input : gamma: the difficulty of the trial
    Output : the rates of right and left clicks
    '''
    ""
    r1 = 40/(1+10**gamma)
    r2 = 40 - r1
    return [r1, r2]


def clicks(gamma=gamma, dur = dur, dt = dt):
    '''
    MC 04/11/21
    Input : gamma: the difficulty of the trial
            dur: the duration of this trial
            dt: the time interval on which to compute if there was a stimuli or not

    Output : a boolean array of right and left clicks for this trial and its duration
    '''
    l = rates_of_clicks(gamma) #generate the rates of clicks
    [rR, rL] = random.sample(l, 2) #pick randomly which size corresponds
                                   #to which rate (otherwise it's always the
                                   #right size that has more clicks)
    nL = scipy.stats.poisson(rL*dur).rvs()#Number of left clicks
    nR = scipy.stats.poisson(rR*dur).rvs()#Number of right clicks
    times_l = np.sort((dur*scipy.stats.uniform.rvs(0,1,((nL,1)))).T[0])
    #timestamps of left clicks
    times_r = np.sort((dur*scipy.stats.uniform.rvs(0,1,((nR,1)))).T[0])
    #timestamps of right clicks
    stim_R = [any((times_l>(i-dt/2))*(times_l<(i+dt/2))) for i in np.arange(dt/2, dur, dt)]
    stim_L = [any((times_l>(i-dt/2))*(times_l<(i+dt/2))) for i in np.arange(dt/2, dur, dt)]
    #boolean array for times where there's a stimuli (L or R)
    return stim_R, stim_L, dur
