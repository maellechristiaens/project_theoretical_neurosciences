#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
STIMULI
'''

import numpy as np
import scipy.stats
import random

import parameters

params_sim = parameters.params_sim
theta_test = parameters.theta_test
N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_sim)
rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)


print('simulated_stimuli imported')

# -----------------------------------------------------------

def rates_of_clicks(gamma=gamma, N=N):
    '''
    MC 04/11/21
    Input : gamma: difficulty of the trial
    Output : rates of right and left clicks
    '''
    r1 = N/(1+10**gamma)
    r2 = N - r1
    return [r1, r2]


def clicks(gamma=gamma, dur=dur):
    '''
    MC 04/11/21
    Input : gamma: difficulty of the trial
            dur: duration of this trial
    Output : arrays containing the times of pulses.
    '''
    l = rates_of_clicks(gamma) # generate click rates
    [rR, rL] = random.sample(l, 2) #pick randomly which size corresponds to which rate (otherwise always the right size has more clicks)
    nL = scipy.stats.poisson(rL*dur).rvs() # number of left clicks
    nR = scipy.stats.poisson(rR*dur).rvs() # number of right clicks
    times_l = np.sort((dur*scipy.stats.uniform.rvs(0,1,((nL,1)))).T[0]) # timestamps of left clicks
    times_r = np.sort((dur*scipy.stats.uniform.rvs(0,1,((nR,1)))).T[0]) # timestamps of right clicks
    return times_r, times_l

def convert_times_to_array(times_r, times_l, dur=dur, dt=dt):
    '''
    Converts the sequence of stimulation times into a time-discretized sequence.
    Input : 
        times_r, times_l : times of pulses.
        dur: duration of this trial
        dt: the time interval on which to compute if there was a stimuli or not
    Output : 
        stim_R, stim_L : arrays containing the number of pulses occurring during each time bin (L or R)
    '''
    stim_L = np.array([np.sum((times_l>=t)*(times_l<t+dt)) for t in np.arange(0, dur, dt)])
    stim_R = np.array([np.sum((times_r>=t)*(times_r<t+dt)) for t in np.arange(0, dur, dt)])
    return stim_R, stim_L


# -----------------------------------------------------------

def generate_experiment(params=params_sim, ntrials=10, gamma_seq=None, dur_seq=None):
    N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    trials = []
    if gamma_seq is None:
        gamma_seq = [gamma for k in range(ntrials)]
    if dur_seq is None:
        dur_seq = [dur for k in range(ntrials)]
    for k in range(ntrials):
        times_r, times_l = clicks(gamma, dur)
        trials.append({'gamma': gamma, 
                        'dur': dur, 
                        'times_r': times_r,
                        'times_l': times_l})
    return trials

