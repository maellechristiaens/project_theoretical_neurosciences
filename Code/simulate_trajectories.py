#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SIMULATE TRAJECTORIES
'''

import numpy as np

import parameters
import simulate_stimuli as stimuli

params_sim = parameters.params_sim
theta_test = parameters.theta_test
theta_opt = parameters.theta_opt
N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_sim)
rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)

print('simulate_trajectories imported')


# -----------------------------------------------------------

def accumulator_evolution(a, C, stim_R_t, stim_L_t, lbda=lbda, sgm2a=sgm2a, sgm2s=sgm2s, b=b, dt=dt):
    '''Dynamics of the memory accumulator.
    Method : finite differences integration.
    WARNING     In np.random.normal(loc,scale), scale is the standard deviation and not the variance.
                Therefore, the root of parameters (sgm2a, sgm2s) is taken.
    Inputs : 
    a           Value of the accumulator at time t
    C           Value of the stimulus magnitude at time t
    stim_R_t    Number of pulses occuring during the current time bin.
    stim_L_t
    lbda        Unit : s**(-1)
    sgm2a       Units : q**2/s
    Output : 
    a + da      New value of the accumulator at time t+dt.'''
    if a >= b:
        return b
    elif a <= -b:
        return -b
    else :
        if stim_R_t!=0 and stim_R_t==stim_L_t : # sumultaneous clicks cancel
            input_R = 0
            input_L = 0
        else :
            input_R = sum([np.random.normal(1,np.sqrt(sgm2s)) for k in range(stim_R_t)])
            input_L = sum([np.random.normal(1,np.sqrt(sgm2s)) for k in range(stim_L_t)])
        da = lbda*a*dt + (input_R - input_L)*C + np.random.normal(0,np.sqrt(sgm2a*dt))
        # inputs are not multiplied by dt, as q is the quantity to be added at each time step (in the model equation, implicit in the units of the parameters)
        # 
        return a + da

def adaptation_evolution(C, stim_R_t, stim_L_t, phi=phi, tau_phi=tau_phi, dt=dt):
    '''Dynamics of the adaptation, i.e. change in the stimulus magnitude.
    Method : evaluation of the explicit (deterministic) solution at time point t+dt.
    Inputs : 
        C : value of the stimulus magnitude at time t
        stim_R_t, stim_L_t : number of pulses occuring during the current time bin.
    Output : new value of the stimulus magnitude at t+dt, after adaptation.'''
    inpt = stim_R_t + stim_L_t
    Ceq = 1/(1 - tau_phi*inpt*(phi-1))
    C = Ceq + (C-Ceq)*np.exp(-dt*(1/tau_phi-inpt*(phi-1)))
    return C

def euler(times_r, times_l, dur, lbda=lbda, sgm2a=sgm2a, sgm2s=sgm2s, sgm2i=sgm2i, b=b, phi=phi, tau_phi=tau_phi, dt=dt):
    stim_R, stim_L = stimuli.convert_times_to_array(times_r, times_l, dur, dt)
    T = len(stim_R)
    a_evol = np.zeros(T)
    C_evol = np.zeros(T)
    # Initialisation
    C_evol[0] = 1 # or 0 if simultaneous pulses at t=0 ? not necessary in our model
    a_evol[0] = np.random.normal(0,np.sqrt(sgm2i))
    # Integration
    for t in range(1,T):
        C_evol[t] = adaptation_evolution(C_evol[t-1], stim_R[t], stim_L[t], phi, tau_phi, dt)
        a_evol[t] = accumulator_evolution(a_evol[t-1], C_evol[t-1], stim_R[t], stim_L[t], lbda, sgm2a, sgm2s, b, dt)
    return a_evol, C_evol


# -----------------------------------------------------------

def generate_trajectories(trials, params=params_sim, theta=theta_test, theta_opt=theta_opt, ntraj=2):
    rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    N, Durs, dur, Gamma, gamma, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    trajectories = []
    for i in range(len(trials)):
        times_r, times_l, dur = trials[i]['times_r'], trials[i]['times_l'], trials[i]['dur']
        trajectories.append({'a_evols': [], 'C_evol': None})
        for k in range(ntraj):
            a_evol, C_evol = euler(times_r, times_l, dur, lbda, sgm2a, sgm2s, sgm2i, b, phi, tau_phi, dt)
            trajectories[i]['a_evols'].append(a_evol)
        trajectories[i]['C_evol'] = C_evol
    return trajectories

def decision_making(a, rho=rho, l=l):
    if np.random.random() < l: # decision at random
        if np.random.random() < 0.5:
            return 'L'
        else:
            return 'R'
    else:
        if a < rho:
            return 'L'
        else:
            return 'R'

def generate_data(trials, params=params_sim, theta=theta_test):
    rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    D = trials.copy()
    for k in range(len(trials)):
        a_evol, _ = euler(trials[k]['times_r'], trials[k]['times_l'], trials[k]['dur'], lbda, sgm2a, sgm2s, sgm2i, b, phi, tau_phi, dt)
        d = decision_making(a_evol[-1], rho, l)
        D[k]['d'] = d
    return D

