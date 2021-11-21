#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SIMULATE TRAJECTORIES
'''

import numpy as np

import parameters

params_sim = parameters.params_sim
theta_test = parameters.theta_test
theta_opt = parameters.theta_opt
N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_sim)
rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)

print('simulate_trajectories imported')


# -----------------------------------------------------------

def accumulator_evolution(a, C, stim_R_t, stim_L_t, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, b=b, dt=dt, q=q):
    '''Dynamics of the memory accumulator.
    Inputs : 
        a : value of the accumulator at time t
        C : value of the stimulus magnitude at time t
        t : time step during the course simulation
        stim_R, stim_L : stimulation sequences, containing the number of pulses occurring during each time step
    Output : new value of the accumulator at t+dt.'''
    if stim_R_t!=0 and stim_R_t==stim_L_t : # sumultaneous clicks cancel
        # print('cancelation')
        return a
    if a >= b:
        return b
    elif a <= -b:
        return -b
    else :
        input_R = sum([q*np.random.normal(1,sgm_s) for k in range(stim_R_t)])
        input_L = sum([q*np.random.normal(1,sgm_s) for k in range(stim_L_t)])
        da = lbda*a*dt + (input_R - input_L)*C + np.random.normal(0,sgm_a*q*dt**0.5)
        # inputs are not multiplied by dt, as q is the quantity to be added at each time step (in the model equation, implicit in the units of the parameters)
        # sgm_a**2 in units of q**2/sec -> SDT = sgm_a*q*dt**0.5
        return a + da

def adaptation_evolution(C, stim_R_t, stim_L_t, phi=phi, tau_phi=tau_phi, dt=dt):
    '''Dynamics of the adaptation, i.e. change in the stimulus magnitude.
    Inputs : 
        C : value of the stimulus magnitude at time t
        t : time step during the simulation
        stim_R, stim_L : stimulation sequences, containing the number of pulses occurring during each time step
    Output : new value of the stimulus magnitude at t+dt, after adaptation.'''
    dC = ((1-C)/tau_phi + (phi-1)*C*(stim_R_t+stim_L_t))*dt
    return C + dC

def euler(stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, sgm_i=sgm_i, b=b, phi=phi, tau_phi=tau_phi, dt=dt, q=q):
    T = len(stim_R)
    a_evol = np.zeros(T)
    C_evol = np.zeros(T)
    # initialisation
    C_evol[0] = 1 # or 0 if simultaneous pulses at t=0 ? not necessary in our model
    a_evol[0] = np.random.normal(0,sgm_i)
    for t in range(1,T):
        C_evol[t] = adaptation_evolution(C_evol[t-1], stim_R[t], stim_L[t], phi, tau_phi, dt)
        a_evol[t] = accumulator_evolution(a_evol[t-1], C_evol[t-1], stim_R[t], stim_L[t], lbda, sgm_a, sgm_s, b, dt, q)
    return a_evol, C_evol


# -----------------------------------------------------------

def generate_trajectories(trials, params=params_sim, theta=theta_test, theta_opt=theta_opt, ntraj=2):
    rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    trajectories = []
    for i in range(len(trials)):
        stim_R, stim_L = trials[i]['stim_R'], trials[i]['stim_L']
        trajectories.append({'a_evols': [], 'C_evol': None})
        for k in range(ntraj):
            a_evol, C_evol = euler(stim_R, stim_L, lbda, sgm_a, sgm_s, sgm_i, b, phi, tau_phi, dt, q)
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
    rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    D = []
    for trial in trials:
        stim_R, stim_L = trial['stim_R'], trial['stim_L']
        a_evol, _ = euler(stim_R, stim_L, params, theta)
        D.append(decision_making(a_evol[-1], rho, l))
    return D

