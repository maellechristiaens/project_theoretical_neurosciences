import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
import simulate_stimuli

print('imported')


######## PARAMETERS ########

### Fixed parameters ###

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


### Parameters to fit to data ###

def extract_params(theta):
    '''Define global variables for the whole code.
    Input : theta, dictionary containing the set of paramters of one model.'''
    rho = theta['rho'] # biais, threshold to output a decision by comparing the value of a at the end of the trial
    b = theta['b'] # sticky decision bound, amount of evidence necessary to commit to a decision
    l = theta['l'] # lapse rate, fraction of trials on which a random response is made
    sgm_i = theta['sgm_i'] # noise in the initial value of a
    sgm_a = theta['sgm_a'] # diffusion constant, noise in a
    sgm_s = theta['sgm_s'] # noise when adding evidence from one pulse (scaled by C amplitude)
    lbda = theta['lbda'] # consistent drift in a
    # leaky or forgetful case : lbda < 0 (drift toward a = 0, and later pulses affect the decision more than earlier pulses)
    # unstable or impulsive case : lbda > 0 (drift away from a = 0, and earlier pulses affect the decision more than later pulses)
    if lbda != 0:
        tau = abs(1/lbda) # memory time constant
    psi = theta['psi'] # adaptation strength, factor by which C is multiplied following a pulse
    # facilitation : psi > 1
    # depression : psi < 1
    tau_psi = theta['tau_psi'] # adaptation time constant for C recovery to its unadapted value, 1
    return rho, b, l, sgm_i, sgm_a, sgm_s, lbda, psi, tau_psi

paramsA = {'lbda': 0,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'psi': 0.34,
            'tau_psi': 0.04,
            'rho': 0,
            'l': 0}
paramsB = {'lbda': -20,
            'b': 4,
            'sgm_a': 0,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}
paramsC = {'lbda': -10,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 140**0.5,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}
paramsD = {'lbda': 10,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 140**0.5,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}
paramsE = {'lbda': 0,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 140**0.5,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}
paramsF = {'lbda': 0,
            'b': 30,
            'sgm_a': 140**0.5,
            'sgm_s': 0,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}
paramsG = {'lbda': 0,
            'b': 30,
            'sgm_a': 70**0.5,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}

params_test = {'lbda': 0.1,
            'b': 30,
            'sgm_a': 70**0.5,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'psi': 1,
            'tau_psi': 1, # Na ?
            'rho': 0,
            'l': 0}

models = {'accumulator_depression': paramsA,
        'burst_detector': paramsB,
        'leaky_accumulator': paramsC,
        'unstable_accumulator': paramsD,
        'accumulator_all_sensory_noise': paramsE,
        'accumulator_all_accumulating_noise': paramsF,
        'accumulator_mixed_sensory_accumulating_noise': paramsG}

rho, b, l, sgm_i, sgm_a, sgm_s, lbda, psi, tau_psi = extract_params(params_test)


######## EULER METHOD ########

def accumulator_evolution(a, C, t, stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, b=b, dt=dt):
    '''Dynamics of the memory accumulator.
    Inputs :
        a : value of the accumulator at time t
        C : value of the stimulus magnitude at time t
        t : time step during the course simulation
        stim_R, stim_L : stimulation sequences, containing the number of pulses occurring during each time step
    Output : new value of the accumulator at t+dt.'''
    if a >= b:
        return b
    elif a <= -b:
        return -b
    else :
        da = lbda*a*dt + (stim_R[t]*np.random.normal(1,sgm_s) + stim_L[t]*np.random.normal(1,sgm_s))*C*dt + np.random.normal(0,sgm_a*dt**0.5)
        return a + da

def adaptation_evolution(C, t, stim_R, stim_L, psi=psi, tau_psi=tau_psi, dt=dt):
    '''Dynamics of the adaptation, i.e. change in the stimulus magnitude.
    Inputs :
        C : value of the stimulus magnitude at time t
        t : time step during the simulation
        stim_R, stim_L : stimulation sequences, containing the number of pulses occurring during each time step
    Output : new value of the stimulus magnitude at t+dt, after adaptation.'''
    if stim_R[t]!=0 and stim_R[t]==stim_R[t] : # sumultaneous clicks cancel
        return 0
    else:
        dC = ((1-C)/tau_psi + (psi-1)*C*abs((stim_R[t]-stim_L[t])))*dt
        return C + dC

def euler(stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, sgm_i=sgm_i, b=b, psi=psi, tau_psi=tau_psi, dt=dt):
    T = len(stim_R)
    a_history = np.zeros(T)
    C_history = np.zeros(T)
    # initialisation
    C_history[0] = 0 # simultaneous pulses
    a_history[0] = np.random.normal(0,sgm_i)
    for t in range(1,T):
        C_history[t] = adaptation_evolution(C_history[t-1], t, stim_R, stim_L, psi, tau_psi, dt)
        a_history[t] = accumulator_evolution(a_history[t-1], C_history[t-1], t, stim_R, stim_L, lbda, sgm_a, sgm_s, b, dt)
    return a_history, C_history