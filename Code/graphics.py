import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
import simulate_stimuli
import simulate_trajectories
import find_best_params

print('imported')

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



####  GRAPHICAL RESULTS ####

def hide_spines(ax, sides=['right', 'top']):
    for side in sides :
            ax.spines[side].set_visible(False)

def hide_ticks(ax, axis):
    if axis == 'x':
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
    if  axis == 'y':
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])

def show_trial(stim_R, stim_L, dt=dt):
    pos_R = [t*dt for t in range(len(stim_R)) if stim_R[t]!=0]
    pos_L = [t*dt for t in range(len(stim_R)) if stim_L[t]!=0]
    plt.eventplot(pos_R, lineoffsets=1, color='red')
    plt.eventplot(pos_L, lineoffsets=-1, color='green')
    plt.xlabel('Time (s)')

def show_trajectory(a_history, C_history, stim_R, stim_L, b=b, rho=rho, dt=dt):
    times = [t*dt for t in range(len(stim_R))]
    pos_R = [t*dt for t in range(len(stim_R)) if stim_R[t]!=0]
    pos_L = [t*dt for t in range(len(stim_R)) if stim_L[t]!=0]
    stim_L_y = [b for t in range(len(pos_L))]
    stim_R_y = [-b for t in range(len(pos_R))]

    fig, ax = plt.subplots(2)
    ax[0].plot(times, a_history, color='k')
    ax[0].scatter(pos_R, stim_R_y, color='red', marker='^')
    ax[0].scatter(pos_L, stim_L_y, color='green', marker='v')
    ax[0].axhline(rho, color='gray', linestyle='dashed')
    ax[0].set_ylabel('Accumulator value, a')
    hide_ticks(ax[0], 'x')
    hide_spines(ax[0], sides=['right', 'top','bottom'])
    ax[1].plot(times, C_history, color='gray')
    ax[1].set_ylabel('Adapatation, C', color='gray')
    hide_spines(ax[1], sides=['right', 'top'])
    ax[1].set_xlabel('Time (s)')
    plt.show()