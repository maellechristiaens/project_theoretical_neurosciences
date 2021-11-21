#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PARAMETERS

>> Free parameters, to be fitted to data
rho     Biais, threshold to output a decision by comparing the value of a at the end of the trial.
b       Sticky decision bound, amount of evidence necessary to commit to a decision.
l       Lapse rate, fraction of trials on which a random response is made.
sgm_i   Noise in the initial value of a.
sgm_a   Diffusion constant, noise in a.
sgm_s   Noise when adding evidence from one pulse (scaled by C amplitude).
lbda    Consistent drift in a.
        - leaky or forgetful case : lbda < 0 (drift toward a = 0, and later pulses affect the decision more than earlier pulses).
        - unstable or impulsive case : lbda > 0 (drift away from a = 0, and earlier pulses affect the decision more than later pulses).
tau     Memory time constant, tau = abs(1/lbda)
phi     Adaptation strength, factor by which C is multiplied following a pulse.
        - facilitation : phi > 1
        - depression : phi < 1
tau_phi Adaptation time constant for C recovery to its unadapted value, 1.
NB : In the article, noise parameters are defined as sgm_i**2, sgm_a**2, sgm_s**2 (variance instead of standard deviation).

>> Stimulation parameters
N       Average number of pulses (n_pulses/s).
Durs    List of stimuli durations (in s).
Gamma   List of stimuli difficulties, defined as abs(log(r_r/r_l)).

>> Integration parameters
q       Unit of evidence, magnitude of one click without adaptation (phi = 1).
        An ideal observer adds one q at every right pulse and subtracts one q at every left pulse.
dt      Time step for simulations (in s).
dx      Bin size for space discretization (values of vector x).
nbs     Number of bins to discretize the gaussian.
dphi    Bin size for finite difference method.
dtphi   Idem.
'''

import numpy as np

print('parameters imported')

# -----------------------------------------------------------
### Fixed parameters

N = 40
Durs = [0.1, 0.25, 0.5, 0.75, 1, 1.2]
Gamma = [0.5, 1.2, 2.5, 4]
gamma = Gamma[1] # default for tests
dur = Durs[1] # default for tests
dt = 0.002 # for Euler method (arbitrary), 0.02 for Fokker Plank method (article)
q = 1 
dx = 0.25*q
nbs = 451 # +1 for the center of the gaussian
dphi = (2.5-0.1)/10
dtphi = (1-0.005)/10

params_sim = {'N': N,
            'Durs': Durs,
            'dur': dur,
            'Gamma': Gamma,
            'gamma': gamma,
            'q': q,
            'dt': dt,
            'dx': dx,
            'nbs': nbs,
            'dphi': dphi,
            'dtphi': dtphi}
params_fkp = {'N': N,
            'Durs': Durs,
            'dur': dur,
            'Gamma': Gamma,
            'gamma': gamma,
            'q': q,
            'dt': 0.02,
            'dx': dx,
            'nbs': nbs,
            'dphi': dphi,
            'dtphi': dtphi}

def extract_fixed_params(params_sim=params_sim):
    '''Define global variables for the whole code.
    Input : params_sim, dictionary containing the set of paramters of one model.'''
    N = params_sim['N']
    Durs = params_sim['Durs']
    dur = params_sim['dur']
    Gamma = params_sim['Gamma']
    gamma = params_sim['gamma']
    q = params_sim['q']
    dt = params_sim['dt'] 
    dx = params_sim['dx']
    nbs = params_sim['nbs']
    dphi = params_sim['dphi']
    dtphi = params_sim['dtphi']
    return N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi

N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = extract_fixed_params(params_sim)


# -----------------------------------------------------------
### Free parameters, to be fitted to data 

theta_test = {'lbda': 1, # for Euler method
            'b': 30,
            'sgm_a': 70**0.5,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'phi': 1.2,
            'tau_phi': 0.04,
            'rho': 0,
            'l': 0}
theta_test1 = {'lbda': 1, # for Euler method
            'b': 30,
            'sgm_a': 0**0.5,
            'sgm_s': 1**0.5,
            'sgm_i': 0,
            'phi': 1.2,
            'tau_phi': 0.04,
            'rho': 0,
            'l': 0}
theta_test2 = {'lbda': 1, # for Euler method
            'b': 30,
            'sgm_a': 70**0.5,
            'sgm_s': 0**0.5,
            'sgm_i': 0,
            'phi': 1.2,
            'tau_phi': 0.04,
            'rho': 0,
            'l': 0}
theta_fkp = {'lbda': 1, # for Fokker Plank method
            'b': 30,
            'sgm_a': 0.5**0.5,
            'sgm_s': 0.5**0.5,
            'sgm_i': 0.5**0.5,
            'phi': 1,
            'tau_phi': 1,
            'rho': 0.5,
            'l': 0.1}
theta_opt = {'lbda': 0,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 0,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': 0.1,
            'rho': 0,
            'l': 0}

thetaA = {'lbda': 0,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'phi': 0.34,
            'tau_phi': 0.04,
            'rho': 0,
            'l': 0}
thetaB = {'lbda': -20,
            'b': 4,
            'sgm_a': 0,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': None,
            'rho': 0,
            'l': 0}
thetaC = {'lbda': -10,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 140**0.5,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': None,
            'rho': 0,
            'l': 0}
thetaD = {'lbda': 10,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 140**0.5,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': None,
            'rho': 0,
            'l': 0}
thetaE = {'lbda': 0,
            'b': 30,
            'sgm_a': 0,
            'sgm_s': 140**0.5,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': None,
            'rho': 0,
            'l': 0}
thetaF = {'lbda': 0,
            'b': 30,
            'sgm_a': 140**0.5,
            'sgm_s': 0,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': None,
            'rho': 0,
            'l': 0}
thetaG = {'lbda': 0,
            'b': 30,
            'sgm_a': 70**0.5,
            'sgm_s': 70**0.5,
            'sgm_i': 0,
            'phi': 1,
            'tau_phi': None,
            'rho': 0,
            'l': 0}

models = {'accumulator_depression': thetaA,
        'burst_detector': thetaB,
        'leaky_accumulator': thetaC,
        'unstable_accumulator': thetaD,
        'accumulator_all_sensory_noise': thetaE,
        'accumulator_all_accumulating_noise': thetaF,
        'accumulator_mixed_sensory_accumulating_noise': thetaG}


def extract_free_params(theta=theta_test):
    '''Define global variables for the whole code.
    Input : theta, dictionary containing the set of paramters of one model.'''
    rho = theta['rho'] 
    b = theta['b']
    l = theta['l']
    sgm_i = theta['sgm_i']
    sgm_a = theta['sgm_a']
    sgm_s = theta['sgm_s']
    lbda = theta['lbda']
    if lbda != 0:
        tau = abs(1/lbda)
    phi = theta['phi']
    tau_phi = theta['tau_phi']
    return rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi

rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = extract_free_params(theta_test)


