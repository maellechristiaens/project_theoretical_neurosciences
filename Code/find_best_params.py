import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
import simulate_stimuli
import simulate_trajectories

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


######## FOKKER PLANK METHOD ########

def discretize_space(b=b, dx=dx):
    X = [-k for k in np.arange(0,b,dx)][:0:-1] + [k for k in np.arange(0,b,dx)]
    w = 2*(b - X[-1])
    X = [X[0]-w] + X + [X[-1]+w]
    return np.array(X)

def discretize_gaussian(sgm_min, sgm_max, dx=dx, scale_sgm=10, scale_dx=2, nsgm=4):
    if dx < sgm_min :
        ds = dx/scale_dx
    else :
        dx = sgm_min/scale_sgm # bin size for discretizing the gaussian probability (values of s)
    Ds = nsgm*sgm_max # maximum distance from the mean for discretizing the gaussian probability
    return ds, Ds

def compute_nbins(ds, Ds):
    if (Ds/ds)%1 != 0 :
        nbins_S = int(2*Ds/ds)+1
    else :
        nbins_S = int(2*Ds/ds)-1
    return nbins_S

def total_input(C_history, stim_R, stim_L):
    CR_history = C_history*stim_R
    CL_history = C_history*stim_L
    c_input = CR_history - CL_history
    return CR_history, CL_history, c_input

def total_variance(CR_history, CL_history, sgm_a=sgm_a, sgm_s=sgm_s, dt=dt):
    sgm_dt_input = (sgm_a**2*dt + (CR_history + CL_history)*sgm_s**2)**0.5 # already takes into account dt**0.5
    return sgm_dt_input

def deterministic_drift(x, c, lbda=lbda, dt=dt):
    m = np.exp(lbda*dt)*(x + c/lbda) - c/lbda
    return m

def gaussian(s, m, sgm):
    return (1/(2*np.pi*sgm**2)**0.5)*np.exp(-(s-m)**2/(2*sgm**2))

def positions(x, m, ds, Ds):
    s = np.array([m-k for k in np.arange(0,Ds,ds)][:0:-1] + [m+k for k in np.arange(0,Ds,ds)]) # discretization of space for approximating the probability distribution
    return s

def ornstein_uhlenbeck_process(X, c_input, sgm_dt_input, lbda=lbda, dt=dt, dx=dx, scale_sgm=10, scale_dx=2, nsgm=4, ds=None, Ds=None):
    T = len(c_input)
    nbins_X = len(X) # number of positions, including 2 extra bins outside boundaries
    if ds is None:
        ds, Ds = discretize_gaussian(np.min(sgm_dt_input), np.max(sgm_dt_input), dx, scale_sgm, scale_dx, nsgm)
    s0 = np.array([k for k in np.arange(0,Ds,ds)][:0:-1] + [k for k in np.arange(0,Ds,ds)])
    nbins_S = len(s0)
    print('dx', dx, 'ds', ds, 'Ds', Ds)
    S = np.zeros((nbins_X, nbins_S, T))
    P_s = np.zeros((nbins_X, nbins_S, T))
    for t in range(T):
        for j in range(1,nbins_X-1):
            S[j,:,t] = positions(X[j], deterministic_drift(X[j], c_input[t], lbda, dt), ds, Ds)
            P_s[j,:,t] = gaussian(s0, 0, sgm_dt_input[t])*ds # P([s, s+ds]) = p(s)ds -> multiply by ds
    S[0,:,:] = X[0]  # positions outside bound assigned to b (after computing the probability distribution for ease)
    S[-1,:,:] = X[-1]
    print(S.shape)
    return S, P_s

def split_mass_low(s, x_low, x_up):
    norm = x_up - x_low
    p_low = (x_up - s)/norm
    return p_low

def split_mass_up(s, x_low, x_up):
    norm = x_up - x_low
    p_up = (s - x_low)/norm
    return p_up

def forward_transition_matrix(X, S, P_s):
    nbins_X = len(X)
    T = S.shape[2]
    F = np.zeros((nbins_X,nbins_X,T))
    # transient states
    for i in range(1,nbins_X-1):
        print(i)
        cond_low = (S>X[i-1]) & (S<X[i])
        cond_up = (S>X[i]) & (S<X[i+1])
        cond_eq = (S==X[i])
        S_low = S*cond_low
        S_up = S*cond_up
        S_eq = S*cond_eq
        P_s_low = P_s*cond_low
        P_s_up = P_s*cond_up
        P_s_eq = P_s*cond_eq
        W_low = split_mass_low(S_up, X[i], X[i+1])
        W_up = split_mass_up(S_low, X[i-1], X[i])
        P_tot = np.sum(W_up*P_s_low, axis=1) + np.sum(W_low*P_s_up, axis=1) + np.sum(P_s_eq, axis=1)
        F[i,:,:] = P_tot
    # absorbant state i = 0
    cond_low = (S<=X[0])
    cond_up = (S>X[0]) & (S<X[1])
    S_low = S*cond_low
    S_up = S*cond_up
    P_s_low = P_s*cond_low
    P_s_up = P_s*cond_up
    W_low = split_mass_low(S_up, X[0], X[1])
    P_tot = np.sum(P_s_low, axis=1) + np.sum(W_low*P_s_up, axis=1)
    F[0,:,:] = P_tot
    F[0,0,:] = 1 # absorbant state
    # absorbant state i = nbins_X-1
    cond_low = (S>X[-2]) & (S<X[-1])
    cond_up = (S>=X[-1])
    S_low = S*cond_low
    S_up = S*cond_up
    P_s_low = P_s*cond_low
    P_s_up = P_s*cond_up
    W_up = split_mass_up(S_low, X[-2], X[-1])
    P_tot = np.sum(W_up*P_s_low, axis=1) + np.sum(P_s_up, axis=1)
    F[-1,:,:] = P_tot
    F[-1,-1,:] = 1
    return F

def fokker_plank(F, X, sgm_i=sgm_i, dx=dx):
    nbins_X = len(X)
    T = F.shape[2]
    P_a = np.zeros((nbins_X,T))
    if sgm_i != 0 :
        P_a[:,0] = gaussian(X, 0, sgm_i)*dx
    else:
        P_a[:,0][X==0] = 1
    for t in range(1,T):
        P_a[:,t] = np.dot(F[:,:,t],P_a[:,t-1])
    return P_a

def proba_left(P_a, X, rho=rho, dx=dx):
    return np.sum(P_a[X<=rho,-1]*dx)

def solve_numerical(stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, sgm_i=sgm_i, b=b, rho=rho, psi=psi, tau_psi=tau_psi, dt=dt, dx=dx, scale_sgm=10, scale_dx=2, nsgm=4, ds=None, Ds=None):
    X = discretize_space(b, dx)
    C_history = euler(stim_R, stim_L, lbda, sgm_a, sgm_s, sgm_i, b, psi, tau_psi, dt)[1]
    CR_history, CL_history, c_input = total_input(C_history, stim_R, stim_L)
    sgm_dt_input = total_variance(CR_history, CL_history, sgm_a, sgm_s, dt)
    print('Ornstein Uhlenbeck')
    S, P_s = ornstein_uhlenbeck_process(X, c_input, sgm_dt_input, lbda, dt, dx, scale_sgm, scale_dx, nsgm, ds, Ds)
    print('Forward transition matrix')
    F = forward_transition_matrix(X, S, P_s)
    print('Fokker Plank')
    P_a = fokker_plank(F, X, sgm_i, dx)
    d = proba_left(P_a, X, rho, dx)
    return X, F, P_a, d