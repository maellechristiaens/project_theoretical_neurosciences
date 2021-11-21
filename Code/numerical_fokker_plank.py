#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
FOKKER PLANK METHOD
'''


import numpy as np
import matplotlib.pyplot as plt

import parameters
import simulate_trajectories as traj

params_fkp = parameters.params_fkp
theta_fkp = parameters.theta_fkp
N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_fkp)
rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta_fkp)

print('numerical_fokker_plank imported')


# -----------------------------------------------------------

def discretize_space(b=b, dx=dx):
    X = [-k for k in np.arange(0,b,dx)][:0:-1] + [k for k in np.arange(0,b,dx)]
    w = 2*(b - X[-1])
    X = [X[0]-w] + X + [X[-1]+w]
    return np.array(X)

def euler_C(stim_R, stim_L, phi=phi, tau_phi=tau_phi, dt=dt, q=q):
    T = len(stim_R)
    C_evol = np.zeros(T)
    C_evol[0] = 1
    for t in range(1,T):
        C_evol[t] = traj.adaptation_evolution(C_evol[t-1], stim_R[t], stim_L[t], phi, tau_phi, dt)
    return C_evol

def input_pulses(stim_R, stim_L, phi, tau_phi, dt, q):
    C_evol = euler_C(stim_R, stim_L, phi, tau_phi, dt, q)
    cR = C_evol*stim_R 
    cL = C_evol*stim_L
    return cR, cL

def total_input(cR, cL):
    return cR - cL

def total_variance(cR, cL, sgm_a=sgm_a, sgm_s=sgm_s, dt=dt):
    sgmdt_in = (sgm_a**2*dt + (cR+cL)*sgm_s**2)**0.5 # already takes into account dt**0.5
    return sgmdt_in

def deterministic_drift(x, c, lbda=lbda, dt=dt):
    m = np.exp(lbda*dt)*(x + c/lbda) - c/lbda
    return m

def gaussian(k, nbs=nbs):
    p0 = 8/(nbs*(2*np.pi)**0.5)*np.exp(-1/2*(8*k/nbs)**2)
    return p0

def positions(sgm, k, nbs=nbs):
    ds = 8*sgm/nbs
    s0 = k*ds
    return s0


def ornstein_uhlenbeck_process(X, cR, cL, lbda=lbda, dt=dt, dx=dx, nbs=nbs, renormalize=True, check=True):
    T = len(cR)
    nbx = len(X) # number of accumulator values, including 2 extra bins outside boundaries
    
    k = np.arange(-int(nbs/2), int(nbs/2)+1)
    c_in = total_input(cR, cL)
    sgmdt_in = total_variance(cR, cL, sgm_a, sgm_s, dt)

    Xx = X[:,np.newaxis,np.newaxis]
    K = k[np.newaxis,:,np.newaxis]
    C = c_in[np.newaxis,np.newaxis,:]
    SGM = sgmdt_in[np.newaxis,np.newaxis,:]

    M = deterministic_drift(Xx, C, lbda, dt)
    S0 = positions(SGM, K, nbs)
    S = M + S0

    ps = gaussian(k, nbs)
    Ps = ps[np.newaxis,:,np.newaxis]*np.ones(nbx)[:,np.newaxis,np.newaxis]
    if renormalize :
        norm = np.sum(Ps[0,:,0])
        Ps /= norm
    Ps[0,1:,:] = 0
    Ps[0,0,:] = 1
    Ps[-1,:-1,:] = 0
    Ps[-1,-1,:] = 1

    if check :
        print('Sum Ps : ', np.sum(Ps[:,:,0], axis=1))
        # plt.plot(C[2,0,:])
        # plt.show()
    return S, Ps

# -----------------------------------------------------------

def split_mass_low(s, x_low, x_up):
    norm = x_up - x_low
    p_low = (x_up - s)/norm
    return p_low

def split_mass_up(s, x_low, x_up):
    norm = x_up - x_low
    p_up = (s - x_low)/norm
    return p_up


def forward_transition_matrix(X, S, Ps, check=False, gradient=False):
    nbx = len(X)
    T = S.shape[2]
    F = np.zeros((nbx,nbx,T))
    dF_ds = np.zeros((nbx,nbx,T))
    # P0 = Ps[0,:,0][np.newaxis,:,np.newaxis]
    
    print('Intervals and Masks')
    X_low = np.zeros(S.shape)
    X_up = np.zeros(S.shape)
    for i in range(0,nbx-1): # start at 0 this time
        # RAM overload : do not store masks, since each one contains ~13 000 000 values
        # print(i)
        mask = (S>=X[i])&(S<X[i+1]) # here >= has no importance (only for filling the array)
        X_low[mask] = X[i]
        X_up[mask] = X[i+1]
    mask_b_low = (S<X[0])
    X_low[mask_b_low] = -1 # specific value with no importance (avoid dividing by 0 when computing W)
    X_up[mask_b_low] = X[0]
    mask_b_up = (S>=X[-1])
    X_low[mask_b_up] = X[-1]
    X_up[mask_b_up] = -1 # specific value idem

    print('Weights')
    W_low = split_mass_low(S, X_low, X_up)
    W_up = split_mass_up(S, X_low, X_up)
    P_up = W_up*Ps
    P_low = W_low*Ps
    if gradient:
        Q = Ps/(X_up-X_low)

    print('Transient states')
    # Transient states
    for i in range(1,nbx-1): 
        print(i)
        mask_low = (S>X[i-1]) & (S<X[i]) # here < and > are required
        mask_up = (S>X[i]) & (S<X[i+1])
        mask_eq = (S==X[i])
        F[i,:,:] = np.sum(P_up*mask_low, axis=1) + np.sum(P_low*mask_up, axis=1) + np.sum(Ps*mask_eq, axis=1)
        # opposite up and low : if s is below x (mask low), it is affected upwards (i.e. to x) with probability P_up
        if gradient:
            dF_ds[i,:,:] = np.sum(Q*mask_low, axis=1) - np.sum(Q*mask_up, axis=1) + np.sum(Ps*mask_eq, axis=1)

    print('Absorbant states')
    # Absorbant state i = 0
    mask_low = (S<=X[0]) # takes into account the equality case
    mask_up = (S>X[0]) & (S<X[1])
    F[0,:,:] = np.sum(Ps*mask_low, axis=1) + np.sum(P_low*mask_up, axis=1)
    # unweighted Ps for all values below the lowest position
    F[0,0,:] = 1 # ensure absorbant state
    if gradient:
        dF_ds[0,:,:] = np.sum(Ps*mask_low, axis=1) + np.sum(Q*mask_up, axis=1)
    
    # Absorbant state i = nbx-1
    mask_low = (S>X[-2]) & (S<X[-1])
    mask_up = (S>=X[-1])
    F[-1,:,:] = np.sum(P_up*mask_low, axis=1) + np.sum(Ps*mask_up, axis=1)
    # unweighted Ps for all values above the uppest position
    F[-1,-1,:] = 1 # ensure absorbant state
    if gradient:
        dF_ds[-1,:,:] = np.sum(Q*mask_low, axis=1) + np.sum(Ps*mask_up, axis=1)

    return F, dF_ds


# -----------------------------------------------------------

def initialize_prior_distribution(X, sgm_i=sgm_i, dx=dx):
    if sgm_i != 0 :
        p0 = 1/((2*np.pi)**0.5*sgm_i)*np.exp(-1/2*(X/sgm_i)**2)*dx
    else:
        p0 = np.zeros(len(X))
        p0[X==0] = 1
    return p0

def prior_probability_distribution(p0, F):
    nbx = F.shape[0]
    T = F.shape[2]
    Pa = np.zeros((nbx,T))
    Pa[:,0] = p0
    for t in range(1,T):
        Pa[:,t] = np.dot(F[:,:,t],Pa[:,t-1])
    return Pa

def proba_left(Pa, X, rho=rho, dx=dx):
    return np.sum(Pa[X<=rho,-1]*dx)

def solve_fokker_plank(stim_R, stim_L, params=params_fkp, theta=theta_fkp, gradient=False):
    rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    X = discretize_space(b, dx)
    print('Inputs history')
    cR, cL = input_pulses(stim_R, stim_L, phi, tau_phi, dt, q)
    print('Ornstein Uhlenbeck')
    S, Ps = ornstein_uhlenbeck_process(X, cR, cL, lbda, dt, dx, nbs)
    print('Forward transition matrix')
    F, dF_ds = forward_transition_matrix(X, S, Ps, gradient=gradient)
    print('Prior probability distribution')
    p0 = initialize_prior_distribution(X, sgm_i, dx)
    Pa = prior_probability_distribution(p0, F)
    p_L = proba_left(Pa, X, rho, dx)
    return X, F, Pa, p_L, dF_ds