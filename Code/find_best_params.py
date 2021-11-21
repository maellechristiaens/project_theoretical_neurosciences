#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PARAMETER FITTING
'''

import numpy as np

import parameters
import simulate_trajectories as traj
import numerical_fokker_plank as fkp


params_sim = parameters.params_sim
theta_test = parameters.theta_test
N, Durs, dur, Gamma, gamma, q, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_sim)
rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)

print('find_best_params imported')


# -----------------------------------------------------------

def backward_transition_matrix(F, Pa, check=True, renorm=True):
    nbx = F.shape[0]
    T = F.shape[2]
    B = np.zeros((nbx,nbx,T-1)) # initialize to 0 : values replacing *unproblematic* divisions by 0 (i.e. Pj = 0 but either Pi = 0 or Ft = 0)
    # T-1 : B goes from t=1 to t=T, whereas F starts at t=0 (and the last time is not used)
    Ft = np.transpose(F.copy(), axes=(1,0,2)) # permutation on dimensions 0 and 1
    Pi = Pa.copy()[:,np.newaxis,:]
    Pj = Pa.copy()[np.newaxis,:,:]
    Pjcorr = Pj.copy()
    Pjcorr[Pj==0] = 1 # avoid division by 0
    B = Ft[:,:,1:]*Pi[:,:,:-1]/Pjcorr[:,:,1:] # translation by one time step
    
    null_cols = np.argwhere(np.sum(B, axis=0)==0) # detect columns which sum to 0
    for a, t in null_cols:
        B[a,a,t] = 1 # approximate dirac ?

    if renorm:
        Bs = np.sum(B, axis=0) # dim0: xi, dim1: T
        Bs = Bs[np.newaxis,:,:]
        B /= Bs

    if check: 
        print('Maximum value in B :', np.max(B))
        print('Sum across lines :\n',  np.sum(B, axis=0))
        # detail *problematic* divisions by 0 (i.e. Pj = 0 and neither Pi = 0 nor Ft = 0)
        test = (Pj[:,:,1:]==0)*(Pi[:,:,:-1]!=0)*(Ft[:,:,1:]!=0)
        print('Number of problematic divisions by 0 :', sum(test[test==1]))
        ii, jj, tt = np.where(test==1)
        for i, j, t in zip(ii, jj,tt):
            print('Pi = ', Pi[i,0,t], 'Ft = ', Ft[i,j,t])
        print('Negligible -> replaced by 0.')

    return B


def initialize_posterior_distribution(d, PaT, X, dx=dx, rho=rho, renorm=True):
    P_inter = PaT*((X<=rho)*(d=='L') + (X>rho)*(d=='R'))
    pb0 = P_inter/np.sum(P_inter)
    if renorm:
        pb0 /= np.sum(pb0)
    return pb0

def posterior_probability_distribution(pb0, B):
    nbx = B.shape[0]
    T = B.shape[2]+1 # B goes from t=1 to t=T
    Pb = np.zeros((nbx,T))
    Pb[:,-1] = pb0
    for t in np.arange(1,T-1) : 
        Pb[:,-t-1] = np.dot(B[:,:,-t],Pb[:,-t])
    return Pb


def gradient_L(Pa, Pb, dF_dtheta):
    nbx = Pa.shape[0]
    T = Pa.shape[1]
    Pbj = Pb[np.newaxis,:,:]
    Pai = Pa[:,np.newaxis,:]
    Paj = Pa[np.newaxis,:,:]
    Paicorr = Pai.copy()
    Paicorr[Pai==0] = 1 # avoid division by 0
    G = Paj[:,:,:-1]/Paicorr[:,:,1:]*Pbj[:,:,1:]*dF[:,:,1:]
    dL = np.sum(G) # too small or too large values ?
    print(np.max(G))
    print(np.min(G))
    return dL

# -----------------------------------------------------------

def finite_differences(stim_R, stim_L, phi=phi, dphi=dphi, tau_phi=tau_phi, dtphi=dtphi, q=q):
    cR0, cL0 = fkp.input_pulses(stim_R, stim_L, phi, tau_phi, dt, q)
    cR1, cL1 = fkp.input_pulses(stim_R, stim_L, phi+dphi, tau_phi, dt, q) # phi -> phi+dphi
    cR2, cL2 = fkp.input_pulses(stim_R, stim_L, phi, tau_phi+dtphi, dt, q) # tau_phi -> tau_phi+dtphi
    dcR_dphi = (cR1-cR0)/dphi 
    dcL_dphi = (cL1-cL0)/dphi
    dcR_dtphi = (cR2-cR0)/dtphi 
    dcL_dtphi = (cL2-cL0)/dtphi
    return cR0, cL0, dcR_dphi, dcL_dphi, dcR_dtphi, dcL_dtphi # length T

def derivatives_ds(S, X, stim_R, stim_L, dcR_dphi, dcL_dphi, dcR_dtphi, dcL_dtphi, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, nbs=nbs, dt=dt):
    T = len(stim_R)
    nbx = len(X)
    
    Xx = X[:,np.newaxis,np.newaxis] # length nbx
    k = np.arange(-int(nbs/2), int(nbs/2)+1) # length nbs
    k = k[np.newaxis,:,np.newaxis]
    cR, cL, dcR_dphi, dcL_dphi, dcR_dtphi, dcL_dtphi = finite_differences(stim_R, stim_L, phi, dphi, tau_phi, dtphi, q)
    cR = cR[np.newaxis,np.newaxis,:] # length T
    cL = cL[np.newaxis,np.newaxis,:]
    sgmdt_in = fkp.total_variance(cR, cL, sgm_a, sgm_s, dt)

    ds_dlbda = dt*np.exp(lbda*dt)*(Xx+(cR-cL)/lbda) # dim (nbx, T)

    ds_dsgmdt_in = 8*k/nbs # length nbs
    dsgmdt_in_dsgm_a = sgm_a*dt/sgmdt_in**0.5 # length T
    dsgmdt_in_dsgm_s = sgm_a*(cR+cL)/sgmdt_in**0.5
    ds_dsgm_a = ds_dsgmdt_in*dsgmdt_in_dsgm_a # dims (nbs, T)
    ds_dsgm_s = ds_dsgmdt_in*dsgmdt_in_dsgm_s

    dm_dc = 1/lbda*(np.exp(lbda*dt) - 1) # scalar
    dsgmdt_in_dc = sgm_s**2/(2*sgmdt_in**0.5) # length T
    ds_dcR = dm_dc + 8*k*dsgmdt_in_dc # dims (nbs, T)
    ds_dcL = -dm_dc + 8*k*dsgmdt_in_dc # dm_dcR = -dm_dcL = dm_dc and dsgmdt_in_dcR = dsgmdt_in_dcL = dsgmdt_in_dc
    ds_dphi = ds_dcR*dcR_dphi + ds_dcL*dcL_dphi 
    ds_dtphi = ds_dcR*dcR_dtphi + ds_dcL*dcL_dtphi 

    dxL_db = -2 # scalars
    dxR_db = 2
    xbR, xbL, xR, xL = X[-1], X[0], X[-2], X[1]
    ds_dxR = np.abs(S-xR)/(xbR-xR)*(S>xR)*(S<b) # only affects variables at borders
    ds_dxL = np.abs(xL-S)/(xL-xbL)*(S>-b)*(S<xL) # dim (nbx, nbs, T)
    ds_db = ds_dxR*dxR_db + ds_dxL*dxL_db 

    return ds_dlbda, ds_dsgm_a, ds_dsgm_s, ds_dphi, ds_dtphi, ds_db


def gradient_F(dF_ds, X, stim_R, stim_L, phi=phi, dphi=dphi, tau_phi =tau_phi, dtphi=dtphi, q=q):
    cR, cL, dcR_dphi, dcL_dphi, dcR_dtphi, dcL_dtphi = finite_differences(stim_R, stim_L, phi, dphi, tau_phi, dtphi, q)
    S, Ps = fkp.ornstein_uhlenbeck_process(X, cR, cL, lbda, dt, dx, nbs, renormalize=True, check=True)

    ds_dlbda, ds_dsgm_a, ds_dsgm_s, ds_dphi, ds_dtphi, ds_db = derivatives_ds(S, X, stim_R, stim_L, dcR_dphi, dcL_dphi, dcR_dtphi, dcL_dtphi, lbda, sgm_a, sgm_s, nbs, dt)

    dF_dlbda = ds_dlbda*dF_ds
    dF_dsgm_a = ds_dsgm_a*dF_ds
    dF_dsgm_s = ds_dsgm_s*dF_ds
    dF_dphi = ds_dphi*dF_ds
    dF_dtphi = ds_dtphi*dF_ds
    dF_db = ds_db*dF_ds
    return dF_dlbda, dF_dsgm_a, dF_dsgm_s, dF_dphi, dF_dtphi, dF_db



# -----------------------------------------------------------

def log_likelihood(D, trials, theta=parameters.theta_test, dt=dt, q=q, dx=dx, scale_sgm=10, scale_dx=2, nsgm=4, ds=None, Ds=None):
    rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    L = 0
    for k in range(len(trials)):
        stim_R, stim_L = trials[k]['stim_R'], trials[k]['stim_L']
        X, F, Pa, p_L = traj.solve_fokker_plank(stim_R, stim_L, lbda, sgm_a, sgm_s, sgm_i, b, rho, phi, tau_phi, dt, q, dx, scale_sgm, scale_dx, nsgm, ds, Ds)
        if D[k] == 'L':
            L += np.log(p_L)
        else:
            L += np.log(1-p_L)
    return L
