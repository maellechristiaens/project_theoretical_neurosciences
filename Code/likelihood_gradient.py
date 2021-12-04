#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
COMPUTATION OF THE GRADIENT
'''

import numpy as np

import parameters
import simulate_trajectories as traj
import likelihood_fokker_plank as ll_fkp


params_fkp = parameters.params_fkp
theta_test = parameters.theta_test
N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_fkp)
rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)

print('likelihood_gradient imported')


# -----------------------------------------------------------

def finite_differences(times_r, times_l, dur, CR0, CL0, phi=phi, tau_phi=tau_phi, dphi=dphi, dtphi=dtphi, dt=dt, tpulse=tpulse):
    '''
    Approximate the derivative of the input with respects to phi and tau_phi, by the finite difference method.
    WARNING     For better precision, the function input_pulses() integrates automatically with a time step dt_euler = dt/10.
    '''
    CR1, CL1 = ll_fkp.input_pulses(times_r, times_l, dur, phi+dphi, tau_phi, dt, tpulse) # phi -> phi+dphi
    CR2, CL2 = ll_fkp.input_pulses(times_r, times_l, dur, phi, tau_phi+dtphi, dt, tpulse) # tau_phi -> tau_phi+dtphi
    dCR_dphi = (CR1-CR0)/dphi 
    dCL_dphi = (CL1-CL0)/dphi
    dCR_dtphi = (CR2-CR0)/dtphi 
    dCL_dtphi = (CL2-CL0)/dtphi
    return dCR_dphi, dCL_dphi, dCR_dtphi, dCL_dtphi # length T

def derivatives_ds_dtheta(times_r, times_l, dur, CR, CL, X, S, m, lbda=lbda, sgm2a=sgm2a, sgm2s=sgm2s, phi=phi, tau_phi=tau_phi, b=b, dphi=dphi, dtphi=dtphi, dt=dt, tpulse=tpulse):
    T = len(CR)
    nbx = len(X)
    # Format to perform array broadcasting
    Xd = X[:,np.newaxis,np.newaxis] # dimensions (nbx,1,1)
    CRd = CR[np.newaxis,np.newaxis,:] # dim (1,1,T)
    CLd = CL[np.newaxis,np.newaxis,:]
    sgm2dt = ll_fkp.total_variance(CR, CL, sgm2a, sgm2s, dt) 
    SGM = sgm2dt[np.newaxis,np.newaxis,:] # dim (1,1,T)

    ds_dlbda = dt*np.exp(lbda*dt)*(Xd+(CRd-CLd)/lbda) # dim (nbx,1,T)

    ds_dsgm2dt = (S - m)/(2*SGM) # dim (nbx,nbs,T)
    dsgm2dt_dsgm2a = dt # scalar
    dsgm2dt_dsgm2s = CRd + CLd # dim (1,1,T)
    ds_dsgm2a = ds_dsgm2dt*dsgm2dt_dsgm2a # dim (nbx,nbs,T)
    ds_dsgm2s = ds_dsgm2dt*dsgm2dt_dsgm2s # dim (nbx,nbs,T)

    dm_dc = 1/lbda*(np.exp(lbda*dt) - 1) # scalar
    dCR_dphi, dCL_dphi, dCR_dtphi, dCL_dtphi = finite_differences(times_r, times_l, dur, CR, CL, phi, tau_phi, dphi, dtphi, dt, tpulse)
    dc_dphi = (dCR_dphi - dCL_dphi)/dt # dim (1,1,T)    divide by dt or not ?
    dc_dtphi = (dCR_dtphi - dCL_dtphi)/dt # idem
    dsgm2dt_dphi = (dCR_dphi + dCL_dphi)*sgm2s # dim(1,1,T)
    dsgm2dt_dtphi = (dCR_dtphi + dCL_dtphi)*sgm2s # idem
    ds_dphi = dm_dc*dc_dphi + ds_dsgm2dt*dsgm2dt_dphi # dim (1,1,T)
    ds_dtphi = dm_dc*dc_dtphi + ds_dsgm2dt*dsgm2dt_dtphi # idem

    xRout, xLout, xRin, xLin = X[-1], X[0], X[-2], X[1]
    SR = np.ma.masked_where((S>xRin)&(S<b), S) # db only affects variables at borders
    SL = np.ma.masked_where((S>-b)&(S<xLin), S)
    dxL_db = -2 # scalar
    dxR_db = 2
    ds_dxR = np.abs(SR-xRin)/(xRout-xRin) # dim (nbx,nbs,T) 
    ds_dxL = np.abs(xLin-SL)/(xLin-xLout)
    ds_db = ds_dxR*dxR_db + ds_dxL*dxL_db 

    return ds_dlbda, ds_dsgm2a, ds_dsgm2s, ds_dphi, ds_dtphi, ds_db

def weight_factor(S, X_low, X_up):
    # If x_low == x_up, set w_low = 1 and w_up = 0 (avoid counting twice)
    W = np.ones(S.shape)
    norm = X_up - X_low
    mask = (X_low!=X_up) # avoid division by 0
    W[mask] = 1/norm[mask]
    return W

def F_gradient(times_r, times_l, dur, X, lbda=lbda, sgm2a=sgm2a, sgm2s=sgm2s, phi=phi, tau_phi=tau_phi, b=b, dphi=dphi, dtphi=dtphi, dt=dt, dx=dx, tpulse=tpulse):
    '''
    Computes the matrix dF_dtheta for each parameter theta.
    '''
    CR, CL = ll_fkp.input_pulses(times_r, times_l, dur, phi, tau_phi, dt, tpulse)
    c = ll_fkp.total_input(CR, CL, dt)
    sgm2dt = ll_fkp.total_variance(CR, CL, sgm2a, sgm2s, dt)
    s0 = ll_fkp.discretize_gaussian(sgm2dt, dx)
    m = ll_fkp.deterministic_drift(X, c, lbda, dt)
    ps = ll_fkp.gaussian(s0, sgm2dt)
    S = ll_fkp.ornstein_uhlenbeck_process(m, s0)
    I_low, I_up, X_low, X_up = ll_fkp.attribute_closest(S, X, dx)
    W = weight_factor(S, X_low, X_up)

    dF_ds = ps*W
    ds_dlbda, ds_dsgm2a, ds_dsgm2s, ds_dphi, ds_dtphi, ds_db = derivatives_ds_dtheta(times_r, times_l, dur, CR, CL, X, S, m, lbda, sgm2a, sgm2s, phi, tau_phi, b, dphi, dtphi, dt, tpulse)

    G_dlbda = dF_ds*ds_dlbda
    G_dsgm2a = dF_ds*ds_dsgm2a
    G_dsgm2s = dF_ds*ds_dsgm2s
    G_dphi = dF_ds*ds_dphi
    G_dtphi = dF_ds*ds_dtphi
    G_db = dF_ds*ds_db

    nbx, nbs, T = S.shape
    dF_dlbda = np.zeros((nbx,nbx,T))
    dF_dsgm2a = np.zeros((nbx,nbx,T))
    dF_dsgm2s = np.zeros((nbx,nbx,T))
    dF_dphi = np.zeros((nbx,nbx,T))
    dF_dtphi = np.zeros((nbx,nbx,T))
    dF_db = np.zeros((nbx,nbx,T))
    print('Computing lines')
    for i in range(nbx):
        print(i)
        dF_dlbda [i,:,:] = np.sum(np.ma.masked_where(I_low!=i, -G_dlbda), axis=1) + np.sum(np.ma.masked_where(I_up!=i, G_dlbda), axis=1)
        dF_dsgm2a [i,:,:] = np.sum(np.ma.masked_where(I_low!=i, -G_dsgm2a), axis=1) + np.sum(np.ma.masked_where(I_up!=i, G_dsgm2s), axis=1)
        dF_dsgm2s [i,:,:] = np.sum(np.ma.masked_where(I_low!=i, -G_dsgm2s), axis=1) + np.sum(np.ma.masked_where(I_up!=i, G_dlbda), axis=1)
        dF_dphi [i,:,:] = np.sum(np.ma.masked_where(I_low!=i, -G_dphi), axis=1) + np.sum(np.ma.masked_where(I_up!=i, G_dphi), axis=1)
        dF_dtphi [i,:,:] = np.sum(np.ma.masked_where(I_low!=i, -G_dtphi), axis=1) + np.sum(np.ma.masked_where(I_up!=i, G_dtphi), axis=1)
        dF_db [i,:,:] = np.sum(np.ma.masked_where(I_low!=i, -G_db), axis=1) + np.sum(np.ma.masked_where(I_up!=i, G_db), axis=1)
    return dF_dlbda, dF_dsgm2a, dF_dsgm2s, dF_dphi, dF_dtphi, dF_db


# -----------------------------------------------------------

def B_matrix(F, Pa, check=False, renorm=True):
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
        B[a,a,t] = 1 # approximate dirac 

    if renorm:
        Bs = np.sum(B, axis=0) # dim0: xi, dim1: T
        Bs = Bs[np.newaxis,:,:]
        B /= Bs
    check_B_matrix(check, B, Ft, Pi, Pj)
    return B

def check_B_matrix(check, B, Ft, Pi, Pj):
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
    

def initialize_posterior_distribution(d, PaT, X, rho=rho, renorm=True):
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

# -----------------------------------------------------------

def LL_gradient_theta(Pa, Pb, dF):
    '''
    Computes one component of the gradient of the log-lokelihood (i.e. with respects to one parameter).
    '''
    nbx = Pa.shape[0]
    T = Pa.shape[1]
    Pbj = Pb[np.newaxis,:,:]
    Pai = Pa[:,np.newaxis,:]
    Paj = Pa[np.newaxis,:,:]
    Paicorr = Pai.copy()
    Paicorr[Pai==0] = 1 # avoid division by 0
    G = Paj[:,:,:-1]/Paicorr[:,:,1:]*Pbj[:,:,1:]*dF[:,:,1:]
    dLL = np.sum(G) # sum over all times and positions
    return dLL

def LL_gradient(D, lbda, sgm2a, sgm2s, phi, tau_phi, b, sgm2i=sgm2i, rho=rho, dphi=dphi, dtphi=dtphi, dx=dx, dt=dt, tpulse=tpulse):
    '''
    Implements the whole computation of the gradient of the log-likelihood..
    Input : 
    D       Data, dictionary containing, for each trial :
            stim_r, stim_l  Sequences of times of input stimuli.
            dur             Duration of the trial.
            d               Decision at the end of each trial ('L' left, 'R' right)
    Output :
    dLL     Vector containing the partial derivatives of the log-likelihood with respects to each parameter.
    '''
    print('lbda, sgm2a, sgm2s, phi, tau_phi, b = ', lbda, sgm2a, sgm2s, phi, tau_phi, b)
    dLL = np.zeros(6) # 6 parameters to be optimized
    for k in range(len(D)):
        X = ll_fkp.discretize_space(b, dx)

        print('Forward pass')
        F = ll_fkp.F_matrix(D[k]['times_r'], D[k]['times_l'], D[k]['dur'], lbda, b, sgm2a, sgm2s, phi, tau_phi, dt, dx)
        pa0 = ll_fkp.initialize_prior_distribution(X, sgm2i)
        Pa = ll_fkp.prior_probability_distribution(pa0, F)
        
        print('Backward pass')
        B = B_matrix(F, Pa, renorm=True)
        pb0 = initialize_posterior_distribution(D[k]['d'], Pa[:,-1], X, rho, renorm=True)
        Pb = posterior_probability_distribution(pb0, B)

        print('Gradient')
        dF_dlbda, dF_dsgm2a, dF_dsgm2s, dF_dphi, dF_dtphi, dF_db = F_gradient(D[k]['times_r'], D[k]['times_l'], D[k]['dur'], X, lbda, sgm2a, sgm2s, phi, tau_phi, b, dphi, dtphi, dt, dx, tpulse)
        dLL_dlbda = LL_gradient_theta(Pa, Pb, dF_dlbda)
        dLL_dsgm2a = LL_gradient_theta(Pa, Pb, dF_dsgm2a)
        dLL_dsgm2s = LL_gradient_theta(Pa, Pb, dF_dsgm2s)
        dLL_dphi = LL_gradient_theta(Pa, Pb, dF_dphi)
        dLL_dtphi = LL_gradient_theta(Pa, Pb, dF_dtphi)
        dLL_db = LL_gradient_theta(Pa, Pb, dF_db)

        dLL += np.array([dLL_dlbda, dLL_dsgm2a, dLL_dsgm2s, dLL_dphi, dLL_dtphi, dLL_db])
    return dLL