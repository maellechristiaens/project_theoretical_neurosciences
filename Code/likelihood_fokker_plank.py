#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
FOKKER PLANK METHOD
'''


import numpy as np
import matplotlib.pyplot as plt

import parameters
import simulate_stimuli as stimuli
import simulate_trajectories as traj

params_fkp = parameters.params_fkp
theta_fkp = parameters.theta_fkp
N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_fkp)
rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta_fkp)

dt_fkp = 0.02
dt = dt_fkp

print('likelihood_fokker_plank imported')


# -----------------------------------------------------------

def euler_C(times_r, times_l, dur, phi=phi, tau_phi=tau_phi, dt=dt):
	'''
	Computes the deterministic evolution of C, by integrating with better precision.
	Typically : dt_euler = dt_fkp/10.
	'''
	dt_euler = dt/10
	stim_R, stim_L = stimuli.convert_times_to_array(times_r, times_l, dur, dt_euler)
	T = len(stim_R)
	C_evol = np.zeros(T)
	C_evol[0] = 1
	for t in range(1,T):
		C_evol[t] = traj.adaptation_evolution(C_evol[t-1], stim_R[t], stim_L[t], phi, tau_phi, dt_euler)
	return C_evol, stim_R, stim_L

def input_pulses(times_r, times_l, dur, phi=phi, tau_phi=tau_phi, dt=dt, tpulse=tpulse):
	'''
	Compresses C history in bins adapted to Fokker Plank time discretization.
	'''
	CL = np.array([0 for t in np.arange(0, dur, dt)]) # discretized time course of C
	CR = np.array([0 for t in np.arange(0, dur, dt)])
	C = 1 # initialization
	all_times = np.sort(np.concatenate((times_r, times_l)))
	n_pulses = len(all_times)

	ipulse = 0 # index of time pulse
	for tbin in range(len(CL)): # index of time bin in the discretized time course of C
		t = tbin*dt # time corresponding to the left border of the bin
		while (ipulse<n_pulses) and (all_times[ipulse]>=t) and (all_times[ipulse]<t+dt): # if time pule i falls into the current bin
			if ipulse == 0:
				delta_t = all_times[0] # delay between trial start and first pulse
			elif ipulse == n_pulses-1:
				delta_t = t + dt - all_times[-1] - tpulse # delay between the end of the last pulse and beginning of the following time bin
			else :
				delta_t = all_times[ipulse] - all_times[ipulse-1] # ISI, delay between successive pulses
			C = traj.adaptation_evolution(C, 0, 0, phi, tau_phi, delta_t-tpulse) # decay from the previous input
			C = traj.adaptation_evolution(C, 1, 0, phi, tau_phi, tpulse) # adapatation due to the pulse
			if all_times[ipulse] in times_l : # add input to the current time bin in the appropriate steam
				CL[tbin] += C
			else:
				CR[tbin] += C
			ipulse += 1
	return CR, CL

def total_input(CR, CL, dt=dt):
	'''
	Output :
	c           Unit : q/s
	'''
	return (CR - CL)/dt

def total_variance(CR, CL, sgm2a=sgm2a, sgm2s=sgm2s, dt=dt):
	'''
	Input :
	CR, CL      Sums of C values for right/left input pulses occuring in each time bin.
	sgm2a       Unit : q^2/s -> multiplied by dt.
	sgm2s       Unit : q^2 -> not multiplied by dt.
	Output : 
	sgm2dt      Variance of the gaussian, sgm^2.dt
	'''
	sgm2dt = sgm2a*dt + (CR+CL)*sgm2s
	return sgm2dt


# -----------------------------------------------------------

def discretize_space(b, dx):
	X = [-k for k in np.arange(0,b,dx)][:0:-1] + [k for k in np.arange(0,b,dx)]
	w = 2*(b - X[-1])
	X = [X[0]-w] + X + [X[-1]+w]
	return np.array(X)

def discretize_gaussian(sgm2dt, dx=dx, nbs_min=70, nbs_max=501, nbs_scale=10, s_range=4):
	'''
	Creates a vector of bins for s values.
	
	Inputs :
	sgm2dt      Vector containing the variance of the process across time.
				Dimensions: T.
	dx          Bin size for discretizing the values of the accumulator variable a.
	nbs_min     Minimum number of s bins (in case of small sgm).
	nbs_scale   Scaling factor to ensure nbs is large enough.
	Output :
	s0          Vector containing the positions which can be reached by a diffusion process during dt, starting from 0, at each time step.
				Dimensions (for future broadcasting) : (1, nbs, 1)
	Variables computed during execution (but not returned) :
	nbs         Number of bins for s values.
				The wider the standard deviation of the gaussian, the more bins are used.
				In order to perform vectorized computations (contrary to the article, probably), nbs is kept fixed across all time points of a trial.
				Thus nbs is computed based on the *maximum* value of sgm2dt for the whole trial.
	Ds          Extremum distance from the center of the gaussian : proportional to the standard deviation.
	ds          Size of a bin, equal to the total range (2 Ds, i.e. from each side of the center) divided by the number of bins.
	'''
	nbs = 2*max(nbs_min, np.ceil(nbs_scale*np.sqrt(np.max(sgm2dt))/dx))+1
	nbs = min(nbs, nbs_max)
	SGM = sgm2dt[np.newaxis,np.newaxis,:] # add dimensions for nbx and nbs
	Ds = s_range*np.sqrt(sgm2dt)
	ds = 2*Ds/nbs
	k = np.arange(nbs)[np.newaxis,:,np.newaxis] # add dimensions for nbx and T
	s0 = (-Ds + k*ds)
	# s0 = np.arange(-Ds, Ds, ds)
	return s0

def gaussian(s0, sgm2dt):
	'''
	Computes gaussian probability distribution over the (centerd) vector of positions of the variable s.
	Output :
	ps      Vector containing the values of the probability distribution over variables s, at each time step during a trial.
			Dimensions (for future broadcasting) : (1, nbs, T)
	'''
	ps = np.exp(-s0**2/(2*sgm2dt[np.newaxis,np.newaxis,:])) # add dimensions to sgm2dt for nbx and nbs
	ps /= np.sum(ps,axis=1) # normalization
	return ps

def deterministic_drift(X, c, lbda, dt):
	'''
	Computes the deterministic mean of the new position reached from every x value.
	Input :
	X       Vector of positions containing the bins centers for the variable a.
			Dimensions: nbx.
	c       Net input to the system, computed by total_input().
			Unit : q/s
			Dimensions: T.
	lbda    Unit : s**(-1)
	Output :
	m       Matrix containing the values of the center of mass of the gaussians.
			Dimensions (for future broadcasting) : (nbx, 1, T)
	'''
	# Format for array broadcating :
	Xd = X[:,np.newaxis,np.newaxis] # add dimensions for nbs and T
	cd = c[np.newaxis,np.newaxis,:] # add dimensions for nbx and nbs
	if lbda == 0 :
		m = Xd + cd*dt
	else :
		m = np.exp(lbda*dt)*(Xd + cd/lbda) - cd/lbda
	return m

def ornstein_uhlenbeck_process(m, s0):
	'''
	Inputs :
	m       Matrix containing the deterministic positions reached from each bin center X at each time step.
			Dimensions: (nbx, 1, T)
	s0      Vector containing the range of positions reachable from each side of a deterministic centered position.
			Dimensions: (1, nbs, 1)
	Set the position s which can be reached from each bin center X at each time step.
	For a given x, each position s reachable from x will correspond to a point of the discretized gaussian.
	Output
	S       Matrix whose element j,k,t corresponds to the kth position s reachable from X[j] at time bin t.
			Dimensions: (nbx, nbs, T)
	'''
	S = s0 + m
	return S


# ----------- version vectorized --------------------

def attribute_closest(S, X, dx):
	'''
	For a given position s, find the index of the x positions just below and above s.
	'''
	nbx = len(X)
	I_low = np.zeros(S.shape)
	I_up = np.zeros(S.shape)
	# Cases X[i+1] - X[i] == dx :
	I_low = np.floor((S-X[1])/dx).astype(int) + 1 # +1 for X[0]
	I_up = np.ceil((S-X[1])/dx).astype(int) + 1
	# Cases X[i+1] - X[i] != dx :
	I_low[(S > X[0]) & (S < X[1])] = 0
	I_up[(S > X[0]) & (S < X[1])] = 1
	I_low[(S > X[-2]) & (S < X[-1])] = nbx - 2
	I_up[(S > X[-2]) & (S < X[-1])] = nbx - 1
	# Cases out of extremums X : attribute same i_low and i_up
	I_low[(S <= X[0])] = 0
	I_up[(S <= X[0])] = 0
	I_low[(S >= X[-1])] = nbx-1
	I_up[(S >= X[-1])] = nbx-1
	# Corresponding positions
	X_low = X[list(I_low.flatten())].reshape(S.shape)
	X_up = X[list(I_up.flatten())].reshape(S.shape)
	return I_low, I_up, X_low, X_up

def split_mass(S, X_low, X_up):
	# If x_low == x_up, set w_low = 1 and w_up = 0 (avoid counting twice)
	W_low = np.ones(S.shape) 
	W_up = np.zeros(S.shape)
	norm = X_up - X_low
	mask = (X_low!=X_up) # avoid division by 0
	W_low[mask] = (X_up - S)[mask]/norm[mask]
	W_up[mask] = (S - X_low)[mask]/norm[mask]
	return W_low, W_up

def F_matrix(times_r, times_l, dur, lbda=lbda, b=b, sgm2a=sgm2a, sgm2s=sgm2s, phi=phi, tau_phi=tau_phi, dt=dt, dx=dx, renorm=True, check=False, check_all=False):
	'''
	Computes the transition matrix between x positions at each time step.
	Output :
	F       All matrices of transition probabilities between bins centers of the variable a, at each time step.
			F[i,j] = P(at+1=xi|at=xj)
			Dimensions : (nbx, nbx, T)
	'''
	X = discretize_space(b, dx)
	CR, CL = input_pulses(times_r, times_l, dur, phi, tau_phi, dt)
	c = total_input(CR, CL, dt)
	sgm2dt = total_variance(CR, CL, sgm2a, sgm2s, dt)
	s0 = discretize_gaussian(sgm2dt, dx)
	m = deterministic_drift(X, c, lbda, dt)
	ps = gaussian(s0, sgm2dt)
	S = ornstein_uhlenbeck_process(m, s0)    
	I_low, I_up, X_low, X_up = attribute_closest(S, X, dx)
	W_low, W_up = split_mass(S, X_low, X_up)
	P_low, P_up = W_low*ps, W_up*ps
	print('nbx, nbs, T = ', S.shape)

	nbx, nbs, T = S.shape
	F = np.zeros((nbx,nbx,T))
	# Transient states
	print('Computing lines')
	for i in range(nbx):
		print(i)
		F[i,:,:] = np.sum(np.ma.masked_where(I_low!=i, P_low), axis=1) + np.sum(np.ma.masked_where(I_up!=i, P_up), axis=1)
	# Absorbant states
	F[0,0,:] = 1
	F[-1,-1,:] = 1
	F[1:,0,:] = 0
	F[:-1,-1,:] = 0
	# Correct approximation errors induced by small sgm2dt
	if renorm: 
		F /= np.sum(F, axis=0)
	check_F_matrix(check, check_all, F, ps, W_low, W_up, P_low, P_up)
	return F

def check_F_matrix(check, check_all, F, ps, W_low, W_up, P_low, P_up):
	if check:
		print('Maximum F :', np.max(F), 'Minimum F :', np.min(F))
		Fs = np.sum(F, axis=0)
		print('Sum F across lines')
		print('Max :', np.max(Fs), 'Min :', np.min(Fs))
		# print(np.where(Fs==np.max(Fs)))
		# print(np.where(Fs==np.min(Fs)))
		F0 = F[:,:,0]
		plt.imshow(F0)
		plt.colorbar()
		plt.show()
		for i in [0,2,-1,-2]:
			plt.plot(F0[i,:], label=i)
		plt.legend()
		plt.title('P(at+1 = {}| at = ...)'.format([0,2,-1,-2]))
		plt.show()
		for i in [1,-2, 100]:
			plt.plot(F0[:,i], label=i)
		plt.legend()
		plt.title('P(at+1 = ... | at = {})'.format([1,-2, 100]))
		plt.show()
	if check_all:
		print('Sum ps :', np.sum(ps))
		print('W_low + W_up != 1 ?', np.sum(W_low + W_up != 1))
		print('Sum P_low + P_up != 1 ?', np.sum(np.sum(P_low+P_up, axis=1) != 1))
		print(np.sum(P_low+P_up, axis=1))

# -----------------------------------------------------------

def initialize_prior_distribution(X, sgm2i=sgm2i):
	if sgm2i != 0 :
		p0 = 1/np.sqrt(2*np.pi*sgm2i)*np.exp(-(1/2)*X**2/sgm2i)
		p0 /= np.sum(p0)
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


# -----------------------------------------------------------


def Likelihood_trial(d, times_r, times_l, dur, lbda=lbda, b=b, sgm2a=sgm2a, sgm2s=sgm2s, sgm2i=sgm2i, phi=phi, tau_phi=tau_phi, rho=rho, dt=dt, dx=dx):
	X = discretize_space(b, dx)
	print('F matrix')
	F = F_matrix(times_r, times_l, dur, lbda, b, sgm2a, sgm2s, phi, tau_phi, dt, dx)
	p0 = initialize_prior_distribution(X, sgm2i)
	print('Prior distribution')
	Pa = prior_probability_distribution(p0, F)
	pL = proba_left(Pa, X, rho, dx)
	if d=='L':
		return pL
	else :
		return 1 - pL

def Log_Likelihood(D, lbda=lbda, b=b, sgm2a=sgm2a, sgm2s=sgm2s, sgm2i=sgm2i, phi=phi, tau_phi=tau_phi, rho=rho, dt=dt, dx=dx):
	'''
	Input :
	D           Data, dictionary containing, for each trial :
				stim_r, stim_l  Sequences of times of input stimuli.
				dur             Duration of the trial.
				d               Decision at the end of each trial ('L' left, 'R' right)
	'''
	N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_fkp)
	LL = 0
	for k in range(len(D)): 
		L = Likelihood_trial(D[k]['d'], D[k]['times_r'], D[k]['times_l'], D[k]['dur'], lbda, b, sgm2a, sgm2s, sgm2i, phi, tau_phi, rho, dt, dx)
		LL += np.log(L)
	return LL
