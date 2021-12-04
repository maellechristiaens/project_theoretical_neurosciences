#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PARAMETER FITTING
'''

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from functools import partial

import parameters
import simulate_trajectories as traj
import likelihood_fokker_plank as ll_fkp
import likelihood_gradient as ll_grad

params_fkp = parameters.params_fkp
theta_test = parameters.theta_test
N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_fkp)
rho, b, l, sgm2i, sgm2a, sgm2s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)

print('optimization_parameters imported')

# Constraints on parameter space
lba_min_max = [-5, 5]
sgm2a_min_max = [0, 200]
sgm2s_min_max = [0, 200]
phi_min_max = [0.1, 2.5]
tau_phi_min_max = [0.005, 1]
b_min_max = [2, 32]
# scipy function : Bounds(lb, ub)
# lb, ub : Lower and upper bounds, each array must have the same size as theta.
all_bounds = [lba_min_max, sgm2a_min_max, sgm2s_min_max, phi_min_max, tau_phi_min_max, b_min_max]
lb = [bound[0] for bound in all_bounds]
ub = [bound[1] for bound in all_bounds]
bounds = Bounds(lb, ub)


def fit_parameters(D, theta0, bounds=bounds):
    '''
    Input
    theta0      Dictionary of parameters used as origin.
    
    Output 
    fit.x       Vector containing the solution of the optimization, i.e. values of the fitted parameters.
    fit.fun     Values of objective function (likelihood).
    fit.jac     Values of the Jacobian (gradient)
    '''
    # 1) Creating the vector of 6 parameters to optimize, to be passed as argument in objective functions
    # Order : lbda, sgm2a, sgm2s, phi, tau_phi, b
    theta0 = params.convert_dict_to_vect(theta0)
    
    # 2) Fixing several parameters in the Likelihood and Gradient functions 
        # Data : D 
        # Parameters which are not optimized : rho, sgm2i
        # Other fixed parameters : dphi, dtphi, dx, dt, tpulse

    def Log_Likelihood_fix(theta):
        lbda, sgm2a, sgm2s, phi, tau_phi, b = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
        LL = ll_fkp.Log_Likelihood(D, lbda=lbda, b=b, sgm2a=sgm2a, sgm2s=sgm2s, sgm2i=sgm2i, phi=phi, tau_phi=tau_phi, rho=rho, dt=dt, dx=dx)
        return LL

    def LL_gradient_fix(theta):
        lbda, sgm2a, sgm2s, phi, tau_phi, b = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
        dLL = ll_grad.LL_gradient(D, lbda, sgm2a, sgm2s, phi, tau_phi, b, rho=rho, dphi=dphi, dtphi=dtphi, dx=dx, dt=dt, tpulse=tpulse)
        return dLL

    # 3) Fitting with scipy function minimize, with bound constraints
    # verbose : 
        # 0 : work silently
        # 1 : display a termination report
        # 2 : display progress during iterations
        # 3 : display progress during iterations (more complete report).

    fit = minimize(Log_Likelihood_fix, theta0, jac=LL_gradient_fix, bounds=bounds, method='trust-constr', options={'verbose': 3})

    return fit.x, fit.fun, fit.jac