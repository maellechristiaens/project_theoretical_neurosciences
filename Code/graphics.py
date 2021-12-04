#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PLOTS OF THE RESULTS
'''
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import parameters

params_sim = parameters.params_sim
theta_test = parameters.theta_test
theta_opt = parameters.theta_opt
N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params_sim)
rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta_test)

print('graphics imported')


# -----------------------------------------------------------

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

def legend_outside(ax, title=''):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(title=title, loc='center left', bbox_to_anchor=(1, 0.5))


# -----------------------------------------------------------

def show_trial(times_r, times_l, params=params_sim):
    N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    fig = plt.subplots(1)
    plt.eventplot(times_r, lineoffsets=0.6, linelengths=1, color='red', label='Right')
    plt.eventplot(times_l, lineoffsets=-0.6, linelengths=1, color='green', label='Left')
    plt.ylim(-1.2, 1.2)
    plt.xlim(0,dur)
    plt.xlabel('Time (s)')
    legend_outside(plt.gca(), title='Stimuli \n (side of headphone)')
    hide_spines(plt.gca(), sides=['right', 'left', 'top'])
    hide_ticks(plt.gca(), 'y')
    plt.show()

def show_trajectories(trajectory, times_r, times_l, params=params_sim, theta=theta_test, ymax=''):
    N, Durs, dur, Gamma, gamma, tpulse, dt, dx, nbs, dphi, dtphi = parameters.extract_fixed_params(params)
    rho, b, l, sgm_i, sgm_a, sgm_s, lbda, phi, tau_phi = parameters.extract_free_params(theta)
    a_evols, C_evol = trajectory['a_evols'], trajectory['C_evol']
    times = [t*dt for t in range(len(C_evol))]
    ntraj = len(a_evols)
    if ymax == 'biais':
        ymax = b
        ymin = -b
    else :
        ymax = max([np.max(a_evols[k]) for k in range(ntraj)])
        ymin = min([np.min(a_evols[k]) for k in range(ntraj)])
    y_L = [ymin for t in range(len(times_l))]
    y_R = [ymax for t in range(len(times_r))]
    
    fig, ax = plt.subplots(2)
    ax[0].plot(times, a_evols[0], color='k')
    for k in range(1,ntraj):
        ax[0].plot(times, a_evols[k], color='gray', alpha=0.5)
    ax[0].scatter(times_r, y_R, color='red', marker='v')
    ax[0].scatter(times_l, y_L, color='green', marker='^')
    ax[0].axhline(rho, color='gray', linestyle='dashed')
    ax[0].set_ylabel('Accumulator value, a')
    hide_ticks(ax[0], 'x')
    hide_spines(ax[0], sides=['right', 'top','bottom'])
    ax[1].plot(times, C_evol, color='gray')
    ax[1].axhline(1, color='gray', linestyle='dashed')
    ax[1].set_ylabel('Adapatation, C', color='gray')
    hide_spines(ax[1], sides=['right', 'top'])
    ax[1].set_xlabel('Time (s)')
    plt.show()

def show_distribution(P_a, X, dt=dt):
    times = np.arange(1, P_a.shape[1]) # exclude t=0 for better vizualization
    times = np.array([t*dt for t in range(P_a.shape[1])][1:]) # exclude t=0 for better vizualization
    gridX, gridY = np.meshgrid(times, X) # invert intuitive side
    Z = P_a[:,1:] # exclude t=0
    # print(Z.shape)
    # print(X.shape)
    # print(times.shape)
    # print(gridX.shape)
    # print(gridY.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # cmap = cm.autumn_r
    cmap = cm.coolwarm
    surf = ax.plot_surface(gridX, gridY, Z, cmap=cmap, linestyles="solid", lw=0, antialiased=True, alpha=0.7)
    bottom = -0.1*np.max(Z)
    cset = ax.contour(gridX, gridY, Z, zdir='x', offset=0, cmap=cm.coolwarm_r)
    cset = ax.contourf(gridX, gridY, Z, zdir='y', offset=30, cmap=cm.coolwarm_r)
    cset = ax.contour(gridX, gridY, Z, zdir='z', offset=bottom, cmap=cm.coolwarm)
    # alternative : contourf
    ax.set_zlim(bottom, np.max(Z))
    ax.set_ylabel('Accumulator value')
    ax.set_xlabel('Time (s)')
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label('Probability', rotation=270)
    plt.show()