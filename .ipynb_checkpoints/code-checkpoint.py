import numpy as np
import matplotlib.pyplot as plt

print('imported')
a = 1


#### PARAMETERS #### 

N = 40 # average number of pulses (n_pulses/s)
rho = 0.1 # biais, threshold to output a decision by comparing the value of a at the end of the trial
b = 30 # sticky decision bound, amount of evidence necessary to commit to a decision
l = 0.01 # lapse rate, fraction of trials on which a random response is made
sgm_i = 0.1 # noise in the initial value of a
sgm_a = 70 # diffusion constant, noise in a
sgm_s = 70 # noise when adding evidence from one pulse (scaled by C amplitude)
lbda = 0.1 # consistent drift in a
# leaky or forgetful case : lbda < 0 (drift toward a = 0, and later pulses affect the decision more than earlier pulses)
# unstable or impulsive case : lbda > 0 (drift away from a = 0, and earlier pulses affect the decision more than later pulses)
tau = abs(1/lbda) # memory time constant
q = 1 # unit of evidence, magnitude of one click without adaptation (psi = 1)
# an ideal observer adds one q at every right pulse and subtracts one q at every left pulse
psi = 0.34 # adaptation strength, factor by which C is multiplied following a pulse
# facilitation : psi > 1
# depression : psi < 1
tau_psi = 0.04 # adaptation time constant for C recovery to its unadapted value, 1
dt = 0.02 # time step for simulations (in s)
dx = 0.25*q # bin size for space discretization (values of vector x)
ds = sgm_a**2*dt/100 # bin size for discretizing the gaussian probability (values of s)
delta_s = 4*sgm_a**2*dt # maximum distance from the mean for discretizing the gaussian probability 


# Parameters to fit to data
theta = {'a': a,
        'b': b,
        'l': l,
        'sgm_i': sgm_i,
        'sgm_s': sgm_s,
        'lbda': lbda,
        'tau': tau,
        'psi': psi,
        'tau_psi': tau_psi}

# Free parameters
sim_params = {'q': q,
            'N': N,
            'dt': dt,
            'dx': dx,
            'ds': ds,
            'delta_s': delta_s}

dur = 1 # stimulus time (in s)

def stim_test(dur=dur, dt=dt, N=N):
    T = int(dur/dt) # number of time steps
    p = N*dur/T # probability to generate a pulse in time step dt, so that the average number of pulses is N
    stim_R = [np.random.random()<p for t in range(T)]
    stim_L = [np.random.random()<p for t in range(T)]
    return stim_R, stim_L


#### EULER METHOD ####

def accumulator_evolution(a, C, t, stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, b=b, dt=dt):
    '''Dynamics of the memory accumulator.
    Inputs : 
        a : value of the accumulator at time t
        C : value of the stimulus magnitude at time t
        t : time step during the course simulation
        stim_R, stim_L : stimulation sequences, containing the number of pulses occurring during each time step
    Output : new value of the accumulator at t+dt.'''
    if abs(a) >= b:
        return a
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
    a = np.random.normal(0,sgm_i)
    C = 0 # simultaneous pulses
    a_history[0] = a
    C_history[0] = C
    for t in range(1,T):
        a = accumulator_evolution(a, C, t, stim_R, stim_L, lbda, sgm_a, sgm_s, b, dt)
        C = adaptation_evolution(C, t, stim_R, stim_L, psi, tau_psi, dt)
        a_history[t] = a
        C_history[t] = C
    return a_history, C_history


#### FOKKER PLANK METHOD ####

def discretize_space(b=b, dx=dx):
    X = [-k for k in np.arange(0,b,dx)][:0:-1] + [k for k in np.arange(0,b,dx)]
    w = 2*(b - X[-1])
    X = [X[0]-w] + X + [X[-1]+w]
    return np.array(X)

def gaussian(s, m, sgm):
    return (1/(2*np.pi*sgm**2)**0.5)*np.exp(-(s-m)**2/(2*sgm**2))

def ornstein_uhlenbeck_process(x, c, sgm, lbda=lbda, sgm_a=sgm_a, dt=dt, ds=ds, delta_s=delta_s):
    m = np.exp(lbda*dt)*(x + c/lbda) - c/lbda
    s = np.arange(m-delta_s, m+delta_s, ds) # discretization of space for approximating the probability distribution
    ps = gaussian(s, m, sgm) # sgm already takes into account dt**0.5
    s[s>=b] = b # positions outside bound assigned to b (after computing the probability distribution for ease)
    s[s<=-b] = -b 
    return s, ps

def split_mass(s, x_low, x_up):
    norm = x_up - x_low
    p_low = (x_up - s)/norm
    p_up = (s - x_low)/norm
    return p_low, p_up

def forward_transition_matrix(X, stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, sgm_i=sgm_i, b=b, psi=psi, tau_psi=tau_psi, dt=dt, ds=ds, delta_s=delta_s):
    T = len(stim_R)
    M = len(X) # number of positions, including 2 extra bins outside boundaries
    C_history = euler(stim_R, stim_L, lbda, sgm_a, sgm_s, sgm_i, b, psi, tau_psi, dt)[1]
    CR_history = C_history*stim_R 
    CL_history = C_history*stim_L
    c_input = CR_history - CL_history
    sgm_input = (sgm_a**2*dt + (CR_history + CL_history)*sgm_s**2)**0.5

    F = np.zeros((M,M,T)) 
    F[0,0,:] = 1 # absorbant state
    F[M-1,M-1,:] = 1
    for t in range(T):
        c = c_input[t]
        sgm = sgm_input[t]
        S = np.zeros((M, int(2*delta_s/ds)))
        P_s = np.zeros((M, int(2*delta_s/ds)))
        for j in range(1,M-1):
            S[j], P_s[j] = ornstein_uhlenbeck_process(X[j], c, sgm, lbda, sgm_a, dt, ds, delta_s)
        for i in range(1,M-1):
            for j in range(1,M-1):
                F[i,j,t] = np.sum(P_s[j][(S>=X[i])*(S<X[i+1])]*split_mass(S[j], X[i], X[i+1])[0]) + np.sum(P_s[j][S>=X[i-1] and S<X[i]]*split_mass(S[j], X[i-1], X[i])[1])
    return F

def fokker_plank(stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, sgm_i=sgm_i, b=b, psi=psi, tau_psi=tau_psi, dt=dt, ds=ds, delta_s=delta_s, dx=dx):
    T = len(stim_R)
    X = discretize_space(b, dx)
    M = len(X)
    F = forward_transition_matrix(X, stim_R, stim_L, lbda, sgm_a, sgm_s, sgm_i, b, psi, tau_psi, dt, ds, delta_s)
    P_a = np.zeros((M,T))
    P_a[:,0] = gaussian(X, 0, sgm_i)
    for t in range(t):
        P_a[:,t] = np.dot(F,P_a[:,t-1])
    return P_a, X

def proba_left(P_a, X, rho=rho):
    return np.sum(P_a[X<=rho])


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

def show_trajectory(a_history, C_history, stim_R, stim_L, dt=dt):
    times = [t*dt for t in range(len(stim_R))]
    pos_R = [t*dt for t in range(len(stim_R)) if stim_R[t]!=0]
    pos_L = [t*dt for t in range(len(stim_R)) if stim_L[t]!=0]
    stim_L_y = [np.max(a_history) for t in range(len(pos_L))]
    stim_R_y = [np.min(a_history) for t in range(len(pos_R))]

    fig, ax = plt.subplots(2)
    ax[0].plot(times, a_history, color='k')
    ax[0].scatter(pos_R, stim_R_y, color='red', marker='^')
    ax[0].scatter(pos_L, stim_L_y, color='green', marker='v')
    ax[0].set_ylabel('Accumulator value, a')
    hide_ticks(ax[0], 'x')
    hide_spines(ax[0], sides=['right', 'top','bottom'])
    ax[1].plot(times, C_history, color='gray')
    ax[1].set_ylabel('Adapatation, C', color='gray')
    hide_spines(ax[1], sides=['right', 'top'])
    ax[1].set_xlabel('Time (s)')
    plt.show()