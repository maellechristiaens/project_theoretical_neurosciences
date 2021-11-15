from manim import * 
import random
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats 
        
    
class slide6(Scene):
        
    #PARAMETERS
    N = 40 # average number of pulses (n_pulses/s)
    rho = 0.1 # biais, threshold to output a decision by comparing the value of a at the end of the trial
    b = 30 # sticky decision bound, amount of evidence necessary to commit to a decision
    l = 0.01 # lapse rate, fraction of trials on which a random response is made
    sgm_i = 0.1 # noise in the initial value of a
    sgm_a = 70 # diffusion constant, noise in a
    sgm_s = 70 # noise when adding evidence from one pulse (scaled by C amplitude)
    lbda = 0.1 # consistent drift in a
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

    def construct(self):
        ax = Axes(x_range=[0, 1.2, 0.05], y_range=[1, 2, 2], tips=False).shift(LEFT*0.4)
        times_r, times_l,_  = self.clicks(0.5)
        nbr = Tex(f'nb of right = {len(times_r)}').shift(RIGHT*3)
        nbl = Tex(f'nb of left = {len(times_l)}').shift(UP*1, RIGHT*3)
        self.add(ax, nbr, nbl)
        evolution_a,_ = self.euler(times_r, times_l)
        left= Line(start=ax.c2p(0, 1, 0), end = ax.c2p(0, 1.1, 0), color=BLUE)
        right= Line(start=ax.c2p(0, 1.9, 0), end = ax.c2p(0, 2, 0), color=GREEN)
        self.play(FadeIn(left), FadeIn(right))
        for i in np.arange(0, 1.2, 0.01):
            self.wait(0.01*10)
            if any((times_l>(i-0.005))*(times_l<(i+0.005))):
                left = Line(start=ax.c2p(i, 1, 0), end = ax.c2p(i, 1.1, 0), color=BLUE)
                self.play(FadeIn(left), run_time=0.01*5)
            elif any((times_r>(i-0.005))*(times_r<(i+0.005))):
                right = Line(start=ax.c2p(i, 1.9, 0), end = ax.c2p(i, 2, 0), color=GREEN)
                self.play(FadeIn(right), run_time=0.01*5)
        graph_a = ax.get_line_graph(x_values = evolution_a, y_values = np.arange(0,1.2, 0.02), add_vertex_dots = False)
    
    def rates_of_clicks(self,gamma):
        '''
        MC 04/11/21
        Input : gamma = the difficulty of the trial
        Output : the rates of right and left clicks
        '''
        r1 = 40/(1+10**gamma)
        r2 = 40 - r1
        return [r1, r2]
    
    def clicks(self,gamma):
        '''
        MC 04/11/21
        Input : the difficulty of the trial

        Output : the timestamps of right and left clicks for this trial and the duration range
        ''' 
        duration = np.random.uniform(0.1, 1.2) #compute the duration of the trial
        l = self.rates_of_clicks(gamma) #generate the rates of clicks 
        [rR, rL] = random.sample(l, 2) #pick randomly which size corresponds to which rate (otherwise it's always the
                                       #right size that has more clicks)
        nL = scipy.stats.poisson(rL*duration).rvs()#Number of left clicks
        nR = scipy.stats.poisson(rR*duration).rvs()#Number of right clicks
        times_l = np.sort((duration*scipy.stats.uniform.rvs(0,1,((nL,1)))).T[0]) #timestamps of left clicks
        times_r = np.sort((duration*scipy.stats.uniform.rvs(0,1,((nR,1)))).T[0]) #timestamps of right clicks 
        return times_r, times_l, duration
    
    def accumulator_evolution(self,a, C, t, stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, b=b, dt=dt):
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
        
        
    def adaptation_evolution(self,C, t, stim_R, stim_L, psi=psi, tau_psi=tau_psi, dt=dt):
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

    def euler(self,stim_R, stim_L, lbda=lbda, sgm_a=sgm_a, sgm_s=sgm_s, sgm_i=sgm_i, b=b, psi=psi, tau_psi=tau_psi, dt=dt):
        T = len(stim_R)
        a_history = np.zeros(T)
        C_history = np.zeros(T)
        # initialisation
        a = np.random.normal(0,sgm_i)
        C = 0 # simultaneous pulses
        a_history[0] = a
        C_history[0] = C
        for t in range(1,T):
            a = self.accumulator_evolution(a, C, t, stim_R, stim_L, lbda, sgm_a, sgm_s, b, dt)
            C = self.adaptation_evolution(C, t, stim_R, stim_L, psi, tau_psi, dt)
            a_history[t] = a
            C_history[t] = C
        return a_history, C_history
    

    
    
class slide5(Scene):
    def construct(self):
        da1 = Tex(f"Let's define $a$, the accumulator memory that represents")
        da2 = Tex(f"an estimate of the right versus left evidence accrued so far").shift(DOWN)
        da = VGroup(da1,da2)
        
        line1 = Line(start = [-3, -3, 0], end = [3, -3, 0])
        line2 = Line(start = [-3, 0, 0], end = [3, 0, 0])
        b1 = Tex('-B').next_to(line1, LEFT)
        b2 = Tex('B').next_to(line2, LEFT)
        schema = VGroup(line1, line2, b1, b2)
        func_left = FunctionGraph(lambda t: np.cos(t) + 0.5 * np.cos(7 * t) + (1 / 7) * np.cos(14 * t) -0.5, x_range=[-3, 0] ,color=RED).shift(DOWN)
        left = Tex(f'Choose left').next_to(line2, RIGHT)
        
        a = Tex(r'a: accumulator memory', font_size = 30).shift(UP*2.2, RIGHT*4)
        b = Tex(r'B: amount of evidence necessary to commit to a decision', font_size = 30).shift(UP*3.2, LEFT*3)
        formula_a = MathTex(r'd',r'a', r'=', r'\sigma_{a}', r' dW', r' + (',r'\delta_{t,t_R}', r' \cdot', r' \eta_R', r' \cdot', r' C' , r'- ',r'\delta_{t,t_L}', r' \cdot', r' \eta_L', r' \cdot', r' C', r')dt +',r' \lambda', r' a', r'dt')
        formula_a[1].set_color(YELLOW)
        formula_a[3].set_color(BLUE)
        formula_a[4].set_color(GREEN)
        formula_a[8].set_color(PURPLE)
        formula_a[10].set_color(RED)
        formula_a[14].set_color(PURPLE)
        formula_a[16].set_color(RED)
        formula_a[18].set_color(PINK)
        formula_a[19].set_color(YELLOW)
        framebox_a = SurroundingRectangle(formula_a, buff = .2, color=WHITE)
        
        formula_c = MathTex(r'\frac{dC}{dt}', r'=', r'frac{1-C}{\tau_{\phi}}', r'+ (' , r'phi' , r'- 1)', r'C', r'(', r'\delta_{t,t_R}', r'+', r'\delta_{t,t_L}', r')')
        framebox_c = SurroundingRectangle(formula_c, buff = .2, color=WHITE)
        
        
        
        self.play(Write(da), run_time = 3)
        self.wait(2)
        self.play(da.animate.shift(UP*2))
        self.play(FadeIn(schema))
        self.play(Create(func_left))
        self.play(Write(left))
        self.play(Write(b))
        self.play(FadeOut(schema), FadeOut(func_left), FadeOut(left))
        self.play(Write(formula_a), FadeIn(framebox_a))
        self.wait(5)
        
        formula_box_a = VGroup(formula_a, framebox_a)
        self.play(FadeOut(da), FadeIn(a))
        self.play(formula_box.animate.shift(UP*3.3), b.animate.shift(DOWN))
        self.wait(3)
        
        sigma1 = formula_a[3].copy().move_to(ORIGIN)
        
        lambda1 = formula_a[18].copy().move_to(ORIGIN)
        
        dw = formula_a[4].copy().align_to(b, LEFT)
        
        self.play(Write(formula_c), FadeIn(framebox_c))
        #self.play(FadeIn(dw), scale = 0.5)
        #exp_dw = Tex(f': White noise Weiner process', font_size = 30).next_to(dw, RIGHT)
        #self.play(Write(exp_dw))
        self.wait(3)
        
        