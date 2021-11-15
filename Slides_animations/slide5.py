from manim import * 
import random
import numpy as np
import scipy.stats 
        
    
class simulate_stimuli(Scene):
    def construct(self):
        ax = Axes(x_range=[0, 1.2, 0.05], y_range=[0, 2, 1], tips=False).shift(LEFT*0.4)
        times_r, times_l,_  = self.clicks(0.5)
        
        self.add(ax)
        right = VGroup(*[Line(start=ax.c2p(time_r, 1, 0), end = ax.c2p(time_r, 1.1, 0), color=BLUE) for time_r in np.sort(times_r[0]) ])
        left = VGroup(*[Line(start=ax.c2p(time_l, 1.5, 0), end = ax.c2p(time_l, 1.6, 0), color=GREEN) for time_l in np.sort(times_l[0]) ])
        self.wait(2)
        self.play(Create(left), Create(right), run_time = 1.2*5)
        self.wait(5)
    
    def rates_of_clicks(self,gamma):
        '''
        MC 04/11/21
        Input : gamma = the difficulty of the trial
        Output : the rates of right and left clicks
        '''
        ""
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
        times_l = duration*scipy.stats.uniform.rvs(0,1,((nL,1))) #timestamps of left clicks
        times_r = duration*scipy.stats.uniform.rvs(0,1,((nR,1))) #timestamps of right clicks
        return times_r, times_l, duration

    
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
        self.play(formula_box_a.animate.shift(UP*3.3), b.animate.shift(DOWN))
        self.wait(3)
        
        sigma1 = formula_a[3].copy().move_to(ORIGIN)
        
        lambda1 = formula_a[18].copy().move_to(ORIGIN)
        
        dw = formula_a[4].copy().align_to(b, LEFT)
        
        self.play(Write(formula_c), FadeIn(framebox_c))
        #self.play(FadeIn(dw), scale = 0.5)
        #exp_dw = Tex(f': White noise Weiner process', font_size = 30).next_to(dw, RIGHT)
        #self.play(Write(exp_dw))
        self.wait(3)
        
        