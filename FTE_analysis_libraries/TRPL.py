# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:07:17 2020

@author: dreickem
"""

import math
import numpy as np
import pandas as pd
from scipy.sparse import spdiags
from scipy.optimize import curve_fit, least_squares
from matplotlib import pyplot as plt
from matplotlib import animation
from os import listdir
from os.path import join
import sys
from IPython import embed
from importlib import reload
import pkg_resources
system_dir = pkg_resources.resource_filename( 'FTE_analysis_libraries', 'System_data' )

from .General import h, c, pi, k, q, T_RT, f1240, findind, interpolated_array
from .XYdata import xy_data, mxy_data
from .Spectrum import above_bg_photon_flux

def lambeer(alpha, x, P):
    """
    P: Excitation photon density in photons /cm2
    """
    return alpha * P * np.e**(-alpha * x)

def pulse(t, pl = 60e-12):
    """
    Gaussian pulse profile with integral 1

    Parameters
    ----------
    t : FLOAT
        Time in s.
    pl : FLOAT, optional
        Pulse length in s. The default is 60e-12 s.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    # Check if integral is 1
    times = np.linspace(0, 5e-9, 1001)
    print(f'The integral over pulse is: {np.trapz(pulse(times), times):.3f}')
    plt.plot(times, pulse(times))
    plt.show()

    """
    
    # pre factor to make integral over gaussian equal 1
    A = 2/pl * math.sqrt(math.log(2)/pi)
    return A * np.e**(-(4*math.log(2)*((t-1.5*pl)/pl)**2)) # Factor of 1.5 is to determine the start point of the pulse so that the integral is still nearly 1


def one_sun_carrier_conc(lifetime = 1000, bg = f1240/800, film_thickness = 500, print_with_units = False):
    """
    Calculates the carrier concentration at 1 sun in 1/cm3 for a material with a carrier lifetime of lifetime in ns
    and a bandgap of bg in eV and a film thickness in nm.
    """
    cc = above_bg_photon_flux(bg)*lifetime*1e-9/(film_thickness*1e-7)/1e4
    if print_with_units:
        print(f'The one sun carrier concentration is {cc:.2e} 1/cm3')
    return cc

def initial_carrier_conc(wavelength, film_thickness, fluence, print_result = False):

    """
    Initial carrier concentration n0 (1/cm3) 
    after a picosecond laser pulse with 
    fluence F (J/cm2) and 
    wavelength (nm)
    has hit a film with 
    film_thickness (nm)
    It is assumed that the laser pulse is absorbed completely.
    """
    
    # Photon energy in J at laser wavelength
    E = h * c / (wavelength*1e-9)

    n0 = fluence/(E*film_thickness*1e-7) #1/cm3

    if print_result:    
        print(f'The initial carrier concentration after a picosecond laser pulse with wavelength {wavelength:.0f} nm')
        print(f'and a fluence {fluence:.2e} J/cm2 has been entirely absorbed by a film with thickness {film_thickness:.0f} nm')
        print(f'is n0 = {n0:.2e} 1/cm3.')
        
    return n0

# These are the important functions for the numerical simulation of the carrier density and TRPL

def EulerHeatConstBCSparse(u0, x, dt, nr_times, t0, mu, k1, k2, k3, SL, SR, n_exc = 0):
    """ This function solves the equation :
            n_t = D n_xx -k1 n - k2 n^2 - k3 n^3
    by using the Euler method. The function takes 
    an initial condition u0, a domain  x, 
    a time step dt, the number of times to run the loop, 
    the initial time, t0, and the physical parameters as input.
    """
    # Important Constants, etc.
    
    time = t0
    dx = x[1] - x[0]
    numUnknowns = len(u0)
    r = dt/(dx**2)
        
    D = k * T_RT / q * mu
    mainDiagonal = -2*np.ones(numUnknowns)
    mainDiagonal[0] = -2 + 2*(-SL)*dx/D  # -SL so that SL is positive
    mainDiagonal[-1] = -2 - 2*SR*dx/D
    upper_offDiagonal = np.ones(numUnknowns)
    lower_offDiagonal = np.ones(numUnknowns)
    upper_offDiagonal[1] = 2
    lower_offDiagonal[-2] = 2
    
    T_matrix = MakeTridiagonalMatrix(mainDiagonal, upper_offDiagonal, lower_offDiagonal) # We could consider a better place to make T to make the code faster.
    u = u0
    
    # Loop to perform the calculations
    for step in range(nr_times):
        u = u + r * D * T_matrix * u - k1 * dt * u - k2 * dt * u**2 - k3 * dt * u**3 + n_exc * dt
        #u = u - k1 * dt * u # only k1

        time = time + dt # We could reconsider how to do this to minimze rounding errors.

    # Return what we want
    return u, time


def EulerHeatConstBCSparse_simple(u0, x, dt, nr_times, t0, mu, k1, SL, SR, n_exc = 0):
    """ This function solves the equation :
            n_t = D n_xx -k1 n # no bimol. and no Auger recombination
    by using the Euler method. The function takes 
    an initial condition u0, a domain  x, 
    a time step dt, the number of times to run the loop, 
    the initial time, t0, and the physical parameters as input.
    """
    # Important Constants, etc.
    
    time = t0
    dx = x[1] - x[0]
    numUnknowns = len(u0)
    r = dt/(dx**2)
        
    D = k * T_RT / q * mu
    mainDiagonal = -2*np.ones(numUnknowns)
    mainDiagonal[0] = -2 + 2*(-SL)*dx/D  # -SL so that SL is positive
    mainDiagonal[-1] = -2 - 2*SR*dx/D
    upper_offDiagonal = np.ones(numUnknowns)
    lower_offDiagonal = np.ones(numUnknowns)
    upper_offDiagonal[1] = 2
    lower_offDiagonal[-2] = 2
    
    T_matrix = MakeTridiagonalMatrix(mainDiagonal, upper_offDiagonal, lower_offDiagonal) # We could consider a better place to make T to make the code faster.
    u = u0
    
    # Loop to perform the calculations
    for step in range(nr_times):
        u = u + r * D * T_matrix * u - k1 * dt * u + n_exc * dt
        #u = u - k1 * dt * u # only k1

        time = time + dt # We could reconsider how to do this to minimze rounding errors.

    # Return what we want
    return u, time



def MakeTridiagonalMatrix(main, upper_offset_one, lower_offset_one):
    """This function will make a tridiagonal 2D matrix
    which has the main array on its main diagonal and the offset_one 
    array on its super and sub diagonals.
    """
    size = len(main)
    offsets = [0,1,-1]
    data = np.vstack((main, upper_offset_one, lower_offset_one))
    A = spdiags(data, offsets, size, size)    
    return A

def PLsignal(u, dx, k2):
    return np.trapz(k2*u**2,dx=dx)


# Plot animation of carrier concentration

def plot_animation(pset1, pset2, interval = 1, ylim=(1e-2,1.2), normalize_to_end = False):

    global u1, u2, time1, time2 # Necessary. 
    
    finaltime = pset1.finaltime
    time_delta = 0.01e-9 #s

    nr_times = int(time_delta / pset1.dt)

    #pset1.n0 = 0 #no laser excitation
    #pset2.n0 = 0 #no laser excitation

    #u1 = pset1.n0
    if pset1.pulse_len == None:
        u1 = pset1.n0
    else:
        u1 = np.zeros(len(pset1.x))
    time1 = 0

    #u2 = pset2.n0
    if pset2.pulse_len == None:
        u2 = pset2.n0
    else:
        u2 = np.zeros(len(pset2.x))
    time2 = 0

    #n_exc = lambeer(alpha_per, pset1.x, N0_per) # constant illumination

    #Plot two curves 

    fig = plt.figure()
    ax = plt.axes(xlim=(0, pset1.thickness-1), ylim=ylim)

    ax.set_yscale('Log')
    #ax.set_yscale('linear')

    ax.set_title('Charge carrier concentration', color = 'black')
    ax.set_xlabel('nm')
    ax.set_ylabel('n(x,t) = Carrier concentration (norm.)')

    N = 2
    lines = []
    for i in range(N):
        lines.append(plt.plot([], [], label='Plot '+str(i+1))[0])

    time_label = plt.text(0.05, 0.05, '', transform=ax.transAxes) # initialize the time label for the graph

    patches = lines

    def init():
        time_label.set_text('') 
        for i, line in enumerate(lines):
            line.set_data([], [])
        return lines[0], lines[1], time_label #return everything that must be updated

    def animate(i):

        global u1, u2, time1, time2 # Necessary. 

        # constant illumination
        #u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.SL, pset1.SR, n_exc = n_exc)
        #u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.SL, pset2.SR, n_exc = n_exc)              

        #print(f'Elapsed time: {time1*1e12:.0f} ps', end = '\r')
        if pset1.pulse_len == 'None':
            u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)
        else:
            u1, time1 = EulerHeatConstBCSparse(u1 + pulse(time1, pset1.pulse_len)*pset1.dt * pset1.n0, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)

        if pset2.pulse_len == 'None':
            u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)
        else:
            u2, time2 = EulerHeatConstBCSparse(u2 + pulse(time2, pset2.pulse_len)*pset2.dt * pset2.n0, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)
        #u2 = u2 + pulse(time2)*pset2.dt * pset2.n0
        #time2 = time1     

        time1_ns = time1*1e9    
        time2_ns = time2*1e9    
        #time_label.set_text('time1 = %.1f ns' % time1_ns + ', time2 = %.1f ns' % time2_ns) # Display the current time to the accuracy of your liking.
        #time_label.set_text('time1 = %.3f ns' % time1_ns)
        time_label.set_text(f'time 1 = {time1_ns:.3f} ns')

        if normalize_to_end:
            lines[0].set_data(pset1.x*1e7, u1/u1[-1])
            lines[1].set_data(pset2.x*1e7, u2/u2[-1])
       
        else:
            lines[0].set_data(pset1.x*1e7, u1/pset1.n0[0]*5)
            lines[1].set_data(pset2.x*1e7, u2/pset2.n0[0]*5)
            
        return lines[0], lines[1], time_label #return everything that must be updated

    anim = animation.FuncAnimation(fig, animate, frames=int(finaltime/pset1.dt/nr_times), interval=interval, init_func=init, blit=True, repeat=False)
    plt.legend()
    plt.show()
    

# Plot animation of quasi-Fermi level splitting

def plot_animation_QFLS(pset1, pset2, interval = 1, ylim=(1e-2,1.2)):

    global u1, u2, time1, time2 # Necessary. 
    
    finaltime = pset1.finaltime
    time_delta = 0.01e-9 #s

    nr_times = int(time_delta / pset1.dt)

    #pset1.n0 = 0 #no laser excitation
    #pset2.n0 = 0 #no laser excitation

    #u1 = pset1.n0
    if pset1.pulse_len == None:
        u1 = pset1.n0
    else:
        u1 = np.zeros(len(pset1.x))
    time1 = 0

    #u2 = pset2.n0
    if pset2.pulse_len == None:
        u2 = pset2.n0
    else:
        u2 = np.zeros(len(pset2.x))
    time2 = 0

    #n_exc = lambeer(alpha_per, pset1.x, N0_per) # constant illumination

    #Plot two curves 

    fig = plt.figure()
    ax = plt.axes(xlim=(0, pset1.thickness-1), ylim=ylim)

    ax.set_yscale('linear')

    ax.set_title('Quasi-Fermi level splitting', color = 'black')
    ax.set_xlabel('nm')
    ax.set_ylabel('QFLS (eV)')

    N = 2
    lines = []
    for i in range(N):
        lines.append(plt.plot([], [], label='Plot '+str(i+1))[0])

    time_label = plt.text(0.05, 0.05, '', transform=ax.transAxes) # initialize the time label for the graph

    patches = lines

    def init():
        time_label.set_text('') 
        for i, line in enumerate(lines):
            line.set_data([], [])
        return lines[0], lines[1], time_label #return everything that must be updated

    def animate(i):

        global u1, u2, time1, time2 # Necessary. 

        # constant illumination
        #u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.SL, pset1.SR, n_exc = n_exc)
        #u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.SL, pset2.SR, n_exc = n_exc)              

        #print(f'Elapsed time: {time1*1e12:.0f} ps', end = '\r')
        if pset1.pulse_len == 'None':
            u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)
        else:
            u1, time1 = EulerHeatConstBCSparse(u1 + pulse(time1, pset1.pulse_len)*pset1.dt * pset1.n0, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)

        if pset2.pulse_len == 'None':
            u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)
        else:
            u2, time2 = EulerHeatConstBCSparse(u2 + pulse(time2, pset2.pulse_len)*pset2.dt * pset2.n0, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)
        #u2 = u2 + pulse(time2)*pset2.dt * pset2.n0
        #time2 = time1     
        
        # Band gap
        Eg = 1.55*q #J
        
        # Effective density of states
        Nc = 2e18 #1/cm3
        Nv = 2e18 #1/cm3

        QFLS1 = k*T_RT * np.log(u1**2/(Nc*Nv) * math.exp(Eg/(k*T_RT)))/q
        QFLS2 = k*T_RT * np.log(u2**2/(Nc*Nv) * math.exp(Eg/(k*T_RT)))/q

        time1_ns = time1*1e9    
        time2_ns = time2*1e9    
        #time_label.set_text('time1 = %.1f ns' % time1_ns + ', time2 = %.1f ns' % time2_ns) # Display the current time to the accuracy of your liking.
        #time_label.set_text('time1 = %.3f ns' % time1_ns)
        time_label.set_text(f'time 1 = {time1_ns:.3f} ns')

        lines[0].set_data(pset1.x*1e7, QFLS1)
        lines[1].set_data(pset2.x*1e7, QFLS2)
            
        return lines[0], lines[1], time_label #return everything that must be updated

    anim = animation.FuncAnimation(fig, animate, frames=int(finaltime/pset1.dt/nr_times), interval=interval, init_func=init, blit=True, repeat=False)
    plt.legend()
    plt.show()



class TRPL_param:

    def __init__(self, dt = 2e-12, finaltime = 200e-9, thickness = 350, N_points = 50, alpha = 1e5, P_exc = 1e10, pulse_len = 60e-12, mu = 1, k1 = 0, k2 = 1e-10, k3 = 8.8e-29, SL = 0, SR = 0):

        self.dt = dt # s
        self.finaltime = finaltime # s
        self.thickness = thickness # nm
        self.N_points = N_points
        self.x = np.linspace(0, thickness, N_points) * 1e-7 # cm 
        self.dx = self.x[1]-self.x[0]
        self.r = dt / self.dx**2
        #Perovskites: alpha = 1e5 # 1/cm
        self.alpha = alpha
        self.P_exc = P_exc # 1/cm2
        self.pulse_len = pulse_len # pulse length in s
        self.mu = mu # cm2/(Vs)
        #self.D = k * T_RT / q * mu
        self.k1 = k1 # 7.47e6 1/s
        self.k2 = k2 # cm3/s
        self.k3 = k3 # Auger recombination constant in cm6/s
        self.SL = SL # 5.89e4 cm/s
        self.SR = SR # cm/s
        self.n0 = lambeer(self.alpha, self.x, self.P_exc)
        
    def copy(self):
        
        return TRPL_param(dt = self.dt, finaltime = self.finaltime, thickness = self.thickness, N_points = self.N_points, alpha = self.alpha, P_exc = self.P_exc, pulse_len = self.pulse_len, mu = self.mu, k1 = self.k1, k2 = self.k2, k3 = self.k3, SL = self.SL, SR = self.SR)

    @staticmethod
    def D_from_mu(mu, T = T_RT):
        D = k * T / q * mu
        return D
    
    def replace_with_fit(self, what_to_fit, fit_value):
        """
        After a fit of what_to_fit the old parameter is replaced by the fitted parameter.

        Parameters
        ----------
        what_to_fit : string
            Defines the parameter that was fitted.
        fit_value : float
            Fitted value.

        Returns
        -------
        None.

        """
        if what_to_fit == 'mu':
            self.mu = fit_value
            self.unit = 'cm2/(Vs)'
        if what_to_fit == 'k1':
            self.k1 = fit_value
            self.unit = '1/s'
        if what_to_fit =='k2':
            self.k2 = fit_value
            self.unit = 'cm3/s'
        if what_to_fit == 'SR':
            self.SR = fit_value
            self.unit = 'cm/s'
        if what_to_fit == 'SL':
            self.SL = fit_value
            self.unit = 'cm/s'

    
class TRPL_data(xy_data):
    
    def __init__(self, ns, cts, quants = dict(x = "Time", y = "Intensity"), units = dict(x = "ns", y = "cts"),  name = '', FN = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        super().__init__(ns, cts, quants = quants, units = units, name = name, plotstyle = plotstyle, check_data = check_data)
        self.FN = FN
        self.mexp_exist = False
        self.savgol_exist = False
        self.plotrange_left = 0
        self.plotrange_right = 100
        
    def copy(self):
        dat = super().copy()
        dat.FN = self.FN
        dat.mexp_exist = self.mexp_exist
        dat.savgol_exist = self.savgol_exist
        dat.plotrange_left = self.plotrange_left
        dat.plotrange_right = self.plotrange_right     
        return dat
        
    def load(directory, FN = '', name = '', delimiter = ',', header = 'infer', time_unit = 'ns'):

        """
        Loads a sinlge TRPL data.
        """

        if FN == '':
            print('Warning: No filename chosen')

        dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)

        ns = np.array(dat)[:,0]
        if time_unit == 'us':
            ns = ns * 1000
        cts = np.array(dat)[:,1]

        return TRPL_data(ns, cts, name = name, FN = FN)
    
    
    def mono_expfit(self, start = 400, stop = None, p0 = (1, 500), showparam = False):
        
        f = lambda t, a, tau : a * np.e**(-t/tau)
        
        ind_min = findind(self.x, start)
        if stop == None:
            ind_max = len(self.x) - 1
        else:
            ind_max = findind(self.x, stop)
        r = range(ind_min, ind_max+1)
        
        popt, pcov = curve_fit(f, self.x[r], self.y[r], p0)

        mexpfit = self.copy()
        mexpfit.y = f(self.x, *popt)
        mexpfit.name = 'mono-exp. fit'        
                
        #popt[0]: a, popt[1]: tau
        mexpfit.popt = popt
        mexpfit.start = start
        if stop == None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop

        if showparam:
            print('Fit function f = a * e**(-t/tau)')
            print(f'a = {popt[0]:.2f}, tau = {popt[1]:.0f} ns')
        
        return mexpfit
    
    def mult2_expfit(self, start = 0, stop = None, p0 = (1, 1e-1, 10, 100), showparam = False):
        '''
        2-exponential fit of TRPL data self.
        Parameters
        ----------
        start : float, optional
            Start time in ns. The default is 0.
        stop : float, optional
            Stop time in ns. The default is None.
        p0 : 4-tuple 
            starting parameters, p0[0:1]: exponential prefactor, p0[2:3]: time in ns. 
            The default is (1, 1e-1, 10, 100).

        Returns
        -------
        mexpfit : TRPL_data
            Fit curve. In addition the optimized fit parameters popt, start, and stop are returned as parameters.
            
        Example (taken from mul3_expfit, not yet tested)
        -------
        #param = TRPL_param(finaltime = 500e-9, mu = 0.1, k1 = 1e7)
        #example = TRPL_data.from_param(param, show_progress = True)
        #example.equidist(right = 50, delta = 1)
        p0 = (0.8, 0.2, 10, 100)
        ns = np.arange(501)
        example = TRPL_data.gen_m3ed(ns, p0)
        #example.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        ex_fit = example.mult2_expfit(start = 0, stop = 500)
        both = mTRPL_data([example, ex_fit])
        both.label(['orig', 'fit'])
        both.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        d = example.delta(ex_fit, left = 2, right = 400)
        d.plot()
        '''
        f = lambda t, a1, a2, tau1, tau2 : a1 * np.e**(-t/tau1) + a2 * np.e**(-t/tau2)
        
        ind_min = findind(self.x, start)
        if stop == None:
            ind_max = len(self.x)-1
        else:
            ind_max = findind(self.x, stop)

        r = range(ind_min, ind_max+1)
        
        popt, pcov = curve_fit(f, self.x[r], self.y[r], p0)
        
        mexpfit = self.copy()
        mexpfit.y = f(self.x, *popt)
        mexpfit.name = '2-exp. fit'        
        mexpfit.popt = popt
        mexpfit.start = start
        if stop == None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop
            
        if showparam:
            print('Fit function f = a1 * e**(-t/tau1) + a2 * e**(-t/tau2)')
            print(f'a1 = {popt[0]:.2f}, tau1 = {popt[2]:.0f} ns')
            print(f'a2 = {popt[1]:.2f}, tau1 = {popt[3]:.0f} ns')
            
        return mexpfit
    
    def mult3_expfit(self, start = 0, stop = None, p0 = (1, 1e-1, 1e-2, 5, 20, 100), showparam = False):
        '''
        3-exponential fit of TRPL data self.
        Parameters
        ----------
        start : float, optional
            Start time in ns. The default is 0.
        stop : float, optional
            Stop time in ns. The default is None.
        p0 : 8-tuple 
            starting parameters, p0[0:2]: exponential prefactor, p0[3:5]: time in ns. 
            The default is (1, 1e-1, 1e-2, 5, 20, 100).

        Returns
        -------
        mexpfit : TRPL_data
            Fit curve. In addition the optimized fit parameters popt, start, and stop are returned as parameters.
            
        Example
        -------
        #param = TRPL_param(finaltime = 500e-9, mu = 0.1, k1 = 1e7)
        #example = TRPL_data.from_param(param, show_progress = True)
        #example.equidist(right = 50, delta = 1)
        p0 = (0.4, 0.4, 0.1, 10, 30, 100)
        ns = np.arange(501)
        example = TRPL_data.gen_m3ed(ns, p0)
        #example.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        ex_fit = example.mult3_expfit(start = 0, stop = 500)
        both = mTRPL_data([example, ex_fit])
        both.label(['orig', 'fit'])
        both.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        d = example.delta(ex_fit, left = 2, right = 400)
        d.plot()
        '''
        f = lambda t, a1, a2, a3, tau1, tau2, tau3 : a1 * np.e**(-t/tau1) + a2 * np.e**(-t/tau2) + a3 * np.e**(-t/tau3)
        
        ind_min = findind(self.x, start)
        if stop == None:
            ind_max = len(self.x)-1
        else:
            ind_max = findind(self.x, stop)

        r = range(ind_min, ind_max+1)
        
        popt, pcov = curve_fit(f, self.x[r], self.y[r], p0)
        
        mexpfit = self.copy()
        mexpfit.y = f(self.x, *popt)
        mexpfit.name = '3-exp. fit'        
        mexpfit.popt = popt
        mexpfit.start = start
        if stop == None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop
            
        if showparam:
            print('Fit function f = a1 * e**(-t/tau1) + a2 * e**(-t/tau2) + a3 * e**(-t/tau3)')
            print(f'a1 = {popt[0]:.2f}, tau1 = {popt[3]:.1f} ns')
            print(f'a2 = {popt[1]:.2f}, tau1 = {popt[4]:.0f} ns')
            print(f'a3 = {popt[2]:.2f}, tau1 = {popt[5]:.0f} ns')
            
        return mexpfit
    
    def mult4_expfit(self, start = 0, stop = None, p0 = (1, 1e-1, 1e-2, 1e-3, 5, 20, 100, 500), showparam = False):
        '''
        4-exponential fit of TRPL data self.
        Parameters
        ----------
        start : float, optional
            Start time in ns. The default is 0.
        stop : float, optional
            Stop time in ns. The default is None.
        p0 : 8-tuple 
            starting parameters, p0[0:3]: exponential prefactor, p0[4:7]: time in ns. 
            The default is (1, 1e-1, 1e-2, 1e-3, 5, 20, 100, 500).

        Returns
        -------
        mexpfit : TRPL_data
            Fit curve. In addition the optimized fit parameters popt, start, and stop are returned as parameters.
            
        Example
        -------
        #param = TRPL_param(finaltime = 500e-9, mu = 0.1, k1 = 1e7)
        #example = TRPL_data.from_param(param, show_progress = True)
        #example.equidist(right = 50, delta = 1)
        p0 = (0.4, 0.4, 0.1, 10, 30, 100)
        ns = np.arange(501)
        example = TRPL_data.gen_m3ed(ns, p0)
        #example.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        p0 = (0.1, 0.01, 0.01, 0.8, 20, 30, 50, 50)
        ex_fit = example.mult4_expfit(start = 2, stop = 500, p0 = p0)
        both = mTRPL_data([example, ex_fit])
        both.label(['orig', 'fit'])
        both.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        d = example.delta(ex_fit, left = 2, right = 400)
        d.plot()

        '''
        f = lambda t, a1, a2, a3, a4, tau1, tau2, tau3, tau4 : a1 * np.e**(-t/tau1) + a2 * np.e**(-t/tau2) + a3 * np.e**(-t/tau3) + a4 * np.e**(-t/tau4)
        
        ind_min = findind(self.x, start)
        if stop == None:
            ind_max = len(self.x)-1
        else:
            ind_max = findind(self.x, stop)

        r = range(ind_min, ind_max+1)
        
        popt, pcov = curve_fit(f, self.x[r], self.y[r], p0)
                
        mexpfit = self.copy()
        mexpfit.y = f(self.x, *popt)
        mexpfit.name = '4-exp. fit'        
        mexpfit.popt = popt
        mexpfit.start = start

        if stop == None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop

        if showparam:
            print('Fit function f = a1 * e**(-t/tau1) + a2 * e**(-t/tau2) + a3 * e**(-t/tau3) + a4 * e**(-t/tau4)')
            print(f'a1 = {popt[0]:.2f}, tau1 = {popt[4]:.1f} ns')
            print(f'a2 = {popt[1]:.2f}, tau1 = {popt[5]:.1f} ns')
            print(f'a3 = {popt[2]:.2f}, tau1 = {popt[6]:.0f} ns')
            print(f'a3 = {popt[3]:.2f}, tau1 = {popt[7]:.0f} ns')

        return mexpfit
    
    @staticmethod
    def gen_med(ns, a, tau):
        """
        Generates a monoexponential data.
        ns: times in ns
        a: exponential prefactor
        tau: time in ns
        """
        d = TRPL_data(ns, a * np.e**(-ns/tau), name = f'a = {a:.1e}, tau = {tau:.0f}')
        d.popt = [a, tau]
        d.start = 0
        d.stop = ns[-1]
        return d

    @staticmethod
    def gen_m2ed(ns, p0):
        """
        Generates a 2 exponential data.
        ns: times in ns
        p0[0:1]: exponential prefactor
        p0[2:3]: time in ns
        """
        a1 = p0[0]
        a2 = p0[1]
        tau1 = p0[3]
        tau2 = p0[4]
        d = TRPL_data(ns, a1 * np.e**(-ns/tau1) + a2 * np.e**(-ns/tau2))
        d.popt = p0
        d.start = 0
        d.stop = ns[-1]
        return d    

    
    @staticmethod
    def gen_m3ed(ns, p0):
        """
        Generates a 3 exponential data.
        ns: times in ns
        p0[0:2]: exponential prefactor
        p0[3:5]: time in ns
        """
        a1 = p0[0]
        a2 = p0[1]
        a3 = p0[2]
        tau1 = p0[3]
        tau2 = p0[4]
        tau3 = p0[5]
        d = TRPL_data(ns, a1 * np.e**(-ns/tau1) + a2 * np.e**(-ns/tau2) + a3 * np.e**(-ns/tau3))
        d.popt = p0
        d.start = 0
        d.stop = ns[-1]
        return d    

    @staticmethod
    def gen_m4ed(ns, p0):
        """
        Generates a 4 exponential data.
        ns: times in ns
        p0[0:3]: exponential prefactor
        p0[4:7]: time in ns
        """
        a1 = p0[0]
        a2 = p0[1]
        a3 = p0[2]
        a4 = p0[3]
        tau1 = p0[4]
        tau2 = p0[5]
        tau3 = p0[6]
        tau4 = p0[7]
        d = TRPL_data(ns, a1 * np.e**(-ns/tau1) + a2 * np.e**(-ns/tau2) + a3 * np.e**(-ns/tau3) + a4 * np.e**(-ns/tau4))
        d.popt = p0
        d.start = 0
        d.stop = ns[-1]
        return d
    
    def dlifetime(self, x = 'time', m = 2, wavelength = 510, film_thickness = 500, fluence = 5e-9, ni = 8.05e4):  
        # m = 1 for low level injection, m = 2 for high level injection
        # ni = 8.05e4 cm-3
        # fluence in J/cm2
        
        ln_self = self.copy()
        ln_self.y = np.log(self.y)
        diff_tau = ln_self.diff()
        diff_tau.name = self.name
        diff_tau.y = -m/(diff_tau.y) * 1e-9
        #diff_tau.name = 'Differential lifetime'
        diff_tau.qy = 'Decay time'
        diff_tau.uy = 's'
        if x == 'QFLS':
            # QFLS in eV
            QFLS_0 = k * T_RT / q * np.log(initial_carrier_conc(wavelength = wavelength, film_thickness = film_thickness, fluence = fluence)**2/ni**2)
            #print(QFLS_0)
            diff_tau.x = QFLS_0 + k * T_RT / q * np.log(self.y/self.y[0])
            diff_tau.qx = 'Quasi-Fermi level splitting'
            diff_tau.ux = 'eV'
            diff_tau.reverse()
        return diff_tau

    
    @staticmethod
    def from_param(p, time_delta = 0.01e-9, name = '', normalize_ns = None, normalize_cts = None, model = 'simple', show_progress = False):
        """
        Generate a TRPL curve from the TRPL parameter p.

        Parameters
        ----------
        p : TYPE
            DESCRIPTION.
        time_delta : float, optional
            Delta time of the fime array, in s. The default is 0.01 ns.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
    
        t=0
        if p.pulse_len == None:
            n = p.n0
        else:
            n = np.zeros(len(p.x))
    
        TRPL_list = []
        ns_list = []
    
        TRPL_list.append(PLsignal(n, p.dx, p.k2))   
        ns_list.append(0)
            
        if p.pulse_len != None:
    
            # Start the calculation over the pulse length * 3 in 1ps steps
            #dt = 1e-12
            nr_times = 1
    
            for i in range(int(p.pulse_len * 3 / p.dt)):
    
                if model == 'simple':
                    n, t = EulerHeatConstBCSparse_simple(n + pulse(t, p.pulse_len)*p.dt * p.n0, p.x, p.dt, nr_times, t, p.mu, p.k1, p.SL, p.SR)
   
                else:
                    n, t = EulerHeatConstBCSparse(n + pulse(t, p.pulse_len)*p.dt * p.n0, p.x, p.dt, nr_times, t, p.mu, p.k1, p.k2, p.k3, p.SL, p.SR)
    
                TRPL_list.append(PLsignal(n, p.dx, p.k2))   
                ns_list.append(t*1e9)
    
            # Now the rest of the time 
        
        time_delta = time_delta #s # origianlly 0.1 ns, display PL intensity every time_delta s
        nr_times = int(time_delta / p.dt)
        
        for i in range(int(p.finaltime/(p.dt*nr_times))):
    
            if model == 'simple':
                n, t = EulerHeatConstBCSparse_simple(n, p.x, p.dt, nr_times, t, p.mu, p.k1, p.SL, p.SR)
   
            else:
                n, t = EulerHeatConstBCSparse(n, p.x, p.dt, nr_times, t, p.mu, p.k1, p.k2, p.k3, p.SL, p.SR)

            TRPL_list.append(PLsignal(n, p.dx, p.k2))  
            ns_list.append(t*1e9)
            
            if show_progress == True:
                print(f'{t*1e9:.0f} ns of {p.finaltime*1e9:.0f} ns', end = '\r')
    
        cts = np.array(TRPL_list)
        ns= np.array(ns_list)
    
        data = TRPL_data(ns, cts, name = name)
    
        if normalize_ns != None:
            data.y = data.y * normalize_cts / data.y_of(normalize_ns)
            
        else:
            data.y = data.y/max(data.y)
    
        return data
    

    # Fit for the full continuity equation
    def model_fit(self, param, fit_from = 'end', fit_range_ns = [30, 40], what = 'SL', start_value= 0, verbose = 2, gtol = 1e-12):
        #Automized fitting routine
        
        fit_range_ns = np.array(fit_range_ns)
        
        #Parameters p have to be duplicated because finaltime is changed
        p = param.copy()
        #Necessary to redefine finaltime, because this is the time until the TRPL curve will be calculated
        #in the function TRPL_data.from_param
        p.finaltime = fit_range_ns[1]*1e-9
    
        # Time array in ns 
        begin_ns = 0 #ns
        end_ns = fit_range_ns[1] #ns
        Nval = int(end_ns)+1
        ns = np.linspace(begin_ns, end_ns, Nval)
    
        datatobefitted = interpolated_array(self.x, self.y, ns)
        
        if fit_from =='begin':
            datatobefitted = datatobefitted/datatobefitted[fit_range_ns[0]]
        if fit_from =='end':
            datatobefitted = datatobefitted/datatobefitted[-1]
    
            
        def data_minus_fit(args):
    
            if what == 'mu':
                p.mu = args
            if what == 'k1':
                p.k1 = args
            if what =='k2':
                p.k2 = args
            if what == 'SR':
                p.SR = args
            if what == 'SL':
                p.SL = args
                
            d = TRPL_data.from_param(p, time_delta = 0.01e-9)
            t_ns, TRPL = d.x, d.y
    
            fitd = interpolated_array(t_ns, TRPL, ns)
            
            if fit_from =='begin':
                fitdata = fitd/fitd[fit_range_ns[0]]
            if fit_from =='end':
                fitdata = fitd/fitd[-1]
            
            delta = datatobefitted[fit_range_ns[0]:] - fitdata[fit_range_ns[0]:]
            criteria = np.sqrt(np.dot(delta, delta) / len(delta))
            return criteria
    
        result = least_squares(fun = data_minus_fit, x0 = [start_value], verbose = verbose, gtol = gtol)
        
        return result.x
    
#________________________________________________________________________________________
# simplified model dn/dt = -k1*n - k2*n**2 begin

    def k1_k2_fit(self, start = None, stop = None, x0 = [3e5, 2.4e4], used_for_fit = 'savgol', savgol_param = None, show_all = False, **kwargs):
        """
        Fits k1, k2 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """

        if start == None:
            start = 0
        if stop == None:
            stop = self.x[-1]

        if used_for_fit == 'savgol':
            if savgol_param == None:
                savgol_param = dict(n1 = 51, n2 = 1, name = 'Savgol')
            dat = self.savgol(**savgol_param)
        else:
            dat = self

        start_idx = dat.x_idx_of(start)
        stop_idx = dat.x_idx_of(stop)
        r = range(start_idx, stop_idx+1)

        t = dat.x[r] - start
        n0 = dat.y[start_idx]
        p0 = [n0, x0[0], x0[1]]

        def n_of_t(t, n0, k1, k2):
            # n**2 is proportional to PL
            #t in ns
            return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)

        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0 = p0, bounds=(0, np.inf), **kwargs)

        n0 = popt[0]
        k1 = popt[1]
        k2 = popt[2]

        fit = TRPL_data(dat.x[r], n_of_t(dat.x[r], *popt))

        if show_all:
            dta = mTRPL_data([self, dat, fit])
            dta.label([self.name, 'savgol', 'fit'])
            dta.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
            dta.sa[1].plotstyle = dict(linestyle = '-', color = 'orange', linewidth = 3)
            dta.sa[2].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)
            m = dta.max_within(left = start, right = stop)
            dta.plot(yscale = 'log', left = 0, right = stop, bottom = m/100, top = m*1.1, plotstyle = 'individual')

        da_new = mTRPL_data([self, fit])
        da_new.label([self.name, f'fit, k1 = {k1:.2e} s-1, k2 = {k2:.2e} cm3 s-1'])
        da_new.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
        da_new.sa[1].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)

        return da_new, popt


    def k1_fit(self, start = None, stop = None, x0 = [2.4e4], k2 = 1e-8, used_for_fit = 'savgol', savgol_param = None, show_all = False, **kwargs):
        """
        Fits k1 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """

        if start == None:
            start = 0
        if stop == None:
            stop = self.x[-1]

        if used_for_fit == 'savgol':
            if savgol_param == None:
                savgol_param = dict(n1 = 51, n2 = 1, name = 'Savgol')
            dat = self.savgol(**savgol_param)
        else:
            dat = self

        start_idx = dat.x_idx_of(start)
        stop_idx = dat.x_idx_of(stop)
        r = range(start_idx, stop_idx+1)

        t = dat.x[r] - start
        n0 = dat.y[start_idx]
        p0 = [n0, x0[0]]

        def n_of_t(t, n0, k1):
            # n**2 is proportional to PL
            #t in ns
            if k2 != 0:
                return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)
            else:
                return n0*np.exp(-k1*t*1e-9)

        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0 = p0, bounds=(0, np.inf), **kwargs)

        n0 = popt[0]
        k1 = popt[1]

        fit = TRPL_data(dat.x[r], n_of_t(dat.x[r], *popt))

        if show_all:
            dta = mTRPL_data([self, dat, fit])
            dta.label([self.name, 'savgol', 'fit'])
            dta.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
            dta.sa[1].plotstyle = dict(linestyle = '-', color = 'orange', linewidth = 3)
            dta.sa[2].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)
            m = dta.max_within(left = start, right = stop)
            dta.plot(yscale = 'log', left = 0, right = stop, bottom = m/100, top = m*1.1, plotstyle = 'individual')

        da_new = mTRPL_data([self, fit])
        da_new.label([self.name, f'fit, k1 = {k1:.2e} s-1, k2 = {k2:.2e} cm3 s-1'])
        da_new.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
        da_new.sa[1].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)

        return da_new, popt


    def k2_fit(self, start = None, stop = None, x0 = [1e-8], k1 = 1e6, used_for_fit = 'savgol', savgol_param = None, show_all = False, **kwargs):
        """
        Fits k2 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """

        if start == None:
            start = 0
        if stop == None:
            stop = self.x[-1]

        if used_for_fit == 'savgol':
            if savgol_param == None:
                savgol_param = dict(n1 = 51, n2 = 1, name = 'Savgol')
            dat = self.savgol(**savgol_param)
        else:
            dat = self

        start_idx = dat.x_idx_of(start)
        stop_idx = dat.x_idx_of(stop)
        r = range(start_idx, stop_idx+1)

        t = dat.x[r] - start
        n0 = dat.y[start_idx]
        p0 = [n0, x0[0]]

        def n_of_t(t, n0, k2):
            # n**2 is proportional to PL
            #t in ns
            return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)
            
        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0 = p0, bounds=([n0/10, 0], np.inf), **kwargs) #if the lower n0 bound is set to 0 then n0 goes towards 0 sometimes and results in an error

        n0 = popt[0]
        k2 = popt[1]

        fit = TRPL_data(dat.x[r], n_of_t(dat.x[r], *popt))

        if show_all:
            dta = mTRPL_data([self, dat, fit])
            dta.label([self.name, 'savgol', 'fit'])
            dta.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
            dta.sa[1].plotstyle = dict(linestyle = '-', color = 'orange', linewidth = 3)
            dta.sa[2].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)
            m = dta.max_within(left = start, right = stop)
            dta.plot(yscale = 'log', left = 0, right = stop, bottom = m/100, top = m*1.1, plotstyle = 'individual')

        da_new = mTRPL_data([self, fit])
        da_new.label([self.name, f'fit, k1 = {k1:.2e} s-1, k2 = {k2:.2e} cm3 s-1'])
        da_new.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
        da_new.sa[1].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)

        return da_new, popt



    def n0_fit(self, start = None, stop = None, n0 = 1e13, k1 = 1e6, k2 = 1e-8, used_for_fit = 'savgol', savgol_param = None, show_all = False, **kwargs):
        """
        Fits n0 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """

        if start == None:
            start = 0
        if stop == None:
            stop = self.x[-1]

        if used_for_fit == 'savgol':
            if savgol_param == None:
                savgol_param = dict(n1 = 51, n2 = 1, name = 'Savgol')
            dat = self.savgol(**savgol_param)
        else:
            dat = self

        start_idx = dat.x_idx_of(start)
        stop_idx = dat.x_idx_of(stop)
        r = range(start_idx, stop_idx+1)

        t = dat.x[r] - start
        n0 = dat.y[start_idx]
        p0 = [n0]

        def n_of_t(t, n0):
            # n**2 is proportional to PL
            #t in ns
            return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)

        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0 = p0, bounds=(0, np.inf), **kwargs)

        n0 = popt[0]

        fit = TRPL_data(dat.x[r], n_of_t(dat.x[r], *popt))

        if show_all:
            dta = mTRPL_data([self, dat, fit])
            dta.label([self.name, 'savgol', 'fit'])
            dta.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
            dta.sa[1].plotstyle = dict(linestyle = '-', color = 'orange', linewidth = 3)
            dta.sa[2].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)
            m = dta.max_within(left = start, right = stop)
            dta.plot(yscale = 'log', left = 0, right = stop, bottom = m/100, top = m*1.1, plotstyle = 'individual')

        da_new = mTRPL_data([self, fit])
        da_new.label([self.name, f'fit, k1 = {k1:.2e} s-1, k2 = {k2:.2e} cm3 s-1'])
        da_new.sa[0].plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 1)
        da_new.sa[1].plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)

        return da_new, popt
    
    def k1_k2_model_fit(self, what_to_fit = ['k1', 'k2'], start = None, stop = None, n0 = 1e-15, k1 = 1e6, k2 = 1e-7, show = None):
    
        if start == None:
            start = 0
        if stop == None:
            stop = self.x[-1]
            
        #n00 will be used for the transformations from PL intensity into carrier concentration and back
        n00 = n0
            
        # Transform from PL instensity into  carrier concentration
        cc = self.copy()
        cc.y = np.sqrt(abs(cc.y))
        cc.y = cc.y/max(cc.y)*n00
        cc.qy = 'Carrier concentration'
        cc.uy = '1/cm3'
        
        if show != None:
            if (show == 'all') or ('step 1' in show):
                #m_max = cc.max_within(left = start, right = stop)
                #m_min = cc.min_within(left = start, right = stop)
                #cc.plot(yscale = 'log', left = start, right = stop, bottom = abs(m_min)*0.9, top = m_max*1.1)
                print('Step 1: Transformation from PL intensity into carrier concentration')
                cc.plot(yscale = 'log')
            # show_all is used for the actual calculation of the fit curves
            if (show == 'all') or ('step 2' in show):
                show_all = True
                print('Step 2: Fit')    
        else:
            show_all = False
                
        if what_to_fit == None:
            # no fit, i.e. only n0        
            dta, [n0] = cc.n0_fit(start, stop, n0 = n0, k1 = k1, k2 = k2, show_all = show_all)
    
        else:
            # parameters for the curve_fit routine
            kwargs = dict(verbose = 0, gtol = 1e-12, xtol = None)
    
            if ('k1' in what_to_fit) and ('k2' in what_to_fit):
                # fit k1 and k2:
                dta, [n0, k1, k2] = cc.k1_k2_fit(start, stop, x0 = [k1, k2], show_all = show_all, used_for_fit = 'savgol', **kwargs)
    
            elif ('k1' in what_to_fit):
                # fit k1 only
                dta, [n0, k1] = cc.k1_fit(start, stop, x0 = [k1], k2 = k2, show_all = show_all, used_for_fit = 'savgol', **kwargs)
    
            elif ('k2' in what_to_fit):
                # fit k2 only
                dta, [n0, k2] = cc.k2_fit(start, stop, x0 = [k2], k1 = k1, show_all = show_all, used_for_fit = 'savgol', **kwargs)
    
        if show != None:
            if (show == 'all') or ('step 3' in show):
                print('Step 3: Show fit.')
                m_max = dta.sa[0].y[0]
                m_min = dta.sa[1].y[-1]
                dta.plot(yscale = 'log', left = 0, right = stop, bottom = m_min*0.8, top = m_max*1.2, plotstyle = 'individual')
                print(f'Carrier concentration n(t={start} ns) = {n0:.2e} 1/cm3')
        
        # Transform back from carrier concentration to PL intensity
    
        PL_dta = dta.copy()
        for idx, sp in enumerate(PL_dta.sa):
            sp.y = sp.y / n00
            sp.y = sp.y**2 * self.y[0]
            sp.qy = 'PL intensity'
            sp.uy = 'cts.'
    
        if show != None:
            if (show == 'all') or ('step 4' in show):
                print('Step 4: Back-transformation from carrier concentration into PL intensity.')
                PL_dta.plot(yscale = 'log', right = stop)
        
        # return self and fit (with the right label) and all parameters
        return PL_dta, [n0, k1, k2]

# simplified model dn/dt = -k1*n - k2*n**2 end
#________________________________________________________________________________________

    def del_bg(self, plot_details = False, norm_val = None, start = None, stop = None):
        """
        Deletes the background of raw TRPL data. 
        """

        def bg_idx_start():
            idx = 0
            stop = False
            while not stop:
                if self.y[idx+1] >= self.y[idx]:
                    idx += 1
                else:
                    stop = True
            return idx

        def bg_idx_stop():
            idx = findind(self.y,max(self.y))
            stop = False
            while not stop:
                if self.y[idx-1] <= self.y[idx]:
                    idx -= 1
                else:
                    stop = True
            return idx

        if start == None:
            start = bg_idx_start()
        else:
            start = self.x_idx_of(start)
        if stop == None:
            stop = bg_idx_stop()
        else:
            stop = self.x_idx_of(stop)
        if stop < start:
            print('Attention [TRPL_data.del_bg()]: stop < start, hence the alternative routine is chosen!')
            #This can happen, if there is very little noise data points are selected. In this case take the first value > 0 as start and the maximum - 1 as stop
            start = 0
            while self.y[start] == 0:
                start += 1
            stop = findind(self.y, max(self.y))
            stop -= 1
        r = range(start, stop + 1)
        m = max(self.y[r])
        av = np.average(self.y[r])
        dat = self.copy()
        dat.y = dat.y - av
        if norm_val != None:
            dat.normalize(norm_val = norm_val)
        
        if plot_details:
            print(f'______{self.name}________')
            self.plot(yscale = 'linear', left = self.x[start]*0, right = self.x[stop]*1.2, bottom = 0, top = m*2, vline = [self.x[start], self.x[stop]], title = 'noise')
            #Show original and bg subtracted data
            dat_all = mTRPL_data([self, dat])
            dat_all.label(['original', 'corrected'])
            dat_all.plot(yscale = 'log', title = self.name, divisor = 1e5)
        
        return dat
    
    def shift_zero(self, ns):
        ind = findind(self.x, ns)
        self.x = self.x[ind:] - self.x[ind]
        self.y = self.y[ind:]
            
    def shift_to_max(self, plot_details = False, left = None, right = None):
        """
        Shift the data so that it starts at maximum.
        """
        idx_shift = findind(self.y,max(self.y))
        
        dat = self.copy()
        dat.shift_zero(dat.x[idx_shift])

        if plot_details:
            if left == None:
                left = 0
            if right == None:
                right = max(dat.x)
            dat.plotstyle = dict(linestyle = '-', linewidth = 5, markersize = 5)
            dat_max = dat.max_within(left = left, right = right)
            dat_min = dat.min_within(left = left, right = right, absolute = True)
            #dat_test.plot(yscale = 'log', bottom = dat_min*0.9, top = dat_max*1.1, left = left, right = right)
            dat.plot(yscale = 'log', bottom = dat_min*0.9, top = dat_max*1.1, left = left, right = right)
        
        return dat




class mTRPL_data(mxy_data):
    """
    sa is a list of TRPL_data.
    """
    
    def __init__(self, sa):
        super().__init__(sa)
        
    @classmethod
    def load_individual(cls, directory, FNs = [], delimiter = ',', header = 'infer', quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False):

        """
        Loads all xy data in individual files in directory.
        """
        if len(FNs) == 0:
            FNs = listdir(directory)

        sa = []
        
        for i, FN in enumerate(FNs):
            
            dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)
            
            npdat = np.array(dat, dtype = np.float64)
            x = npdat[:,0]
            y = npdat[:,1]

            if cls.__name__ == 'mTRPL_data':
                sp = TRPL_data(x, y, quants, units, FN)
       
            if take_quants_and_units_from_file:

                col0 = list(dat)[0]
                sp.qx = col0.split(' (')[0]
                if ' (' in col0:
                    sp.ux = col0.split(' (')[1].split(')')[0]

                col1 = list(dat)[1]
                sp.qy = col1.split(' (')[0]
                if ' (' in col1:
                    sp.uy = col1.split(' (')[1].split(')')[0]

            sa.append(sp)    
        spectra = cls(sa)
    
        return spectra
    
    def copy_old(self):
        sa_new = []
        for i, sp in enumerate(self.sa):
            sa_new.append(sp.copy())
        return type(self)(sa_new)
        
    def mono_expfit(self, start = 400, stop = None, p0 = (1, 500), showparam = False):
        dafit_sa = []
        for idx, d in enumerate(self.sa):
            if showparam:
                print(d.name)
            dfit = d.mono_expfit(start = start, stop = stop, p0 = p0, showparam = showparam)
            dfit.name = 'mono exp fit_' + d.name
            dafit_sa.append(dfit)
            if showparam:
                dfit.plotstyle = dict(linestyle = '--', color = 'tab:red', linewidth = 2)
                delta = d.residual(dfit)
                print(f'chi**2 = {xy_data.chisquare(d, dfit, right = stop):.2e}')
                delta.plot(right = stop, hline = 0, title = 'Residual plot')
                md = mTRPL_data([d, dfit])
                md.label(['original', 'fit'])
        dafit = mTRPL_data(dafit_sa)  
        dafit.names_to_label(split_ch = '.csv')
        return dafit   
        
    def mult2_expfit(self, start = 0, stop = None, p0 = (1, 1e-1, 10, 100), showparam = False):
        dafit_sa = []
        for idx, d in enumerate(self.sa):
            if showparam:
                print(d.name)
            dfit = d.mult2_expfit(start = start, stop = stop, p0 = p0, showparam = showparam)
            dfit.name = '2 exp fit_' + d.name
            dafit_sa.append(dfit)
            if showparam:
                dfit.plotstyle = dict(linestyle = '--', color = 'tab:red', linewidth = 2)
                delta = d.residual(dfit)
                print(f'chi**2 = {xy_data.chisquare(d, dfit, right = stop):.2e}')
                delta.plot(right = stop, hline = 0, title = 'Residual plot')
                md = mTRPL_data([d, dfit])
                md.label(['original', 'fit'])
        dafit = mTRPL_data(dafit_sa)  
        dafit.names_to_label(split_ch = '.csv')
        return dafit
    
    def mult3_expfit(self, start = 0, stop = None, p0 = (1, 1e-1, 1e-2, 5, 20, 100), showparam = False):
        dafit_sa = []
        for idx, d in enumerate(self.sa):
            if showparam:
                print(d.name)
            dfit = d.mult3_expfit(start = start, stop = stop, p0 = p0, showparam = showparam)
            dfit.name = '3 exp fit_' + d.name
            dafit_sa.append(dfit)
            if showparam:
                dfit.plotstyle = dict(linestyle = '--', color = 'tab:red', linewidth = 2)
                delta = d.residual(dfit)
                print(f'chi**2 = {xy_data.chisquare(d, dfit, right = stop):.2e}')
                delta.plot(right = stop, hline = 0, title = 'Residual plot')
                md = mTRPL_data([d, dfit])
                md.label(['original', 'fit'])
    
        dafit = mTRPL_data(dafit_sa)  
        dafit.names_to_label(split_ch = '.csv')
        return dafit

    def mult4_expfit(self, start = 0, stop = None, p0 = (1, 1e-1, 1e-2, 1e-3, 5, 20, 100, 500), showparam = False):
        dafit_sa = []
        for idx, d in enumerate(self.sa):
            if showparam:
                print(d.name)
            dfit = d.mult4_expfit(start = start, stop = stop, p0 = p0, showparam = showparam)
            dfit.name = '4 exp fit_' + d.name
            dafit_sa.append(dfit)
            if showparam:
                dfit.plotstyle = dict(linestyle = '--', color = 'tab:red', linewidth = 2)
                delta = d.residual(dfit)
                print(f'chi**2 = {xy_data.chisquare(d, dfit, right = stop):.2e}')
                delta.plot(right = stop, hline = 0, title = 'Residual plot')
                md = mTRPL_data([d, dfit])
                md.label(['original', 'fit'])
        dafit = mTRPL_data(dafit_sa)  
        dafit.names_to_label(split_ch = '.csv')
        return dafit


    def dlifetime(self, x = 'time', m = 2, wavelength = 510, film_thickness = 500, fluence = 5e-9, ni = 8.05e4):  
        diff_tau_sa = [] 
        for idx, sp in enumerate(self.sa):
            diff_tau = sp.dlifetime(x = x, m = m, wavelength = wavelength, film_thickness = film_thickness, fluence = fluence, ni = ni)
            diff_tau_sa.append(diff_tau)
        return mTRPL_data(diff_tau_sa)
        
if __name__ == "__main__":

    #Perovskites: alpha = 1e5 # 1/cm
    alpha_per = 8e5
    bg = f1240/800 #eV
    # bg 800nm: above bg photonflux P(0) = 1.7e21 1/(s m2) = 1.7e17 1/(s cm2) = N0
    N0_per = above_bg_photon_flux(f1240/800) * 1e-4 # 1/(s cm2)
    
    # Excitation laser fluence
    F_exc = 3e-9 #3e-9 # J/cm2
    
    #Excitation wavelength and photon energy
    l_exc = 510 # nm
    E_l = f1240 * q / l_exc # J
    
    # TRPL laser pulse length FWHM
    pl = 40e-12 # s
    
    # Excitation photon density
    P_exc = F_exc / E_l # photons / cm2
    
    print(f'Average photon flux during pulse excitation: {P_exc/pl:.2e} 1/(s cm2)')
    print(f'Average number of suns during pulse excitation: {P_exc/pl/N0_per:.2f} suns')
    
    # absorption depth
    ad = 1/alpha_per #cm
    print(f'The absorption depth is {ad * 1e7:.1f} nm ')
     
    # carrier concentration
    c = P_exc / ad
    print(f'Average carrier concentration within absoprtion depth is {c:.2e} 1/cm3')
    
    k1 = 1e6 #1/s
    k2 = 1e-10 #cm3/s
    k3 = 8.8e-29 #cm6/s
    print(f'SRH recombination rate: {k1 * c:.2e} 1/(cm3 s)')
    print(f'Radiative recombination rate: {k2 * c**2:.2e} 1/(cm3 s)')
    print(f'Auger recombination rate: {k3 * c**3:.2e} 1/(cm3 s)')
    
    # Plot the initial carrier concentration profile after the pulse has been absorbed
    plot_ini = True
    if plot_ini:
        p = TRPL_param()
        c_prof = lambeer(alpha_per, p.x, P_exc)
        plt.plot(p.x, c_prof)
        plt.show()
        
    mu = 1
    k1 = 0 #1/s
    SR = 0 #cm/s
    SL = 100
    thickness = 500 #640
    finaltime = 100e-9 #s
    alpha = 1e-3
    P_exc = 1.0
    pulse_len = None
    fit_range_stop_ns = 10 #ns
    
    # Starting parameters
    
    # Excitation laser fluence
    F_exc = 11000e-9 #3e-9 # J/cm2
    #Excitation wavelength and photon energy
    l_exc = 510 # nm
    E_l = f1240 * q / l_exc # J
    # TRPL laser pulse length FWHM
    pulse_len = 100e-12 # s
    # Excitation photon density
    P_exc = F_exc / E_l # photons / cm2
    
    P_exc = 1.0
    pulse_len = None
    
    mu_0 = mu #5 #cm2/Vs
    k1_0 = k1 #1e6 #1/s
    k2_0 = 1e-10 #cm3/s
    k3_0 = 8.8e-29 #cm3/s
    SR_0 = SR #0 #cm/s
    SL_2 = 12.6 #SL_d2 #1400 #cm/s
    SL_1 = 20.5 #SL_d1 # 400
    thickness = thickness #570
    finaltime = finaltime #40e-9
    # Carry out calculation
    
    pset1 = TRPL_param(thickness = thickness, finaltime = finaltime, N_points = 50, alpha = alpha, P_exc = P_exc, pulse_len = pulse_len, mu = mu_0, k1 = k1_0, k2 = k2_0, k3 = k3_0, SL = SL_1)
    # Standard values for k2 and k3: k2 = 1e-10, k3 = 8.8e-29
    pset2 = TRPL_param(thickness = thickness, finaltime = finaltime, N_points = 50, alpha = alpha, P_exc = P_exc, pulse_len = pulse_len, mu = mu_0, k1 = k1_0, k2 = k2_0 * 1e-0, k3 = k3_0 * 1e-1, SL = SL_1)
    
