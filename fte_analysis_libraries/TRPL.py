# -*- coding: utf-8 -*-

import math
from importlib.resources import files as _resource_files
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.sparse import spdiags

system_dir = str(_resource_files('fte_analysis_libraries').joinpath('System_data'))

from typing import Any

from .General import T_RT, c, f1240, findind, h, interpolated_array, k, pi, q
from .Spectrum import above_bg_photon_flux
from .XYdata import MXYData, XYData


def lambeer(alpha: Any, x: np.ndarray, P: float) -> Any:
    """
    P: Excitation photon density in photons /cm2
    """
    return alpha * P * np.e**(-alpha * x)

def pulse(t: float, pl: Any = 60e-12) -> Any:
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
    print(f'The integral over pulse is: {np.trapezoid(pulse(times), times):.3f}')
    plt.plot(times, pulse(times))
    plt.show()

    """

    # pre factor to make integral over gaussian equal 1
    A = 2/pl * math.sqrt(math.log(2)/pi)
    return A * np.e**(-(4*math.log(2)*((t-1.5*pl)/pl)**2)) # Factor of 1.5 is to determine the start point of the pulse so that the integral is still nearly 1


def one_sun_carrier_conc(lifetime: Any = 1000, bg: float = f1240/800, film_thickness: Any = 500, print_with_units: Any = False) -> Any:
    """
    Calculates the carrier concentration at 1 sun in 1/cm3 for a material with a carrier lifetime of lifetime in ns
    and a bandgap of bg in eV and a film thickness in nm.
    """
    cc = above_bg_photon_flux(bg)*lifetime*1e-9/(film_thickness*1e-7)/1e4
    if print_with_units:
        print(f'The one sun carrier concentration is {cc:.2e} 1/cm3')
    return cc

def initial_carrier_conc(wavelength: np.ndarray, film_thickness: Any, fluence: float, print_result: Any = False) -> Any:

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

def EulerHeatConstBCSparse(u0: Any, x: np.ndarray, dt: Any, nr_times: Any, t0: Any, mu: Any, k1: float, k2: float, k3: Any, SL: Any, SR: Any, n_exc: Any = 0) -> Any:
    """Solve n_t = D n_xx - k1 n - k2 n^2 - k3 n^3 using the Euler method.

    Takes an initial condition u0, a domain x, a time step dt, the number
    of iterations nr_times, the initial time t0, and the physical parameters.
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


def EulerHeatConstBCSparse_simple(u0: Any, x: np.ndarray, dt: Any, nr_times: Any, t0: Any, mu: Any, k1: float, SL: Any, SR: Any, n_exc: Any = 0) -> Any:
    """Solve n_t = D n_xx - k1 n (no bimolecular/Auger) using the Euler method.

    Takes an initial condition u0, a domain x, a time step dt, the number
    of iterations nr_times, the initial time t0, and the physical parameters.
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



def MakeTridiagonalMatrix(main: Any, upper_offset_one: Any, lower_offset_one: Any) -> Any:
    """This function will make a tridiagonal 2D matrix
    which has the main array on its main diagonal and the offset_one 
    array on its super and sub diagonals.
    """
    size = len(main)
    offsets = [0,1,-1]
    data = np.vstack((main, upper_offset_one, lower_offset_one))
    A = spdiags(data, offsets, size, size)
    return A

def PLsignal(u: float, dx: Any, k2: float) -> Any:
    """
    P Lsignal.
    
    Parameters
    ----------
    u : float
        U.
    dx : Any
        Dx.
    k2 : float
        K2.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> PLsignal()
    """
    return np.trapezoid(k2*u**2,dx=dx)  # type: ignore


# Plot animation of carrier concentration

def plot_animation(pset1: Any, pset2: Any, interval: Any = 1, ylim: Any=(1e-2,1.2), normalize_to_end: Any = False) -> Any:
    """
    Plot animation.
    
    Parameters
    ----------
    pset1 : Any
        Pset1.
    pset2 : Any
        Pset2.
    interval : Any
        Interval.
    ylim : Any
        Ylim.
    normalize_to_end : Any
        Normalize to end.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> plot_animation()
    """

    global u1, u2, time1, time2 # Necessary.

    finaltime = pset1.finaltime
    time_delta = 0.01e-9 #s

    nr_times = int(time_delta / pset1.dt)

    #pset1.n0 = 0 #no laser excitation
    #pset2.n0 = 0 #no laser excitation

    #u1 = pset1.n0
    if pset1.pulse_len is None:
        u1 = pset1.n0  # type: ignore
    else:
        u1 = np.zeros(len(pset1.x))  # type: ignore
    time1 = 0  # type: ignore

    #u2 = pset2.n0
    if pset2.pulse_len is None:
        u2 = pset2.n0  # type: ignore
    else:
        u2 = np.zeros(len(pset2.x))  # type: ignore
    time2 = 0  # type: ignore

    #n_exc = lambeer(alpha_per, pset1.x, N0_per) # constant illumination

    #Plot two curves

    fig = plt.figure()
    ax = plt.axes(xlim=(0, pset1.thickness-1), ylim=ylim)

    ax.set_yscale('log')
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

    def init() -> Any:
        time_label.set_text('')
        for i, line in enumerate(lines):
            line.set_data([], [])
        return lines[0], lines[1], time_label #return everything that must be updated

    def animate(i: float) -> Any:

        global u1, u2, time1, time2 # Necessary.

        # constant illumination
        #u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.SL, pset1.SR, n_exc = n_exc)
        #u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.SL, pset2.SR, n_exc = n_exc)

        #print(f'Elapsed time: {time1*1e12:.0f} ps', end = '\r')
        if pset1.pulse_len is None:
            u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)  # type: ignore
        else:
            u1, time1 = EulerHeatConstBCSparse(u1 + pulse(time1, pset1.pulse_len)*pset1.dt * pset1.n0, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)  # type: ignore

        if pset2.pulse_len is None:
            u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)  # type: ignore
        else:
            u2, time2 = EulerHeatConstBCSparse(u2 + pulse(time2, pset2.pulse_len)*pset2.dt * pset2.n0, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)  # type: ignore
        #u2 = u2 + pulse(time2)*pset2.dt * pset2.n0
        #time2 = time1

        time1_ns = time1*1e9      # type: ignore
        time2_ns = time2*1e9      # type: ignore
        #time_label.set_text('time1 = %.1f ns' % time1_ns + ', time2 = %.1f ns' % time2_ns) # Display the current time to the accuracy of your liking.
        #time_label.set_text('time1 = %.3f ns' % time1_ns)
        time_label.set_text(f'time 1 = {time1_ns:.3f} ns')

        if normalize_to_end:
            lines[0].set_data(pset1.x*1e7, u1/u1[-1])  # type: ignore
            lines[1].set_data(pset2.x*1e7, u2/u2[-1])  # type: ignore

        else:
            lines[0].set_data(pset1.x*1e7, u1/pset1.n0[0]*5)  # type: ignore
            lines[1].set_data(pset2.x*1e7, u2/pset2.n0[0]*5)  # type: ignore

        return lines[0], lines[1], time_label #return everything that must be updated

    print(int(finaltime/pset1.dt/nr_times))
    anim = animation.FuncAnimation(fig, animate, frames=int(finaltime/pset1.dt/nr_times), interval=interval, init_func=init, blit=True, repeat=False)

    plt.legend()
    plt.show()

    return anim

# Plot animation of quasi-Fermi level splitting

def plot_animation_QFLS(pset1: Any, pset2: Any, interval: Any = 1, ylim: Any=(1e-2,1.2)) -> Any:
    """
    Plot animation QFLS.
    
    Parameters
    ----------
    pset1 : Any
        Pset1.
    pset2 : Any
        Pset2.
    interval : Any
        Interval.
    ylim : Any
        Ylim.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> plot_animation_QFLS()
    """

    global u1, u2, time1, time2 # Necessary.

    finaltime = pset1.finaltime
    time_delta = 0.01e-9 #s

    nr_times = int(time_delta / pset1.dt)

    #pset1.n0 = 0 #no laser excitation
    #pset2.n0 = 0 #no laser excitation

    #u1 = pset1.n0
    if pset1.pulse_len is None:
        u1 = pset1.n0  # type: ignore
    else:
        u1 = np.zeros(len(pset1.x))  # type: ignore
    time1 = 0  # type: ignore

    #u2 = pset2.n0
    if pset2.pulse_len is None:
        u2 = pset2.n0  # type: ignore
    else:
        u2 = np.zeros(len(pset2.x))  # type: ignore
    time2 = 0  # type: ignore

    #n_exc = lambeer(alpha_per, pset1.x, N0_per) # constant illumination

    #Plot two curves

    fig = plt.figure()
    ax = plt.axes(xlim=(0, pset1.thickness-1), ylim=ylim)

    ax.set_yscale('linear')

    ax.set_title('Quasi-Fermi level splitting', color = 'black')
    ax.set_xlabel('nm')
    ax.set_ylabel('qfls (eV)')

    N = 2
    lines = []
    for i in range(N):
        lines.append(plt.plot([], [], label='Plot '+str(i+1))[0])

    time_label = plt.text(0.05, 0.05, '', transform=ax.transAxes) # initialize the time label for the graph

    patches = lines

    def init() -> Any:
        time_label.set_text('')
        for i, line in enumerate(lines):
            line.set_data([], [])
        return lines[0], lines[1], time_label #return everything that must be updated

    def animate(i: float) -> Any:

        global u1, u2, time1, time2 # Necessary.

        # constant illumination
        #u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.SL, pset1.SR, n_exc = n_exc)
        #u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.SL, pset2.SR, n_exc = n_exc)

        #print(f'Elapsed time: {time1*1e12:.0f} ps', end = '\r')
        if pset1.pulse_len is None:
            u1, time1 = EulerHeatConstBCSparse(u1, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)  # type: ignore
        else:
            u1, time1 = EulerHeatConstBCSparse(u1 + pulse(time1, pset1.pulse_len)*pset1.dt * pset1.n0, pset1.x, pset1.dt, nr_times, time1, pset1.mu, pset1.k1, pset1.k2, pset1.k3, pset1.SL, pset1.SR)  # type: ignore

        if pset2.pulse_len is None:
            u2, time2 = EulerHeatConstBCSparse(u2, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)  # type: ignore
        else:
            u2, time2 = EulerHeatConstBCSparse(u2 + pulse(time2, pset2.pulse_len)*pset2.dt * pset2.n0, pset2.x, pset2.dt, nr_times, time2, pset2.mu, pset2.k1, pset2.k2, pset2.k3, pset2.SL, pset2.SR)  # type: ignore
        #u2 = u2 + pulse(time2)*pset2.dt * pset2.n0
        #time2 = time1

        # Band gap
        Eg = 1.55*q #J

        # Effective density of states
        Nc = 2e18 #1/cm3
        Nv = 2e18 #1/cm3

        QFLS1 = k*T_RT * np.log(u1**2/(Nc*Nv) * math.exp(Eg/(k*T_RT)))/q  # type: ignore
        QFLS2 = k*T_RT * np.log(u2**2/(Nc*Nv) * math.exp(Eg/(k*T_RT)))/q  # type: ignore

        time1_ns = time1*1e9      # type: ignore
        time2_ns = time2*1e9      # type: ignore
        #time_label.set_text('time1 = %.1f ns' % time1_ns + ', time2 = %.1f ns' % time2_ns) # Display the current time to the accuracy of your liking.
        #time_label.set_text('time1 = %.3f ns' % time1_ns)
        time_label.set_text(f'time 1 = {time1_ns:.3f} ns')

        lines[0].set_data(pset1.x*1e7, QFLS1)
        lines[1].set_data(pset2.x*1e7, QFLS2)

        return lines[0], lines[1], time_label #return everything that must be updated

    anim = animation.FuncAnimation(fig, animate, frames=int(finaltime/pset1.dt/nr_times), interval=interval, init_func=init, blit=True, repeat=False)
    plt.legend()
    plt.show()



class TRPLParam:
    """
    Container class for TRPLParam data and operations.
    """

    def __init__(self, dt: Any = 2e-12, finaltime: Any = 200e-9, thickness: Any = 350, N_points: Any = 50, alpha: Any = 1e5, P_exc: Any = 1e10, pulse_len: Any = 60e-12, mu: Any = 1, k1: float = 0, k2: float = 1e-10, k3: Any = 8.8e-29, SL: Any = 0, SR: Any = 0) -> None:
        """
        Initialize the object.
        
        Parameters
        ----------
        dt : Any
            Dt.
        finaltime : Any
            Finaltime.
        thickness : Any
            Thickness.
        N_points : Any
            N points.
        alpha : Any
            Alpha.
        P_exc : Any
            P exc.
        pulse_len : Any
            Pulse len.
        mu : Any
            Mu.
        k1 : float
            K1.
        k2 : float
            K2.
        k3 : Any
            K3.
        SL : Any
            Sl.
        SR : Any
            Sr.
        
        Examples
        --------
        >>> obj.__init__()
        """

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

    def copy(self) -> Any:
        """
        Return a deep copy of this object.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.copy()
        """

        return TRPLParam(dt = self.dt, finaltime = self.finaltime, thickness = self.thickness, N_points = self.N_points, alpha = self.alpha, P_exc = self.P_exc, pulse_len = self.pulse_len, mu = self.mu, k1 = self.k1, k2 = self.k2, k3 = self.k3, SL = self.SL, SR = self.SR)

    @staticmethod
    def D_from_mu(mu: Any, T: float = T_RT) -> Any:
        """
        D from mu.
        
        Parameters
        ----------
        mu : Any
            Mu.
        T : float
            T.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.D_from_mu()
        """
        D = k * T / q * mu
        return D

    def replace_with_fit(self, what_to_fit: Any, fit_value: Any) -> Any:
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


class TRPLData(XYData):
    """
    Single time-resolved photoluminescence decay trace.
    """

    def __init__(self, ns: Any, cts: Any, quants: Any = dict(x = "Time", y = "Intensity"), units: Any = dict(x = "ns", y = "cts"),  name: str = '', filepath: str = '', plotstyle: str = dict(linestyle = '-', color = 'black', linewidth = 3), check_data: bool = True) -> None:  # type: ignore
        """
        Initialize the object.
        
        Parameters
        ----------
        ns : Any
            Ns.
        cts : Any
            Cts.
        quants : Any
            Quants.
        units : Any
            Units.
        name : str
            Name.
        filepath : str
            Filepath.
        plotstyle : str
            Plotstyle.
        check_data : bool
            Check data.
        
        Examples
        --------
        >>> obj.__init__()
        """
        super().__init__(ns, cts, quants = quants, units = units, name = name, plotstyle = plotstyle, check_data = check_data)
        self.filepath = filepath
        self.mexp_exist = False
        self.savgol_exist = False
        self.plotrange_left = 0
        self.plotrange_right = 100

    def copy(self) -> Any:
        """
        Copy.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.copy()
        """
        dat = super().copy()
        dat.filepath = self.filepath
        dat.mexp_exist = self.mexp_exist
        dat.savgol_exist = self.savgol_exist
        dat.plotrange_left = self.plotrange_left
        dat.plotrange_right = self.plotrange_right
        return dat

    @staticmethod
    def load(directory: str, filepath: str = '', name: str = '', delimiter: str = ',', header: Any = 'infer', time_unit: str = 'ns') -> Any:  # type: ignore

        """
        Loads a sinlge TRPL data.
        """

        if filepath == '':
            print('Warning: No filename chosen')

        dat = pd.read_csv(join(directory, filepath), delimiter = delimiter, header = header)

        ns = np.array(dat)[:,0]
        if time_unit == 'us':
            ns = ns * 1000
        cts = np.array(dat)[:,1]

        return TRPLData(ns, cts, name = name, filepath = filepath)


    def mono_expfit(self, start: float = 400, stop: float | None = None, p0: Any = (1, 500), showparam: bool = False) -> Any:
        """
        Mono exponential fit.
        
        Parameters
        ----------
        start : float
            Start.
        stop : float | None
            Stop.
        p0 : Any
            P0.
        showparam : bool
            Showparam.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.mono_expfit()
        """

        f = lambda t, a, tau : a * np.e**(-t/tau)

        ind_min = findind(self.x, start)
        if stop is None:
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
        if stop is None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop

        if showparam:
            print('Fit function f = a * e**(-t/tau)')
            print(f'a = {popt[0]:.2f}, tau = {popt[1]:.0f} ns')

        return mexpfit

    def mult2_expfit(self, start: float = 0, stop: float | None = None, p0: Any = (1, 1e-1, 10, 100), showparam: bool = False) -> Any:
        '''
        2-exponential fit of TRPL data self.
        Parameters
        ----------
        start : float, optional
            Start time in ns. The default is 0.
        stop : float, optional
            Stop time in ns. The default is None.
        p0 : 4-tuple
            Starting parameters: p0[0:1] exponential prefactor, p0[2:3] time in ns.
            The default is (1, 1e-1, 10, 100).

        Returns
        -------
        mexpfit : TRPLData
            Fit curve. In addition the optimized fit parameters popt, start, and stop are returned as parameters.

        Examples
        --------
        #param = TRPLParam(finaltime = 500e-9, mu = 0.1, k1 = 1e7)
        #example = TRPLData.from_param(param, show_progress = True)
        #example.equidist(right = 50, delta = 1)
        p0 = (0.8, 0.2, 10, 100)
        ns = np.arange(501)
        example = TRPLData.gen_m3ed(ns, p0)
        #example.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        ex_fit = example.mult2_expfit(start = 0, stop = 500)
        both = MTRPLData([example, ex_fit])
        both.label(['orig', 'fit'])
        both.plot(yscale = 'log', left = 2, right = 500, divisor = 1e3, figsize = (7, 5))
        d = example.delta(ex_fit, left = 2, right = 400)
        d.plot()
        '''
        f = lambda t, a1, a2, tau1, tau2 : a1 * np.e**(-t/tau1) + a2 * np.e**(-t/tau2)

        ind_min = findind(self.x, start)
        if stop is None:
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
        if stop is None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop

        if showparam:
            print('Fit function f = a1 * e**(-t/tau1) + a2 * e**(-t/tau2)')
            print(f'a1 = {popt[0]:.2f}, tau1 = {popt[2]:.0f} ns')
            print(f'a2 = {popt[1]:.2f}, tau2 = {popt[3]:.0f} ns')

        return mexpfit

    def mult3_expfit(self, start: float = 0, stop: float | None = None, p0: Any = (1, 1e-1, 1e-2, 5, 20, 100), showparam: bool = False) -> Any:
        '''
        3-exponential fit of TRPL data self.
        Parameters
        ----------
        start : float, optional
            Start time in ns. The default is 0.
        stop : float, optional
            Stop time in ns. The default is None.
        p0 : 8-tuple
            Starting parameters: p0[0:2] exponential prefactor, p0[3:5] time in ns.
            The default is (1, 1e-1, 1e-2, 5, 20, 100).

        Returns
        -------
        mexpfit : TRPLData
            Fit curve. In addition the optimized fit parameters popt, start, and stop are returned as parameters.

        Examples
        --------
        >>> p0 = (0.4, 0.4, 0.1, 10, 30, 100)
        '''
        f = lambda t, a1, a2, a3, tau1, tau2, tau3 : a1 * np.e**(-t/tau1) + a2 * np.e**(-t/tau2) + a3 * np.e**(-t/tau3)

        ind_min = findind(self.x, start)
        if stop is None:
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
        if stop is None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop

        if showparam:
            print('Fit function f = a1 * e**(-t/tau1) + a2 * e**(-t/tau2) + a3 * e**(-t/tau3)')
            print(f'a1 = {popt[0]:.2f}, tau1 = {popt[3]:.1f} ns')
            print(f'a2 = {popt[1]:.2f}, tau2 = {popt[4]:.0f} ns')
            print(f'a3 = {popt[2]:.2f}, tau3 = {popt[5]:.0f} ns')

        return mexpfit

    def mult4_expfit(self, start: float = 0, stop: float | None = None, p0: Any = (1, 1e-1, 1e-2, 1e-3, 5, 20, 100, 500), showparam: bool = False) -> Any:
        '''
        4-exponential fit of TRPL data self.
        Parameters
        ----------
        start : float, optional
            Start time in ns. The default is 0.
        stop : float, optional
            Stop time in ns. The default is None.
        p0 : 8-tuple
            Starting parameters: p0[0:3] exponential prefactor, p0[4:7] time in ns.
            The default is (1, 1e-1, 1e-2, 1e-3, 5, 20, 100, 500).

        Returns
        -------
        mexpfit : TRPLData
            Fit curve. In addition the optimized fit parameters popt, start, and stop are returned as parameters.

        Examples
        --------
        >>> p0 = (0.1, 0.01, 0.01, 0.8, 20, 30, 50, 50)
        '''
        f = lambda t, a1, a2, a3, a4, tau1, tau2, tau3, tau4 : a1 * np.e**(-t/tau1) + a2 * np.e**(-t/tau2) + a3 * np.e**(-t/tau3) + a4 * np.e**(-t/tau4)

        ind_min = findind(self.x, start)
        if stop is None:
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

        if stop is None:
            mexpfit.stop = self.x[-1]
        else:
            mexpfit.stop = stop

        if showparam:
            print('Fit function f = a1 * e**(-t/tau1) + a2 * e**(-t/tau2) + a3 * e**(-t/tau3) + a4 * e**(-t/tau4)')
            print(f'a1 = {popt[0]:.2f}, tau1 = {popt[4]:.1f} ns')
            print(f'a2 = {popt[1]:.2f}, tau2 = {popt[5]:.1f} ns')
            print(f'a3 = {popt[2]:.2f}, tau3 = {popt[6]:.0f} ns')
            print(f'a4 = {popt[3]:.2f}, tau4 = {popt[7]:.0f} ns')

        return mexpfit

    @staticmethod
    def gen_med(ns: Any, a: float, tau: Any) -> Any:
        """
        Generates a monoexponential data.
        ns: times in ns
        a: exponential prefactor
        tau: time in ns
        """
        d = TRPLData(ns, a * np.e**(-ns/tau), name = f'a = {a:.1e}, tau = {tau:.0f}')
        d.popt = [a, tau]  # type: ignore
        d.start = 0  # type: ignore
        d.stop = ns[-1]  # type: ignore
        return d

    @staticmethod
    def gen_m2ed(ns: Any, p0: Any) -> Any:
        """
        Generates a 2 exponential data.
        ns: times in ns
        p0[0:1]: exponential prefactor
        p0[2:3]: time in ns
        """
        a1 = p0[0]
        a2 = p0[1]
        tau1 = p0[2]
        tau2 = p0[3]
        d = TRPLData(ns, a1 * np.e**(-ns/tau1) + a2 * np.e**(-ns/tau2))
        d.popt = p0  # type: ignore
        d.start = 0  # type: ignore
        d.stop = ns[-1]  # type: ignore
        return d


    @staticmethod
    def gen_m3ed(ns: Any, p0: Any) -> Any:
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
        d = TRPLData(ns, a1 * np.e**(-ns/tau1) + a2 * np.e**(-ns/tau2) + a3 * np.e**(-ns/tau3))
        d.popt = p0  # type: ignore
        d.start = 0  # type: ignore
        d.stop = ns[-1]  # type: ignore
        return d

    @staticmethod
    def gen_m4ed(ns: Any, p0: Any) -> Any:
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
        d = TRPLData(ns, a1 * np.e**(-ns/tau1) + a2 * np.e**(-ns/tau2) + a3 * np.e**(-ns/tau3) + a4 * np.e**(-ns/tau4))
        d.popt = p0  # type: ignore
        d.start = 0  # type: ignore
        d.stop = ns[-1]  # type: ignore
        return d

    def dlifetime(self, x: np.ndarray = 'time', m: float = 2, wavelength: np.ndarray = 510, film_thickness: Any = 500, fluence: float = 5e-9, ni: float = 8.05e4) -> Any:    # type: ignore
        """
        Dlifetime.
        
        Parameters
        ----------
        x : np.ndarray
            X.
        m : float
            M.
        wavelength : np.ndarray
            Wavelength, in nm.
        film_thickness : Any
            Film thickness.
        fluence : float
            Fluence, in photons/cm².
        ni : float
            Ni.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.dlifetime()
        """
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
        if x == 'qfls':
            # qfls in eV
            QFLS_0 = k * T_RT / q * np.log(initial_carrier_conc(wavelength = wavelength, film_thickness = film_thickness, fluence = fluence)**2/ni**2)
            #print(QFLS_0)
            diff_tau.x = QFLS_0 + k * T_RT / q * np.log(self.y/self.y[0])
            diff_tau.qx = 'Quasi-Fermi level splitting'
            diff_tau.ux = 'eV'
            diff_tau.reverse()
        return diff_tau


    @staticmethod
    def from_param(p: float, time_delta: Any = 0.01e-9, name: str = '', normalize_ns: Any | None = None, normalize_cts: Any | None = None, model: Any = 'simple', show_progress: bool = False) -> Any:
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
        if p.pulse_len is None:  # type: ignore
            n = p.n0  # type: ignore
        else:
            n = np.zeros(len(p.x))  # type: ignore

        TRPL_list = []
        ns_list = []

        TRPL_list.append(PLsignal(n, p.dx, p.k2))     # type: ignore
        ns_list.append(0)

        if p.pulse_len is not None:  # type: ignore

            # Start the calculation over the pulse length * 3 in 1ps steps
            #dt = 1e-12
            nr_times = 1

            for i in range(int(p.pulse_len * 3 / p.dt)):  # type: ignore

                if model == 'simple':
                    n, t = EulerHeatConstBCSparse_simple(n + pulse(t, p.pulse_len)*p.dt * p.n0, p.x, p.dt, nr_times, t, p.mu, p.k1, p.SL, p.SR)  # type: ignore

                else:
                    n, t = EulerHeatConstBCSparse(n + pulse(t, p.pulse_len)*p.dt * p.n0, p.x, p.dt, nr_times, t, p.mu, p.k1, p.k2, p.k3, p.SL, p.SR)  # type: ignore

                TRPL_list.append(PLsignal(n, p.dx, p.k2))     # type: ignore
                ns_list.append(t*1e9)

            # Now the rest of the time

        # time_delta in s; originally 0.1 ns, display PL intensity every time_delta s
        nr_times = int(time_delta / p.dt)  # type: ignore

        for i in range(int(p.finaltime/(p.dt*nr_times))):  # type: ignore

            if model == 'simple':
                n, t = EulerHeatConstBCSparse_simple(n, p.x, p.dt, nr_times, t, p.mu, p.k1, p.SL, p.SR)  # type: ignore

            else:
                n, t = EulerHeatConstBCSparse(n, p.x, p.dt, nr_times, t, p.mu, p.k1, p.k2, p.k3, p.SL, p.SR)  # type: ignore

            TRPL_list.append(PLsignal(n, p.dx, p.k2))    # type: ignore
            ns_list.append(t*1e9)

            if show_progress == True:
                print(f'{t*1e9:.0f} ns of {p.finaltime*1e9:.0f} ns', end = '\r')  # type: ignore

        cts = np.array(TRPL_list)
        ns= np.array(ns_list)

        data = TRPLData(ns, cts, name = name)

        if normalize_ns is not None:
            data.y = data.y * normalize_cts / data.y_of(normalize_ns)

        else:
            data.y = data.y/max(data.y)

        return data


    # Fit for the full continuity equation
    def model_fit(self, param: Any, fit_from: Any = 'end', fit_range_ns: Any = [30, 40], what: Any = 'SL', start_value: Any= 0, verbose: bool = 2, gtol: Any = 1e-12) -> Any:  # type: ignore
        """
        Model fit.
        
        Parameters
        ----------
        param : Any
            Param.
        fit_from : Any
            Fit from.
        fit_range_ns : Any
            Fit range ns.
        what : Any
            What.
        start_value : Any
            Start value.
        verbose : bool
            Verbose.
        gtol : Any
            Gtol.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.model_fit()
        """
        #Automized fitting routine

        fit_range_ns = np.array(fit_range_ns)

        #Parameters p have to be duplicated because finaltime is changed
        p = param.copy()
        #Necessary to redefine finaltime, because this is the time until the TRPL curve will be calculated
        #in the function TRPLData.from_param
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


        def data_minus_fit(args) -> Any:

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

            d = TRPLData.from_param(p, time_delta = 0.01e-9)
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

    def _trpl_prepare(self, start: float, stop: float, used_for_fit: str, savgol_param: Any) -> None:
        """Common setup for k-fit methods: defaults, smoothing, index range, n0 initial."""
        if start is None:
            start = 0
        if stop is None:
            stop = self.x[-1]
        if used_for_fit == 'savgol':
            if savgol_param is None:
                savgol_param = dict(n1=51, n2=1, name='Savgol')
            dat = self.savgol(**savgol_param)
        else:
            dat = self
        start_idx = dat.x_idx_of(start)
        stop_idx = dat.x_idx_of(stop)
        r = range(start_idx, stop_idx + 1)
        return start, stop, dat, r, dat.y[start_idx]  # type: ignore

    def _trpl_show_debug(self, dat: Any, fit: Any, start: float, stop: float) -> None:
        """Show raw + smoothed + fit trace on log scale."""
        dta = MTRPLData([self, dat, fit])
        dta.label([self.name, 'savgol', 'fit'])
        dta.sa[0].plotstyle = dict(linestyle='-', color='blue', linewidth=1)
        dta.sa[1].plotstyle = dict(linestyle='-', color='orange', linewidth=3)
        dta.sa[2].plotstyle = dict(linestyle='-', color='red', linewidth=3)
        m = dta.max_within(left=start, right=stop)
        dta.plot(yscale='log', left=0, right=stop, bottom=m/100, top=m*1.1, plotstyle='individual')

    def _trpl_make_result(self, fit: Any, k1: float, k2: float) -> None:
        """Assemble a labeled MTRPLData containing measured + fit traces."""
        da_new = MTRPLData([self, fit])
        da_new.label([self.name, f'fit, k1 = {k1:.2e} s-1, k2 = {k2:.2e} cm3 s-1'])
        da_new.sa[0].plotstyle = dict(linestyle='-', color='blue', linewidth=1)
        da_new.sa[1].plotstyle = dict(linestyle='-', color='red', linewidth=3)
        return da_new  # type: ignore

    def k1_k2_fit(self, start: float | None=None, stop: float | None=None, x0: float=[3e5, 2.4e4], used_for_fit: str='savgol', savgol_param: Any | None=None, show_all: bool=False, **kwargs) -> Any:  # type: ignore
        """
        Fits k1, k2 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """
        start, stop, dat, r, n0 = self._trpl_prepare(start, stop, used_for_fit, savgol_param)  # type: ignore

        def n_of_t(t: float, n0: float, k1: float, k2: float) -> Any:
            return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)

        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0=[n0, x0[0], x0[1]], bounds=(0, np.inf), **kwargs)  # type: ignore
        k1, k2 = popt[1], popt[2]
        fit = TRPLData(dat.x[r], n_of_t(dat.x[r], *popt))  # type: ignore
        if show_all:
            self._trpl_show_debug(dat, fit, start, stop)  # type: ignore
        return self._trpl_make_result(fit, k1, k2), popt  # type: ignore

    def k1_fit(self, start: float | None=None, stop: float | None=None, x0: float=[2.4e4], k2: float=1e-8, used_for_fit: str='savgol', savgol_param: Any | None=None, show_all: bool=False, **kwargs) -> Any:  # type: ignore
        """
        Fits k1 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """
        start, stop, dat, r, n0 = self._trpl_prepare(start, stop, used_for_fit, savgol_param)  # type: ignore

        def n_of_t(t: float, n0: float, k1: float) -> Any:
            if k2 != 0:
                return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)
            else:
                return n0*np.exp(-k1*t*1e-9)

        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0=[n0, x0[0]], bounds=(0, np.inf), **kwargs)  # type: ignore
        k1 = popt[1]
        fit = TRPLData(dat.x[r], n_of_t(dat.x[r], *popt))  # type: ignore
        if show_all:
            self._trpl_show_debug(dat, fit, start, stop)  # type: ignore
        return self._trpl_make_result(fit, k1, k2), popt  # type: ignore

    def k2_fit(self, start: float | None=None, stop: float | None=None, x0: float=[1e-8], k1: float=1e6, used_for_fit: str='savgol', savgol_param: Any | None=None, show_all: bool=False, **kwargs) -> Any:  # type: ignore
        """
        Fits k2 to a TRPL trace.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """
        start, stop, dat, r, n0 = self._trpl_prepare(start, stop, used_for_fit, savgol_param)  # type: ignore

        def n_of_t(t: float, n0: float, k2: float) -> Any:
            return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)

        # lower n0 bound kept above 0 to prevent n0 from collapsing
        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0=[n0, x0[0]], bounds=([n0/10, 0], np.inf), **kwargs)  # type: ignore
        k2 = popt[1]
        fit = TRPLData(dat.x[r], n_of_t(dat.x[r], *popt))  # type: ignore
        if show_all:
            self._trpl_show_debug(dat, fit, start, stop)  # type: ignore
        return self._trpl_make_result(fit, k1, k2), popt  # type: ignore

    def n0_fit(self, start: float | None=None, stop: float | None=None, n0: float=1e13, k1: float=1e6, k2: float=1e-8, used_for_fit: str='savgol', savgol_param: Any | None=None, show_all: bool=False, **kwargs) -> Any:
        """
        Fit the initial carrier density n0 to TRPL data.
        Rate equation: dn/dt = -k1*n - k2*n**2
        """
        start, stop, dat, r, n0_initial = self._trpl_prepare(start, stop, used_for_fit, savgol_param)  # type: ignore

        def n_of_t(t: float, n0: float) -> Any:
            return k1/k2 * 1/(np.exp(k1*t*1e-9)*(1+k1/(n0*k2))-1)

        popt, pcov = curve_fit(n_of_t, dat.x[r], dat.y[r], p0=[n0_initial], bounds=(0, np.inf), **kwargs)  # type: ignore
        fit = TRPLData(dat.x[r], n_of_t(dat.x[r], *popt))  # type: ignore
        if show_all:
            self._trpl_show_debug(dat, fit, start, stop)  # type: ignore
        return self._trpl_make_result(fit, k1, k2), popt  # type: ignore

    def k1_k2_model_fit(self, what_to_fit: Any = ['k1', 'k2'], start: float | None = None, stop: float | None = None, n0: float = 1e-15, k1: float = 1e6, k2: float = 1e-7, show: bool = None) -> Any:
        """
        K 1 k 2 model fit.
        
        Parameters
        ----------
        what_to_fit : Any
            What to fit.
        start : float | None
            Start.
        stop : float | None
            Stop.
        n0 : float
            N0.
        k1 : float
            K1.
        k2 : float
            K2.
        show : bool
            Show.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.k1_k2_model_fit()
        """

        if start is None:
            start = 0
        if stop is None:
            stop = self.x[-1]

        #n00 will be used for the transformations from PL intensity into carrier concentration and back
        n00 = n0

        # Transform from PL instensity into  carrier concentration
        cc = self.copy()
        cc.y = np.sqrt(abs(cc.y))
        cc.y = cc.y/max(cc.y)*n00
        cc.qy = 'Carrier concentration'
        cc.uy = '1/cm3'

        if show is not None:
            if (show == 'all') or ('step 1' in show):  # type: ignore
                #m_max = cc.max_within(left = start, right = stop)
                #m_min = cc.min_within(left = start, right = stop)
                #cc.plot(yscale = 'log', left = start, right = stop, bottom = abs(m_min)*0.9, top = m_max*1.1)
                print('Step 1: Transformation from PL intensity into carrier concentration')
                cc.plot(yscale = 'log')
            # show_all is used for the actual calculation of the fit curves
            if (show == 'all') or ('step 2' in show):  # type: ignore
                show_all = True
                print('Step 2: Fit')
        else:
            show_all = False

        if what_to_fit is None:
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

        if show is not None:
            if (show == 'all') or ('step 3' in show):  # type: ignore
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

        if show is not None:
            if (show == 'all') or ('step 4' in show):  # type: ignore
                print('Step 4: Back-transformation from carrier concentration into PL intensity.')
                PL_dta.plot(yscale = 'log', right = stop)

        # return self and fit (with the right label) and all parameters
        return PL_dta, [n0, k1, k2]

# simplified model dn/dt = -k1*n - k2*n**2 end
#________________________________________________________________________________________

    def del_bg(self, plot_details: bool = False, norm_val: float | None = None, start: float | None = None, stop: float | None = None) -> Any:
        """
        Deletes the background of raw TRPL data. 
        """

        def bg_idx_start() -> Any:
            idx = 0
            stop = False
            while not stop:
                if self.y[idx+1] >= self.y[idx]:
                    idx += 1
                else:
                    stop = True
            return idx

        def bg_idx_stop() -> Any:
            idx = findind(self.y,max(self.y))
            stop = False
            while not stop:
                if self.y[idx-1] <= self.y[idx]:
                    idx -= 1
                else:
                    stop = True
            return idx

        if start is None:
            start = bg_idx_start()
        else:
            start = self.x_idx_of(start)
        if stop is None:
            stop = bg_idx_stop()
        else:
            stop = self.x_idx_of(stop)
        if stop < start:
            print('Attention [TRPLData.del_bg()]: stop < start, hence the alternative routine is chosen!')
            #This can happen, if there is very little noise data points are selected. In this case take the first value > 0 as start and the maximum - 1 as stop
            start = 0
            while self.y[start] == 0:
                start += 1
            stop = findind(self.y, max(self.y))
            stop -= 1
        r = range(start, stop + 1)  # type: ignore
        m = max(self.y[r])
        av = np.average(self.y[r])
        dat = self.copy()
        dat.y = dat.y - av
        if norm_val is not None:
            dat.normalize(norm_val = norm_val)

        if plot_details:
            print(f'______{self.name}________')
            self.plot(yscale = 'linear', left = self.x[start]*0, right = self.x[stop]*1.2, bottom = 0, top = m*2, vline = [self.x[start], self.x[stop]], title = 'noise')  # type: ignore
            #Show original and bg subtracted data
            dat_all = MTRPLData([self, dat])
            if norm_val is not None:
                dat_all = dat_all.copy()
                dat_all.normalize()
            dat_all.label(['original', 'corrected'])
            dat_all.plot(yscale = 'log', title = self.name, divisor = 1e5)

        return dat

    def shift_zero(self, ns: Any) -> Any:
        """
        Shift zero.
        
        Parameters
        ----------
        ns : Any
            Ns.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.shift_zero()
        """
        ind = findind(self.x, ns)
        self.x = self.x[ind:] - self.x[ind]
        self.y = self.y[ind:]

    def shift_to_max(self, plot_details: bool = False, left: float | None = None, right: float | None = None) -> Any:
        """
        Shift the data so that the maximum value is at x = 0.
        """
        idx_shift = findind(self.y,max(self.y))
        x_shift = self.x[idx_shift]

        dat = self.copy()
        #if start_with_xeq0:
        #    dat.shift_zero(dat.x[idx_shift])
        #else:
        dat.x -= x_shift


        if plot_details:
            if left is None:
                left = 0
            if right is None:
                right = max(dat.x)
            dat.plotstyle = dict(linestyle='-', linewidth=5, markersize=5)
            #dat_max = dat.max_within(left = left, right = right)
            #dat_min = dat.min_within(left = left, right = right, absolute = True)
            #dat_test.plot(yscale = 'log', bottom = dat_min*0.9, top = dat_max*1.1, left = left, right = right)
            dat.plot(yscale = 'log', left=left, right=right)

        return dat




class MTRPLData(MXYData):
    """
    sa is a list of TRPLData.
    """

    def __init__(self, sa: Any) -> None:
        """
        Initialize the object.
        
        Parameters
        ----------
        sa : Any
            Sa.
        
        Examples
        --------
        >>> obj.__init__()
        """
        super().__init__(sa)

    @classmethod
    def load_individual(cls, directory: str, FNs: str = [], delimiter: str = ',', header: Any = 'infer', quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, take_quants_and_units_from_file: bool = False) -> Any:  # type: ignore

        """
        Loads all xy data in individual files in directory.
        """
        if len(FNs) == 0:
            FNs = listdir(directory)  # type: ignore

        sa = []

        for i, filepath in enumerate(FNs):

            dat = pd.read_csv(join(directory, filepath), delimiter = delimiter, header = header)

            npdat = np.array(dat, dtype = np.float64)
            x = npdat[:,0]
            y = npdat[:,1]

            if cls.__name__ == 'MTRPLData':
                sp = TRPLData(x, y, quants, units, filepath)

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
        result = cls(sa)

        return result

    def _batch_expfit(self, method_name: str, name_prefix: str, start: float, stop: float, p0: Any, showparam: bool) -> None:
        """Shared loop scaffold for all batch multi-exponential fit wrappers."""
        dafit_sa = []
        for d in self.sa:
            if showparam:
                print(d.name)
            dfit = getattr(d, method_name)(start=start, stop=stop, p0=p0, showparam=showparam)
            dfit.name = name_prefix + d.name
            dafit_sa.append(dfit)
            if showparam:
                dfit.plotstyle = dict(linestyle='--', color='tab:red', linewidth=2)
                delta = d.residual(dfit, relative=True)
                print(f'chi**2 = {XYData.chisquare(d, dfit, right=stop):.2e}')
                delta.plot(right=stop, hline=0, title='Residual plot')
                MTRPLData([d, dfit]).label(['original', 'fit'])
        dafit = MTRPLData(dafit_sa)
        dafit.names_to_label(split_ch='.csv')
        return dafit  # type: ignore

    def mono_expfit(self, start: float=400, stop: float | None=None, p0: Any=(1, 500), showparam: bool=False) -> Any:
        """
        Mono exponential fit.
        
        Parameters
        ----------
        start : float
            Start.
        stop : float | None
            Stop.
        p0 : Any
            P0.
        showparam : bool
            Showparam.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.mono_expfit()
        """
        return self._batch_expfit('mono_expfit', 'mono exp fit_', start, stop, p0, showparam)  # type: ignore

    def mult2_expfit(self, start: float=0, stop: float | None=None, p0: Any=(1, 1e-1, 10, 100), showparam: bool=False) -> Any:
        """
        Mult 2 exponential fit.
        
        Parameters
        ----------
        start : float
            Start.
        stop : float | None
            Stop.
        p0 : Any
            P0.
        showparam : bool
            Showparam.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.mult2_expfit()
        """
        return self._batch_expfit('mult2_expfit', '2 exp fit_', start, stop, p0, showparam)  # type: ignore

    def mult3_expfit(self, start: float=0, stop: float | None=None, p0: Any=(1, 1e-1, 1e-2, 5, 20, 100), showparam: bool=False) -> Any:
        """
        Mult 3 exponential fit.
        
        Parameters
        ----------
        start : float
            Start.
        stop : float | None
            Stop.
        p0 : Any
            P0.
        showparam : bool
            Showparam.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.mult3_expfit()
        """
        return self._batch_expfit('mult3_expfit', '3 exp fit_', start, stop, p0, showparam)  # type: ignore

    def mult4_expfit(self, start: float=0, stop: float | None=None, p0: Any=(1, 1e-1, 1e-2, 1e-3, 5, 20, 100, 500), showparam: bool=False) -> Any:
        """
        Mult 4 exponential fit.
        
        Parameters
        ----------
        start : float
            Start.
        stop : float | None
            Stop.
        p0 : Any
            P0.
        showparam : bool
            Showparam.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.mult4_expfit()
        """
        return self._batch_expfit('mult4_expfit', '4 exp fit_', start, stop, p0, showparam)  # type: ignore


    def dlifetime(self, x: np.ndarray = 'time', m: float = 2, wavelength: np.ndarray = 510, film_thickness: Any = 500, fluence: float = 5e-9, ni: float = 8.05e4) -> Any:    # type: ignore
        """
        Dlifetime.
        
        Parameters
        ----------
        x : np.ndarray
            X.
        m : float
            M.
        wavelength : np.ndarray
            Wavelength, in nm.
        film_thickness : Any
            Film thickness.
        fluence : float
            Fluence, in photons/cm².
        ni : float
            Ni.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.dlifetime()
        """
        diff_tau_sa = []
        for idx, sp in enumerate(self.sa):
            diff_tau = sp.dlifetime(x = x, m = m, wavelength = wavelength, film_thickness = film_thickness, fluence = fluence, ni = ni)
            diff_tau_sa.append(diff_tau)
        return MTRPLData(diff_tau_sa)
