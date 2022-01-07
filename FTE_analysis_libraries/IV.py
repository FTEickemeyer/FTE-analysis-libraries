# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:41:51 2020

@author: dreickem
"""

from dataclasses import dataclass
from scipy.special import lambertw
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import sys
import numpy as np
import pandas as pd
from os.path import join
import math
import matplotlib.pyplot as plt
from importlib import reload
from IPython import embed
from scipy.optimize import fsolve, fmin
import pkg_resources
system_dir = pkg_resources.resource_filename( 'FTE_analysis_libraries', 'System_data' )

from .General import linfit, findind, save_ok, plx, q, k, T_RT
from .XYdata import xy_data, mxy_data
from .Spectrum import diff_spectrum, abs_spectrum


@dataclass
class fivep:
    """
    fivep stands for five parameters.
    Class for keeping track of the 5param of an IV curve.

    Parameters
    ----------
    cell_area : FLOAT
        cell (mask) area in cm2.
    Voc : FlOAT
        Voc in V.
    Jsc : FLOAT
        Jsc in mA/cm2 (or mA if cell_area == None)
    nid : dimensionless
        ideality factor
    Rs : FLOAT
        Series resistance in kOhm * cm2 (or kOhm if cell_area == None)
    Rsh : FLOAT
        Shunt resistnace in kOhm * cm2 (or kOhm if cell_area == None
    
    """
    
    cell_area: float = None
    Voc: float = None
    Jsc: float = None
    nid: float = None
    Rs: float = None
    Rsh: float = None
    
    def __init__(self, cell_area = 1, Voc = 1, Jsc = 20, nid = 1, Rs = 0, Rsh = np.inf):

        self.cell_area = cell_area
        self.Voc = Voc
        self.Jsc = Jsc
        self.nid = nid
        self.Rs = Rs
        self.Rsh = Rsh
        
    def copy(self):
        return fivep(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)
    
@dataclass
class perf_dat:

    cell_area: float = None # cm2
    Vmpp: float = None # V
    Jmpp: float = None # mA/cm2 (or mA if cell_area == None)
    Pmpp: float = None # mW/cm2 (or mW if cell_area == None)
    PCE: float = None # % (makes no sense if cell_area == None)
    FF: float = None # %
    Voc: float = None # V
    Jsc: float = None # mA/cm2 (or mA if cell_area == None)
    nid: float = None # dimension less
    Rs: float = None # kOhm * cm2 (or kOhm if cell_area == None)
    Rsh: float = None # kOhm * cm2 (or kOhm if cell_area == None)
    light_int: float = 100 # mW/cm2
    
    def copy(self):
        return perf_dat(cell_area = self.cell_area, Vmpp = self.Vmpp, Jmpp = self.Jmpp, Pmpp = self.Pmpp, 
                      PCE = self.PCE, FF = self.FF, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, 
                      Rs = self.Rs, Rsh = self.Rsh, light_int = self.light_int)
    
    def Jmpp_text(self, uA = False):
        if uA:
            text = f'$J_{{mpp}} = {self.Jmpp*1000:.2f} \; \mu A/cm^2$'
        else:
            text = f'$J_{{mpp}} = {self.Jmpp:.2f} \; mA/cm^2$'
        return text        
    
    def Vmpp_text(self):
        return f'$V_{{mpp}} = {self.Vmpp:.3f} \; V$'
    
    def Pmpp_text(self):
        return f'$P_{{mpp}} = {self.Pmpp:.2f} \; mW/cm^2$'
    
    def PCE_text(self):
        return f'$PCE = {self.PCE:.1f}\%$'
    
    def FF_text(self):
        return f'$FF = {self.FF:.1f}\%$'
    
    def Voc_text(self):
        return f'$V_{{oc}} = {self.Voc:.3f} \; V$'

    def Jsc_text(self, uA = False):
        if uA:
            text = f'$J_{{sc}} = {self.Jsc*1000:.2f} \; \mu A/cm^2$'
        else:
            text = f'$J_{{sc}} = {self.Jsc:.2f} \; mA/cm^2$'
        return text

    def nid_text(self):
        return f'$n_{{id}} = {self.nid:.2f}$'
    
    def Rs_text(self):
        return f'$R_{{s}} = {self.Rs*1e3:.2e} \; \Omega \cdot cm^2$'
    
    def Rsh_text(self):
        return f'$R_{{sh}} = {self.Rsh*1e3:.2e} \; \Omega \cdot cm^2$'
    
    def cell_area_text(self):
        if not(self.cell_area is None):
            return f'Cell area $= {self.cell_area:.5f} \; cm^2$'
        else:
            return ''
    
    def light_int_text(self, uW = False):
        if uW:
            return f'Light intensity $= {self.light_int*1000:.2f} \; uW/cm^2$'
        else:
            return f'Light intensity $= {self.light_int:.2f} \; mW/cm^2$'
        
    @staticmethod
    def SQ_limit(bg, illumspec_eV = None, light_int = 100, show = False):
        """
        Calculates the performance data of the Shockley-Queisser limit.

        Parameters
        ----------
        bg : FLOAT
            Bandgap in eV.
        illumspec_eV: diff_spectrum
            Illumination spectrum other than AM1.5GT        
            If None then AM1.5GT spectrum is taken
        light_int: FLOAT
            If AM1.5GT spectrum as illumination spectrum used, then with light_int the light intensity in mW/cm2 can be chosen.
            If None than 100 mW/cm2 (1 sun) is used.
        show : BOOLEAN, optional
            If True it will show the performance data as formatted text. The default is False.
    
        Returns
        -------
        fp : instance of perf_dat.
        
        Example
        -------
        bg = 2.30 #eV
        pd = perf_dat.SQ_limit(bg, show = True)

        """
        fp = IV_data.SQ_limit(bg, illumspec_eV = illumspec_eV, light_int = light_int)
        x_arr = np.linspace(0, fp.Voc*1.01,  int(round((fp.Voc*1.01)/0.001)+1))
        IV = IV_data.from_fp(x_arr, fp, name = '', light_int = light_int, T = T_RT, perfparam = True)
        IV.det_perfparam(show = show)
        return IV.pd
    

class IV_data(xy_data):
    
    
    def __init__(self, x, y, cell_area = None, light_int = None, sweep_dir = 'rev', name = '', Voc = None, Jsc = None, quants = {"x": "Voltage", "y": "Current density"}, units = {"x": "V", "y": "mA/cm2"}, plotstyle = dict(linestyle = 'None', marker = 'o', color = 'blue', markersize = 5), check_data = True):
        """
        x is a numpy array for voltage in V
        y is a numpy array for current density in mA/cm2
        cell_area in cm2
        light_int in mW/cm2
        name is the name of the data, e.g. the file name
        quants and units are redefined but they are used to be compatible with the load routine.
        """
        super().__init__(x, y, quants = quants, units = units, name = name, check_data = check_data)
        self.cell_area = cell_area
        self.sweep_dir = sweep_dir
        self.light_int = light_int
        self.Jsc = Jsc
        self.Voc = Voc
        self.plotstyle = plotstyle
        
    def copy_old(self):
        """
        replaced by the new version 210108.
        """
        dat = IV_data(self.x.copy(), self.y.copy(), cell_area = self.cell_area, light_int = self.light_int, sweep_dir = self.sweep_dir, name = self.name, Voc = self.Voc, Jsc = self.Jsc)

        dat.plotstyle = self.plotstyle.copy()
        
        if hasattr(self, 'nid'):
            dat.nid = self.nid
        if hasattr(self, 'Rs'):
            dat.Rs = self.Rs
        if hasattr(self, 'Rsh'):
            dat.Rsh = self.Rsh
        if hasattr(self, 'pd'):
            dat.pd = self.pd

        return dat
    
    def copy(self):
        dat = super().copy()
        if hasattr(self, 'nid'):
            dat.nid = self.nid
        if hasattr(self, 'Rs'):
            dat.Rs = self.Rs
        if hasattr(self, 'Rsh'):
            dat.Rsh = self.Rsh
        if hasattr(self, 'pd'):
            dat.pd = self.pd.copy()

        return dat
    
    def convert_from_mA_to_uA(self):
        self.y *= 1000
        self.uy = 'uA/cm2'
        
    @staticmethod
    def from_J0(V, J0, Jph, nid, Rs, Rsh, cell_area = None, light_int = None, name = '', T = T_RT):
        Gp = 1/Rsh
        Vth = k * T / q
        expW = (V + Rs * (J0 + Jph)) / (nid * Vth * (1 + Rs * Gp))
        argW = J0 * Rs / (nid * Vth * (1 + Rs * Gp)) * np.exp(expW)
        J = nid * Vth / Rs * lambertw(argW) + (V * Gp - (J0 + Jph)) / (1 + Rs * Gp)
        J = J.real
        return IV_data(V, J, cell_area = cell_area, light_int = light_int, sweep_dir = None, name = name, Voc = None, Jsc = None)
        
    def to_fp(self):
        return fivep(self.cell_area, self.Voc, self.Jsc, self.nid, self.Rs, self.Rsh)
    
    def norm_to_onesun(self, MMF = 1):
        """
        Normalizes current density to one sun. Takes into account the MMF by dividing the current by MMF.

        Returns
        -------
        None.

        """
        self.y = self.y * 100 / self.light_int / MMF
        self.light_int = 100
    

    @staticmethod
    def load_Igor_IV(dir, FN, print_lines = False):
        """
        Function that opens an igor IV file and returns a 2-dim data array, column 1: Voltage (V), column 2: Current (A)
        """
    
        TFN = join(dir,FN)
    
        # Check if photocurrent data and load data
        PCdata = False
        toks = []
        data = False 
            
        with open(TFN) as z:
            
            for line in z:
                
                if 'PhotoCurrent' in line:
                    PCdata = True
                    
                if PCdata:
    
                    # Get y-values and parameters
    
                    if print_lines:
                        print(line)
    
                    if line.startswith('BEGIN'):
                        data = True
                        continue
    
                    if line.startswith('END'):
                        data = False
    
                    if data:
                        #split('\t')[0] is necessary because some data shows two identical columns separated by \t
                        #(maybe it's only the one of the OrielIV system)
                        #and only one column is relevant
                        toks.append(line.strip('\t').strip('\n').split('\t')[0])
    
                    #elif line.startswith('X SetScale'):
                    elif 'X SetScale' in line:
                        startV = float(line.split("SetScale/P x ")[1].split(",")[0])
                        deltaV = float(line.split(",")[1])
    
                    #elif line.startswith('X Note'):
                    elif 'X Note' in line:
                        light_int = float(line.split("IT:")[1].split(";")[0]) * 1000 # mW/cm2
                        cell_area = float(line.split("AR:")[1].split(";")[0]) # cm2
    
        if not(PCdata):
    
            print('load_Igor_IV error: Data is not photocurrent data!')
    
        else:
                
            y = np.asarray(toks, dtype = np.float64) / cell_area * 1e3 #mA/cm2
    
            x = np.linspace(startV, startV + (len(y) - 1) * deltaV, num=len(y))
    
            if len(x) != len(y):
                print(FN + ': ATTENTION: Number of lines of column voltage (' + str(len(x)) + ' lines) does not match number of lines of column current (' + str(len(y)) +' lines)!')
                x = x[:len(y)]
    
            # If reverse measurement reverse the order of array
            if x[1] < x[0]:
                x = x[::-1]
                y = y[::-1]
                sweep_dir = 'rev'
            else:
                sweep_dir = 'fwd'
    
        return IV_data(x, y, cell_area = cell_area, light_int = light_int, sweep_dir = sweep_dir, name = FN)


    @staticmethod
    def load_Biologic_CV_old(dir, FN, cell_area, light_int = 100, J_1sun = None, raw_data = False, both_scans = False, reverse_scan = True, warning = True):
        """
        shifted to module biologic!
        Loads CV data measured with a Biologic potentiostat. It works if the CV was measured first forward and then backward.
    
        Parameters
        ----------
        dir : string
            data directory.
        FN : string
            data file name.
        cell_area : float
            active cell area in cm2.
        light_int : float, optional
            light intensity in mW/cm2. If none then the light intensity is calculated from the Jsc and the current at 1 sun
            given by the variable current_1sun. The default is 1 sun (=100 mW/cm2)
        J_1sun : float, optional
            Current density at 1 sun, used to calculate the light intensity. The default is None.
        raw_data : boolean, optional
            True: The raw data is taken with forward and reverse scan. In this case no evaluation of the IV data is possible. The default is False.
        both_scans : boolean, optional
            True if both scans (reverse and forward) are taken The default is False
        reverse_scan : boolean, optional
            True if the reverse scan is taken, False if the forward scan is taken. The default is True.
        warning: boolean, optional
            True: prints a warning that the CV has to be first forward, then backward.
    
        Returns
        -------
        Returns an instance of IV_data if both_scans is False or raw_data is True, otherwise a tuple of two instances of IV_data
    
        """
        if warning:
            print('Function IV_data.load_Biologic_CV: The CV scan has to be first in forward then in reverse sweep direction!')
            print('To switch off this message, set argument warning = False!')
    
        TFN = join(dir,FN)
    
        # Returns the number of header lines in the .mpt file
        def header_lines(TFN):
    
            with open(TFN, encoding = "ISO-8859-1") as mpt_file:
            #with open(Dir_FN) as mpt_file:
    
                for line in mpt_file:
    
                    if 'Nb header lines' in line:
                        header_lines = int(line[17:].strip())
                        #print(f'Number of header_lines = {header_lines}')
                        break
    
            return header_lines
    
    
        raw = pd.read_csv(TFN, delimiter='\t', header = header_lines(TFN))
        raw_V = np.array(raw)[:,7]
        raw_J = np.array(raw)[:,8] / cell_area    
    
        if raw_data:
            V_array = raw_V
            J_array = raw_J
            sweep_dir = None
            IV = IV_data(V_array, J_array, cell_area = cell_area, light_int = light_int, sweep_dir = sweep_dir, name = FN, check_data = False)
    
        else:
            
    
            def IV_from_raw(data, sweep_dir, cell_area, light_int, J_1sun):
            
                # Sort data by ascending V (important to find the mpp voltage and Voc)
                data2 = data[data[:,0].argsort()]
                
                
                # Only take such data points for wich voltages are not equal, 
                # i.e. truly ascending (important to find the mpp voltage and Voc)
                data3 = []
                for i in range(len(data2[:,0])-1):
                    if data2[i+1,0] > data2[i,0]:
                        data3.append(data2[i,:])
                        
                data3 = np.array(data3)
                
                # V data as an array in 0.001 V steps
                begin_V = round(min(data3[:,0]) * 1000) / 1000
                end_V = round(max(data3[:,0]) * 1000) / 1000
                V_array = np.arange(begin_V, end_V, 0.001)
             
                JVinterp = interp1d(data3[:, 0], data3[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
                #J_array = np.zeros(len(V_array))
                #for i in range(len(V_array)):
                #    J_array[i] = JVinterp(V_array[i])[0]
                J_array = JVinterp(V_array)
                
                IV = IV_data(V_array, J_array, cell_area = cell_area, light_int = light_int, sweep_dir = sweep_dir, name = FN)
    
                if light_int == None:
                    Jsc = IV.det_Jsc(fit_to = None, show_fit = False)
                    light_int = Jsc/J_1sun*100
                    IV.light_int = light_int
            
                return IV
            
            
            idx = findind(raw_V, max(raw_V))        
    
            # Get forward scan
            data_fwd = np.zeros((idx, 2), dtype = float)
            data_fwd[:,0] = raw_V[:idx]
            data_fwd[:,1] = raw_J[:idx]
            IV_fwd = IV_from_raw(data_fwd, 'fwd', cell_area, light_int, J_1sun)
            
            # Get reverse scan
            data_rev = np.zeros((len(raw_V)-idx, 2), dtype = float)
            data_rev[:,0] = raw_V[idx:]
            data_rev[:,1] = raw_J[idx:]
            IV_rev = IV_from_raw(data_rev, 'rev', cell_area, light_int, J_1sun)        
    
            if both_scans:
                IV = (IV_fwd, IV_rev)
            else:
                if reverse_scan:
                    IV = IV_rev
                else:
                    IV = IV_fwd
            
        return IV

    @staticmethod
    def load(data_dir, sample, data_format, cell_area = 1, light_int = 100, delimiter = ',', header = 'infer', quants = {"x": "Voltage", "y": "Current density"}, units = {"x": "V", "y": "mA/cm2"}, 
         take_quants_and_units_from_file = False, J_1sun = None, reverse_scan = True, raw_data = False, print_lines = False):

        if data_format == 'Igor':
            IV = IV_data.load_Igor_IV(data_dir, sample, print_lines = print_lines)
        elif data_format == 'csv':
            xy = xy_data.load(data_dir, sample, delimiter = delimiter, header = header, quants = quants, units = units, 
                                   take_quants_and_units_from_file = take_quants_and_units_from_file)
            IV = IV_data(xy.x, xy.y, cell_area = cell_area, light_int = light_int, sweep_dir = None, name = sample, Voc = None, Jsc = None, quants = {"x": "Voltage", "y": "Current density"}, units = {"x": "V", "y": "mA/cm2"}, )
            #IV.y = -IV.y
        elif data_format == 'Biologic-CV':
            IV = IV_data.load_Biologic_CV(data_dir, sample, cell_area = cell_area, light_int = light_int, J_1sun = J_1sun, reverse_scan = reverse_scan, raw_data = raw_data)
    
        return IV
    
    def det_J0(self, T = T_RT):
        Jsc = self.Jsc
        Voc = self.Voc
        Rs = self.Rs
        Rsh = self.Rsh
        nid = self.nid
        return (Jsc + (Rs*Jsc-Voc)/Rsh) * math.exp(-q*Voc / (nid * k * T)) / (1 - math.exp(q * (Rs * Jsc - Voc) / (nid * k * T)))
      
        
    def det_Voc(self):
        JVinterp = interp1d(self.x, self.y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        Voc = fsolve(JVinterp,.95*max(self.x))[0]
        self.Voc = Voc
        return Voc
        
    def det_Jsc(self, fit_to = None, show_fit = False):
        """
        Returns the short circuit current. Also stores the value in self.Jsc.
        fit_to: specifies the voltage up to which the IV curve is fitted. If None then the maximum V / 10 is taken.
        show_fit: if True the IV curve and the fit from which Jsc is taken is plotted.
        """

        if fit_to == None:
            fit_to = max(self.x)/10
        m, b = linfit(self.x, self.y, 0, fit_to)
        self.Jsc = -b
        
        if show_fit:
            plt.figure(figsize=(7,5))
            plt.title('Fit for determination of Jsc')
            plt.plot(self.x, self.y, 'o', label = 'IV curve')
            plt.plot(self.x, self.x * m - self.Jsc, '-', label = f'fit (Jsc = {self.Jsc:.1f} mA/cm2)')
            plt.ylim(-self.Jsc*1.2, -self.Jsc*0.8)
            plt.legend()
            plt.show()
            
        return -b
    
    def ini_guess_Rsh(self, fit_to = None, show_fit = False):
        """
        Returns an initial guess for the shunt resistance Rsh in Ohm * cm2. Also stores the value in self.Rsh.
        fit_to: specifies the voltage up to which the IV curve is fitted. If None then the maximum V / 5 is taken.
        show_fit: if True the IV curve and the fit from which Rsh is taken is plotted.
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 11
        """
        if fit_to == None:
            fit_to = max(self.x)/5

        m, b = linfit(self.x, self.y, 0, fit_to)
        if abs(m) < 1e-12:
            Rsh = 1e15 # in kOhm cm2
        else:
            Rsh = 1/m # in kOhm cm2
            
        if Rsh < 0:
            Rsh = 1e15
            
        self.Rsh = Rsh # in kOhm cm2
        
        if show_fit:
            plt.figure(figsize=(7,5))
            plt.title('Fit for determination of Rsh')
            plt.plot(self.x, self.y, 'o', label = 'IV curve')
            plt.plot(self.x, self.x / self.Rsh - self.Jsc, '-', label = f'fit (Rsh = {self.Rsh*1e3:.2e} Ohm cm2)')
            plt.ylim(-self.Jsc*1.2, -self.Jsc*0.8)
            plt.legend()
            plt.show()
        
        return Rsh
    
    def ini_guess_nid_and_Rs(self, show_fit = False):
        """
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 12
        """
        dIdV = np.gradient(self.y, self.x)
                    
        r = range(findind(self.x, self.Voc*0.95), findind(self.x, self.Voc*1.05)+1)
        dVdI = 1 / dIdV[r]
        new_x = k * T_RT / (q * (self.Jsc + self.y[r] - self.x[r] / self.Rsh))
        
        n, Rs = np.polyfit(new_x, dVdI, 1)
        if n < 1:
            n = 1
        self.nid = n
        if Rs < 0:
            Rs = 0
        self.Rs = Rs

        if show_fit:
                        
            plt.figure(figsize=(7,5))
            plt.title('Derivative dI/DV')            
            plt.plot(self.x, self.y, 'o', label = 'IV curve')
            plt.plot(self.x, dIdV, label = 'derivative')
            plt.legend()
            plt.show()

            plt.figure(figsize=(7,5))
            plt.title('Fit for determination of nid and Rsh')
            plt.plot(new_x, dVdI, 'o')
            plt.plot(new_x, n * new_x + Rs, '-', label = f'fit: nid = {n:.2f}, Rs = {Rs:.2e} Ohm cm2' '\n' f'(range: {r})')
            plt.legend()
            plt.show()
                        
        return n, Rs
    
    def check_assumption(self, T = T_RT):
        """
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 6
        """
        delta = np.exp(q * (self.Rs * self.Jsc - self.Voc) / (k * T))
        if delta > 0.1:
            text1 = "IV_data.check_assumption: \n"
            text2 = 'Attention: The assumption $\Delta \ll 1$ (eq. 6) is not satisfied!'
            plx(text1+text2)
        return delta
    
    def det_ini_5param(self, show_fit = False):
        self.det_Voc()
        self.det_Jsc(fit_to = None, show_fit = show_fit)
        self.ini_guess_Rsh(fit_to = None, show_fit = show_fit)
        self.ini_guess_nid_and_Rs(show_fit = show_fit)
        self.check_assumption()
        
    
    def det_perfparam(self, show = False, uA = False, uW = False, minimal = False):
        #uA: calculate currents in uA/cm2
        #uW: show light intensity in uW/cm2
        #If minimal: only Voc, Jsc, FF, Vmpp, Jmpp and PCE is determined
        if self.Voc == None:
            self.det_Voc()
        if self.Jsc == None:
            self.det_Jsc()
        JVinterp = interp1d(self.x, self.y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        Vmpp = fmin(lambda x: x*JVinterp(x),.8*self.Voc,disp=False, maxiter = 100)[0]
        Jmpp = abs(JVinterp(Vmpp))
        #Pmpp = abs(Vmpp*Jmpp)
        Pmpp = abs(Vmpp*Jmpp)
        PCE = Pmpp / self.light_int * 100
        FF = abs(Pmpp/(self.Jsc*self.Voc) * 100)
        if not(minimal) and not(hasattr(self, 'nid')):
            self.fit_param()
        if not(hasattr(self, 'light_int')):
            self.light_int = 100
        if not minimal:
            pd = perf_dat(cell_area = self.cell_area, Vmpp = Vmpp, Jmpp = Jmpp, Pmpp = Pmpp, 
                          PCE = PCE, FF = FF, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, 
                          Rs = self.Rs, Rsh = self.Rsh, light_int = self.light_int)
            text = pd.light_int_text(uW) + ', ' + pd.Vmpp_text() + ', ' + pd.Jmpp_text(uA) + '\n' + pd.Voc_text() + ', ' + pd.Jsc_text(uA) + ', ' + pd.FF_text() + ', ' + pd.PCE_text() + '\n' + pd.nid_text() + ', ' + pd.Rs_text() + ', ' + pd.Rsh_text() + '\n' + pd.cell_area_text()

        else:
            pd = perf_dat(cell_area = self.cell_area, Vmpp = Vmpp, Jmpp = Jmpp, Pmpp = Pmpp, 
                          PCE = PCE, FF = FF, Voc = self.Voc, Jsc = self.Jsc, light_int = self.light_int)            
            text = pd.light_int_text(uW) + ', ' + pd.Vmpp_text() + ', ' + pd.Jmpp_text(uA) + '\n' + pd.Voc_text() + ', ' + pd.Jsc_text(uA) + ', ' + pd.FF_text() + ', ' + pd.PCE_text() + '\n' + pd.cell_area_text()

            
        self.pd = pd
        if show:
            plx(text)
    
 
    def fit_param(self, T = T_RT, bounds = ([0, 0, 0], [10, np.inf, np.inf]), p0 = None, verbose = 0, xtol = None):
        """
        Fits  nid, Rs, Rsh to the JV curve. Saves the values in self.nid, self.Rs, self.Rsh.
        It also saves the initial fit parameters in self.ini_fp.
    
        Parameters
        ----------
        T : FLOAT, optional
            Temperature. The default is T_RT.
        bounds : tuple of two 3-arrays ([nid_min, Rs_min, Rsh_min], [nid_max, Rs_max, Rsh_max]), optional
            Boundary values for nid, Rs, and Rsh. The default is ([0, 0, 0], [3, np.inf, np.inf]).
    
        Returns
        -------
        3-ARRAY [nid, Rs, Rsh]
            Returns the fit parameters for nid, Rs and Rsh.
    
        """
    
        def func(V_arr, nid, Rs, Rsh):
            #J = np.array([IV_data.I_of_V(self.x[i], self.Jsc, self.Voc, nid, Rs, Rsh, T = T) for i in range(len(self.x))])
            J = np.array([IV_data.I_of_V(V_arr[i], self.Jsc, self.Voc, nid, Rs, Rsh, T = T) for i in range(len(V_arr))])
            return J
        #IV_data.I_of_V(V, Isc, Voc, n, Rs, Rsh, T = T_RT)
            
        if p0 == None:
            if not(hasattr(self, 'nid')) or not(hasattr(self, 'Rs')) or not(hasattr(self, 'Rsh')):
                self.det_ini_5param()
            p0 = [self.nid, self.Rs, self.Rsh]
            bounds = ([1, 0, 0], [10, np.inf, np.inf])
    
        popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds, verbose = verbose, xtol = xtol)
        #popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds)
    
        # store the initial 5 parameters in ini_fp
        self.ini_fp = fivep(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)
    
        self.nid = popt[0]
        self.Rs = popt[1]
        self.Rsh = popt[2]
    
        return popt


    def fit_param_old(self, T = T_RT, bounds = ([0, 0, 0], [10, np.inf, np.inf]), p0 = None):
        """
        Fits  nid, Rs, Rsh to the JV curve. Saves the values in self.nid, self.Rs, self.Rsh.
        It also saves the initial fit parameters in self.ini_fp.

        Parameters
        ----------
        T : FLOAT, optional
            Temperature. The default is T_RT.
        bounds : tuple of two 3-arrays ([nid_min, Rs_min, Rsh_min], [nid_max, Rs_max, Rsh_max]), optional
            Boundary values for nid, Rs, and Rsh. The default is ([0, 0, 0], [3, np.inf, np.inf]).

        Returns
        -------
        3-ARRAY [nid, Rs, Rsh]
            Returns the fit parameters for nid, Rs and Rsh.

        """
        
        def func(V, nid, Rs, Rsh):
            J = np.array([IV_data.I_of_V(self.x[i], self.Jsc, self.Voc, nid, Rs, Rsh, T = T) for i in range(len(self.x))])
            return J
        #IV_data.I_of_V(V, Isc, Voc, n, Rs, Rsh, T = T_RT)
        
        if not(hasattr(self, 'nid')):
            self.det_ini_5param()
        p0 = [self.nid, self.Rs, self.Rsh]
        
        
        popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds)
        
        # store the initial 5 parameters in ini_fp
        self.ini_fp = fivep(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)
        
        self.nid = popt[0]
        self.Rs = popt[1]
        self.Rsh = popt[2]
        
        return popt      
    
    
    def fit_fivep(self, T = T_RT, bounds = ([0, 0, 0, 0, 0], [100, 10, 10, np.inf, np.inf]), p0 = None):
        """
        Fits all five parameters (Voc, Jsc, nid, Rs, Rsh) of the single diode curve to the JV curve. 
        It saves the initial fit parameters in self.ini_fp.
    
        Parameters
        ----------
        T : FLOAT, optional
            Temperature. The default is T_RT.
        bounds : tuple of two 5-arrays ([Jsc_min, Voc_min, nid_min, Rs_min, Rsh_min], [Jsc_max, Voc_max, nid_max, Rs_max, Rsh_max]), optional
            Boundary values for Voc, Jsc, nid, Rs, and Rsh. The default is ([0, 0, 0, 0, 0], [100, 10, 10, np.inf, np.inf]).
        p0 : 5-array of FLOAT
            initial fit parameters. The default is None, in this case 
    
        Returns
        -------
        5-ARRAY [Jsc, Voc, nid, Rs, Rsh]
        Returns the fit parameters for Jsc, Voc, nid, Rs and Rsh.
    
        """
    
        def func(V, Jsc, Voc, nid, Rs, Rsh):
            J = np.array([IV_data.I_of_V(V[i], Jsc, Voc, nid, Rs, Rsh, T = T) for i in range(len(V))])
            return J
    
        if p0 == None:
            if self.x[1] < self.x[0]:
                print('Attention [IV-data.fit_fivep()]: Voltage array (self.x) is not strictly ascending!')
            p0 = [abs(self.y[0]), self.x[-1], 1.5, 0, 1e18]
    
    
        popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds, verbose = 0, xtol = None)
    
        # store the initial 5 parameters in ini_fp
        self.ini_fp = fivep(cell_area = self.cell_area, Voc = p0[1], Jsc = p0[0], nid = p0[2], Rs = p0[3], Rsh = p0[4])
    
        self.Jsc = popt[0]
        self.Voc = popt[1]
        self.nid = popt[2]
        self.Rs = popt[3]
        self.Rsh = popt[4]
    
        return popt


    
    def get_fp(self):
        return fivep(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)
    
    def table_param(PCE, Voc, Jsc, FF, Vmpp, Jmpp, light_int, cell_area):
            
        row_labels = ['$PCE\ (\%)$', '$V_{OC}\ (V)$', '$J_{SC}\ (mA/cm^2)$', '$FF\ (\%)$', 
                      '$V_{mpp} \ (V)$', '$J_{mpp} \ (mA/cm^2)$', 
                      '$Light \ int \ (mW/cm^2)$', '$Cell \ area \ (cm^2)$']
        cell_text = []
        cell_text.append([f'{PCE:1.1f}'])
        cell_text.append([f'{Voc:1.3f}'])
        cell_text.append([f'{Jsc:1.2f}'])
        cell_text.append([f'{FF:1.1f}'])
        cell_text.append([f'{Vmpp:1.3f}'])
        cell_text.append([f'{Jmpp:1.2f}'])
        cell_text.append([f'{light_int:.2f}'])
        cell_text.append([f'{cell_area}'])
        return cell_text, row_labels
         
    def plot(self, title = 'self.name', xscale = 'linear', yscale = 'linear', 
             left = None, right = None, bottom = None, top = None, plot_table = False, hline = None, vline = None, figsize=(9,6), return_fig = False, show_plot = True, **kwargs):
        
        if plot_table:
            cell_text, row_labels = IV_data.table_param(self.pd.PCE, self.Voc, self.Jsc, self.pd.FF, 
                                                        self.pd.Vmpp, self.pd.Jmpp, self.light_int, self.cell_area)
        else:
            cell_text = None
            row_labels = None
        
        if title == None:
            title = self.name
            
        fig = xy_data.plot(self, title = title, xscale = xscale, yscale = yscale, left = left, right = right, 
                           bottom = bottom, top = top, plot_table = plot_table, cell_text = cell_text, row_labels = row_labels, hline = hline, vline = vline, figsize = figsize, return_fig = return_fig, show_plot = show_plot, **kwargs)        
        if return_fig:      
            return fig

        
    def plot_fit(self, xscale = 'linear', yscale = 'linear', left = None, right = None, bottom = None, top = None,  title = None, plot_table = False):
        """
        Plot the IV curve and the fit with the internal 5 parameters.
        """
        def print5param():
            text = f'$n_{{id}} = {self.nid:.2f}, \ R_s = {self.Rs*1e3:.2e} \ \Omega \cdot cm^2, \ R_{{sh}} = {self.Rsh*1e3:.2e} \ \Omega \cdot cm^2$'
            return text
        J = np.array([IV_data.I_of_V(self.x[i], self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh, T = T_RT) for i in range(len(self.x))])
        IVp = IV_data(self.x, J, name = print5param())
        IVp.plotstyle = dict(linestyle = '-', color = 'red')
        mIV = mIV_data([self, IVp])
        mIV.label(['Measurement', IVp.name])
        
        if plot_table:
            cell_text, row_labels = IV_data.table_param(self.pd.PCE, self.Voc, self.Jsc, self.pd.FF, 
                                                        self.pd.Vmpp, self.pd.Jmpp, self.light_int, self.cell_area)
        else:
            cell_text = None
            row_labels = None
            
        if title == None:
            title = self.name
        
        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, left = left, right = right, bottom = bottom, top = top, 
                 title = title, plot_table = plot_table, cell_text = cell_text, row_labels = row_labels)
        
    def plot_ini_and_fit(self, xscale = 'linear', yscale = 'linear', bottom = None, top = None):
        """
        Plot the IV curve, the fit with the internal 5 parameters and the fit with the ini_fp parameters.
        """
        def print5param(Jsc, Voc, nid, Rs, Rsh):
            #text = f'Voc = {Voc:.3f} V, Jsc = {Jsc:.2f} mA/cm2' '\n' f'nid = {nid:.2f}, Rs = {Rs:.2e} Ohm cm2, Rsh = {Rsh:.2e} Ohm cm2'
            text = f'$n_{{id}} = {nid:.2f}, \ R_s = {Rs*1e3:.2e} \ \Omega \cdot cm^2, \ R_{{sh}} = {Rsh*1e3:.2e} \ \Omega \cdot cm^2$'
            return text
        J_int = np.array([IV_data.I_of_V(self.x[i], self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh, T = T_RT) for i in range(len(self.x))])
        J_ini = np.array([IV_data.I_of_V(self.x[i], self.ini_fp.Jsc, self.ini_fp.Voc, self.ini_fp.nid, self.ini_fp.Rs, self.ini_fp.Rsh, T = T_RT) for i in range(len(self.x))])
        IV_int = IV_data(self.x, J_int, name = 'Fitted parameters:\n' + print5param(self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh))
        IV_ini = IV_data(self.x, J_ini, name = 'Initial parameters:\n' + print5param(self.ini_fp.Jsc, self.ini_fp.Voc, self.ini_fp.nid, self.ini_fp.Rs, self.ini_fp.Rsh))
        IV_int.plotstyle = dict(linestyle = '-', color = 'red')
        IV_ini.plotstyle = dict(linestyle = '-', color = 'green')
        mIV = mIV_data([self, IV_ini, IV_int])
        #mIV.names_to_label()
        mIV.label(['Measurement', IV_ini.name, IV_int.name])
        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, bottom = bottom, top = top, title = self.name)
        
    def save_loss_param_old(self, IVsq, IVrad, IVtrans, row_labels, col_labels, save_dir, FN):
            
        cell_text = []
        cell_text.append([IVsq.Voc, IVsq.Jsc, IVsq.pd.FF, IVsq.pd.PCE])
        cell_text.append([IVrad.Voc, IVrad.Jsc, IVrad.pd.FF, IVrad.pd.PCE])
        cell_text.append([IVtrans.Voc, IVtrans.Jsc, IVtrans.pd.FF, IVtrans.pd.PCE])
        cell_text.append([self.Voc, self.Jsc, self.pd.FF, self.pd.PCE])
        
        alldata = np.array(cell_text, dtype = np.float64)
                    
        df = pd.DataFrame(data=alldata[0:,0:], columns = col_labels, index = row_labels)
                        
        TFN = join(save_dir, FN)
        if save_ok(TFN):
            df.to_csv(join(save_dir, FN), header = True, index = True)
     
    @staticmethod
    def save_loss_param(sa, row_labels, col_labels, save_dir, FN):
            
        cell_text = []

        for IV in sa:
            cell_text.append([IV.Voc, IV.Jsc, IV.pd.FF, IV.pd.PCE])
        
        alldata = np.array(cell_text, dtype = np.float64)
                    
        df = pd.DataFrame(data=alldata[0:,0:], columns = col_labels, index = row_labels)
                        
        TFN = join(save_dir, FN)
        if save_ok(TFN):
            df.to_csv(join(save_dir, FN), header = True, index = True)

    @staticmethod
    def IVsq(bg, x_max = None, light_int = None):
        if light_int == None:
            fp_sq = IV_data.SQ_limit(bg)
        else:
            fp_sq = IV_data.SQ_limit(bg, light_int = light_int)
        if (x_max == None) or (x_max < (fp_sq.Voc+0.01)):
            #x_max = max(self.x)
            x_max = fp_sq.Voc + 0.01
        new_x = np.arange(0, x_max, step = 0.001, dtype = np.float64)
        IVsq = IV_data.from_fp(new_x, fp_sq, name = f'Shockely-Queisser limit ($E_g = {bg:.3f} \ eV): \ V_{{oc,SQ}} = {fp_sq.Voc:.3f} \ V, \ J_{{sc,SQ}} = {fp_sq.Jsc:.2f} \ mA/cm^2, \ n_{{id}} = 1,\ R_s = 0,\ R_{{sh}} = \infty$')
        IVsq.det_perfparam()
        return IVsq
    
    @staticmethod
    def IVrad(Vocrad, Jsc, light_int = 100, x_max = None):
        if (x_max == None) or (x_max < Vocrad):
            x_max = Vocrad
        new_x = np.arange(0, x_max, step = 0.001, dtype = np.float64)
        Jrad = np.array([IV_data.I_of_V(new_x[i], Jsc, Voc = Vocrad, nid = 1, Rs = 0, Rsh = 1e15, T = T_RT) for i in range(len(new_x))])
        IVrad = IV_data(new_x, Jrad, light_int = light_int, name = f'Radiative limit: $V_{{oc,rad}} = {Vocrad:.3f} \ V, \ J_{{sc,rad}} = {Jsc:.2f} \ mA/cm^2, \ n_{{id}} = 1,\ R_s = 0,\ R_{{sh}} = \infty$')
        IVrad.det_perfparam()
        return IVrad
    
    @staticmethod
    def IVtrans(V_arr, Voc, Jsc, nid_rec, light_int = 100):
        Jtrans = np.array([IV_data.I_of_V(V_arr[i], Jsc, Voc = Voc, nid = nid_rec, Rs = 0, Rsh = 1e10, T = T_RT) for i in range(len(V_arr))])
        IVtrans = IV_data(V_arr, Jtrans, light_int = light_int, name = f'Transport limit: $V_{{oc}} = {Voc:.3f} \ V, \ J_{{sc,trans}} = {Jsc:.2f} \ mA/cm^2, \ n_{{id}} = {nid_rec:.2f},\ R_s = 0,\ R_{{sh}} = \infty$')
        IVtrans.det_perfparam()
        return IVtrans
        
        
    def loss_plot(self, bg = None, Vocrad = None, nid_rec = None, x_max = None, IVsq = None, IVrad = None, IVtrans = None, IVfit = None, title = None, xscale = 'linear', yscale = 'linear', 
                  left = None, right = None, bottom = None, top = None, what_to_show = ['measurement', 'fit', 'SQ limit', 'rad. limit', 'transp. limit'],
                  plot_table = False, figsize=(12,8), save = False, save_dir = '', bbox = [0.15, 0.25, 0.5, 0.3], **kwargs):
        """
        Plot the IV curve, the fit with the internal 5 parameters and the fit with the ini_fp parameters.
        """
        def print5param(Jsc, Voc, nid, Rs, Rsh):
            #text = f'Voc = {Voc:.3f} V, Jsc = {Jsc:.2f} mA/cm2' '\n' f'nid = {nid:.2f}, Rs = {Rs:.2e} Ohm cm2, Rsh = {Rsh:.2e} Ohm cm2'
            text = f'Voc = {Voc:.3f} V, Jsc = {Jsc:.2f} mA/cm$^2$, $n_{{id}} = {nid:.2f}, \ R_s = {Rs*1e3:.2e} \ \Omega \cdot cm^2, \ R_{{sh}} = {Rsh*1e3:.2e} \ \Omega \cdot cm^2$'
            return text
        
        #fp_sq = IV_data.SQ_limit(bg)

        if x_max == None:
            x_max = max(self.x)
            
        new_x = np.arange(0, x_max, step = 0.001, dtype = np.float64)

        linewidth = 2
        if 'linewidth' in kwargs:
            linewidth = kwargs['linewidth']

        if IVsq == None:
            if self.light_int == 100:
                IVsq = IV_data.IVsq(bg, x_max = x_max)
            else:
                IVsq = IV_data.IVsq(bg, x_max = x_max, light_int = self.light_int)
                print(f'Attention: light intensity {self.light_int:.1e} mW/cm2 is used to calculate Shockley-Queisser limit!')
        IVsq.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'black')
        
        if IVrad == None:
            IVrad = IV_data.IVrad(Vocrad, self.Jsc, self.light_int, x_max = Vocrad + 0.01)
        IVrad.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'green')
        
        if IVtrans == None:
            IVtrans = IV_data.IVtrans(self.x, self.Voc, self.Jsc, nid_rec, light_int = self.light_int)
        IVtrans.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'cyan')
        
        if IVfit == None:
            Jfit = np.array([IV_data.I_of_V(new_x[i], self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh, T = T_RT) for i in range(len(new_x))])
            IVfit = IV_data(new_x, Jfit, light_int = self.light_int, name = 'Fit: ' + print5param(self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh))
            IVfit.det_perfparam()
        IVfit.plotstyle = dict(linestyle = '-', linewidth = 1, color = 'red')
        
        #IVsq.det_perfparam()
        #IVrad.det_perfparam()
        #IVtrans.det_perfparam()
        
        show_this = []
        lab = []
        if 'measurement' in what_to_show:
            show_this.append(self)
            lab.append('Measurement')
        if 'fit' in what_to_show:
            show_this.append(IVfit)
            lab.append(IVfit.name)
        if 'SQ limit' in what_to_show:
            show_this.append(IVsq)
            lab.append(IVsq.name)
        if 'rad. limit' in what_to_show:
            show_this.append(IVrad)
            lab.append(IVrad.name)
        if 'transp. limit' in what_to_show:
            show_this.append(IVtrans)
            lab.append(IVtrans.name)            

        mIV = mIV_data(show_this)
        #mIV.names_to_label()
        mIV.label(lab)
        if 'legend' in kwargs:
            if kwargs['legend'] == False:
                mIV.no_label()

        
        row_labels = ['SQ lim.', 'Rad. lim.', 'Transp. lim.', 'Meas.']
        col_labels = ['Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'PCE (%)']
        
        if save:
            
            FN = f'{title} - losses.csv'

            row_labels = []
            if 'measurement' in what_to_show:
                row_labels.append('Meas.')
            if 'fit' in what_to_show:
                row_labels.append('fit')
            if 'SQ limit' in what_to_show:
                row_labels.append('SQ lim.')
            if 'rad. limit' in what_to_show:
                row_labels.append('Rad. lim.')
            if 'transp. limit' in what_to_show:
                row_labels.append('Transp. lim.')
                
            IV_data.save_loss_param(mIV.sa, row_labels, col_labels, save_dir, FN)
            mIV.save(save_dir, title, row_labels)
        
        if plot_table:
            
            cell_text = []
            row_labels = []
            
            if 'SQ limit' in what_to_show:
                cell_text.append([f'{IVsq.Voc:.3f}', f'{IVsq.Jsc:.2f}', f'{IVsq.pd.FF:2.1f}', f'{IVsq.pd.PCE:2.1f}'])
                row_labels.append('SQ lim.')
            if 'rad. limit' in what_to_show:
                cell_text.append([f'{IVrad.Voc:.3f}', f'{IVrad.Jsc:.2f}', f'{IVrad.pd.FF:2.1f}', f'{IVrad.pd.PCE:2.1f}'])
                row_labels.append('Rad. lim.')
            if 'transp. limit' in what_to_show:
                cell_text.append([f'{IVtrans.Voc:.3f}', f'{IVtrans.Jsc:.2f}', f'{IVtrans.pd.FF:2.1f}', f'{IVtrans.pd.PCE:2.1f}'])
                row_labels.append('Transp. lim.')
            if 'measurement' in what_to_show:
                cell_text.append([f'{self.Voc:.3f}', f'{self.Jsc:.2f}', f'{self.pd.FF:2.1f}', f'{self.pd.PCE:2.1f}'])
                row_labels.append('Meas.')


        else:
            cell_text = None
            row_labels = None
            col_labels = None
            
        if title == None:
            title = self.name
                
        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, bottom = bottom, left = left, right = right, top = top,
                 title = title, plot_table = plot_table, cell_text = cell_text, row_labels = row_labels, 
                 col_labels = col_labels, bbox = bbox, figsize = figsize, **kwargs)
        
        return IVsq, IVrad, IVtrans
    
    
    @staticmethod
    def loss_barplot(IV1, IVtrans1, IVrad1, IVsq1, IV2, IVtrans2, IVrad2, IVsq2, sample_names = ['Control', 'Treated'], save = False, save_dir = None, FN = None):
    
        # Losses (for barplot)
    
        Voc1_rad_SQ = (1 - IVrad1.Voc / IVsq1.Voc) * 100
        Jsc1_rad_SQ = (1 - IVrad1.Jsc / IVsq1.Jsc) * 100
        Voc1_TL_rad = (1 - IVtrans1.Voc / IVrad1.Voc) * 100
        FF1_TL_rad = (1 - IVtrans1.pd.FF / IVrad1.pd.FF) * 100
        FF1_meas_TL = (1 - IV1.pd.FF / IVtrans1.pd.FF) * 100
    
        Voc2_rad_SQ = (1 - IVrad2.Voc / IVsq2.Voc) * 100
        Jsc2_rad_SQ = (1 - IVrad2.Jsc / IVsq2.Jsc) * 100
        Voc2_TL_rad = (1 - IVtrans2.Voc / IVrad2.Voc) * 100
        FF2_TL_rad = (1 - IVtrans2.pd.FF / IVrad2.pd.FF) * 100
        FF2_meas_TL = (1 - IV2.pd.FF / IVtrans2.pd.FF) * 100
    
        # abbrevations
    
        o1 = Voc1_rad_SQ
        y1 = Jsc1_rad_SQ
        dg1 = Voc1_TL_rad
        lg1 = FF1_TL_rad
        b1 = FF1_meas_TL
    
        o2 = Voc2_rad_SQ
        y2 = Jsc2_rad_SQ
        dg2 = Voc2_TL_rad
        lg2 = FF2_TL_rad
        b2 = FF2_meas_TL
    
        # Data
    
        orangeBars = np.array([o1, o2])
        yellowBars = np.array([y1, y2])
        darkgreenBars = np.array([dg1, dg2])
        lightgreenBars = np.array([lg1, lg2])
        blueBars = np.array([b1, b2])    
    
        allBars = np.array([orangeBars, yellowBars, darkgreenBars, lightgreenBars, blueBars])
        #labels = [f'$1 - (V_{{OC,rad}} / V_{{OC,SQ}})$', f'$1 - (J_{{SC,rad}} / J_{{SC,SQ}})$', f'$1 - (V_{{OC,TL}} / V_{{OC,rad}})$', f'$1 - (FF_{{TL}} / FF_{{rad}})$', f'$1 - (FF_{{meas}} / FF_{{TL}})$']
        labels = ['$V_{{OC,SQ}} \\rightarrow V_{{OC,rad}}$', f'$J_{{SC,SQ}} \\rightarrow J_{{SC,rad}}$', f'$V_{{OC,rad}} \\rightarrow V_{{OC,TL}}$', f'$FF_{{rad}} \\rightarrow FF_{{TL}}$', f'$FF_{{TL}} \\rightarrow FF_{{meas.}}$']
    
    
        # plot
        barWidth = 0.85
        samples = [0,1]
        
        # calculate the bottom of each bar
        bot = np.array([[allBars[0:i, 0].sum(), allBars[0:i, 1].sum()] for i, val in enumerate(allBars)])
    
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(8,8))
    
        # calculate the bottom of each number
        bot_no = np.array([[allBars[0:i, 0].sum() + allBars[i,0]/2, allBars[0:i, 1].sum() + allBars[i,1]/2] for i, val in enumerate(allBars)])
    
        # Create orange Bars
        plt.bar(samples, orangeBars, bottom = bot[0], color='orange', edgecolor='orange', width=barWidth, label=labels[0])
    
        # Create yellow Bars
        plt.bar(samples, yellowBars, bottom = bot[1], color = 'yellow', edgecolor= 'yellow', width=barWidth, label=labels[1])
    
        # Create dark green Bars
        #plt.bar(samples, darkgreenBars, bottom = bot[2], color = '#b5ffb9', edgecolor= 'grey', hatch="///", width=barWidth, label=f'$V_{{OC,TL}} / J_{{OC,rad}}$')
        plt.bar(samples, darkgreenBars, bottom = bot[2], color = 'green', edgecolor= 'green', width=barWidth, label=labels[2])
    
        # Create light green Bars
        #plt.bar(samples, lightgreenBars, bottom = bot[3], color = '#b5ffb9', edgecolor= 'grey', hatch = '\\\\\\', width=barWidth, label=f'$FF_{{TL}} / FF_{{rad}}$')
        plt.bar(samples, lightgreenBars, bottom = bot[3], color = '#b5ffb9', edgecolor= '#b5ffb9', width=barWidth, label=labels[3])
    
        # Create blue Bars
        plt.bar(samples, blueBars, bottom = bot[4], color = 'skyblue', edgecolor= 'skyblue', width=barWidth, label=labels[4]) 
    
        # Plot numbers in each bar
        for sample in samples:
            for colors, bottom in enumerate(bot_no[:, sample]):
                plt.text(x = sample-0.05 , y = bottom - 0.5 , s=f'{allBars[colors, sample]:.1f}')
    
        # Custom x axis
        plt.xticks(samples, sample_names)
    
        plt.ylabel("Losses (%)")
    
        # Add a legend
        plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
            
        # Show graphic
        plt.show()
        
        if save:
                                
            df = pd.DataFrame(data=allBars, columns = sample_names, index = labels)
                            
            TFN = join(save_dir, FN)
            if save_ok(TFN):
                df.to_csv(join(save_dir, FN), header = True, index = True)

        
    def plot_fp(self, fp, title = None, xscale = 'linear', yscale = 'linear', bottom = None, top = None):
        """
        Plot the IV curve and the fit with the fp 5 parameters.
        """
        def print5param():
            #text = f'Voc = {fp.Voc:.3f} V, Jsc = {fp.Jsc:.2f} mA/cm2' '\n' f'nid = {fp.nid:.2f}, Rs = {fp.Rs:.2e} Ohm cm2, Rsh = {fp.Rsh:.2e} Ohm cm2'
            text = f'$n_{{id}} = {fp.nid:.2f}, \ R_s = {fp.Rs*1e3:.2e} \ \Omega \cdot cm^2, \ R_{{sh}} = {fp.Rsh*1e3:.2e} \ \Omega \cdot cm^2$'
            return text

        J = np.array([IV_data.I_of_V(self.x[i], fp.Jsc, fp.Voc, fp.nid, fp.Rs, fp.Rsh, T = T_RT) for i in range(len(self.x))])
        IVp = IV_data(self.x, J, name = print5param())
        IVp.plotstyle = dict(linestyle = '-', color = 'red')
        mIV = mIV_data([self, IVp])
        #mIV.names_to_label()
        mIV.label(['Measurement', IVp.name])
        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, bottom = bottom, top = top, title = self.name)

        
    @staticmethod
    def from_fp(x_arr, fp, name = '', light_int = 100, T = T_RT, perfparam = True):
        """
        Returns an instance of IV_data from the five parameters in fp.

        Parameters
        ----------
        x_arr : numpy array
            array of Voltage in V.
        fp : instance of fivep.
            contains the five parameters Voc, Jsc, nid, Rs, Rsh (and cell_area).
        name: string
            Name of the curve.
        light_int : float, optional
            Light intensity in mW/cm2. The default is 100.
        T : float, optional
            Cell temperature in K. The default is T_RT.

        Returns
        -------
        IV : Instance of IV_data
            
        """
        J = np.array([IV_data.I_of_V(x_arr[i], fp.Jsc, fp.Voc, fp.nid, fp.Rs, fp.Rsh, T = T) for i in range(len(x_arr))])
        IV = IV_data(x_arr, J, light_int = light_int, name = name)
        if perfparam:
            IV.det_ini_5param()
            IV.det_perfparam()
        if fp.cell_area != None:
            IV.cell_area = fp.cell_area
        return IV
    
    @staticmethod
    def I_of_V(V, Isc, Voc, nid, Rs, Rsh, T = T_RT):
        """
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 8
        """
        if Rs * Isc > 700:
            print(f'I_of_V error: The factor Rs * Isc is {Rs*Isc:.1e}, i.e. > 700, np.exp(Rs*Isc) is too large, decrease Rs (in kOhm cm2) to below {700/Isc:.0f}!')
        if Rs == 0:
            Rs = 1e-10
        if Rsh == np.inf:
            Rsh = 1e20
        argW = q * Rs / (nid * k * T) * (Isc - Voc / (Rs + Rsh)) * math.exp(- q * Voc / (nid * k * T)) * math.exp(q / (nid * k * T) * (Rs * Isc +  Rsh * V / (Rsh + Rs)))
        I = nid * k * T / (q * Rs) * lambertw(argW) + V / Rs - Isc - Rsh * V / (Rs * (Rsh + Rs))
        return I.real
    
    @staticmethod
    def SQ_limit_Voc(bg, illumspec_eV = None, light_int = None, from_file = True):
        
        if from_file:
            FN = 'Vocsq.csv'
            d = xy_data.load(system_dir, FN, take_quants_and_units_from_file = True)
            Vocsq = d.y_of(bg, interpolate = True)
        else:
            if illumspec_eV == None:
                # Load Astmg173 spectrum
                AM15_nm = diff_spectrum.load_ASTMG173()
                AM15_nm.equidist()
                #AM15_nm.plot(left = 300, right = 1000)    
                AM15_eV = AM15_nm.nm_to_eV()
                AM15_eV.equidist(left = 0.414, right = 4.27, delta = 0.001)
                #AM15_eV.plot()
                if light_int != None:
                    AM15_eV.y = AM15_eV.y * light_int/100
                illumspec_eV = AM15_eV
                
            left = min(illumspec_eV.x)
            right = max(illumspec_eV.x)
            eV_arr = np.arange(left, right, 0.001)
            #Absorptance spectrum
            ab = abs_spectrum(eV_arr, np.array([0 if eV < bg else 1 for eV in eV_arr]))
            ab.calc_Jradlim(illumspec_eV, start_eV = bg, handover_eV = bg+0.1)
            ab.calc_Vocrad(E_start = bg-0.3, E_stop = bg+0.3, T = T_RT, show_table = False)
            Vocsq = ab.Vocrad
        
        return Vocsq
    
    @staticmethod
    def SQ_limit_Jsc(bg, illumspec_eV = None, light_int = None):
                
        if illumspec_eV == None:
            AM15 = diff_spectrum.AM15_eV()
            if light_int != None:
                AM15.y = AM15.y * light_int/100
            illumspec_eV = AM15

        # Calculation of above bandgap photon flux                    
        abpf = illumspec_eV.photonflux(start = bg, stop = illumspec_eV.x[-1])  
        # Calculation of Shockley-Queisser photocurrent
        return abpf * q * 1e3/1e4
        
        
    @staticmethod
    def SQ_limit(bg, illumspec_eV = None, light_int = None):
        """
        Returns the SQ-limit five parameters. 
    
        Parameters
        ----------
        bg : FLOAT
            Bandgap in eV.
        illumspec_eV: diff_spectrum
            Illumination spectrum other than AM1.5GT        
            If None then AM1.5GT spectrum is taken
        light_int: FLOAT
            If AM1.5GT spectrum as illumination spectrum used, then with light_int the light intensity in mW/cm2 can be chosen.
            If None than 100 mW/cm2 (1 sun) is used.
    
        Returns
        -------
        fp : instance of fivep.
    
        """
            
        if illumspec_eV == None:
            AM15_eV = diff_spectrum.AM15_eV(left = 0.310, right = 4.428, delta = 0.001, y_unit = 'Spectral photon flux')
            if light_int != None:
                AM15_eV.y = AM15_eV.y * light_int/100
            illumspec_eV = AM15_eV
            
        Vsq = IV_data.SQ_limit_Voc(bg, illumspec_eV)
        Jsq = IV_data.SQ_limit_Jsc(bg, illumspec_eV)
        
        return fivep(cell_area = 1, Voc = Vsq, Jsc = Jsq, nid = 1, Rs = 0, Rsh = np.inf)
    
        
    def calc_resistance_curve(self, left = None, right = None):
        if left == None:
            left = min(self.x)
        if right == None:
            right = max(self.x)
        ra = range(self.x_idx_of(left), self.x_idx_of(right)+1)
        curve = self.copy()
        curve.x = curve.x[ra]
        curve.y = curve.y[ra]
        dIV = curve.diff()
        #dIV.plot()
        resist_arr = 1 / dIV.y * 1000 # Ohm * cm2
        R = IV_data(dIV.x, resist_arr, quants = dict(x = 'Voltage', y = 'Resistance'), units = dict(x = 'V', y = 'Ohm cm2'))        
        return R
    
    def resistance_plot(self, V_rel, left = None, right = None, bottom = None, top = None, vline = None, yscale = 'linear', title = None):
    
        if left == None:
            left = V_rel - 0.3
        if right == None:
            right = V_rel+0.3
        R = self.calc_resistance_curve(left = left, right = right)
        if bottom == None:
            bottom = R.min_within(left = left, right = right) * 0.9
        if top == None:
            top = R.max_within(left = left, right = right) * 1.1
            #top = 10*self.Rs*1000
        if vline == None:
            vline = V_rel
        R.plot(yscale = yscale, left = left, right = right, bottom = bottom, top = top, vline = vline, title = title)
        R_rel = R.y_of(V_rel)
        print(f'The resistance at {V_rel:.3f} V is: {R_rel:.2f} Ohm cm2')
        
        return R_rel
        
        
class mIV_data(mxy_data):
    
    def __init__(self, sa):
        super().__init__(sa)
        
