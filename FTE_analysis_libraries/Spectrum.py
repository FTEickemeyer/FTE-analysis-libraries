# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:05:55 2020

@author: dreickem
"""

from scipy.optimize import curve_fit, least_squares
import sys
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import join
import math
import matplotlib.pyplot as plt
from importlib import reload
from IPython import embed
import pkg_resources
system_dir = pkg_resources.resource_filename( 'FTE_analysis_libraries', 'System_data' )


from .General import findind, int_arr, linfit, save_ok, q, k, T_RT, h, c, f1240, pi

from .XYdata import xy_data, mxy_data

class spectrum(xy_data):
    
    def __init__(self, x, y, quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, name = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
                  
    
    @classmethod  
    def load_Andor(cls, directory, FN = '', meta_data = None):
        """
        Loads a sinlge Andor spectrum. If a filename is given it will be used, if not the first file in the directory will be used.
        If meta_data is provided, integration time (int_s) and accumulations (acc) will be taken from metadata.
        if not, it will take the integration time and number of accumulations from the filename.
        """
        
        if FN == '':
            FN = listdir(directory)[0]        

        dat = pd.read_csv(join(directory, FN))
        x = np.array(dat, dtype = np.float64)[:,0]

        if type(meta_data) == dict:
            y = np.array(dat, dtype = np.float64)[:,1] / meta_data['int_s'] / meta_data['acc']
        else:
            def get_int_time(FN):
                it = FN.split('--')[3].split('_')[1].split('s')[0]
                return float(it)

            def get_accum(FN):
                acc = FN.split('--')[3].split('_')[2].split('acc')[0]
                #acc_raw = FN.split('--')[3].split('_')[2].split('av')[0]
                return int(acc)
    
            y = np.array(dat, dtype = np.float64)[:,1] / get_int_time(FN) / get_accum(FN)
        
        return cls(x, y, quants = dict(x = 'Wavelength', y = 'Intensity'), units = dict(x = 'nm', y = 'cps'), name = FN)
        
            
    def nm_to_eV(self):
    
        """
        Transforms a single non-differential spectrum of type spectrum from wavelength to photon energy. 
        """
        
        x = self.x.copy()
        y = self.y.copy()
        x_eV = h * c / (x[::-1] * 1e-9) / q
        y_eV = y[::-1]
                
        name = self.name
        quants = {"x": "Photon energy", "y": self.qy}
        units = {"x": "eV", "y": self.uy}
        
        if self.__class__.__name__ == 'spectrum':
            sp_new = spectrum(x_eV, y_eV, quants = quants, units = units, name = name)
        elif self.__class__.__name__ == 'EQE_spectrum':
            sp_new = EQE_spectrum(x_eV, y_eV, quants = quants, units = units, name = name)
        elif self.__class__.__name__ == 'abs_spectrum':
            sp_new = abs_spectrum(x_eV, y_eV, quants = quants, units = units, name = name)
            
        return sp_new
    

    
    def old_max_within(self, left = None, right = None):
        r = range(findind(self.x, left), findind(self.x,right)+1)
        max_y = max(self.y[r])
        max_x = self.x[findind(self.x, left) + findind(self.y[r], max_y) - 1]
        return [max_x, max_y]

    
    @staticmethod
    def luminosity_fn(y_unit = 'Spectral photon flux'):
        lum_FN = 'luminosity function CIE 1931.csv'
        return spectrum.load(system_dir, lum_FN, quants = dict(x = 'Wavelength', y = 'Conversion factor'), units = dict(x = 'nm', y = 'lx/(W/m2)'))



class EQE_spectrum(spectrum):
    
    def __init__(self, x, y, quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, name = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
        
    @staticmethod
    def EQE100(Eg, start=300, stop=4001, step=0.5, name = ''):
        """
        Returns an EQE spectrum in % as a function of wavelength in nm from start to stop with a step size of 1 nm.
        Eg is the bandgap in eV.
        """
        Eg_nm = f1240/Eg
        x_arr = np.arange(start=start, stop=stop, step=step)
        y_arr = np.array([0 if x>Eg_nm else 100 for x in x_arr])
        if name == '':
            name = f'100 % EQE until cut off wavelength {Eg_nm: .0f} nm'
        return EQE_spectrum(x = x_arr, y = y_arr, quants = dict(x='Wavelength', y='EQE'), units = dict(x='nm', y='%'), name = name)

    @staticmethod    
    def MMF_Eg(Eg, ref_EQE, sim_PF, ref_PF = 'AM15GT', left = 300, right = None, delta = 0.5):
        """
        Calculates the spectral mismatch factor as a function of Eg in eV.
        It is assumed a EQE spectrum which is 100% above Eg and 0 below.
        """
        Eg_nm = f1240/Eg
        x_arr = np.arange(start = left, stop = int(Eg_nm), step = delta)
        y_arr = np.ones(len(x_arr))
        sp = EQE_spectrum(x = x_arr, y = y_arr, quants = dict(x='Wavelength', y='EQE'), units = dict(x='nm', y=''))
        #return sp.MMF_test(ref_EQE, sim_PF, ref_PF, left = 300, right = Eg_nm, delta = 1)
        return sp.MMF(ref_EQE, sim_PF, ref_PF = ref_PF, left = left, right = right, delta = delta)

    def MMF(self, ref_EQE, sim_PF, ref_PF = 'AM15GT', left = None, right = None, delta = None):
        """
        Calculates the spectral mismatsch factor of the EQE spectrum self.
        How to apply it: The photocurrent of the Si reference diode has to be the photocurrent of 
        the Si ref. diode under AM1.5GT multiplied with the mismatch factor
        """
        num = ref_EQE.calc_Jsc(left = left, right = right, delta = delta, sp = ref_PF) * self.calc_Jsc(left = left, right = right, delta = delta, sp = sim_PF)
        denom = ref_EQE.calc_Jsc(left = left, right = right, delta = delta, sp = sim_PF) * self.calc_Jsc(left = left, right = right, delta = delta, sp = ref_PF)
        return num / denom
    
    def MMF_test(self, ref_EQE, sim_PF, ref_PF = 'AM15GT', left = None, right = None, delta = None):
        """
        Calculates the spectral mismatsch factor of the EQE spectrum self.
        How to apply it: The photocurrent of the Si reference diode has to be the photocurrent of 
        the Si ref. diode under AM1.5GT multiplied with the mismatch factor
        """
        print(f'ref_EQE.calc_Jsc(sp = ref_PF) = {ref_EQE.calc_Jsc(left = left, right = right, delta = delta, sp = ref_PF):.3f}')
        print(f'self.calc_Jsc(sp = sim_PF) = {self.calc_Jsc(left = left, right = right, delta = delta, sp = sim_PF):.3f}')
        print(f'ref_EQE.calc_Jsc(sp = sim_PF) = {ref_EQE.calc_Jsc(left = left, right = right, delta = delta, sp = sim_PF):.3f}')
        print(f'self.calc_Jsc(sp = ref_PF) = {self.calc_Jsc(left = left, right = right, delta = delta, sp = ref_PF):.3f}')
        num = ref_EQE.calc_Jsc(left = left, right = right, delta = delta, sp = ref_PF) * self.calc_Jsc(left = left, right = right, delta = delta, sp = sim_PF)
        denom = ref_EQE.calc_Jsc(left = left, right = right, delta = delta, sp = sim_PF) * self.calc_Jsc(left = left, right = right, delta = delta, sp = ref_PF)
        return num / denom
    
    def old_second_diff(self, left = None, right = None):
        
        if left == None:
            left = min(self.x)
        if right == None:
            right = max(self.x)

        le = findind(self.x, left)
        ri = findind(self.x, right)

        ra = range(le, ri+1)
        x = self.x[ra]
        
        d2ydx2 = np.gradient(np.gradient(self.y[ra], x), x)
        name = f'Second derivative of: {self.name}'
        quants = dict(x = self.qx, y = f'$d^2$({self.qy})/$d$({self.qx})$^2$')
        units = dict(x = self.ux, y = f'{self.uy}/({self.ux})$^2$')
        
        return EQE_spectrum(x, d2ydx2, quants = quants, units = units, name = name)
        
    
    def bg_from_ip(self, left = None, right = None, showplot = None):
        """
        Calculates the bandgap from the inflection point of the EQE spectrum.

        """
            
        if left == None:
            left = min(self.x)
        if right == None:
            right = max(self.x)
            
        dEQE = self.diff(left = left, right = right)
        Eg = dEQE.x[findind(dEQE.y, max(dEQE.y))]
        
        if 'diff' in showplot:
            dEQE.plot(left = left, right = right, vline = Eg)
        
        if 'orig' in showplot:
            self.plot(left = left, right = right, vline = Eg)
        
        self.Eg_ip = Eg
        
        return Eg
        
    def calc_Jsc(self, left = None, right = None, delta = None, sp = 'AM15GT'):    
        """
        #Calculates the integrated Jsc of an EQE spectrum. sp is the spectral photon flux (type diff_spectrum) of the light source.
        #It works if self and sp are both functions of nm or eV.
        """
        if sp != 'AM15GT':
            sp = sp.copy() # Otherwise the original lamp spectrum is changed with the function equidist
            if 'W' in sp.uy:
                print('Attention(EQE_spectrum.calc_Jsc()): sp expects a spectral photon flux not an irradiance spectrum!')

        if self.ux == 'nm':
            if delta==None:
                delta = 0.5
            EQE_nm = self.copy()
            if left == None:
                left = min(EQE_nm.x)
            if right == None:
                right = max(EQE_nm.x)
            EQE_nm.equidist(left = left, right = right, delta = delta)
            if sp == 'AM15GT':
                sp = diff_spectrum.AM15_nm(left = left, right = right, delta = delta)
            else:
                if sp.ux == 'nm':
                    sp.equidist(left = left, right = right, delta = delta)
                elif sp.ux == 'eV':
                    print('Attention (EQE_spectrum.calc_Jsc): EQE is a function of wavelength in nm but sp is a function of photon energy in eV!')
                else:
                    print('Attention: EQE_spectrum.calc_Jsc() requires ux to be either in nm or eV!')
                    print('Your ux is {sp.ux}.')
                    print(f'self.name = {self.name}')
                    print(f'sp.name = {sp.name}')
        
            if self.uy == '%':
                return np.trapz(EQE_nm.y/100 * sp.y * q * 1e3/1e4, dx = delta) #mA/cm2
            elif (self.uy == '') or (self.uy =='abs.'):
                return np.trapz(EQE_nm.y * sp.y * q * 1e3/1e4, dx = delta)
        
        if self.ux == 'eV':
            if delta == None:
                delta = 0.001
            EQE_eV = self.copy()
            if left == None:
                left = min(EQE_eV.x)
            if right == None:
                right = max(EQE_eV.x)
            EQE_eV.equidist(left = left, right = right, delta = delta)
            
            if sp == 'AM15GT':
                sp = diff_spectrum.AM15_eV(left = left, right = right, delta = delta)
            else:
                if sp.ux == 'eV':
                    sp.equidist(left = left, right = right, delta = delta)
                elif sp.ux == 'nm':
                    print('Attention (EQE_spectrum.calc_Jsc): EQE is a function of photon energy in eV but sp is a function of wavelength in nm!')
                else:
                    print('Attention: EQE_spectrum.calc_Jsc() requires ux to be either in nm or eV!')
                    print('Your ux is {sp.ux}.')                
        
            if self.uy == '%':
                return np.trapz(EQE_eV.y/100 * sp.y * q * 1e3/1e4, dx = delta)
            elif (self.uy == '') or (self.uy =='abs.'):
                return np.trapz(EQE_eV.y * sp.y * q * 1e3/1e4, dx = delta)


    def normalize_to_Jsc(self, Jsc):
        
        Jsc_EQE = self.calc_Jsc()
        self.y = self.y / Jsc_EQE * Jsc
        
    
    def to_ab(self):
        x = self.x.copy()
        y = self.y.copy() / 100 # absorptance is EQE (in %) / 100
        
        quants = dict(x=self.qx, y='absorptance')
        units = dict(x=self.ux, y='')
        name = self.name
        
        sp_new = abs_spectrum(x, y, quants = quants, units = units, name = name)
            
        return sp_new    
    
    @staticmethod
    def load_Cicci(directory, FN):
        
        return EQE_spectrum.load(directory, FN, delimiter = '\t', header = 0, quants = {"x": "Wavelength", "y": "EQE"}, units = {"x": "nm", "y": "%"})
    
class abs_spectrum(spectrum):
    """
    Absorptance spectrum. Absorptance is dimensionless from 0 to 1.
    """
    
    def __init__(self, x, y, quants = {"x": "Photon energy", "y": "Absorptance"}, units = {"x": "eV", "y": ""}, name = 'Absorptance', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
        
        
    def U_energy_fit(self, Efit_start, Efit_stop):
        """
        Calculates the Urbach energy of an absorptance curve and returns the fitcurve as an instance of the class "spectrum".
        self: absorptance curve, instance of class "spectrum"
        Efit_start: Photon energy (in eV) to begin with the fit.
        Efit_stop: Photon energy to end with the fit.
        """
        m, b = linfit(self.x, np.log(np.abs(self.y)), Efit_start, Efit_stop)
        U_E = 1/m * 1000 #meV
        name = f'Fit: $E_u$ = {U_E:.1} eV'
        UE_fit = spectrum(self.x, np.exp(m * self.x + b), quants = {"x": self.qx, "y": "Urbach energy fit"}, units = {"x": self.ux, "y": ""}, name = name)
        UE_fit.UE = U_E
        return UE_fit
    
    def new_UE(self, UE, E_takeover):
        m = 1000/UE
        idx = self.x_idx_of(E_takeover)
        b = self.y[idx]
        # UE_fit = spectrum(self.x, np.exp(m * (self.x - E_takeover)) * b, quants = {"x": self.qx, "y": "Urbach energy fit"}, units = {"x": self.ux, "y": ""})
        UE_fit = spectrum(self.x, np.exp(m * (self.x - self.x[idx])) * b, quants = {"x": self.qx, "y": "Urbach energy fit"}, units = {"x": self.ux, "y": ""})
        self.y[0:idx] = UE_fit.y[0:idx] 
        
    
    def Tauc_plot(self, Efit_start, Efit_stop, left_offs, right_offs, showplot = True, title = '', save = False, save_dir = '', save_name = None, return_fig = True):
        """
        Plots the Tauc plot, calculates the direct bandgap and Shockley-Queisser limit of Voc.
        """
        alpha_eV = - np.log(np.abs(1-self.y))
        m, b = linfit(self.x, (alpha_eV*self.x)**2, Efit_start, Efit_stop)
        Eg = -b/m
        Vocsq = 0.932*Eg - 0.167
        name_Tp = r'$(\alpha \cdot h \nu)^2 \, \sim \, [\ln(1-a) \cdot h \nu]^2$'
        #print(name_Tp)
        name_Tpfit = f'Fit, $Eg ={Eg:.3f}$ eV (${f1240/Eg:.0f}$ nm)' + ', $V_{{oc,sq}}$ ' + f'$= {Vocsq:.3f}$ eV'

        qy = r'$(\alpha \cdot h \nu)^2$'
        uy = 'a.u.'
        Tp = spectrum(self.x, (alpha_eV * self.x)**2, quants = {"x": self.qx, "y": qy}, units = {"x": self.ux, "y": uy}, 
                      name = name_Tp)

        Tpfit = spectrum(self.x, (m * self.x + b), quants = {"x": self.qx, "y": qy}, units = {"x": self.ux, "y": uy}, 
                         name = name_Tpfit)
        
        sa = spectra([Tp, Tpfit])
        sa.label([name_Tp, name_Tpfit])

        if save:
            if save_name is None:
                FN = 'Tauc plot.csv'
            else:
                FN = save_name
            sa.save(save_dir, FN) 
        
        self.Eg_Tauc = Eg
        self.Vocsq = Vocsq
        
        graph = sa.plot(left = Efit_start + left_offs, right = Efit_stop + right_offs, bottom = 0, top = Tp.y_of(Efit_stop + right_offs) * 1.2, title = title, return_fig = True, show_plot = showplot)

        if return_fig:
            return graph





    def emission_pf(self, E_start, E_stop, T = T_RT):
        """
        Calculate the emission photon flux for a given absorptance spectrum self as a function of photon energy.
        """
        BB = diff_spectrum.PhiBB(self.x * q, T)
        r = range(self.x_idx_of(E_start), self.x_idx_of(E_stop)+1)
        empf = diff_spectrum(self.x[r], self.y[r] * BB[r] * q, quants = dict(x = 'Photon energy', y = 'Spectral photon flux'), units = dict(x = 'eV', y = '1/[s m2 eV]'))        
        return empf

    def calc_Vocrad(self, E_start, E_stop, T = T_RT, show_table = False):
        """
        Calculates the Voc,rad for the absorptance self (of type spectrum) and plots a table with all relevant information.
        E_start: Photon energy to start integration (in eV)
        E_stop: Photon energy to stop integration (in eV)
        Jph: Photo current (in mA/cm2)
        T: Temperature in K
        """
        empf = self.emission_pf(E_start, E_stop, T = T)
        denergies_eV = self.x[1] - self.x[0]
        Jrad0 = q * np.trapz(empf.y, dx = denergies_eV) *1e-4 /1e-3 #in mA/cm2

        Vocrad = k*T/q * math.log(self.Jradlim/Jrad0 + 1)
        
        self.Jrad0 = Jrad0 
        self.Vocrad = Vocrad

        if show_table:
            table_data = [[f'{Vocrad:1.3f}', f'{Jrad0:1.2e}', f'{self.Jradlim:2.1f}', f'{T - 273.15:2.1f}']]
            collabel = [r'$V_{oc, rad} \; (V)$', r'$J_{rad, 0} \; (mA/cm^{2})$', r'$J_{ph} \; (mA/cm^2)$', r'$T \; (°C)$']
            
            #print(table_data)
            # plot table
            fig, ax = plt.subplots()
    
            # Hide axes
            ax.xaxis.set_visible(False) 
            ax.yaxis.set_visible(False)
    
            # Table
            table = ax.table(cellText=table_data, colLabels=collabel, loc='center', cellLoc='center', colLoc='center')
            fig.patch.set_visible(False)
            ax.axis('off')
            table.set_fontsize(40)
            table.scale(1.1, 2.5)
            plt.show()        
 
    def calc_Jradlim(self, illumspec_eV = None, start_eV = None, handover_eV = None):
        """
        Calculates the radiative limit phtocurrent taking the absorptance spectrum (self).
        Absorptance spectrum has to be a function of photon energy in eV.

        Parameters
        ----------
        illumspec_eV : diff_spectrum
            Illumination photon flux spectrum as a function of eV in 1/[s m2 eV] (e.g. AM 1.5 GT spectrum).
            Has to have equidistant x-axis
        start_eV : float, optional
            photon energy from which on the current is integrated.. The default is 1.3.
        handover_eV : float or None, optional            
            All absorptance values above handover_eV will be assumed to be the same as handover_eV. The default is 1.6.
            If None, there will be no handover.

        Returns
        -------
        None.

        """
        if illumspec_eV == None:
            illumspec_eV = diff_spectrum.AM15_eV(left = min(self.x), right = max(self.x))
            
        # New absorption spectrum with the same x-values as illumspec_eV (necessary for integration)
        asp = diff_spectrum(illumspec_eV.x, int_arr(self.x, self.y, illumspec_eV.x))
                
        if handover_eV != None:
            asp.y[asp.x_idx_of(handover_eV):] = asp.y[asp.x_idx_of(handover_eV)]
            
        if start_eV != None:
            start_idx = asp.x_idx_of(start_eV)
        
        else:
            start_idx = 0
        
        dx = (asp.x[1] - asp.x[0])
                
        self.Jradlim = np.trapz(asp.y[start_idx:] * illumspec_eV.y[start_idx:] * q, dx = dx) * 1e3 / 1e4

    def convert_absorbance_to_absorptance(self):
        A = self.copy()
        new_y = 1 - 10**(-A.y)
        ab = abs_spectrum(self.x, new_y, self.quants(), self.units(), self.name)
        ab.qy = 'Absorptance'
        return ab
    
    def convert_absorptance_to_absorbance(self):
        A = self.copy()
        new_y = -np.log10(1-A.y)
        ab = abs_spectrum(self.x, new_y, self.quants(), self.units(), self.name)
        ab.qy = 'Absorbance'
        return ab

    @staticmethod
    def load_absorbance(directory, FN):
        return abs_spectrum.load(directory, FN = FN, delimiter = ',', header = 'infer', take_quants_and_units_from_file = True)

    def absorbed_photonflux(self, left, right, details = True):
        # Calculates the absorbed photon flux in 1/(s m²) of an absoptance spectrum under AM1.5G.
        sp = self.copy()
        sp.equidist(left = left, right = right, delta = 1)
        AM = diff_spectrum.AM15_nm(left = left, right = right, delta = 1)
        AMnorm = AM.copy()
        AMnorm.y = AM.y / AM.y[AM.x_idx_of((left+right)/2)]
        both = spectra([self, AMnorm])
        both.label([self.name, 'AM1.5G'])
        if details:
            both.plot(hline = 0, vline = [left, right])
        spAM = AM * sp.y
        PF_1sun = spAM.calc_integrated_photonflux()
        if details:
            print(f'The absorbed photon flux at 1 sun is: {PF_1sun:.2e} 1/(s m²)')
        return PF_1sun

        
class diff_spectrum(spectrum):   
    
    def __init__(self, x, y, quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, name = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
               

    def spectralflux_to_photonflux(self, factor = 1):
        """
        This is the old version: Use the function "diff_spectrum.irradiance_to_photonflux" in future!
        In the new version self remains unchanged.
        Converts y from spectral flux into photon flux (in 1/[s m2 nm]).
        Expects that x is wavelength in nm and y is spectral flux in factor * W / (m2 nm).
        Example: If y is in uW/(cm2 nm), then factor is 1e-6/1e-4.
        """
        
        old_y = self.y
        self.y = factor * old_y * (self.x * 1e-9) / (h * c)
        self.qx = "Wavelength"
        self.qy = "Spectral photon flux"
        self.ux = "nm"
        self.uy = "1/[s m2 nm]"
            
    def pf_to_sf(self):
        """
        This is the old version: Use the function "diff_spectrum.photonflux_to_irradiance" in future!
        """
        if self.uy == '1/[s m2 nm]':
            if self.ux == 'nm':
                SF = self.y / (self.x * 1e-09 / (h * c)) # spectral flux (SF) in W/[m2 nm]
                new = self.copy()
                new.y = SF
                new.qy = 'Spectral flux'
                new.uy = 'W/[m2 nm]'
                return new
            else:
                print('Attention: pf_to_sf works only if ux = nm!')
        else:
            print('Attention: pf_to_sf works only if uy = 1/[s m2 nm]!')
            
            
    def photonflux_to_irradiance(self):
        """
        Converts y from spectral photon flux into spectral irradiance (in W/[m2 nm]).
        Expects that x is wavelength in nm and y is spectral photon flux in 1/[s m2 nm].
        Attention: This function has been changed 31.10.2020. Before self was changed and returned. Now self stays unchanged.
        """
        if self.uy == '1/[s m2 nm]':
            if self.ux == 'nm':
                SF = self.y / (self.x * 1e-09 / (h * c)) # spectral flux (SF) in W/[m2 nm]
                new = self.copy()
                new.y = SF
                new.qy = 'Spectral irradiance'
                new.uy = 'W/[m2 nm]'
                return new
            else:
                print('Attention: diff_spectrum.photonflux_to_irradiance works only if ux = nm!')
                print(f'Your ux = {self.ux}')
        else:
            print('Attention: diff_spectrum.photonflux_to_irradiance works only if uy = 1/[s m2 nm]!')
            print(f'Your uy = {self.uy}')
    
    
    def irradiance_to_photonflux(self, factor = 1):
        """
        Converts y from spectral irradiance into photon flux (in 1/[s m2 nm]).
        Expects that x is wavelength in nm and y is spectral irradiance in factor * W/[m2 nm].
        Example: If y is in uW/[cm2 nm], then factor is 1e-6/1e-4.
        """
        if factor != 1:
            PF = factor * self.y * (self.x * 1e-9) / (h * c)
            new = self.copy()
            new.y = PF
            new.qy = 'Spectral photon flux'
            new.uy = '1/[s m2 nm]'
            return new
        else:
            if self.uy == 'W/[m2 nm]':
                if self.ux == 'nm':
                    PF = self.y * (self.x * 1e-9) / (h * c)
                    new = self.copy()
                    new.y = PF
                    new.qy = 'Spectral photon flux'
                    new.uy = '1/[s m2 nm]'
                    return new
                else:
                    print('Attention: diff_spectrum.irradiance_to_photonflux works only if ux = nm!')
                    print(f'Your ux = {self.ux}')
            else:
                print('Attention: diff_spectrum.irradiance_to_photonflux works only if uy = W/[m2 nm]!')
                print(f'Your uy = {self.uy}')
    
    def irradiance_to_illuminance(self):
        """
        Returns the illuminance spectrum of the irradiance spectrum self.
        """
        if self.uy == 'W/[m2 nm]':
            cf = spectrum.luminosity_fn()
            return self.product(cf, qy = 'Spectral illuminance', uy = 'lm/[m2 nm]')
        else:
            print('Attention: Function "diff_spectrum.irradiance_to_illuminance" works only if uy is in W/[m2 nm]!')
            print(f'Your uy is {self.uy}.')

    
    
    def illuminance_to_irradiance(self, warning = True):
        """
        Returns the irradiance spectrum of the illuminance spectrum self.
        """
        if (self.uy == 'lm/[m2 nm]') or (self.uy == 'lx/nm'):
            if warning:
                print('Attention! Function "diff_spectrum.illuminance_to_irradiance" has to be used carefully.')
                print('A lot of information could be lost outside the x-limits (400-720 nm) of the photopic luminosity function.')
                print('To switch off this warning, use "warning = False"!')
            cf = spectrum.luminosity_fn()
            #Inverse photopic luminosity function:
            icf = cf.copy()
            icf.y = 1/cf.y 
            return self.product(icf, qy = 'Spectral irradiance', uy = 'W/[m2 nm]')
        else:
            print('Attention: Function "diff_spectrum.illuminance_to_irradiance" works only if uy is in lm/[m2 nm] or in lx/nm!')
            print(f'Your uy is {self.uy}.')
      
    
    def nm_to_eV(self, delta = 0.001):
    
        """
        Transforms a single differential spectrum (dS/dlambda) as a function of wavelength (in nm) into 
        a differential spectrum (dS/dE) as a function of photon energy (in eV). 
        The following calculatio is used:
        self.x = lambda in nm
        E(lambda)*q = h*c/(lambda*1e-9), lambda in nm, E in eV
        E(lambda) = h*c/(lambda*1e-9) * 1/q
        self.y = dS(lambda)/dlambda, lambda in nm, return dS(lambda(E)))/dE, E in eV
        dS(E)/d(E*q) = - dS(E(lambda))/d(lambda*1e-9) * d(lambda*1e-9)/d(E*q) #"-" because integration boundaries are reversed
        dS(E)/dE = dS(lambda)/dlambda * q/1e-9 * h*c/(E*q)**2 
                 = dS(lambda)/dlambda * h*c/(q*1e-9) * 1/E**2        
        """
        
        x = self.x.copy()
        y = self.y.copy()
        x_eV = h * c / (x[::-1] * 1e-9) / q
        y#_eV = y[::-1] * (q * 1e-9) / (h*c) * x[::-1]**2
        y_eV = y[::-1] * h*c/(q*1e-9) * 1/x_eV**2
        
        # quants = {"x": "Photon energy", "y": "Spectral photon flux"}
        # units = {"x": "eV", "y": "1/[s m2 eV]"}

        quants = dict(x = 'Photon energy', y = self.qy)
        units = dict(x = 'eV', y = self.uy.replace('nm', 'eV'))

        name = self.name
        
        # if self.__class__.__name__ == 'diff_spectrum':
        #     sp_new = diff_spectrum(x_eV, y_eV, quants = quants, units = units, name = name)
        # if self.__class__.__name__ == 'PEL_spectrum':
        #     sp_new = PEL_spectrum(x_eV, y_eV, quants = quants, units = units, name = name)

        sp_new = type(self)(x_eV, y_eV, quants = quants, units = units, name = name)
        sp_new.equidist(delta = delta)    
        return sp_new
    
    def eV_to_nm(self, delta = 1):
    
        """
        Transforms a single differential spectrum as a function of photon energy (in eV) into 
        a differential spectrum as a function of wavelength (in nm).  
        The following calculatio is used:
        self.x = E in eV
        lambda(qE)*1e-9 = h*c/(qE), lambda in nm , E in eV
        lambda(E) = h*c/(q*E) * 1e9
        self.y = dS(E)/dE, E in eV, return dS(E(lambda))/dlambda, lambda in nm
        dS(lambda)/d(lambda*1e-9) =  - dS(E)/d(E*q) * d(E*q)/d(lambda*1e-9) #"-" because integration boundaries are reversed
        dS(lambda)/dlambda = dS(E)/dE * 1e-9/q * h*c/(lambda*1e-9)**2 
                           = dS(E)/dE * h*c / (q*1e-9) / lambda**2
        """
        
        x = self.x.copy()
        y = self.y.copy()
        x_nm = h * c / (x[::-1] * q) * 1e9
        #y_nm = y[::-1] * h*c / (q * 1e-9) / x[::-1]**2
        y_nm = y[::-1] * h*c / (q * 1e-9) / x_nm**2
        
        quants = dict(x = 'Wavelength', y=self.qy)
        units = dict(x = 'nm', y = self.uy.replace('eV', 'nm'))
        name = self.name
        
        sp_new = type(self)(x_nm, y_nm, quants = quants, units = units, name = name)        
        sp_new.equidist(delta = delta)    

        return sp_new

    
    @staticmethod
    def load_ASTMG173(y_unit = 'Spectral photon flux', warning = True, spectrum = 'AM1.5GT'):

        """
        spectrum == 'AM1.5GT': Loads the AM1.5GT spectrum.
        spectrum == 'Etr': Loads the extraterrestrial spectrum.
        y_unit: Either 'Spectral photon flux' ('PF') or 'Spectral irradiance ('Spectral flux', 'SF')
        warning: If True it prints a warning that the Wavelengths are not evenly spaced and that rather the function AM15_nm or AM15_eV should be used.
        """
        # Directory for Astmg173 spectrum

        ASTMG173 = 'Astmg173.xls'
        
        dat = pd.read_excel(join(system_dir, ASTMG173), header = 1)
        dat_nm = np.array(dat, dtype = np.float64)[:,0] # wavelength data in nm
        if spectrum == 'AM1.5GT':
            dat_SF = np.array(dat, dtype = np.float64)[:,2] # spectral irradiance in W/(m2 nm)
        elif spectrum == 'Etr':
            dat_SF = np.array(dat, dtype = np.float64)[:,1] # spectral irradiance in W/(m2 nm)

        dat_PF = dat_SF * dat_nm * 1e-09 / (h * c) # photon flux (PF) in photons / (s m2 nm)
        
        x = dat_nm 
        
        if (y_unit == 'Spectral photon flux') or (y_unit == 'PF'):
        
            y = dat_PF
    
            quants = {"x": "Wavelength", "y": "Spectral photon flux"}
            units = {"x": "nm", "y": "1/[s m2 nm]"}
            
        elif (y_unit == 'Spectral irradiance') or (y_unit == 'Spectral flux') or (y_unit == 'SF'):
            
            y = dat_SF
    
            quants = {"x": "Wavelength", "y": "Spectral irradiance"}
            units = {"x": "nm", "y": "W/[m2 nm]"}
            
        else:
            print('Attention (load_ASTMG173): y_unit not known!')
            
        if warning:
            print('load_ASTMG173: Wavelengths are not evenly spaced, use rather the function AM15_nm or AM15_eV.')

        return diff_spectrum(x, y, quants, units, ASTMG173)
    
    @staticmethod
    def AM15_nm(left = 280, right = 4000, delta = 1, y_unit = 'Spectral photon flux'):
        """
        Returns the AM1.5 GT photon flux spectrum as a function of wavlelength in nm.
        """
        
        AM15 = diff_spectrum.load_ASTMG173(y_unit = y_unit, warning = False)
        AM15.equidist(left = left, right = right, delta = delta)
        AM15.name += ' (AM1.5GT)'
        #AM15_nm.plot(left = 300, right = 1000)    
        return AM15
        
    @staticmethod
    def AM15_eV(left = 0.310, right = 4.428, delta = 0.001, y_unit = 'Spectral photon flux'):
        """
        Returns the AM1.5 GT photon flux spectrum as a function of photon energy in eV.
        """
        
        AM15nm = diff_spectrum.AM15_nm(y_unit = y_unit)
        AM15eV = AM15nm.nm_to_eV()
        AM15eV.equidist(left = left, right = right, delta = delta)
        #AM15_eV.plot()
        return AM15eV
    
    @staticmethod
    def Etr_nm(left = 280, right = 4000, delta = 1, y_unit = 'Spectral photon flux'):
        """
        Returns the extraterrestrial photon flux spectrum as a function of wavlelength in nm.
        """
        
        Etr = diff_spectrum.load_ASTMG173(y_unit = y_unit, warning = False, spectrum = 'Etr')
        Etr.equidist(left = left, right = right, delta = delta)
        Etr.name += ' (Extraterrestrial)'
        #AM15_nm.plot(left = 300, right = 1000)    
        return Etr
        
    @staticmethod
    def Etr_eV(left = 0.310, right = 4.428, delta = 0.001, y_unit = 'Spectral photon flux'):
        """
        Returns the extraterrestrial photon flux spectrum as a function of photon energy in eV.
        """
        
        Etrnm = diff_spectrum.Etr_nm(y_unit = y_unit)
        EtreV = Etrnm.nm_to_eV()
        EtreV.equidist(left = left, right = right, delta = delta)
        #AM15_eV.plot()
        return EtreV
    
    @staticmethod
    def load_OSRAM930(y_unit = 'Spectral photon flux'):

        """
        Loads the OSRAM 930 spectrum.
        y_unit: Either 'Spectral photon flux' ('PF') or 'Spectral irradiance' ('Spectral flux', 'SF')
        """
        # Directory for OSRAM 930 spectrum
        OSRAM930 = 'OSRAM_930.txt'
        
        dat = pd.read_csv(join(system_dir, OSRAM930), delimiter = '\t', header = 2)
        dat_nm = np.array(dat, dtype = np.float64)[:,0] # wavelength data in nm
        dat_SF = np.array(dat, dtype = np.float64)[:,1] # spectral irradiance in W/(m2 nm)
        dat_PF = dat_SF * dat_nm * 1e-09 / (h * c) # photon flux (PF) in photons / (s m2 nm)
        
        x = dat_nm 
        
        if (y_unit == 'Spectral photon flux') or (y_unit == 'PF'):
        
            y = dat_PF
    
            quants = {"x": "Wavelength", "y": "Spectral photon flux"}
            units = {"x": "nm", "y": "1/[s m2 nm]"}
            
        elif (y_unit == 'Spectral irradiance') or (y_unit == 'Spectral flux') or (y_unit == 'SF'):
            
            y = dat_SF
    
            quants = {"x": "Wavelength", "y": "Spectral irradiance"}
            units = {"x": "nm", "y": "W/[m2 nm]"}       
            
        else:
            print('Attention: y_unit not known!')
        
        sp = diff_spectrum(x, y, quants, units, OSRAM930)
        sp.equidist(delta = 1)
    
        return sp


    
    @staticmethod
    def load_LED5000K(y_unit = 'Spectral photon flux'):
        
        """
        Loads the high CRI LED (YJ-VTC-2835MX-G2) 5000 K spectrum.
        y_unit: Either 'Spectral photon flux' ('PF') or 'Spectral irradiance ('Spectral flux', 'SF')
        left = 383, right = 779, delta = 1
        """
        # Directory for LED spectrum
        LED = 'high CRI LED (YJ-VTC-2835MX-G2) 5000 K.csv'
        
        dat = pd.read_csv(join(system_dir, LED), header = 0)
        dat_nm = np.array(dat, dtype = np.float64)[:,0] # wavelength data in nm
        dat_SF = np.array(dat, dtype = np.float64)[:,1] # spectral irradiance in W/[m2 nm]
        dat_PF = dat_SF * dat_nm * 1e-09 / (h * c) # photon flux (PF) in photons/[s m2 nm]
        
        x = dat_nm 
        
        if (y_unit == 'Spectral photon flux') or (y_unit == 'PF'):
        
            y = dat_PF
    
            quants = {"x": "Wavelength", "y": "Spectral photon flux"}
            units = {"x": "nm", "y": "1/[s m2 nm]"}
            
        elif (y_unit == 'Spectral irradiance') or (y_unit == 'Spectral flux') or (y_unit == 'SF'):
            
            y = dat_SF
    
            quants = {"x": "Wavelength", "y": "Spectral irradiance"}
            units = {"x": "nm", "y": "W/[m2 nm]"}       
            
        else:
            print('Attention: y_unit not known!')
        
        sp = diff_spectrum(x, y, quants, units, LED)
        sp.equidist(delta = 1)
    
        return sp
    
    
    def calc_integrated_photonflux(self, start = None, stop = None):
        
        """
        Calculates photon flux from self.x=start to self.x=stop. self.x values have to be equidistant. Standard is from min(self.x) to max(self.x).
        """

        if start == None:
            start_x = min(self.x)
        else:
            start_x = start
        
        if stop == None:
            stop_x = max(self.x)
        else:
            stop_x = stop
            
        index_start = findind(self.x, start_x)
        index_stop = findind(self.x, stop_x)
        r = range(index_start, index_stop+1)
        
        dx = (max(self.x) - min(self.x)) / (len(self.x) - 1)

        return np.trapz(self.y[r], dx = dx)
    
    def photonflux(self, start = None, stop = None):
        
        """
        This is the old version, please use calc_integrated_photonflux.
        Calculates photon flux from self.x=start to self.x=stop. self.x values have to be equidistant. Standard is from min(self.x) to max(self.x).
        """

        if start == None:
            start_x = min(self.x)
        else:
            start_x = start
        
        if stop == None:
            stop_x = max(self.x)
        else:
            stop_x = stop
            
        index_start = findind(self.x, start_x)
        index_stop = findind(self.x, stop_x)
        r = range(index_start, index_stop+1)
        
        dx = (max(self.x) - min(self.x)) / (len(self.x) - 1)

        return np.trapz(self.y[r], dx = dx) 
    
    
    def calc_illuminance(self):
        """
        Calculates the illuminance in lx.
        Only works if self.x is wavelength in nm.
        It works for irradiance and illuminance spectra.
        """
        # Directory for photopic luminosity function
        lum_FN = 'luminosity function CIE 1931.csv'
        cf = spectrum.load(system_dir, lum_FN, quants = dict(x = 'Wavelength', y = 'Conversion factor'), units = dict(x = 'nm', y = 'lx/(W/m2)'))
        if self.ux == 'nm':
            if self.uy == 'W/[m2 nm]':
                new = self.product(cf, qy = 'Spectral illuminance', uy = 'lm/[m2 nm]')
                return np.trapz(new.y, dx = new.x[1]-new.x[0]) # lx = lm/m2
            elif self.uy == '1/[s m2 nm]':
                self_sf = self.photonflux_to_irradiance()
                new = self_sf.product(cf, qy = 'Spectral illuminance', uy = 'lm/[m2 nm]')
                return np.trapz(new.y, dx = new.x[1]-new.x[0]) # lx = lm/m2
            elif self.uy == ('lm/[m2 nm]') or (self.uy == 'lx/nm'):
                return np.trapz(self.y, dx = self.x[1]-self.x[0]) # lx = lm/m2

            else:
                print('Attention (calc_lux): This function works only for spectral irradiances in W/m2 or photon flux in 1/[s m2 nm],')
                print('or for spectral illuminance in lm/[m2 nm] or lx/nm!')
        else:
            print('Attention(calc_lux): This function works only if ux = nm!')            
 
    
    def normalize_to_lux(self, lx):
        """
        Normalizes self to lx.
        Works only if self.ux = 'nm' and self.uy = 'W/[m2 nm]' or '1/[s m2 nm]'
        """
        ill = self.calc_illuminance()
        self.y = self.y/ill*lx
        
        
    def calc_irradiance(self):
        """
        Calculates the irradiance in W/m2 of an irradiance spectrum.
        uy has to be either in W/[m2 nm] or W/[m2 eV].

        Returns
        -------
        float
            Irradiance in mW/cm2.

        """
        if ((self.uy == 'W/[m2 nm]') and (self.ux =='nm')) or ((self.uy == 'W/[m2 eV]') and (self.ux =='eV')):
            return np.trapz(self.y, dx = self.x[1]-self.x[0])*1e-1 # mW/cm2
        else:

            print('Attention (calc_irradiance): uy has to be either in W/[m2 nm] or W/[m2 eV] and ux in nm or eV, respectively!')
            print(f'Your ux is {self.ux} ({self.qx}) and your uy is {self.uy} ({self.qy})')
        
    
    #Black body spectral photon flux in 1/(s m2 J), energy in Joule
    @staticmethod
    def PhiBB(E, T):
        return 1/(4 * math.pi**2 * (h/(2 * math.pi))**3 * c**2) * E**2 / (np.exp(E / (k * T)) - 1)

    @staticmethod
    def BBspectrum(sp, T = T_RT, fit_eV = 1.5):
        x_eV = sp.x
        ind = findind(x_eV, fit_eV)
        BB = diff_spectrum.PhiBB(x_eV * q, T) * q
        BB = BB / BB[ind] * sp.y_of(fit_eV)
        quants = {"x": "Photon energy", "y": "BB fit"}
        units = {"x": "eV", "y": ""}
        name = 'BB spectrum'
        return diff_spectrum(x_eV, BB, quants = quants, units = units, name = name)    
    
    def integrated_current(self, EQE=None):
        # Calculates an integrated current plot as a function of wavelength
        # Works only with photon flux spectrum
        # EQE in % and as a function fo wavelength
        # if EQE==1 then EQE is assumed 100% throughout the spectrum
    
        sp = self.copy()
        if EQE is None:
            e = EQE_spectrum.EQE100(Eg=0.1)
        else:
            e = EQE.copy()
        left = max(sp.x[0], e.x[0])
        right = min(sp.x[-1], e.x[-1]) 
        sp.equidist(left=left, right=right, delta=1)
        e.equidist(left=left, right=right, delta=1)
        sp_times_e = sp*e/100
        int_curr_y = q*np.array([np.trapz(sp_times_e.y[0:idx], dx=1)*1e3/1e4 for idx in range(len(sp_times_e.x))]) # mA/cm2]
        return diff_spectrum(x=sp_times_e.x, y=int_curr_y, quants=dict(x='Wavelength', y='Integrated current density'), units=dict(x='nm', y='mA/cm2'))


    def integrated_irradiance(self):
        # Calculates an integrated irradiance plot as a function of wavelength
        # Works only with spectral irradiance
    
        int_irr_y = np.array([np.trapz(self.y[0:idx], dx=self.x[1]-self.x[0])*1e3/1e4 for idx in range(len(self.x))]) # mW/cm2]
        return diff_spectrum(x=self.x, y=int_irr_y, quants=dict(x='Wavelength', y='Integrated irradiance'), units=dict(x='nm', y='mW/cm2'))
    
class PEL_spectrum(diff_spectrum):   
    """
    PL or EL spectrum.
    """
    
    def __init__(self, x, y, quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, name = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
        
        
    def absorptance(self, bb, eV, a_eV):
        """
        Calculates the absorptance spectrum from a PL/EL spectrum (self) by dividing self with the blackbody spectrum (BB).
        Then it is calibrated so that at the energy eV the absorptance is a_eV .
        BB is of type spectrum. self and BB have to have the same x value in eV.
        """
        if not(all(self.x==bb.x)):
            print("Warning: PL/EL spectrum and BB spectrum don't have the same x-values.")
            ab = self
        else:
            ab = abs_spectrum(self.x, self.y / bb.y, quants = {"x": self.qx, "y": "Absorptance"}, units = {"x": self.ux, "y": ""}, name = 'Absorptance')
            ab.y = ab.y / ab.y_of(eV) * a_eV
        return ab
    
    
    def BBT_fit(self, Efit_start, Efit_stop, Tguess = 300):
        """
        Calculates the high energy tail fit with BB temperature, returns the fitcurve as an instance of the class "spectrum".
        self: PL/EL spectrum, instance of class "spectrum"
        Efit_start: Photon energy (in eV) to begin with the fit.
        Efit_stop: Photon energy to end with the fit.
        """
        f = lambda x, T, A : A * x**2 * np.e**(- (x*q) / (k * T))
        r = range(self.x_idx_of(Efit_start), self.x_idx_of(Efit_stop)+1)
        Efit_middle = (Efit_stop + Efit_start) / 2
        A0 = self.y_of(Efit_middle) / f(Efit_middle, Tguess, 1)
        p0 = (Tguess, A0) 
        try:
            popt, pcov = curve_fit(f, self.x[r], self.y[r], p0)
        except:
            popt = p0
            print("Fit didn't converge, Tguess taken!")
        
        T = popt[0] #K
        name = f'Fit: $T$ = {T:.1e} K'
        T_fit = spectrum(self.x, f(self.x, *popt), quants = {"x": self.qx, "y": "Blackbody fit"}, units = {"x": self.ux, "y": self.uy}, name = name)
        T_fit.T = T
        return T_fit
        

class spectra(mxy_data):
    
    def __init__(self, sa):
        super().__init__(sa)
        
    @classmethod
    def generate_empty(cls, empty_spectra_list=[]):
        return cls(empty_spectra_list)   
        
                
    def replace(self, idx, sp_new):
        self.sa[idx] = sp_new
        
    def remain(self, idx_list):
        """
        Return all spectra with indices in list idx_list.
        """
        new_sa = self.copy()
        sa = []
        for i, idx in enumerate(idx_list):
            new = new_sa.sa[idx].copy()
            sa.append(new)
            
        new_sa.sa = sa
        new_sa.names_to_label(split_ch = '.csv')
        return new_sa

    def names_to_label(self, split_ch = None):
        lab = []
        for i, sp in enumerate(self.sa):
            if split_ch == None:
                lab.append(sp.name)
            else:
                lab.append(sp.name.split(split_ch)[0])
        self.label(lab)
        self.label_defined = True

    def nm_to_eV(self):
        """
        Transforms a list of non-differential spectra of type spectrum from wavelength to photon energy. 
        """
        spectra_eV = []
        for i, spectrum_nm in enumerate(self.sa):
            spectra_eV.append(spectrum_nm.nm_to_eV())
            
        spa_new = self.copy()
        spa_new.sa = spectra_eV
    
        return spa_new

    def plot_old(self, title = '', yscale = 'linear', left = None, right = None, bottom = None, top = None, plotstyle = 'auto', showindex = False, in_name = [], hline = None, vline = None, figsize=(8,5)):

        """
        Plots multiple spectra of type spectrum. The axis title are taken from the first spectrum.
        showindex: If True then the index of the sa list will be shown before the regular label. 
        This is helpful when certain curves have to be selected e.g. for PLQY. 
        in_name: List with strings that have to be in the name to be plotted. If [] then everything is plotted
        """
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=figsize)
        
        plt.yscale(yscale)
        
        if left != None:
            plt.xlim(left = left)
        #     left_idx = findind(self.sa[0].x, left)
        # else:
        #     left_idx = 0
        if right != None:
            plt.xlim(right = right)
        #     right_idx = findind(self.x, right)
        # else:
        #     right_idx = len(self.x)
        # r = range(left_idx, right_idx+1)
        if bottom != None:
            plt.ylim(bottom = bottom)
        if top != None:
            plt.ylim(top = top)
            
        def in_name_in_spec(in_name, spec):
            
            result = False

            if in_name == []:
                result = True
            else:
                for i, name in enumerate(in_name):
                    if name in spec.name:
                        result = True
                        break
            
            return result
            
        for i, spec in enumerate(self.sa):

            if in_name_in_spec(in_name, spec):
            
                if self.label_defined:
                    if plotstyle == 'auto':
                        if showindex == True:
                            plt.plot(spec.x, spec.y, label = f'{i}: {self.lab[i]}')
                        else:
                            plt.plot(spec.x, spec.y, label = self.lab[i])
                    else:
                        if showindex == True:
                            plt.plot(spec.x, spec.y, **spec.plotstyle, label = f'{i}: {self.lab[i]}')
                        else:
                            plt.plot(spec.x, spec.y, **spec.plotstyle, label = self.lab[i])
                    plt.legend()
                else:
                    if plotstyle == 'auto':
                        plt.plot(spec.x, spec.y)
                    else:
                        plt.plot(spec.x, spec.y, **spec.plotstyle)
    
        sp = self.sa[0]
        plt.xlabel(f'{sp.qx} ({sp.ux})')
        
        if sp.uy == '':
            plt.ylabel(f'{sp.qy}')
        else:
            plt.ylabel(f'{sp.qy} ({sp.uy})')

        if title != '':
            plt.title(title)

        if hline != None:
            plt.axhline(y = hline, color='black', linestyle='-')
        if vline != None:
            plt.axvline(x = vline, color='r', linestyle='-')
            
        plt.show()
            
    def save(self, save_dir, FN):
        
        if not(self.all_spectra_have_same_x_range()):
            
            print('Warning: Data have not the same x range, data is not saved!')

        else:
            
            alldata = np.zeros((self.n_x, self.n_y + 1), dtype = float) # Initialize array
            alldata[:,0] = self.sa[0].x
            
            x_col_name = self.sa[0].qx
            if self.sa[0].ux != "":
                x_col_name = x_col_name + f' ({self.sa[0].ux})'
            
            columns = [x_col_name] 
            
            for i, dat in enumerate(self.sa):
                alldata[:,i + 1] = dat.y
                y_col_name = self.sa[i].qy
                if self.sa[i].uy != "":
                    y_col_name = y_col_name + f' ({self.sa[i].uy})'
                columns.append(y_col_name) 
                
            df = pd.DataFrame(data=alldata[0:,0:], columns = columns)
                            
            TFN = join(save_dir, FN)
            if save_ok(TFN):
                df.to_csv(join(save_dir, FN), header = True, index = False)
                
                
    def save_individual_old(self, save_dir, FNs = None):
        
        quitted = False
        for i, sp in enumerate(self.sa):
                    
            x_col_name = sp.qx
            if sp.ux != "":
                x_col_name = x_col_name + f' ({sp.ux})'
            y_col_name = sp.qy
            if sp.uy != "":
                y_col_name = y_col_name + f' ({sp.uy})'
                                    
            if FNs == None:
                FN = sp.name
            else:
                FN = FNs[i]
                
            TFN = join(save_dir, FN)
            ok_to_save, quitted = save_ok(TFN, quitted)
            if ok_to_save and not(quitted):
                df = pd.DataFrame({x_col_name : sp.x, y_col_name : sp.y})
                df.to_csv(join(save_dir, FN), header = True, index = False)
                
    @staticmethod
    def load_multiple(directory, FN, delimiter = ',', header = 'infer', quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False):

        """
        Loads multiple spectra, all spectra in one file FN.
        """
        
        dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)
        
        npdat = np.array(dat, dtype = np.float64)
        x = npdat[:,0]

        sa = []
        for i in range(np.shape(npdat)[1]-1): #"-1" because x-column disregarded

            y = np.array(dat, dtype = np.float64)[:,i+1]
            sp = spectrum(x, y, quants, units, FN)
       
            if take_quants_and_units_from_file:

                col0 = list(dat)[0]
                sp.qx = col0.split(' (')[0]
                if ' (' in col0:
                    sp.ux = col0.split(' (')[1].split(')')[0]

                col1 = list(dat)[i+1]
                sp.qy = col1.split(' (')[0]
                if ' (' in col1:
                    sp.uy = col1.split(' (')[1].split(')')[0]

            sa.append(sp)
            
            new_spectra = spectra(sa)

        return new_spectra


    def load_old(self, directory, FN, delimiter = ',', header = 'infer', quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False):

        """
        Loads multiple spectra, all spectra in one file FN.
        """
        
        dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)
        
        npdat = np.array(dat, dtype = np.float64)
        x = npdat[:,0]

        sa = []
        for i in range(np.shape(npdat)[1]-1): #"-1" because x-column disregarded

            y = np.array(dat, dtype = np.float64)[:,i+1]
            sp = spectrum(x, y, quants, units, FN)
       
            if take_quants_and_units_from_file:

                col0 = list(dat)[0]
                sp.qx = col0.split(' (')[0]
                if ' (' in col0:
                    sp.ux = col0.split(' (')[1].split(')')[0]

                col1 = list(dat)[i+1]
                sp.qy = col1.split(' (')[0]
                if ' (' in col1:
                    sp.uy = col1.split(' (')[1].split(')')[0]

            sa.append(sp)
            self.sa = sa

        return self 
    
    @classmethod
    def load_individual(cls, directory, FNs = [], delimiter = ',', header = 'infer', quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False, check_data = False):

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

            if cls.__name__ == 'spectra':
                sp = spectrum(x, y, quants, units, FN, check_data = False)
            elif cls.__name__ == 'abs_spectra':
                sp = abs_spectrum(x, y, quants, units, FN, check_data = False)
            elif cls.__name__ == 'diff_spectra':
                sp = diff_spectrum(x, y, quants, units, FN, check_data = False)
            elif cls.__name__ == 'PEL_spectra':
                sp = PEL_spectrum(x, y, quants, units, FN, check_data = False)
       
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
       
                
    def save_names(self, save_dir, FN):
        TFN = join(save_dir, FN)
        if save_ok(TFN):
            names  = []
            for i, sp in enumerate(self.sa):
                names.append(sp.name)
            with open(join(save_dir, FN), 'w') as f:
                f.write("\n".join(names))    
                
    def load_names(self, dir, FN):
        TFN = join(dir, FN)
        with open(TFN, 'r') as f:
            FNstring = f.read()
            FN_list = FNstring.split("\n")
        for i, sp in enumerate(self.sa):
            sp.name = FN_list[i]
            
    
    def all_spectra_have_same_x_range(self):
        """
        Returns True if all spectra have the same x-range, otherwise False.
        """
        same_range = True
        
        for i, spec in enumerate(self.sa):
            if not(np.array_equal(spec.x, self.sa[0].x)):
                same_range = False
        return same_range
    
    
    def equidist(self, left, right, delta):
        """
        Change x values so that they are equidistant with a delta of delta for all spectra in the same way. 
        x ranges from start to stop. 
        If start (stop) = None then the new start (stop) value is the old one. Then it could happen that the different spectra have different start and stop values.
        """
        if not(spectra.all_spectra_have_same_x_range(self)):
            print('Warning: Not all spectra have the same x-range!')
            
        for i, spec in enumerate(self.sa):
            spec.equidist(left = left, right = right, delta = delta)
            
            self.n_y = len(self.sa)
            self.n_x = len(self.sa[0].x)
           
    @classmethod
    def load_Andor(cls, directory, meta_data = None, sel_list = None):
    
        """
        Loads all Andor spectra in the directory.
        """
    
        FNs = listdir(directory)
        
        if sel_list != None:
            FNs = [FNs[idx] for idx in sel_list]
        
        sa = []
        for i, FN in enumerate(FNs):
            if cls.__name__ == 'spectra':
                sa.append(spectrum.load_Andor(directory, FN, meta_data = meta_data))
            if cls.__name__ == 'diff_spectra':
                sa.append(diff_spectrum.load_Andor(directory, FN, meta_data = meta_data))
            if cls.__name__ == 'PEL_spectra':
                sa.append(PEL_spectrum.load_Andor(directory, FN, meta_data = meta_data))
                
        return cls(sa)
    

class abs_spectra(spectra):
    
    def __init__(self, sa):
        super().__init__(sa)

            
class diff_spectra(spectra):
    
    def __init__(self, sa):
        super().__init__(sa)
        
    
class PEL_spectra(diff_spectra):
    
    def __init__(self, sa):
        super().__init__(sa)
        
    
    def calc_calfn(mspec, calspec):
        
        calib = []
        for i, sp in enumerate(mspec.sa):
            # make sure that there are no negatve values and no 0 values
            sp.y = np.array([abs(z) if z !=0 else np.average(sp.y) for z in sp.y], dtype = float)
            calib.append(PEL_spectrum(sp.x, int_arr(calspec.x, calspec.y, sp.x) / sp.y, 
                                  quants = {"x": "Wavelength", "y": "Calibration factor"}, 
                                  units = {"x": "nm", "y": "1/[cps s m2 nm]"}, name = sp.name))
        return PEL_spectra(calib)

            
    def calibrate_single(self, calib):
        
        """
        Calibrates all PL spectra with the calibration function calib.
        """
    
        calibrated = []
        for i, rawPLspectrum in enumerate(self.sa):
            calibrated_spectrum = PEL_spectrum(rawPLspectrum.x, rawPLspectrum.y * int_arr(calib.x, calib.y, rawPLspectrum.x), 
                                           quants = {"x": "Wavelength", "y": "Spectral photon flux"}, units = {"x": "nm", "y": "1/[s m2 nm]"}, name = rawPLspectrum.name)
            calibrated.append(calibrated_spectrum)
            
        return PEL_spectra(calibrated)

    def calibrate(self, calib, check = False):
        
        """
        Calibrates all PL spectra with the respective calibration function calib.
        check: True if the calibration function should be printed for each spectrum. 
        """

        def right_cal_fn(rawPLsp, calsp):
            
            result = False
            PLn = rawPLsp.name
            caln = calsp.name
            # if ip or fs in both
            if caln.split('--')[1] in PLn.split('--')[1]: 
                # same grating
                if caln.split('--Andor')[1].split('_')[3] == PLn.split('--Andor')[1].split('_')[3]: 
                    # same center wavelength
                    if caln.split('--Andor')[1].split('_')[4] == PLn.split('--Andor')[1].split('_')[4]: 
                        # same filter
                        if caln.split('--Andor')[1].split('_')[-1] == PLn.split('--Andor')[1].split('_')[-1]:                     
                            result = True
            return result

        calibrated = []
        calib_ok = False
        for i, rawPLsp in enumerate(self.sa):

            for j, calsp in enumerate(calib.sa):
                if right_cal_fn(rawPLsp, calsp):
                    calibrated_spectrum = PEL_spectrum(rawPLsp.x, rawPLsp.y * int_arr(calsp.x, calsp.y, rawPLsp.x), 
                                                   quants = {"x": "Wavelength", "y": "Spectral photon flux"}, units = {"x": "nm", "y": "1/[s m2 nm]"}, name = rawPLsp.name)
                    calibrated.append(calibrated_spectrum)
                    calib_ok = True
                    if check:
                        print(f'{rawPLsp.name}: {calsp.name}')
            if not(calib_ok):
                print(f'Warning: No adequate calibration function found for {rawPLsp.name}.')
            
        return PEL_spectra(calibrated)
        
    def choose_for_PLQY(self, name, laser_marker, PL_marker):
        """
        All spectra with name in its name will be put into a new instance of spectra.
        The following indexing:
        La: laser, no sample
        Lb: laser, outofbeam
        Lc: laser, inbeam
        Pa: PL, no sample
        Pb: PL, outofbeam
        Pc: PL, inbeam
        P_fs: free space PL
        """
        sa = []
        sa_idx = 0
        expl = {}
        
        for i, sp in enumerate(self.sa):
                                        
            if 'ip_laser' in sp.name:
                
                if laser_marker in sp.name:

                    sa.append(sp)
                    expl.update(La = sa_idx)
                    sa_idx += 1

                if PL_marker in sp.name:

                    sa.append(sp)
                    expl.update(Pa = sa_idx)
                    sa_idx += 1
                    
            
            if name == sp.name.split('--')[0]:
                
                if laser_marker in sp.name:

                    if 'outofbeam' in sp.name:
                        sa.append(sp)
                        expl.update(Lb = sa_idx)
                        sa_idx += 1
                    if 'inbeam' in sp.name:
                        sa.append(sp)
                        expl.update(Lc = sa_idx)
                        sa_idx += 1
                        
                if PL_marker in sp.name:
                    
                    if 'outofbeam' in sp.name:
                        sa.append(sp)
                        expl.update(Pb = sa_idx)
                        sa_idx += 1
                    if 'inbeam' in sp.name:
                        sa.append(sp)
                        expl.update(Pc = sa_idx)
                        sa_idx += 1
                        
                if '--fs--' in sp.name:
                    sa.append(sp)
                    expl.update(P_fs = sa_idx)
                    sa_idx += 1
        
        PLQY_sa = PEL_spectra(sa)
        PLQY_sa.expl =expl
        
        lab = [None] * len(expl)
        #embed()
        for i, (l, v) in enumerate(expl.items()):
            lab[v] = l
            
        PLQY_sa.label(lab)
        return PLQY_sa
    
    def calc_PLQY_param(self, laser_marker, left_laser, right_laser, PL_marker, left_PL, right_PL, eval_Pa = False, eval_Pb = True, show_errmsg = True):
        
        if 'La' in self.expl:
            La = self.sa[self.expl['La']].photonflux(start = left_laser, stop = right_laser)
        else:
            La = 0
            if show_errmsg:
                print('Attention: No La signal!')

        if 'Lb' in self.expl:
            Lb = self.sa[self.expl['Lb']].photonflux(start = left_laser, stop = right_laser)
        else:
            Lb = 0
            if show_errmsg:
                print('Attention: No Lb signal!')

        if 'Lc' in self.expl:
            Lc = self.sa[self.expl['Lc']].photonflux(start = left_laser, stop = right_laser)
        else:
            Lc = 0
            if show_errmsg:
                print('Attention: No Lc signal!')

        if 'Pa' in self.expl and eval_Pa:
            Pa = self.sa[self.expl['Pa']].photonflux(start = left_PL, stop = right_PL)
        else:
            Pa = 0
            if show_errmsg:
                print('Attention: No Pa signal evaluated!')

        if 'Pb' in self.expl and eval_Pb:
            Pb = self.sa[self.expl['Pb']].photonflux(start = left_PL, stop = right_PL)
        else:
            Pb = 0
            if show_errmsg:
                print('Attention: No Pb signal!')

        if 'Pc' in self.expl:
            Pc = self.sa[self.expl['Pc']].photonflux(start = left_PL, stop = right_PL)
    
        else:
            Pc = 0
            if show_errmsg:
                print('Attention: No Pc signal!')
         
        Pb = Pb - Pa
        Pc = Pc - Pa
        A = 1 - Lc/Lb
        PLQY = (Pc - (1 - A) * Pb) / (La * A)

        return (PLQY, A, La, Lb, Lc, Pa, Pb, Pc)
    
    def guess_factor(self, left, right):
        """
        Returns the inbeam-free space adjustment factor.
        self.sa[0] has to be the ip spectrum.
        self.sa[1] has to be the fs spectrum.
        Both spectra have to have the same x array.
        """
        
        scpy = self.copy()
        delta = self.sa[0].x[1] - self.sa[0].x[0] 
        scpy.equidist(left = left, right = right, delta = delta)

        #r = range(findind(self.sa[0].x, left), findind(self.sa[0].x, right)+1)        

        def f(fac): 
            diff = scpy.sa[0].y - fac * scpy.sa[1].y
            return math.sqrt(1/len(diff) * np.dot(diff, diff))
            
        result = least_squares(fun = f, x0 = [1])
        
        return result.x[0]
    
    def guess_factor_old(self, left, right):
        """
        Returns the inbeam-free space adjustment factor.
        self.sa[0] has to be the ip spectrum.
        self.sa[1] has to be the fs spectrum.
        Both spectra have to have the same x array.
        """
        
        r = range(findind(self.sa[0].x, left), findind(self.sa[0].x, right)+1)        

        def f(fac): 
            diff = self.sa[0].y[r] - fac * self.sa[1].y[r]
            return math.sqrt(1/len(diff) * np.dot(diff, diff))
            
        result = least_squares(fun = f, x0 = [1])
        
        return result.x[0]

        
    def _inb_oob_adjust(self, factor = 1):
        """
        Probably a very old version, the current one is in module PLQY.py
        Multiplies the freespace spectrum with constant factor.
        The aim is to have the low energy tail of fs and ip measurement equal.
        The first spectrum of self.sa should be the ip one and the secon one the fs one.
        """
        self.sa[1].y = self.sa[1].y * factor  
        
    def Udata_plot(self, overlap, yscale = 'log', left = None, right = None, save = False, save_dir = '', save_name = None, return_fig = False, show_plot = True):
        ab = self.sa[0]
        uf = self.sa[1]
        sp = self.sa[2].copy()
        bb = self.sa[3].copy()
        rescaling = 1 / sp.y_of(overlap) * ab.y_of(overlap)
        sp.y = sp.y * rescaling
        bb.y = bb.y * rescaling

        ab.plotstyle = dict(linestyle = '-', color = 'black', linewidth = 5)
        uf.plotstyle = dict(linestyle = '--', color = 'lawngreen', linewidth = 5)
        sp.plotstyle = dict(linestyle = '-', color = 'red', linewidth = 3)
        bb.plotstyle = dict(linestyle = '-', color = 'blue', linewidth = 3)
        
        sa = spectra([ab, uf, sp, bb])
        sa.label(['Absorptance', 'Exponential fit', 'Luminescence', 'Blackbody radiation'])
        
        graph = sa.plot(yscale = yscale, left = left, right = right, bottom = ab.y_of(left), top = ab.y_of(overlap) * 4, plotstyle = 'not auto', return_fig = return_fig, show_plot = show_plot)
        
        if save:
            if save_name is None:
                FN = 'Urbach_data.csv'
            else:
                FN = save_name
            sa.save(save_dir, FN) 
            
        if return_fig:
            return(graph)
            
            

            
            

def above_bg_photon_flux(bg, illumspec_eV = None):
    """
    Returns the above bandgap photon flux for an illumination spectrum illumspec_eV. If no illumination spectrum is given, 1 sun condition is assumed. 

    Parameters
    ----------
    bg : FLOAT
        Bandgap in eV.

    Returns
    -------
    Above bandgap photon flux in 1/[s m2].

    """
        
    if illumspec_eV == None:
        illumspec_eV = diff_spectrum.AM15_eV()
        
    return illumspec_eV.photonflux(start = bg, stop = illumspec_eV.x[-1])


def calc_laser_power(laser_nm, bg_eV = None, pf = None, A = None, Nsun = None, only90deg = True, details = True):
        """
        Calculates the laser power of the laser on the optical table as a function of bandgap or photonflux.
        laser_nm: 405, 420 or 660
        bg_eV: bandgap in eV
        pf: if no bandgap is given then the photonflux in 1/[s m2] has to be provided
        A: beam size in m2
        Nsun: Number of suns
        example: calc_laser_power(660, bg_eV = f1240/800, A = 1e-6, details = True)
        """
        if (A is None) and (not(laser_nm in [405, 420, 660])):
            print('Attention (calc_laser_power(): laser_nm has to be either 405, 420 or 660.')
        if Nsun == None:
            Nsun = 1
        if bg_eV != None:
            pf = above_bg_photon_flux(bg_eV)
        if A == None:            
            #FWHM laser beam diameters
            if laser_nm == 405:
                A = pi/4 * 533.5e-6 * 715.0e-6 #m2 for 403 nm laser
            elif laser_nm == 420:
                A = pi/4 * 418e-6 * 495e-6 #m2 for 422 nm laser
            elif laser_nm == 660:
                A = pi/4 * 693.0e-6 * 891.0e-6  #m2 for 657 nm laser
            d = math.sqrt(4*A/pi)*1e6
            if details:
                print(f'The size of the eliptical laser spot equals a circular diameter of {d:.0f} um')
            
        #Photon energy in J
        PE = f1240/laser_nm * q

        #Intesity within FWHM is 76% of total intensity
        factor = 0.76

        #Average power within FWHM
        #P = factor * LP * att #W
        #print(P*1e4)
        #Average photon flux within FWHM
        #PF = P / (PE * A)

        # Laser power

        LP = pf * PE * A / factor

        if details:
            if bg_eV != None:
                print(f'For {Nsun} sun conditions with a bandgap of {bg_eV:.3f} eV (= {f1240/bg_eV:.0f} nm) the laser power at {laser_nm} nm has to be {LP * 1000:.2e} mW (90° sample geometry).')
                if not(only90deg):
                    print(f'... at 45°: {LP * math.sqrt(2) * 1000:.2e} mW')
                print(f'The above bandgap photon flux is: {pf:.2e} 1/(s m²)')
            else:
                print(f'For {Nsun} sun conditions with a photonflux of {pf:.2e} 1/(s m2) the laser power at {laser_nm} nm has to be {LP * 1000:.2e} mW (90° sample geometry).')
                if not(only90deg):
                    print(f'... at 45°: {LP * math.sqrt(2) * 1000:.2e} mW')
        return LP
