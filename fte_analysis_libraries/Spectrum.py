# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:05:55 2020

@author: dreickem
"""

from scipy.optimize import curve_fit, least_squares
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import join
import math
import matplotlib.pyplot as plt
from importlib.resources import files as _resource_files
system_dir = str(_resource_files('fte_analysis_libraries').joinpath('System_data'))


from .General import findind, int_arr, linfit, save_ok, q, k, T_RT, h, c, f1240, pi

from .XYdata import XYData, MXYData
from typing import Any

class Spectrum(XYData):
    
    def __init__(self, x: np.ndarray, y: np.ndarray, quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, name: str = '', plotstyle: str = dict(linestyle = '-', color = 'black', linewidth = 3), check_data: bool = True) -> None:  # type: ignore
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
                  
    
    @classmethod  
    def load_andor(cls, directory: str, filepath: str = '', meta_data: np.ndarray | None = None) -> Any:
        """
        Loads a sinlge Andor Spectrum. If a filename is given it will be used, if not the first file in the directory will be used.
        If meta_data is provided, integration time (int_s) and accumulations (acc) will be taken from metadata.
        if not, it will take the integration time and number of accumulations from the filename.
        """
        
        if filepath == '':
            filepath = listdir(directory)[0]        

        dat = pd.read_csv(join(directory, filepath))
        x = np.array(dat, dtype = np.float64)[:,0]

        if type(meta_data) == dict:
            y = np.array(dat, dtype = np.float64)[:,1] / meta_data['int_s'] / meta_data['acc']
        else:
            def get_int_time(filepath: str) -> Any:
                it = filepath.split('--')[3].split('_')[1].split('s')[0]
                return float(it)

            def get_accum(filepath: str) -> Any:
                acc = filepath.split('--')[3].split('_')[2].split('acc')[0]
                #acc_raw = filepath.split('--')[3].split('_')[2].split('av')[0]
                return int(acc)
    
            y = np.array(dat, dtype = np.float64)[:,1] / get_int_time(filepath) / get_accum(filepath)
        
        return cls(x, y, quants = dict(x = 'Wavelength', y = 'Intensity'), units = dict(x = 'nm', y = 'cps'), name = filepath)
        
            
    def nm_to_ev(self) -> Any:
    
        """
        Transforms a single non-differential Spectrum of type Spectrum from wavelength to photon energy. 
        """
        
        x = self.x.copy()
        y = self.y.copy()
        x_eV = h * c / (x[::-1] * 1e-9) / q
        y_eV = y[::-1]
                
        name = self.name
        quants = {"x": "Photon energy", "y": self.qy}
        units = {"x": "eV", "y": self.uy}
        
        if self.__class__.__name__ == 'Spectrum':
            sp_new = Spectrum(x_eV, y_eV, quants = quants, units = units, name = name)
        elif self.__class__.__name__ == 'EQESpectrum':
            sp_new = EQESpectrum(x_eV, y_eV, quants = quants, units = units, name = name)
        elif self.__class__.__name__ == 'AbsSpectrum':
            sp_new = AbsSpectrum(x_eV, y_eV, quants = quants, units = units, name = name)
            
        return sp_new

    
    @staticmethod
    def luminosity_fn(y_unit: str = 'Spectral photon flux') -> Any:
        lum_FN = 'luminosity function CIE 1931.csv'
        return Spectrum.load(system_dir, lum_FN, quants = dict(x = 'Wavelength', y = 'Conversion factor'), units = dict(x = 'nm', y = 'lx/(W/m2)'))



class EQESpectrum(Spectrum):
    
    def __init__(self, x: np.ndarray, y: np.ndarray, quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, name: str = '', plotstyle: str = dict(linestyle = '-', color = 'black', linewidth = 3), check_data: bool = True) -> None:  # type: ignore
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
        
    @staticmethod
    def eqe100(Eg: float, start: float=300, stop: float=4001, step: Any=0.5, name: str = '') -> Any:
        """
        Returns an EQE Spectrum in % as a function of wavelength in nm from start to stop with a step size of 1 nm.
        Eg is the bandgap in eV.
        """
        Eg_nm = f1240/Eg
        x_arr = np.arange(start=start, stop=stop, step=step)  # type: ignore
        y_arr = np.array([0 if x>Eg_nm else 100 for x in x_arr])
        if name == '':
            name = f'100 % EQE until cut off wavelength {Eg_nm: .0f} nm'
        return EQESpectrum(x = x_arr, y = y_arr, quants = dict(x='Wavelength', y='EQE'), units = dict(x='nm', y='%'), name = name)

    @staticmethod    
    def mmf_eg(Eg: float, ref_EQE: Any, sim_PF: np.ndarray, ref_PF: Any = 'AM15GT', left: float = 300, right: float | None = None, delta: float = 0.5) -> Any:
        """
        Calculates the spectral mismatch factor as a function of Eg in eV.
        It is assumed a EQE Spectrum which is 100% above Eg and 0 below.
        """
        Eg_nm = f1240/Eg
        x_arr = np.arange(start = left, stop = int(Eg_nm), step = delta)  # type: ignore
        y_arr = np.ones(len(x_arr))
        sp = EQESpectrum(x = x_arr, y = y_arr, quants = dict(x='Wavelength', y='EQE'), units = dict(x='nm', y=''))
        #return sp.mmf_test(ref_EQE, sim_PF, ref_PF, left = 300, right = Eg_nm, delta = 1)
        return sp.mmf(ref_EQE, sim_PF, ref_PF = ref_PF, left = left, right = right, delta = delta)

    def mmf(self, ref_EQE: Any, sim_PF: np.ndarray, ref_PF: Any = 'AM15GT', left: float | None = None, right: float | None = None, delta: float | None = None) -> Any:
        """
        Calculates the spectral mismatsch factor of the EQE Spectrum self.
        How to apply it: The photocurrent of the Si reference diode has to be the photocurrent of 
        the Si ref. diode under AM1.5GT multiplied with the mismatch factor
        """
        num = ref_EQE.calc_jsc(left = left, right = right, delta = delta, sp = ref_PF) * self.calc_jsc(left = left, right = right, delta = delta, sp = sim_PF)
        denom = ref_EQE.calc_jsc(left = left, right = right, delta = delta, sp = sim_PF) * self.calc_jsc(left = left, right = right, delta = delta, sp = ref_PF)
        return num / denom
    
    def mmf_test(self, ref_EQE: Any, sim_PF: np.ndarray, ref_PF: Any = 'AM15GT', left: float | None = None, right: float | None = None, delta: float | None = None) -> Any:
        """
        Calculates the spectral mismatsch factor of the EQE Spectrum self.
        How to apply it: The photocurrent of the Si reference diode has to be the photocurrent of 
        the Si ref. diode under AM1.5GT multiplied with the mismatch factor
        """
        print(f'ref_EQE.calc_jsc(sp = ref_PF) = {ref_EQE.calc_jsc(left = left, right = right, delta = delta, sp = ref_PF):.3f}')
        print(f'self.calc_jsc(sp = sim_PF) = {self.calc_jsc(left = left, right = right, delta = delta, sp = sim_PF):.3f}')
        print(f'ref_EQE.calc_jsc(sp = sim_PF) = {ref_EQE.calc_jsc(left = left, right = right, delta = delta, sp = sim_PF):.3f}')
        print(f'self.calc_jsc(sp = ref_PF) = {self.calc_jsc(left = left, right = right, delta = delta, sp = ref_PF):.3f}')
        num = ref_EQE.calc_jsc(left = left, right = right, delta = delta, sp = ref_PF) * self.calc_jsc(left = left, right = right, delta = delta, sp = sim_PF)
        denom = ref_EQE.calc_jsc(left = left, right = right, delta = delta, sp = sim_PF) * self.calc_jsc(left = left, right = right, delta = delta, sp = ref_PF)
        return num / denom
        
    
    def bg_from_ip(self, left: float | None = None, right: float | None = None, showplot: Any = None) -> Any:
        """
        Calculates the bandgap from the inflection point of the EQE Spectrum.

        """
            
        if left is None:
            left = min(self.x)
        if right is None:
            right = max(self.x)
            
        dEQE = self.diff(left = left, right = right)
        Eg = dEQE.x[findind(dEQE.y, max(dEQE.y))]
        
        if 'diff' in showplot:  # type: ignore
            dEQE.plot(left = left, right = right, vline = Eg)
        
        if 'orig' in showplot:  # type: ignore
            self.plot(left = left, right = right, vline = Eg)
        
        self.Eg_ip = Eg
        
        return Eg
        
    def calc_jsc(self, left: float | None = None, right: float | None = None, delta: float | None = None, sp: Any = 'AM15GT') -> Any:    
        """
        #Calculates the integrated Jsc of an EQE Spectrum. sp is the spectral photon flux (type DiffSpectrum) of the light source.
        #It works if self and sp are both functions of nm or eV.
        """
        if sp != 'AM15GT':
            sp = sp.copy() # Otherwise the original lamp Spectrum is changed with the function equidist
            if 'W' in sp.uy:
                print('Attention(EQESpectrum.calc_jsc()): sp expects a spectral photon flux not an irradiance Spectrum!')

        if self.ux == 'nm':
            if delta==None:
                delta = 0.5
            EQE_nm = self.copy()
            if left is None:
                left = min(EQE_nm.x)
            if right is None:
                right = max(EQE_nm.x)
            EQE_nm.equidist(left = left, right = right, delta = delta)
            if sp == 'AM15GT':
                sp = DiffSpectrum.am15_nm(left = left, right = right, delta = delta)
            else:
                if sp.ux == 'nm':
                    sp.equidist(left = left, right = right, delta = delta)
                elif sp.ux == 'eV':
                    print('Attention (EQESpectrum.calc_jsc): EQE is a function of wavelength in nm but sp is a function of photon energy in eV!')
                else:
                    print('Attention: EQESpectrum.calc_jsc() requires ux to be either in nm or eV!')
                    print('Your ux is {sp.ux}.')
                    print(f'self.name = {self.name}')
                    print(f'sp.name = {sp.name}')
        
            if self.uy == '%':
                return np.trapz(EQE_nm.y/100 * sp.y * q * 1e3/1e4, dx = delta)  # type: ignore
            elif (self.uy == '') or (self.uy =='abs.'):
                return np.trapz(EQE_nm.y * sp.y * q * 1e3/1e4, dx = delta)  # type: ignore
        
        if self.ux == 'eV':
            if delta is None:
                delta = 0.001
            EQE_eV = self.copy()
            if left is None:
                left = min(EQE_eV.x)
            if right is None:
                right = max(EQE_eV.x)
            EQE_eV.equidist(left = left, right = right, delta = delta)
            
            if sp == 'AM15GT':
                sp = DiffSpectrum.am15_ev(left = left, right = right, delta = delta)
            else:
                if sp.ux == 'eV':
                    sp.equidist(left = left, right = right, delta = delta)
                elif sp.ux == 'nm':
                    print('Attention (EQESpectrum.calc_jsc): EQE is a function of photon energy in eV but sp is a function of wavelength in nm!')
                else:
                    print('Attention: EQESpectrum.calc_jsc() requires ux to be either in nm or eV!')
                    print('Your ux is {sp.ux}.')                
        
            if self.uy == '%':
                return np.trapz(EQE_eV.y/100 * sp.y * q * 1e3/1e4, dx = delta)  # type: ignore
            elif (self.uy == '') or (self.uy =='abs.'):
                return np.trapz(EQE_eV.y * sp.y * q * 1e3/1e4, dx = delta)  # type: ignore


    def normalize_to_jsc(self, Jsc: float) -> Any:
        
        Jsc_EQE = self.calc_jsc()
        self.y = self.y / Jsc_EQE * Jsc
        
    
    def to_ab(self) -> Any:
        x = self.x.copy()
        y = self.y.copy() / 100 # absorptance is EQE (in %) / 100
        
        quants = dict(x=self.qx, y='absorptance')
        units = dict(x=self.ux, y='')
        name = self.name
        
        sp_new = AbsSpectrum(x, y, quants = quants, units = units, name = name)
            
        return sp_new    
    
    @staticmethod
    def load_cicci(directory: str, filepath: str) -> Any:
        
        return EQESpectrum.load(directory, filepath, delimiter = '\t', header = 0, quants = {"x": "Wavelength", "y": "EQE"}, units = {"x": "nm", "y": "%"})
    
class AbsSpectrum(Spectrum):
    """
    Absorptance Spectrum. Absorptance is dimensionless from 0 to 1.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, quants: Any = {"x": "Photon energy", "y": "Absorptance"}, units: Any = {"x": "eV", "y": ""}, name: str = 'Absorptance', plotstyle: str = dict(linestyle = '-', color = 'black', linewidth = 3), check_data: bool = True) -> None:  # type: ignore
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
        
        
    def u_energy_fit(self, Efit_start: Any, Efit_stop: Any) -> Any:
        """
        Calculates the Urbach energy of an absorptance curve and returns the fitcurve as an instance of the class "Spectrum".
        self: absorptance curve, instance of class "Spectrum"
        Efit_start: Photon energy (in eV) to begin with the fit.
        Efit_stop: Photon energy to end with the fit.
        """
        m, b = linfit(self.x, np.log(np.abs(self.y)), Efit_start, Efit_stop)
        U_E = 1/m * 1000 #meV
        name = f'Fit: $E_u$ = {U_E:.1} eV'
        UE_fit = Spectrum(self.x, np.exp(m * self.x + b), quants = {"x": self.qx, "y": "Urbach energy fit"}, units = {"x": self.ux, "y": ""}, name = name)
        UE_fit.UE = U_E  # type: ignore
        return UE_fit
    
    def new_ue(self, UE: float, E_takeover: Any) -> Any:
        m = 1000/UE
        idx = self.x_idx_of(E_takeover)
        b = self.y[idx]
        # UE_fit = Spectrum(self.x, np.exp(m * (self.x - E_takeover)) * b, quants = {"x": self.qx, "y": "Urbach energy fit"}, units = {"x": self.ux, "y": ""})
        UE_fit = Spectrum(self.x, np.exp(m * (self.x - self.x[idx])) * b, quants = {"x": self.qx, "y": "Urbach energy fit"}, units = {"x": self.ux, "y": ""})
        self.y[0:idx] = UE_fit.y[0:idx] 
        
    
    def tauc_plot(self, Efit_start: Any, Efit_stop: Any, left_offs: Any, right_offs: Any, showplot: Any = True, title: str = '', save: bool = False, save_dir: str = '', save_name: str | None = None, return_fig: bool = True) -> Any:
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
        Tp = Spectrum(self.x, (alpha_eV * self.x)**2, quants = {"x": self.qx, "y": qy}, units = {"x": self.ux, "y": uy}, 
                      name = name_Tp)

        Tpfit = Spectrum(self.x, (m * self.x + b), quants = {"x": self.qx, "y": qy}, units = {"x": self.ux, "y": uy}, 
                         name = name_Tpfit)
        
        sa = Spectra([Tp, Tpfit])
        sa.label([name_Tp, name_Tpfit])

        if save:
            if save_name is None:
                filepath = 'Tauc plot.csv'
            else:
                filepath = save_name
            sa.save(save_dir, filepath) 
        
        self.Eg_Tauc = Eg
        self.Vocsq = Vocsq
        
        graph = sa.plot(left = Efit_start + left_offs, right = Efit_stop + right_offs, bottom = 0, top = Tp.y_of(Efit_stop + right_offs) * 1.2, title = title, return_fig = True, show_plot = showplot)  # type: ignore

        if return_fig:
            return graph


    def emission_pf(self, E_start: Any, E_stop: Any, T: float = T_RT) -> Any:
        """
        Calculate the emission photon flux for a given absorptance Spectrum self as a function of photon energy.
        """
        BB = DiffSpectrum.phi_bb(self.x * q, T)  # type: ignore
        r = range(self.x_idx_of(E_start), self.x_idx_of(E_stop)+1)
        empf = DiffSpectrum(self.x[r], self.y[r] * BB[r] * q, quants = dict(x = 'Photon energy', y = 'Spectral photon flux'), units = dict(x = 'eV', y = '1/[s m2 eV]'))        
        return empf

    def calc_vocrad(self, E_start: Any, E_stop: Any, T: float = T_RT, show_table: bool = False) -> None:
        """
        Calculates the Voc,rad for the absorptance self (of type Spectrum) and plots a table with all relevant information.
        E_start: Photon energy to start integration (in eV)
        E_stop: Photon energy to stop integration (in eV)
        Jph: Photo current (in mA/cm2)
        T: Temperature in K

        Note: call calc_jradlim() before this method to set self.Jradlim.
        """
        if not hasattr(self, 'Jradlim'):
            raise AttributeError("calc_jradlim() must be called before calc_vocrad() to set self.Jradlim.")
        empf = self.emission_pf(E_start, E_stop, T = T)
        denergies_eV = self.x[1] - self.x[0]
        Jrad0 = q * np.trapz(empf.y, dx = denergies_eV) *1e-4 /1e-3  # type: ignore

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
 
    def calc_jradlim(self, illumspec_eV: Any | None = None, start_eV: Any | None = None, handover_eV: Any | None = None) -> None:
        """
        Calculates the radiative limit phtocurrent taking the absorptance Spectrum (self).
        Absorptance Spectrum has to be a function of photon energy in eV.

        Parameters
        ----------
        illumspec_eV : DiffSpectrum
            Illumination photon flux Spectrum as a function of eV in 1/[s m2 eV] (e.g. AM 1.5 GT Spectrum).
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
        if illumspec_eV is None:
            illumspec_eV = DiffSpectrum.am15_ev(left = min(self.x), right = max(self.x))
            
        # New absorption Spectrum with the same x-values as illumspec_eV (necessary for integration)
        asp = DiffSpectrum(illumspec_eV.x, int_arr(self.x, self.y, illumspec_eV.x))
                
        if handover_eV is not None:
            asp.y[asp.x_idx_of(handover_eV):] = asp.y[asp.x_idx_of(handover_eV)]
            
        if start_eV is not None:
            start_idx = asp.x_idx_of(start_eV)
        
        else:
            start_idx = 0
        
        dx = (asp.x[1] - asp.x[0])
                
        self.Jradlim = np.trapz(asp.y[start_idx:] * illumspec_eV.y[start_idx:] * q, dx = dx) * 1e3 / 1e4  # type: ignore

    def convert_absorbance_to_absorptance(self) -> Any:
        A = self.copy()
        new_y = 1 - 10**(-A.y)
        ab = AbsSpectrum(self.x, new_y, self.quants(), self.units(), self.name)
        ab.qy = 'Absorptance'
        return ab
    
    def convert_absorptance_to_absorbance(self) -> Any:
        A = self.copy()
        new_y = -np.log10(1-A.y)
        ab = AbsSpectrum(self.x, new_y, self.quants(), self.units(), self.name)
        ab.qy = 'Absorbance'
        return ab

    @staticmethod
    def load_absorbance(directory: str, filepath: str) -> Any:
        return AbsSpectrum.load(directory, filepath = filepath, delimiter = ',', header = 'infer', take_quants_and_units_from_file = True)

    def absorbed_photonflux(self, left: float, right: float, details: Any = True) -> Any:
        # Calculates the absorbed photon flux in 1/(s m²) of an absoptance Spectrum under AM1.5G.
        sp = self.copy()
        sp.equidist(left = left, right = right, delta = 1)
        AM = DiffSpectrum.am15_nm(left = left, right = right, delta = 1)
        AMnorm = AM.copy()
        AMnorm.y = AM.y / AM.y[AM.x_idx_of((left+right)/2)]
        both = Spectra([self, AMnorm])
        both.label([self.name, 'AM1.5G'])
        if details:
            both.plot(hline = 0, vline = [left, right])
        spAM = AM * sp.y
        PF_1sun = spAM.calc_integrated_photonflux()
        if details:
            print(f'The absorbed photon flux at 1 sun is: {PF_1sun:.2e} 1/(s m²)')
        return PF_1sun

        
class DiffSpectrum(Spectrum):   
    
    def __init__(self, x: np.ndarray, y: np.ndarray, quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, name: str = '', plotstyle: str = dict(linestyle = '-', color = 'black', linewidth = 3), check_data: bool = True) -> None:  # type: ignore
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
            
            
    def photonflux_to_irradiance(self) -> Any:
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
                print('Attention: DiffSpectrum.photonflux_to_irradiance works only if ux = nm!')
                print(f'Your ux = {self.ux}')
        else:
            print('Attention: DiffSpectrum.photonflux_to_irradiance works only if uy = 1/[s m2 nm]!')
            print(f'Your uy = {self.uy}')
    
    
    def irradiance_to_photonflux(self, factor: float = 1) -> Any:
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
                    print('Attention: DiffSpectrum.irradiance_to_photonflux works only if ux = nm!')
                    print(f'Your ux = {self.ux}')
            else:
                print('Attention: DiffSpectrum.irradiance_to_photonflux works only if uy = W/[m2 nm]!')
                print(f'Your uy = {self.uy}')
    
    def irradiance_to_illuminance(self) -> Any:
        """
        Returns the illuminance Spectrum of the irradiance Spectrum self.
        """
        if self.uy == 'W/[m2 nm]':
            cf = Spectrum.luminosity_fn()
            return self.product(cf, qy = 'Spectral illuminance', uy = 'lm/[m2 nm]')
        else:
            print('Attention: Function "DiffSpectrum.irradiance_to_illuminance" works only if uy is in W/[m2 nm]!')
            print(f'Your uy is {self.uy}.')

    
    
    def illuminance_to_irradiance(self, warning: bool = True) -> Any:
        """
        Returns the irradiance Spectrum of the illuminance Spectrum self.
        """
        if (self.uy == 'lm/[m2 nm]') or (self.uy == 'lx/nm'):
            if warning:
                print('Attention! Function "DiffSpectrum.illuminance_to_irradiance" has to be used carefully.')
                print('A lot of information could be lost outside the x-limits (400-720 nm) of the photopic luminosity function.')
                print('To switch off this warning, use "warning = False"!')
            cf = Spectrum.luminosity_fn()
            #Inverse photopic luminosity function:
            icf = cf.copy()
            icf.y = 1/cf.y 
            return self.product(icf, qy = 'Spectral irradiance', uy = 'W/[m2 nm]')
        else:
            print('Attention: Function "DiffSpectrum.illuminance_to_irradiance" works only if uy is in lm/[m2 nm] or in lx/nm!')
            print(f'Your uy is {self.uy}.')
      
    
    def nm_to_ev(self, delta: float = 0.001) -> Any:
    
        """
        Transforms a single differential Spectrum (dS/dlambda) as a function of wavelength (in nm) into 
        a differential Spectrum (dS/dE) as a function of photon energy (in eV). 
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
        y_eV = y[::-1] * h*c/(q*1e-9) * 1/x_eV**2
        
        # quants = {"x": "Photon energy", "y": "Spectral photon flux"}
        # units = {"x": "eV", "y": "1/[s m2 eV]"}

        quants = dict(x = 'Photon energy', y = self.qy)
        units = dict(x = 'eV', y = self.uy.replace('nm', 'eV'))

        name = self.name
        
        # if self.__class__.__name__ == 'DiffSpectrum':
        #     sp_new = DiffSpectrum(x_eV, y_eV, quants = quants, units = units, name = name)
        # if self.__class__.__name__ == 'PELSpectrum':
        #     sp_new = PELSpectrum(x_eV, y_eV, quants = quants, units = units, name = name)

        sp_new = type(self)(x_eV, y_eV, quants = quants, units = units, name = name)
        sp_new.equidist(delta = delta)    
        return sp_new
    
    def ev_to_nm(self, delta: float = 1) -> Any:
    
        """
        Transforms a single differential Spectrum as a function of photon energy (in eV) into 
        a differential Spectrum as a function of wavelength (in nm).  
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
    def load_astmg173(y_unit: str = 'Spectral photon flux', warning: bool = True, Spectrum: Any = 'AM1.5GT') -> Any:

        """
        Spectrum == 'AM1.5GT': Loads the AM1.5GT Spectrum.
        Spectrum == 'Etr': Loads the extraterrestrial Spectrum.
        Spectrum == 'direct': Loads the direct+circumsolar Spectrum.
        y_unit: Either 'Spectral photon flux' ('PF') or 'Spectral irradiance ('Spectral flux', 'SF')
        warning: If True it prints a warning that the Wavelengths are not evenly spaced and that rather the function am15_nm or am15_ev should be used.
        """
        # Directory for Astmg173 Spectrum

        ASTMG173 = 'Astmg173.xls'
        
        dat = pd.read_excel(join(system_dir, ASTMG173), header = 1)
        dat_nm = np.array(dat, dtype = np.float64)[:,0] # wavelength data in nm
        if Spectrum == 'AM1.5GT':
            dat_SF = np.array(dat, dtype = np.float64)[:,2] # spectral irradiance in W/(m2 nm)
        elif Spectrum == 'Etr':
            dat_SF = np.array(dat, dtype = np.float64)[:,1] # spectral irradiance in W/(m2 nm)
        elif Spectrum == 'direct':
            dat_SF = np.array(dat, dtype = np.float64)[:,3] # spectral irradiance in W/(m2 nm)        

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
            print('Attention (load_astmg173): y_unit not known!')
            
        if warning:
            print('load_astmg173: Wavelengths are not evenly spaced, use rather the function am15_nm or am15_ev.')

        return DiffSpectrum(x, y, quants, units, ASTMG173)
    
    @staticmethod
    def am15_nm(left: float = 280, right: float = 4000, delta: float = 1, y_unit: str = 'Spectral photon flux') -> Any:
        """
        Returns the AM1.5 GT photon flux Spectrum as a function of wavlelength in nm.
        """
        
        AM15 = DiffSpectrum.load_astmg173(y_unit = y_unit, warning = False)
        AM15.equidist(left = left, right = right, delta = delta)
        AM15.name += ' (AM1.5GT)'
        #am15_nm.plot(left = 300, right = 1000)    
        return AM15
        
    @staticmethod
    def am15_ev(left: float = 0.310, right: float = 4.428, delta: float = 0.001, y_unit: str = 'Spectral photon flux') -> Any:
        """
        Returns the AM1.5 GT photon flux Spectrum as a function of photon energy in eV.
        """
        
        AM15nm = DiffSpectrum.am15_nm(y_unit = y_unit)
        AM15eV = AM15nm.nm_to_ev()
        AM15eV.equidist(left = left, right = right, delta = delta)
        #am15_ev.plot()
        return AM15eV
    
    @staticmethod
    def am15direct_nm(left: float = 280, right: float = 4000, delta: float = 1, y_unit: str = 'Spectral photon flux') -> Any:
        """
        Returns the AM1.5 direct and circumsolar photon flux Spectrum as a function of wavlelength in nm.
        """
        
        AM15direct = DiffSpectrum.load_astmg173(y_unit = y_unit, warning = False, Spectrum = 'direct')
        AM15direct.equidist(left = left, right = right, delta = delta)
        AM15direct.name += ' (AM1.5 direct+circumsolar)'
        #am15_nm.plot(left = 300, right = 1000)    
        return AM15direct
        
    @staticmethod
    def am15direct_ev(left: float = 0.310, right: float = 4.428, delta: float = 0.001, y_unit: str = 'Spectral photon flux') -> Any:
        """
        Returns the AM1.5 GT photon flux Spectrum as a function of photon energy in eV.
        """
        
        AM15directnm = DiffSpectrum.am15direct_nm(y_unit = y_unit)
        AM15directeV = AM15directnm.nm_to_ev()
        AM15directeV.equidist(left = left, right = right, delta = delta)
        #am15_ev.plot()
        return AM15directeV
    
    @staticmethod
    def etr_nm(left: float = 280, right: float = 4000, delta: float = 1, y_unit: str = 'Spectral photon flux') -> Any:
        """
        Returns the extraterrestrial photon flux Spectrum as a function of wavlelength in nm.
        """
        
        Etr = DiffSpectrum.load_astmg173(y_unit = y_unit, warning = False, Spectrum = 'Etr')
        Etr.equidist(left = left, right = right, delta = delta)
        Etr.name += ' (Extraterrestrial)'
        #am15_nm.plot(left = 300, right = 1000)    
        return Etr
        
    @staticmethod
    def etr_ev(left: float = 0.310, right: float = 4.428, delta: float = 0.001, y_unit: str = 'Spectral photon flux') -> Any:
        """
        Returns the extraterrestrial photon flux Spectrum as a function of photon energy in eV.
        """
        
        Etrnm = DiffSpectrum.etr_nm(y_unit = y_unit)
        EtreV = Etrnm.nm_to_ev()
        EtreV.equidist(left = left, right = right, delta = delta)
        #am15_ev.plot()
        return EtreV
    
    @staticmethod
    def _load_illuminant_spectrum(filename: str, y_unit: str, delimiter: str=',', header: int=0) -> None:
        """Load an illuminant spectrum from system_dir and return a DiffSpectrum."""
        dat = pd.read_csv(join(system_dir, filename), delimiter=delimiter, header=header)
        dat_nm = np.array(dat, dtype=np.float64)[:, 0]
        dat_SF = np.array(dat, dtype=np.float64)[:, 1]
        dat_PF = dat_SF * dat_nm * 1e-09 / (h * c)
        if (y_unit == 'Spectral photon flux') or (y_unit == 'PF'):
            y = dat_PF
            quants = {"x": "Wavelength", "y": "Spectral photon flux"}
            units = {"x": "nm", "y": "1/[s m2 nm]"}
        elif y_unit in ('Spectral irradiance', 'Spectral flux', 'SF'):
            y = dat_SF
            quants = {"x": "Wavelength", "y": "Spectral irradiance"}
            units = {"x": "nm", "y": "W/[m2 nm]"}
        else:
            print('Attention: y_unit not known!')
        sp = DiffSpectrum(dat_nm, y, quants, units, filename)
        sp.equidist(delta=1)
        return sp  # type: ignore

    @staticmethod
    def load_osram930(y_unit: str='Spectral photon flux') -> Any:
        """
        Loads the OSRAM 930 Spectrum.
        y_unit: Either 'Spectral photon flux' ('PF') or 'Spectral irradiance' ('Spectral flux', 'SF')
        """
        return DiffSpectrum._load_illuminant_spectrum('OSRAM_930.txt', y_unit, delimiter='\t', header=2)

    @staticmethod
    def load_led5000k(y_unit: str='Spectral photon flux') -> Any:
        """
        Loads the high CRI LED (YJ-VTC-2835MX-G2) 5000 K Spectrum.
        y_unit: Either 'Spectral photon flux' ('PF') or 'Spectral irradiance' ('Spectral flux', 'SF')
        left = 383, right = 779, delta = 1
        """
        return DiffSpectrum._load_illuminant_spectrum(
            'high CRI LED (YJ-VTC-2835MX-G2) 5000 K.csv', y_unit, delimiter=',', header=0
        )
    
    
    def calc_integrated_photonflux(self, start: float | None = None, stop: float | None = None) -> Any:
        
        """
        Calculates photon flux from self.x=start to self.x=stop. self.x values have to be equidistant. Standard is from min(self.x) to max(self.x).
        """

        if start is None:
            start_x = min(self.x)
        else:
            start_x = start
        
        if stop is None:
            stop_x = max(self.x)
        else:
            stop_x = stop
            
        index_start = findind(self.x, start_x)
        index_stop = findind(self.x, stop_x)
        r = range(index_start, index_stop+1)
        
        dx = (max(self.x) - min(self.x)) / (len(self.x) - 1)

        return np.trapz(self.y[r], dx = dx)  # type: ignore
    
    def photonflux(self, start: float | None=None, stop: float | None=None) -> Any:
        """Old alias for calc_integrated_photonflux; prefer that method."""
        return self.calc_integrated_photonflux(start=start, stop=stop)
    
    
    def calc_illuminance(self) -> Any:
        """
        Calculates the illuminance in lx.
        Only works if self.x is wavelength in nm.
        It works for irradiance and illuminance Spectra.
        """
        # Directory for photopic luminosity function
        lum_FN = 'luminosity function CIE 1931.csv'
        cf = Spectrum.load(system_dir, lum_FN, quants = dict(x = 'Wavelength', y = 'Conversion factor'), units = dict(x = 'nm', y = 'lx/(W/m2)'))
        if self.ux == 'nm':
            if self.uy == 'W/[m2 nm]':
                new = self.product(cf, qy = 'Spectral illuminance', uy = 'lm/[m2 nm]')
                return np.trapz(new.y, dx = new.x[1]-new.x[0])  # type: ignore
            elif self.uy == '1/[s m2 nm]':
                self_sf = self.photonflux_to_irradiance()
                new = self_sf.product(cf, qy = 'Spectral illuminance', uy = 'lm/[m2 nm]')
                return np.trapz(new.y, dx = new.x[1]-new.x[0])  # type: ignore
            elif self.uy == ('lm/[m2 nm]') or (self.uy == 'lx/nm'):
                return np.trapz(self.y, dx = self.x[1]-self.x[0])  # type: ignore

            else:
                print('Attention (calc_lux): This function works only for spectral irradiances in W/[m2 nm] or photon flux in 1/[s m2 nm],')
                print('or for spectral illuminance in lm/[m2 nm] or lx/nm!')
        else:
            print('Attention(calc_lux): This function works only if ux = nm!')            
 
    
    def normalize_to_lux(self, lx: Any) -> Any:
        """
        Normalizes self to lx.
        Works only if self.ux = 'nm' and self.uy = 'W/[m2 nm]' or '1/[s m2 nm]'
        """
        ill = self.calc_illuminance()
        self.y = self.y/ill*lx
        
        
    def calc_irradiance(self) -> Any:
        """
        Calculates the irradiance in W/m2 of an irradiance Spectrum.
        uy has to be either in W/[m2 nm] or W/[m2 eV].

        Returns
        -------
        float
            Irradiance in mW/cm2.

        """
        if ((self.uy == 'W/[m2 nm]') and (self.ux =='nm')) or ((self.uy == 'W/[m2 eV]') and (self.ux =='eV')):
            return np.trapz(self.y, dx = self.x[1]-self.x[0])*1e-1  # type: ignore
        else:

            print('Attention (calc_irradiance): uy has to be either in W/[m2 nm] or W/[m2 eV] and ux in nm or eV, respectively!')
            print(f'Your ux is {self.ux} ({self.qx}) and your uy is {self.uy} ({self.qy})')
        
    
    #Black body spectral photon flux in 1/(s m2 J), energy in Joule
    @staticmethod
    def phi_bb(E: float, T: float) -> Any:
        return 1/(4 * math.pi**2 * (h/(2 * math.pi))**3 * c**2) * E**2 / (np.exp(E / (k * T)) - 1)

    @staticmethod
    def bb_spectrum(sp: Any, T: float = T_RT, fit_eV: Any = 1.5) -> Any:
        x_eV = sp.x
        ind = findind(x_eV, fit_eV)
        BB = DiffSpectrum.phi_bb(x_eV * q, T) * q
        BB = BB / BB[ind] * sp.y_of(fit_eV)
        quants = {"x": "Photon energy", "y": "BB fit"}
        units = {"x": "eV", "y": ""}
        name = 'BB Spectrum'
        return DiffSpectrum(x_eV, BB, quants = quants, units = units, name = name)    
    
    def integrated_current(self, EQE: Any | None=None) -> Any:
        # Calculates an integrated current plot as a function of wavelength
        # Works only with photon flux Spectrum
        # EQE in % and as a function fo wavelength
        # if EQE==1 then EQE is assumed 100% throughout the Spectrum
    
        sp = self.copy()
        if EQE is None:
            e = EQESpectrum.eqe100(Eg=0.1)
        else:
            e = EQE.copy()
        left = max(sp.x[0], e.x[0])
        right = min(sp.x[-1], e.x[-1]) 
        sp.equidist(left=left, right=right, delta=1)
        e.equidist(left=left, right=right, delta=1)
        sp_times_e = sp*e/100
        int_curr_y = q*np.array([np.trapz(sp_times_e.y[0:idx], dx=1)*1e3/1e4 for idx in range(len(sp_times_e.x))])  # type: ignore
        return DiffSpectrum(x=sp_times_e.x, y=int_curr_y, quants=dict(x='Wavelength', y='Integrated current density'), units=dict(x='nm', y='mA/cm2'))


    def integrated_irradiance(self) -> Any:
        # Calculates an integrated irradiance plot as a function of wavelength
        # Works only with spectral irradiance
    
        int_irr_y = np.array([np.trapz(self.y[0:idx], dx=self.x[1]-self.x[0])*1e3/1e4 for idx in range(len(self.x))])  # type: ignore
        return DiffSpectrum(x=self.x, y=int_irr_y, quants=dict(x='Wavelength', y='Integrated irradiance'), units=dict(x='nm', y='mW/cm2'))
    

    def calculate_irradiance_illuminance(self, left: float=400, right: float=720, show: bool=True) -> Any:

        sp_new = self.copy().cut_data_outside(left=left, right=right)
        sp_irr = sp_new.photonflux_to_irradiance()
        irr = sp_irr.calc_irradiance()
        ill = sp_new.calc_illuminance()
        if show:
            print(f'__{sp_new.name}')
            print(f'The irradiance is {irr:.3e} mW/cm2')
            print(f'The illuminance is {ill:.0f} lx')
        return irr, ill
    
class PELSpectrum(DiffSpectrum):   
    """
    PL or EL Spectrum.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, name: str = '', plotstyle: str = dict(linestyle = '-', color = 'black', linewidth = 3), check_data: bool = True) -> None:  # type: ignore
        super().__init__(x, y, quants, units, name, plotstyle, check_data = check_data)
        
        
    def absorptance(self, bb: Any, eV: Any, a_eV: Any) -> Any:
        """
        Calculates the absorptance Spectrum from a PL/EL Spectrum (self) by dividing self with the blackbody Spectrum (BB).
        Then it is calibrated so that at the energy eV the absorptance is a_eV .
        BB is of type Spectrum. self and BB have to have the same x value in eV.
        """
        if not(all(self.x==bb.x)):
            print("Warning: PL/EL Spectrum and BB Spectrum don't have the same x-values.")
            ab = self
        else:
            ab = AbsSpectrum(self.x, self.y / bb.y, quants = {"x": self.qx, "y": "Absorptance"}, units = {"x": self.ux, "y": ""}, name = 'Absorptance')  # type: ignore
            ab.y = ab.y / ab.y_of(eV) * a_eV
        return ab
    
    
    def bbt_fit(self, Efit_start: Any, Efit_stop: Any, Tguess: Any = 300) -> Any:
        """
        Calculates the high energy tail fit with BB temperature, returns the fitcurve as an instance of the class "Spectrum".
        self: PL/EL Spectrum, instance of class "Spectrum"
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
        except RuntimeError:
            popt = p0
            print("Fit didn't converge, Tguess taken!")
        
        T = popt[0] #K
        name = f'Fit: $T$ = {T:.1e} K'
        T_fit = Spectrum(self.x, f(self.x, *popt), quants = {"x": self.qx, "y": "Blackbody fit"}, units = {"x": self.ux, "y": self.uy}, name = name)
        T_fit.T = T  # type: ignore
        return T_fit
        

class Spectra(MXYData):
    
    def __init__(self, sa: Any) -> None:
        super().__init__(sa)
        
    @classmethod
    def generate_empty(cls, empty_spectra_list: Any=[]) -> Any:
        return cls(empty_spectra_list)   
        
                
    def replace(self, idx: int, sp_new: Any) -> Any:
        self.sa[idx] = sp_new
        
    def remain(self, idx_list: Any) -> Any:
        """
        Return all Spectra with indices in list idx_list.
        """
        new_sa = self.copy()
        sa = []
        for i, idx in enumerate(idx_list):
            new = new_sa.sa[idx].copy()
            sa.append(new)
            
        new_sa.sa = sa
        new_sa.names_to_label(split_ch = '.csv')
        return new_sa

    def names_to_label(self, split_ch: str | None = None) -> Any:
        lab = []
        for i, sp in enumerate(self.sa):
            if split_ch is None:
                lab.append(sp.name)
            else:
                lab.append(sp.name.split(split_ch)[0])
        self.label(lab)
        self.label_defined = True

    def nm_to_ev(self) -> Any:
        """
        Transforms a list of non-differential Spectra of type Spectrum from wavelength to photon energy. 
        """
        spectra_eV = []
        for i, spectrum_nm in enumerate(self.sa):
            spectra_eV.append(spectrum_nm.nm_to_ev())
            
        spa_new = self.copy()
        spa_new.sa = spectra_eV
    
        return spa_new
            
    def save(self, save_dir: str, filepath: str) -> None:  # type: ignore
        
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
                            
            TFN = join(save_dir, filepath)
            if save_ok(TFN):
                df.to_csv(join(save_dir, filepath), header = True, index = False)
                
    @staticmethod
    def load_multiple(directory: str, filepath: str, delimiter: str = ',', header: int = 'infer', quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, take_quants_and_units_from_file: str = False) -> Any:  # type: ignore

        """
        Loads multiple Spectra, all Spectra in one file filepath.
        """
        
        dat = pd.read_csv(join(directory, filepath), delimiter = delimiter, header = header)
        
        npdat = np.array(dat, dtype = np.float64)
        x = npdat[:,0]

        sa = []
        for i in range(np.shape(npdat)[1]-1): #"-1" because x-column disregarded

            y = np.array(dat, dtype = np.float64)[:,i+1]
            sp = Spectrum(x, y, quants, units, filepath)
       
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
            
            new_spectra = Spectra(sa)

        return new_spectra
    
    @classmethod
    def load_individual(cls, directory: str, FNs: str = [], delimiter: str = ',', header: int = 'infer', quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, take_quants_and_units_from_file: str = False, check_data: bool = False) -> Any:  # type: ignore

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

            if cls.__name__ == 'Spectra':
                sp = Spectrum(x, y, quants, units, filepath, check_data = False)
            elif cls.__name__ == 'AbsSpectra':
                sp = AbsSpectrum(x, y, quants, units, filepath, check_data = False)
            elif cls.__name__ == 'DiffSpectra':
                sp = DiffSpectrum(x, y, quants, units, filepath, check_data = False)
            elif cls.__name__ == 'PELSpectra':
                sp = PELSpectrum(x, y, quants, units, filepath, check_data = False)
       
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
       
                
    def save_names(self, save_dir: str, filepath: str) -> Any:
        TFN = join(save_dir, filepath)
        if save_ok(TFN):
            names  = []
            for i, sp in enumerate(self.sa):
                names.append(sp.name)
            with open(join(save_dir, filepath), 'w') as f:
                f.write("\n".join(names))    
                
    def load_names(self, directory: str, filepath: str) -> Any:
        TFN = join(directory, filepath)
        with open(TFN, 'r') as f:
            FNstring = f.read()
            FN_list = FNstring.split("\n")
        for i, sp in enumerate(self.sa):
            sp.name = FN_list[i]
            
    
    def all_spectra_have_same_x_range(self) -> Any:
        """
        Returns True if all Spectra have the same x-range, otherwise False.
        """
        same_range = True
        
        for i, spec in enumerate(self.sa):
            if not(np.array_equal(spec.x, self.sa[0].x)):
                same_range = False
        return same_range
    
    
    def equidist(self, left: float, right: float, delta: float, kind: str='cubic') -> None:  # type: ignore
        """
        Change x values so that they are equidistant with a delta of delta for all Spectra in the same way. 
        x ranges from start to stop. 
        If start (stop) = None then the new start (stop) value is the old one. Then it could happen that the different Spectra have different start and stop values.
        """
        if not(Spectra.all_spectra_have_same_x_range(self)):
            print('Warning: Not all Spectra have the same x-range!')
            
        for i, spec in enumerate(self.sa):
            spec.equidist(left = left, right = right, delta = delta, kind=kind)
            
            self.n_y = len(self.sa)
            self.n_x = len(self.sa[0].x)
           
    @classmethod
    def load_andor(cls, directory: str, meta_data: np.ndarray | None = None, sel_list: Any | None = None) -> Any:
    
        """
        Loads all Andor Spectra in the directory.
        """
    
        FNs = listdir(directory)
        
        if sel_list is not None:
            FNs = [FNs[idx] for idx in sel_list]
        
        sa = []
        for i, filepath in enumerate(FNs):
            if cls.__name__ == 'Spectra':
                sa.append(Spectrum.load_andor(directory, filepath, meta_data = meta_data))
            if cls.__name__ == 'DiffSpectra':
                sa.append(DiffSpectrum.load_andor(directory, filepath, meta_data = meta_data))
            if cls.__name__ == 'PELSpectra':
                sa.append(PELSpectrum.load_andor(directory, filepath, meta_data = meta_data))
                
        return cls(sa)
    

class AbsSpectra(Spectra):
    
    def __init__(self, sa: Any) -> None:
        super().__init__(sa)

            
class DiffSpectra(Spectra):
    
    def __init__(self, sa: Any) -> None:
        super().__init__(sa)
        
    
class PELSpectra(DiffSpectra):
    
    def __init__(self, sa: Any) -> None:
        super().__init__(sa)
        
    
    def calc_calfn(self, calspec: Any) -> Any:

        calib = []
        for i, sp in enumerate(self.sa):
            # make sure that there are no negatve values and no 0 values
            sp.y = np.array([abs(z) if z !=0 else np.average(sp.y) for z in sp.y], dtype = float)
            calib.append(PELSpectrum(sp.x, int_arr(calspec.x, calspec.y, sp.x) / sp.y, 
                                  quants = {"x": "Wavelength", "y": "Calibration factor"}, 
                                  units = {"x": "nm", "y": "1/[cps s m2 nm]"}, name = sp.name))
        return PELSpectra(calib)

            
    def calibrate_single(self, calib: Any) -> Any:
        
        """
        Calibrates all PL Spectra with the calibration function calib.
        """
    
        calibrated = []
        for i, rawPLspectrum in enumerate(self.sa):
            calibrated_spectrum = PELSpectrum(rawPLspectrum.x, rawPLspectrum.y * int_arr(calib.x, calib.y, rawPLspectrum.x), 
                                           quants = {"x": "Wavelength", "y": "Spectral photon flux"}, units = {"x": "nm", "y": "1/[s m2 nm]"}, name = rawPLspectrum.name)
            calibrated.append(calibrated_spectrum)
            
        return PELSpectra(calibrated)

    def calibrate(self, calib: Any, check: Any = False) -> Any:
        
        """
        Calibrates all PL Spectra with the respective calibration function calib.
        check: True if the calibration function should be printed for each Spectrum. 
        """

        def right_cal_fn(rawPLsp: Any, calsp: Any) -> Any:
            
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
                    calibrated_spectrum = PELSpectrum(rawPLsp.x, rawPLsp.y * int_arr(calsp.x, calsp.y, rawPLsp.x), 
                                                   quants = {"x": "Wavelength", "y": "Spectral photon flux"}, units = {"x": "nm", "y": "1/[s m2 nm]"}, name = rawPLsp.name)
                    calibrated.append(calibrated_spectrum)
                    calib_ok = True
                    if check:
                        print(f'{rawPLsp.name}: {calsp.name}')
            if not(calib_ok):
                print(f'Warning: No adequate calibration function found for {rawPLsp.name}.')
            
        return PELSpectra(calibrated)
        
    def choose_for_plqy(self, name: str, laser_marker: Any, PL_marker: Any) -> Any:
        """
        All Spectra with name in its name will be put into a new instance of Spectra.
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
        expl = {}  # type: ignore
        
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
        
        PLQY_sa = PELSpectra(sa)
        PLQY_sa.expl =expl  # type: ignore
        
        lab = [None] * len(expl)
        #embed()
        for i, (l, v) in enumerate(expl.items()):
            lab[v] = l
            
        PLQY_sa.label(lab)
        return PLQY_sa
    
    def calc_plqy_param(self, laser_marker: Any, left_laser: Any, right_laser: Any, PL_marker: Any, left_PL: Any, right_PL: Any, eval_Pa: Any = False, eval_Pb: Any = True, show_errmsg: bool = True) -> Any:
        
        if 'La' in self.expl:  # type: ignore
            La = self.sa[self.expl['La']].photonflux(start = left_laser, stop = right_laser)  # type: ignore
        else:
            La = 0
            if show_errmsg:
                print('Attention: No La signal!')

        if 'Lb' in self.expl:  # type: ignore
            Lb = self.sa[self.expl['Lb']].photonflux(start = left_laser, stop = right_laser)  # type: ignore
        else:
            Lb = 0
            if show_errmsg:
                print('Attention: No Lb signal!')

        if 'Lc' in self.expl:  # type: ignore
            Lc = self.sa[self.expl['Lc']].photonflux(start = left_laser, stop = right_laser)  # type: ignore
        else:
            Lc = 0
            if show_errmsg:
                print('Attention: No Lc signal!')

        if 'Pa' in self.expl and eval_Pa:  # type: ignore
            Pa = self.sa[self.expl['Pa']].photonflux(start = left_PL, stop = right_PL)  # type: ignore
        else:
            Pa = 0
            if show_errmsg:
                print('Attention: No Pa signal evaluated!')

        if 'Pb' in self.expl and eval_Pb:  # type: ignore
            Pb = self.sa[self.expl['Pb']].photonflux(start = left_PL, stop = right_PL)  # type: ignore
        else:
            Pb = 0
            if show_errmsg:
                print('Attention: No Pb signal!')

        if 'Pc' in self.expl:  # type: ignore
            Pc = self.sa[self.expl['Pc']].photonflux(start = left_PL, stop = right_PL)  # type: ignore
    
        else:
            Pc = 0
            if show_errmsg:
                print('Attention: No Pc signal!')
         
        Pb = Pb - Pa
        Pc = Pc - Pa
        A = 1 - Lc/Lb
        PLQY = (Pc - (1 - A) * Pb) / (La * A)

        return (PLQY, A, La, Lb, Lc, Pa, Pb, Pc)
    
    def guess_factor(self, left: float, right: float) -> Any:
        """
        Returns the inbeam-free space adjustment factor.
        self.sa[0] has to be the ip Spectrum.
        self.sa[1] has to be the fs Spectrum.
        Both Spectra have to have the same x array.
        """
        
        scpy = self.copy()
        delta = self.sa[0].x[1] - self.sa[0].x[0] 
        scpy.equidist(left = left, right = right, delta = delta)

        #r = range(findind(self.sa[0].x, left), findind(self.sa[0].x, right)+1)        

        def f(fac: float) -> Any: 
            diff = scpy.sa[0].y - fac * scpy.sa[1].y
            return math.sqrt(1/len(diff) * np.dot(diff, diff))
            
        result = least_squares(fun = f, x0 = [1])
        
        return result.x[0]
        
    def udata_plot(self, overlap: Any, yscale: str = 'log', left: float | None = None, right: float | None = None, save: bool = False, save_dir: str = '', save_name: str | None = None, return_fig: bool = False, show_plot: bool = True) -> Any:
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
        
        sa = Spectra([ab, uf, sp, bb])
        sa.label(['Absorptance', 'Exponential fit', 'Luminescence', 'Blackbody radiation'])
        
        graph = sa.plot(yscale = yscale, left = left, right = right, bottom = ab.y_of(left), top = ab.y_of(overlap) * 4, plotstyle = 'not auto', return_fig = return_fig, show_plot = show_plot)  # type: ignore
        
        if save:
            if save_name is None:
                filepath = 'Urbach_data.csv'
            else:
                filepath = save_name
            sa.save(save_dir, filepath) 
            
        if return_fig:
            return(graph)
            
            

            
            

def above_bg_photon_flux(bg: float, illumspec_eV: Any | None = None) -> Any:
    """
    Returns the above bandgap photon flux for an illumination Spectrum illumspec_eV. If no illumination Spectrum is given, 1 sun condition is assumed. 

    Parameters
    ----------
    bg : FLOAT
        Bandgap in eV.

    Returns
    -------
    Above bandgap photon flux in 1/[s m2].

    """
        
    if illumspec_eV is None:
        illumspec_eV = DiffSpectrum.am15_ev()
        
    return illumspec_eV.photonflux(start = bg, stop = illumspec_eV.x[-1])


def calc_laser_power(laser_nm: np.ndarray, bg_eV: Any | None = None, pf: Any | None = None, A: float | None = None, Nsun: Any | None = None, only90deg: Any = True, details: Any = True) -> Any:
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
        if Nsun is None:
            Nsun = 1
        if bg_eV is not None:
            pf = above_bg_photon_flux(bg_eV)
        if A is None:            
            #FWHM laser beam diameters
            if laser_nm == 405:
                A = pi/4 * 533.5e-6 * 715.0e-6 #m2 for 403 nm laser
            elif laser_nm == 420:
                #A = pi/4 * 418e-6 * 495e-6 #m2 for 422 nm laser, old lab CH G1 522
                A = pi/4 * 1598.87e-6*1876.48e-6 #m2 for 422 nm laser, new lab CH B1 365, measured October 2023                
            elif laser_nm == 660:
                #A = pi/4 * 693.0e-6 * 891.0e-6  #m2 for 657 nm laser, old lab CH G1 522
                A = pi/4 * 1909.56e-6 * 2329.96e-6  #m2 for 657 nm laser, new lab CH B1 365, measured October 2023       
            d = math.sqrt(4*A/pi)*1e6  # type: ignore
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

        LP = pf * PE * A / factor  # type: ignore

        if details:
            if bg_eV is not None:
                print(f'For {Nsun} sun conditions with a bandgap of {bg_eV:.3f} eV (= {f1240/bg_eV:.0f} nm) the laser power at {laser_nm} nm has to be {LP * 1000:.2e} mW (90° sample geometry).')
                if not(only90deg):
                    print(f'... at 45°: {LP * math.sqrt(2) * 1000:.2e} mW')
                print(f'The above bandgap photon flux is: {pf:.2e} 1/(s m²)')
            else:
                print(f'For {Nsun} sun conditions with a photonflux of {pf:.2e} 1/(s m2) the laser power at {laser_nm} nm has to be {LP * 1000:.2e} mW (90° sample geometry).')
                if not(only90deg):
                    print(f'... at 45°: {LP * math.sqrt(2) * 1000:.2e} mW')
        return LP
