# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:41:51 2020

@author: dreickem
"""

import math
import pathlib
from dataclasses import dataclass
from importlib.resources import files as _resource_files
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, fmin, fsolve
from scipy.special import lambertw

system_dir = str(_resource_files('fte_analysis_libraries').joinpath('System_data'))

from typing import Any

from .General import T_RT, findind, k, linfit, plx, q, save_ok
from .Spectrum import AbsSpectrum, DiffSpectrum
from .XYdata import MXYData, XYData


@dataclass
class FiveParam:
    """
    FiveParam stands for five parameters.
    Class for keeping track of the 5param of an IV curve.

    Parameters
    ----------
    cell_area : FLOAT
        cell (mask) area in cm2.
    Voc : FlOAT
        Voc in V.
    Jsc : FLOAT
        Jsc in mA/cm2 (or mA if cell_area is None)
    nid : dimensionless
        ideality factor
    Rs : FLOAT
        Series resistance in kOhm * cm2 (or kOhm if cell_area is None)
    Rsh : FLOAT
        Shunt resistnace in kOhm * cm2 (or kOhm if cell_area is None
    
    """

    cell_area: float | None = None  # type: ignore
    Voc: float | None = None  # type: ignore
    Jsc: float | None = None  # type: ignore
    nid: float | None = None  # type: ignore
    Rs: float | None = None  # type: ignore
    Rsh: float | None = None  # type: ignore

    def __init__(self, cell_area: float = 1, Voc: float = 1, Jsc: float = 20, nid: float = 1, Rs: float = 0, Rsh: float = np.inf) -> None:
        """
        Initialize the object.
        
        Parameters
        ----------
        cell_area : float
            Cell area, in cm².
        Voc : float
            Voc, in v.
        Jsc : float
            Jsc, in ma/cm².
        nid : float
            Nid.
        Rs : float
            Rs, in ω·cm².
        Rsh : float
            Rsh, in ω·cm².
        
        Examples
        --------
        >>> obj.__init__()
        """

        self.cell_area = cell_area
        self.Voc = Voc
        self.Jsc = Jsc
        self.nid = nid
        self.Rs = Rs
        self.Rsh = Rsh

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
        return FiveParam(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)  # type: ignore

@dataclass
class PerfData:
    """
    Solar-cell performance parameters dataclass (Voc, Jsc, FF, PCE, …).
    """

    cell_area: float | None = None # cm2  # type: ignore
    Vmpp: float | None = None # V  # type: ignore
    Jmpp: float | None = None # mA/cm2 (or mA if cell_area is None)  # type: ignore
    Pmpp: float | None = None # mW/cm2 (or mW if cell_area is None)  # type: ignore
    PCE: float | None = None # % (makes no sense if cell_area is None)  # type: ignore
    FF: float | None = None # %  # type: ignore
    Voc: float | None = None # V  # type: ignore
    Jsc: float | None = None # mA/cm2 (or mA if cell_area is None)  # type: ignore
    nid: float | None = None # dimension less  # type: ignore
    Rs: float | None = None # kOhm * cm2 (or kOhm if cell_area is None)  # type: ignore
    Rsh: float | None = None # kOhm * cm2 (or kOhm if cell_area is None)  # type: ignore
    light_int: float = 100 # mW/cm2

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
        return PerfData(cell_area = self.cell_area, Vmpp = self.Vmpp, Jmpp = self.Jmpp, Pmpp = self.Pmpp,
                      PCE = self.PCE, FF = self.FF, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid,
                      Rs = self.Rs, Rsh = self.Rsh, light_int = self.light_int)

    def jmpp_text(self, uA: Any = False) -> Any:
        """
        Jmpp text.
        
        Parameters
        ----------
        uA : Any
            Ua.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.jmpp_text()
        """
        if uA:
            text = f'$J_{{mpp}} = {self.Jmpp*1000:.2f} \\; \\mu A/cm^2$'  # type: ignore
        else:
            text = f'$J_{{mpp}} = {self.Jmpp:.2f} \\; mA/cm^2$'
        return text

    def vmpp_text(self) -> Any:
        """
        Vmpp text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.vmpp_text()
        """
        return f'$V_{{mpp}} = {self.Vmpp:.3f} \\; V$'

    def pmpp_text(self) -> Any:
        """
        Pmpp text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.pmpp_text()
        """
        return f'$P_{{mpp}} = {self.Pmpp:.2f} \\; mW/cm^2$'

    def pce_text(self) -> Any:
        """
        Pce text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.pce_text()
        """
        return f'$PCE = {self.PCE:.1f}\\%$'

    def ff_text(self) -> Any:
        """
        Ff text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.ff_text()
        """
        return f'$FF = {self.FF:.1f}\\%$'

    def voc_text(self) -> Any:
        """
        Voc text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.voc_text()
        """
        return f'$V_{{oc}} = {self.Voc:.3f} \\; V$'

    def jsc_text(self, uA: Any = False) -> Any:
        """
        Jsc text.
        
        Parameters
        ----------
        uA : Any
            Ua.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.jsc_text()
        """
        if uA:
            text = f'$J_{{sc}} = {self.Jsc*1000:.2f} \\; \\mu A/cm^2$'  # type: ignore
        else:
            text = f'$J_{{sc}} = {self.Jsc:.2f} \\; mA/cm^2$'
        return text

    def nid_text(self) -> Any:
        """
        Nid text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.nid_text()
        """
        return f'$n_{{id}} = {self.nid:.2f}$'

    def rs_text(self) -> Any:
        """
        Rs text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.rs_text()
        """
        return f'$R_{{s}} = {self.Rs*1e3:.2e} \\; \\Omega \\cdot cm^2$'  # type: ignore

    def rsh_text(self) -> Any:
        """
        Rsh text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.rsh_text()
        """
        return f'$R_{{sh}} = {self.Rsh*1e3:.2e} \\; \\Omega \\cdot cm^2$'  # type: ignore

    def cell_area_text(self) -> Any:
        """
        Cell area text.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.cell_area_text()
        """
        if self.cell_area is not None:
            return f'Cell area $= {self.cell_area:.5f} \\; cm^2$'
        else:
            return ''

    def light_int_text(self, uW: Any = False) -> Any:
        """
        Light int text.
        
        Parameters
        ----------
        uW : Any
            Uw.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.light_int_text()
        """
        if uW:
            return f'Light intensity $= {self.light_int*1000:.2f} \\; uW/cm^2$'
        else:
            return f'Light intensity $= {self.light_int:.2f} \\; mW/cm^2$'

    @staticmethod
    def sq_limit(bg: float, illumspec_PF_eV: Any | None = None, light_int: float = 100, show: bool = False) -> Any:
        """
        Calculates the performance data of the Shockley-Queisser limit.

        Parameters
        ----------
        bg : FLOAT
            Bandgap in eV.
        illumspec_PF_eV: DiffSpectrum
            Illumination Spectrum ((spectral photon flux as a function of photon energy in eV)) other than AM1.5GT        
            If None then AM1.5GT Spectrum is taken
        light_int: FLOAT
            If AM1.5GT Spectrum as illumination Spectrum used, then with light_int the light intensity in mW/cm2 can be chosen.
            If None than 100 mW/cm2 (1 sun) is used.
        show : BOOLEAN, optional
            If True it will show the performance data as formatted text. The default is False.
    
        Returns
        -------
        fp : instance of PerfData.
        
        Examples
        --------
        bg = 2.30 #eV
        pd = PerfData.sq_limit(bg, show = True)

        """
        fp = IVData.sq_limit(bg, illumspec_PF_eV = illumspec_PF_eV, light_int = light_int)
        x_arr = np.linspace(0, fp.Voc*1.01,  int(round((fp.Voc*1.01)/0.001)+1))
        IV = IVData.from_fp(x_arr, fp, name = '', light_int = light_int, T = T_RT, perfparam = True)
        IV.det_perfparam(show = show)
        return IV.pd


class IVData(XYData):
    """
    Current-voltage (JV) curve with parameter extraction and fitting.
    """


    def __init__(self, x: np.ndarray, y: np.ndarray, cell_area: float | None = None, light_int: float | None = None, sweep_dir: str = 'rev', name: str = '', Voc: float | None = None, Jsc: float | None = None, quants: Any = {"x": "Voltage", "y": "Current density"}, units: Any = {"x": "V", "y": "mA/cm2"}, plotstyle: str = dict(linestyle = 'None', marker = 'o', color = 'blue', markersize = 5), check_data: bool = True) -> None:  # type: ignore
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
        if hasattr(self, 'nid'):
            dat.nid = self.nid
        if hasattr(self, 'Rs'):
            dat.Rs = self.Rs
        if hasattr(self, 'Rsh'):
            dat.Rsh = self.Rsh
        if hasattr(self, 'pd'):
            dat.pd = self.pd.copy()

        return dat

    def convert_from_mA_to_uA(self) -> Any:
        """
        Convert from m A to u A.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.convert_from_mA_to_uA()
        """
        self.y *= 1000
        self.uy = 'uA/cm2'

    @staticmethod
    def from_j0(V: float, J0: Any, Jph: Any, nid: float, Rs: float, Rsh: float, cell_area: float | None = None, light_int: float | None = None, name: str = '', T: float = T_RT) -> Any:
        """
        From j 0.
        
        Parameters
        ----------
        V : float
            V.
        J0 : Any
            J0.
        Jph : Any
            Jph.
        nid : float
            Nid.
        Rs : float
            Rs, in ω·cm².
        Rsh : float
            Rsh, in ω·cm².
        cell_area : float | None
            Cell area, in cm².
        light_int : float | None
            Light int, in mw/cm².
        name : str
            Name.
        T : float
            T.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.from_j0()
        """
        Gp = 1/Rsh
        Vth = k * T / q
        expW = (V + Rs * (J0 + Jph)) / (nid * Vth * (1 + Rs * Gp))
        argW = J0 * Rs / (nid * Vth * (1 + Rs * Gp)) * np.exp(expW)
        J = nid * Vth / Rs * lambertw(argW) + (V * Gp - (J0 + Jph)) / (1 + Rs * Gp)
        J = J.real
        return IVData(V, J, cell_area = cell_area, light_int = light_int, sweep_dir = None, name = name, Voc = None, Jsc = None)  # type: ignore

    def to_fp(self) -> Any:
        """
        To five-parameter.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.to_fp()
        """
        return FiveParam(self.cell_area, self.Voc, self.Jsc, self.nid, self.Rs, self.Rsh)  # type: ignore

    def norm_to_onesun(self, mmf: Any = 1) -> Any:
        """
        Normalizes current density to one sun. Takes into account the mmf by dividing the current by mmf.

        Returns
        -------
        None.

        """
        self.y = self.y * 100 / self.light_int / mmf
        self.light_int = 100


    @staticmethod
    def load_igor_iv(directory: str, filepath: str, print_lines: Any = False) -> Any:
        """
        Function that opens an igor IV file and returns a 2-dim data array, column 1: Voltage (V), column 2: Current (A)
        """

        TFN = join(directory,filepath)

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

            print('load_igor_iv error: Data is not photocurrent data!')

        else:

            y = np.asarray(toks, dtype = np.float64) / cell_area * 1e3 #mA/cm2

            x = np.linspace(startV, startV + (len(y) - 1) * deltaV, num=len(y))

            if len(x) != len(y):
                print(filepath + ': ATTENTION: Number of lines of column voltage (' + str(len(x)) + ' lines) does not match number of lines of column current (' + str(len(y)) +' lines)!')
                x = x[:len(y)]

            # If reverse measurement reverse the order of array
            if x[1] < x[0]:
                x = x[::-1]
                y = y[::-1]
                sweep_dir = 'rev'
            else:
                sweep_dir = 'fwd'

        return IVData(x, y, cell_area = cell_area, light_int = light_int, sweep_dir = sweep_dir, name = filepath)

    @staticmethod
    def load(filepath_or_directory: str, filepath: str = '', data_format: Any = 'csv', cell_area: float = 1, light_int: float = 100, delimiter: str = ',', header: Any = 'infer', quants: Any = {"x": "Voltage", "y": "Current density"}, units: Any = {"x": "V", "y": "mA/cm2"},   # type: ignore
         take_quants_and_units_from_file: bool = False, J_1sun: Any | None = None, reverse_scan: bool = True, raw_data: np.ndarray = False, print_lines: Any = False, **kwargs) -> Any:  # type: ignore
        """
        Load.
        
        Parameters
        ----------
        filepath_or_directory : str
            Filepath or directory.
        filepath : str
            Filepath.
        data_format : Any
            Data format.
        cell_area : float
            Cell area, in cm².
        light_int : float
            Light int, in mw/cm².
        delimiter : str
            Delimiter.
        header : Any
            Header.
        quants : Any
            Quants.
        units : Any
            Units.
        take_quants_and_units_from_file : bool
            Take quants and units from file.
        J_1sun : Any | None
            J 1sun.
        reverse_scan : bool
            Reverse scan.
        raw_data : np.ndarray
            Raw data.
        print_lines : Any
            Print lines.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.load()
        """

        fp = join(filepath_or_directory, filepath)
        filepath = pathlib.Path(fp)  # type: ignore
        directory = dirname(fp)
        if data_format == 'Igor':
            IV = IVData.load_igor_iv(directory, filepath = filepath, print_lines = print_lines)
        elif data_format == 'csv':
            xy = XYData.load(filepath_or_directory, filepath = filepath, delimiter = delimiter, header = header, quants = quants, units = units,
                                   take_quants_and_units_from_file = take_quants_and_units_from_file, **kwargs)
            IV = IVData(xy.x, xy.y, cell_area = cell_area, light_int = light_int, sweep_dir = None, name = filepath.stem, Voc = None, Jsc = None, quants = {"x": "Voltage", "y": "Current density"}, units = {"x": "V", "y": "mA/cm2"}, **kwargs)  # type: ignore
            #IV.y = -IV.y
        elif data_format == 'Biologic-CV':
            IV = IVData.load_Biologic_CV(directory, filepath = filepath, cell_area = cell_area, light_int = light_int, J_1sun = J_1sun, reverse_scan = reverse_scan, raw_data = raw_data)  # type: ignore

        return IV

    def det_j0(self, T: float = T_RT) -> Any:
        """
        Determine j 0.
        
        Parameters
        ----------
        T : float
            T.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.det_j0()
        """
        Jsc = self.Jsc
        Voc = self.Voc
        Rs = self.Rs
        Rsh = self.Rsh
        nid = self.nid
        return (Jsc + (Rs*Jsc-Voc)/Rsh) * math.exp(-q*Voc / (nid * k * T)) / (1 - math.exp(q * (Rs * Jsc - Voc) / (nid * k * T)))  # type: ignore


    @staticmethod
    def _linear_extrapolate(x_target: Any, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Estimate y at x_target by linear interpolation or extrapolation."""
        order = np.argsort(x_data)
        x_data, y_data = x_data[order], y_data[order]
        if x_target <= x_data[0]:
            i1, i2 = 0, 1
        elif x_target >= x_data[-1]:
            i1, i2 = -2, -1
        else:
            i2 = np.searchsorted(x_data, x_target)
            i1 = i2 - 1
        x1, x2 = x_data[i1], x_data[i2]
        y1, y2 = y_data[i1], y_data[i2]
        return y1 + (y2 - y1) / (x2 - x1) * (x_target - x1)

    @staticmethod
    def _format_nid_rs_rsh_label(nid: float, Rs: float, Rsh: float) -> None:
        """
        Format ideality factor series resistance shunt resistance label.
        
        Parameters
        ----------
        nid : float
            Nid.
        Rs : float
            Rs, in ω·cm².
        Rsh : float
            Rsh, in ω·cm².
        
        Examples
        --------
        >>> obj._format_nid_rs_rsh_label()
        """
        return f'$n_{{id}} = {nid:.2f}, \\ R_s = {Rs*1e3:.2e} \\ \\Omega \\cdot cm^2, \\ R_{{sh}} = {Rsh*1e3:.2e} \\ \\Omega \\cdot cm^2$'  # type: ignore

    def det_voc(self, use_interpolate_extrapolate_method: bool=True) -> Any:
        """
        Determine and store the open-circuit voltage Voc in V.
        """

        if use_interpolate_extrapolate_method:
            # Compute Voc (at J = 0) by treating axes as swapped
            self.Voc = IVData._linear_extrapolate(0.0, self.y, self.x)

        else:
            JVinterp = interp1d(self.x, self.y, kind='cubic', bounds_error=False, fill_value='extrapolate')
            Voc = fsolve(JVinterp,.95*max(self.x))[0]
            self.Voc = Voc
        return self.Voc

    def det_jsc(self, fit_to: Any | None= None, show_fit: bool= False, use_interpolate_extrapolate_method: bool= True) -> Any:
        """
        Determine and store the short-circuit current density Jsc in mA/cm².
        fit_to: specifies the voltage up to which the IV curve is fitted. If None then the maximum V / 10 is taken.
        show_fit: if True the IV curve and the fit from which Jsc is taken is plotted.
        use_interpolate_extrapolate_method: Use interpolation/extrapolation method to determine Jsc
        """

        if use_interpolate_extrapolate_method:
            # Compute Jsc (at V = 0)
            self.Jsc = -IVData._linear_extrapolate(0.0, self.x, self.y)  # type: ignore
        else:
            if fit_to is None:
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

        return self.Jsc

    def ini_guess_rsh(self, fit_to: Any | None = None, show_fit: bool = False) -> None:
        """
        Estimate the shunt resistance Rsh from the JV slope near 0 V.
        fit_to: specifies the voltage up to which the IV curve is fitted. If None then the maximum V / 5 is taken.
        show_fit: if True the IV curve and the fit from which Rsh is taken is plotted.
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 11
        """
        if fit_to is None:
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
            plt.plot(self.x, self.x / self.Rsh - self.Jsc, '-', label = f'fit (Rsh = {self.Rsh*1e3:.2e} Ohm cm2)')  # type: ignore
            plt.ylim(-self.Jsc*1.2, -self.Jsc*0.8)  # type: ignore
            plt.legend()
            plt.show()

        return Rsh  # type: ignore

    def ini_guess_nid_and_rs(self, show_fit: bool = False) -> Any:
        """
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 12
        """
        dIdV = np.gradient(self.y, self.x)

        r = range(findind(self.x, self.Voc*0.95), findind(self.x, self.Voc*1.05)+1)  # type: ignore
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

    def check_assumption(self, T: float = T_RT) -> Any:
        """
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 6
        """
        delta = np.exp(q * (self.Rs * self.Jsc - self.Voc) / (k * T))
        if delta > 0.1:
            text1 = "IVData.check_assumption: \n"
            text2 = 'Attention: The assumption $\\Delta \\ll 1$ (eq. 6) is not satisfied!'
            plx(text1+text2)
        return delta

    def det_ini_5param(self, show_fit: bool = False) -> Any:
        """
        Determine ini 5 parameters.
        
        Parameters
        ----------
        show_fit : bool
            Show fit.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.det_ini_5param()
        """
        self.det_voc()
        self.det_jsc(fit_to = None, show_fit = show_fit)
        self.ini_guess_rsh(fit_to = None, show_fit = show_fit)
        self.ini_guess_nid_and_rs(show_fit = show_fit)
        self.check_assumption()


    def det_perfparam(self, show: bool= False, uA: Any= False, uW: Any= False, minimal: Any= False, simple_calc: Any= False, use_interpolate_extrapolate_method: bool= True) -> None:
        """
        Extract performance parameters (Voc, Jsc, FF, PCE, Vmpp, Jmpp).
        
        Parameters
        ----------
        show : bool
            Show.
        uA : Any
            Ua.
        uW : Any
            Uw.
        minimal : Any
            Minimal.
        simple_calc : Any
            Simple calc.
        use_interpolate_extrapolate_method : bool
            Use interpolate extrapolate method.
        
        Examples
        --------
        >>> obj.det_perfparam()
        """
        #uA: calculate currents in uA/cm2
        #uW: show light intensity in uW/cm2
        #If minimal: only Voc, Jsc, FF, Vmpp, Jmpp and PCE is determined
        #minimal: only determine Vmpp, Jmpp, Pmpp, PCE, FF
        #simple_calc: calculate Jsc and Voc in the simplest way , can be done later
        if self.Voc is None:
            self.det_voc(use_interpolate_extrapolate_method=use_interpolate_extrapolate_method)
        if self.Jsc is None:
            self.det_jsc(use_interpolate_extrapolate_method=use_interpolate_extrapolate_method)
        JVinterp = interp1d(self.x, self.y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        Vmpp = fmin(lambda x: x*JVinterp(x),.8*self.Voc,disp=False, maxiter = 100)[0]  # type: ignore
        Jmpp = abs(JVinterp(Vmpp))
        Pmpp = abs(Vmpp*Jmpp)
        PCE = Pmpp / self.light_int * 100
        FF = abs(Pmpp/(self.Jsc*self.Voc) * 100)  # type: ignore
        if not(minimal) and not(hasattr(self, 'nid')):
            self.fit_param()
        if not(hasattr(self, 'light_int')):
            self.light_int = 100
        if not minimal:
            pd = PerfData(cell_area = self.cell_area, Vmpp = Vmpp, Jmpp = Jmpp, Pmpp = Pmpp,   # type: ignore
                          PCE = PCE, FF = FF, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid,   # type: ignore
                          Rs = self.Rs, Rsh = self.Rsh, light_int = self.light_int)  # type: ignore
            text = pd.light_int_text(uW) + ', ' + pd.vmpp_text() + ', ' + pd.jmpp_text(uA) + '\n' + pd.voc_text() + ', ' + pd.jsc_text(uA) + ', ' + pd.ff_text() + ', ' + pd.pce_text() + '\n' + pd.nid_text() + ', ' + pd.rs_text() + ', ' + pd.rsh_text() + '\n' + pd.cell_area_text()

        else:
            pd = PerfData(cell_area = self.cell_area, Vmpp = Vmpp, Jmpp = Jmpp, Pmpp = Pmpp,   # type: ignore
                          PCE = PCE, FF = FF, Voc = self.Voc, Jsc = self.Jsc, light_int = self.light_int)              # type: ignore
            text = pd.light_int_text(uW) + ', ' + pd.vmpp_text() + ', ' + pd.jmpp_text(uA) + '\n' + pd.voc_text() + ', ' + pd.jsc_text(uA) + ', ' + pd.ff_text() + ', ' + pd.pce_text() + '\n' + pd.cell_area_text()


        self.pd = pd
        if show:
            plx(text)


    def fit_param(self, T: float = T_RT, bounds: Any = ([0, 0, 0], [10, np.inf, np.inf]), p0: Any | None = None, verbose: bool = 0, xtol: Any | None = None) -> Any:  # type: ignore
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

        def func(V_arr: np.ndarray, nid: float, Rs: float, Rsh: float) -> Any:
            #J = np.array([IVData.i_of_v(self.x[i], self.Jsc, self.Voc, nid, Rs, Rsh, T = T) for i in range(len(self.x))])
            J = np.array([IVData.i_of_v(V_arr[i], self.Jsc, self.Voc, nid, Rs, Rsh, T = T) for i in range(len(V_arr))])  # type: ignore
            return J
        #IVData.i_of_v(V, Isc, Voc, n, Rs, Rsh, T = T_RT)

        if p0 is None:
            if not(hasattr(self, 'nid')) or not(hasattr(self, 'Rs')) or not(hasattr(self, 'Rsh')):
                self.det_ini_5param()
            p0 = [self.nid, self.Rs, self.Rsh]
            bounds = ([1, 0, 0], [10, np.inf, np.inf])

        popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds, verbose = verbose, xtol = xtol)
        #popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds)

        # store the initial 5 parameters in ini_fp
        self.nid = popt[0]
        self.Rs = popt[1]
        self.Rsh = popt[2]
        self.ini_fp = FiveParam(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)  # type: ignore

        return popt


    def fit_fivep(self, T: float = T_RT, bounds: Any = ([0, 0, 0, 0, 0], [100, 10, 10, np.inf, np.inf]), p0: Any | None = None) -> Any:
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

        def func(V: float, Jsc: float, Voc: float, nid: float, Rs: float, Rsh: float) -> Any:
            J = np.array([IVData.i_of_v(V[i], Jsc, Voc, nid, Rs, Rsh, T = T) for i in range(len(V))])  # type: ignore
            return J

        if p0 is None:
            if self.x[1] < self.x[0]:
                print('Attention [IV-data.fit_fivep()]: Voltage array (self.x) is not strictly ascending!')
            p0 = [abs(self.y[0]), self.x[-1], 1.5, 0, 1e18]


        popt, pcov = curve_fit(func, self.x, self.y, p0 = p0, bounds = bounds, verbose = 0, xtol = None)

        # store the initial 5 parameters in ini_fp
        self.ini_fp = FiveParam(cell_area = self.cell_area, Voc = p0[1], Jsc = p0[0], nid = p0[2], Rs = p0[3], Rsh = p0[4])  # type: ignore

        self.Jsc = popt[0]
        self.Voc = popt[1]
        self.nid = popt[2]
        self.Rs = popt[3]
        self.Rsh = popt[4]

        return popt



    def get_fp(self) -> Any:
        """
        Get five-parameter.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.get_fp()
        """
        return FiveParam(cell_area = self.cell_area, Voc = self.Voc, Jsc = self.Jsc, nid = self.nid, Rs = self.Rs, Rsh = self.Rsh)  # type: ignore

    def table_param(PCE: float, Voc: float, Jsc: float, FF: float, Vmpp: Any, Jmpp: Any, light_int: float, cell_area: float) -> Any:  # type: ignore
        """
        Table parameters.
        
        Parameters
        ----------
        PCE : float
            Pce, in %.
        Voc : float
            Voc, in v.
        Jsc : float
            Jsc, in ma/cm².
        FF : float
            Ff.
        Vmpp : Any
            Vmpp, in v.
        Jmpp : Any
            Jmpp, in ma/cm².
        light_int : float
            Light int, in mw/cm².
        cell_area : float
            Cell area, in cm².
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.table_param()
        """

        row_labels = ['$PCE\\ (\\%)$', '$V_{OC}\\ (V)$', '$J_{SC}\\ (mA/cm^2)$', '$FF\\ (\\%)$',
                      '$V_{mpp} \\ (V)$', '$J_{mpp} \\ (mA/cm^2)$',
                      '$Light \\ int \\ (mW/cm^2)$', '$Cell \\ area \\ (cm^2)$']
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

    def plot(self, title: str = 'self.name', xscale: str = 'linear', yscale: str = 'linear',   # type: ignore
             left: float | None = None, right: float | None = None, bottom: float | None = None, top: float | None = None, plot_table: bool = False, hline: Any | None = None, vline: Any | None = None, figsize: Any=(9,6), return_fig: bool = False, show_plot: bool = True, **kwargs) -> None:
        """
        Plot.
        
        Parameters
        ----------
        title : str
            Title.
        xscale : str
            Xscale.
        yscale : str
            Yscale.
        left : float | None
            Left.
        right : float | None
            Right.
        bottom : float | None
            Bottom.
        top : float | None
            Top.
        plot_table : bool
            Plot table.
        hline : Any | None
            Hline.
        vline : Any | None
            Vline.
        figsize : Any
            Figsize.
        return_fig : bool
            Return fig.
        show_plot : bool
            Show plot.
        
        Examples
        --------
        >>> obj.plot()
        """

        if plot_table:
            cell_text, row_labels = IVData.table_param(self.pd.PCE, self.Voc, self.Jsc, self.pd.FF,   # type: ignore
                                                        self.pd.Vmpp, self.pd.Jmpp, self.light_int, self.cell_area)  # type: ignore
        else:
            cell_text = None
            row_labels = None

        if title is None:
            title = self.name

        fig = XYData.plot(self, title = title, xscale = xscale, yscale = yscale, left = left, right = right,   # type: ignore
                           bottom = bottom, top = top, plot_table = plot_table, cell_text = cell_text, row_labels = row_labels, hline = hline, vline = vline, figsize = figsize, return_fig = return_fig, show_plot = show_plot, **kwargs)
        if return_fig:
            return fig


    def plot_fit(self, xscale: str = 'linear', yscale: str = 'linear', left: float | None = None, right: float | None = None, bottom: float | None = None, top: float | None = None,  title: str | None = None, plot_table: bool = False) -> Any:
        """
        Plot the JV curve with the five-parameter model fit overlaid.
        """
        J = np.array([IVData.i_of_v(self.x[i], self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh, T = T_RT) for i in range(len(self.x))])  # type: ignore
        IVp = IVData(self.x, J, name=IVData._format_nid_rs_rsh_label(self.nid, self.Rs, self.Rsh))  # type: ignore
        IVp.plotstyle = dict(linestyle = '-', color = 'red')
        mIV = MIVData([self, IVp])
        mIV.label(['Measurement', IVp.name])

        if plot_table:
            cell_text, row_labels = IVData.table_param(self.pd.PCE, self.Voc, self.Jsc, self.pd.FF,   # type: ignore
                                                        self.pd.Vmpp, self.pd.Jmpp, self.light_int, self.cell_area)  # type: ignore
        else:
            cell_text = None
            row_labels = None

        if title is None:
            title = self.name

        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, left = left, right = right, bottom = bottom, top = top,
                 title = title, plot_table = plot_table, cell_text = cell_text, row_labels = row_labels)

    def plot_ini_and_fit(self, xscale: str = 'linear', yscale: str = 'linear', bottom: float | None = None, top: float | None = None) -> Any:
        """
        Plot JV data with both initial guess and optimised five-parameter fit.
        """
        J_int = np.array([IVData.i_of_v(self.x[i], self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh, T = T_RT) for i in range(len(self.x))])  # type: ignore
        J_ini = np.array([IVData.i_of_v(self.x[i], self.ini_fp.Jsc, self.ini_fp.Voc, self.ini_fp.nid, self.ini_fp.Rs, self.ini_fp.Rsh, T = T_RT) for i in range(len(self.x))])  # type: ignore
        IV_int = IVData(self.x, J_int, name='Fitted parameters:\n' + IVData._format_nid_rs_rsh_label(self.nid, self.Rs, self.Rsh))  # type: ignore
        IV_ini = IVData(self.x, J_ini, name='Initial parameters:\n' + IVData._format_nid_rs_rsh_label(self.ini_fp.nid, self.ini_fp.Rs, self.ini_fp.Rsh))  # type: ignore
        IV_int.plotstyle = dict(linestyle = '-', color = 'red')
        IV_ini.plotstyle = dict(linestyle = '-', color = 'green')
        mIV = MIVData([self, IV_ini, IV_int])
        #mIV.names_to_label()
        mIV.label(['Measurement', IV_ini.name, IV_int.name])
        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, bottom = bottom, top = top, title = self.name)

    @staticmethod
    def save_perf_data(sa: Any, row_labels: Any, col_labels: Any, save_dir: str, filepath: str) -> Any:
        """
        Save performance data.
        
        Parameters
        ----------
        sa : Any
            Sa.
        row_labels : Any
            Row labels.
        col_labels : Any
            Col labels.
        save_dir : str
            Save dir.
        filepath : str
            Filepath.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.save_perf_data()
        """

        cell_text = []

        for IV in sa:
            cell_text.append([IV.Voc, IV.Jsc, IV.pd.FF, IV.pd.PCE])

        alldata = np.array(cell_text, dtype = np.float64)

        df = pd.DataFrame(data=alldata[0:,0:], columns = col_labels, index = row_labels)

        TFN = join(save_dir, filepath)
        if save_ok(TFN):
            df.to_csv(join(save_dir, filepath), header = True, index = True)

    @staticmethod
    def iv_sq(bg: float, x_max: Any | None = None, light_int: float | None = None) -> Any:
        """
        Iv Shockley-Queisser.
        
        Parameters
        ----------
        bg : float
            Bg.
        x_max : Any | None
            X max.
        light_int : float | None
            Light int, in mw/cm².
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.iv_sq()
        """
        if light_int is None:
            fp_sq = IVData.sq_limit(bg)
        else:
            fp_sq = IVData.sq_limit(bg, light_int = light_int)
        if (x_max is None) or (x_max < (fp_sq.Voc+0.01)):
            #x_max = max(self.x)
            x_max = fp_sq.Voc + 0.01
        new_x = np.arange(0, x_max, step = 0.001, dtype = np.float64)
        iv_sq = IVData.from_fp(new_x, fp_sq, name = f'SQ limit ($E_g = {bg:.3f} \\ eV): \\ V_{{oc,SQ}} = {fp_sq.Voc:.3f} \\ V, \\ J_{{sc,SQ}} = {fp_sq.Jsc:.2f} \\ mA/cm^2, \\ n_{{id}} = 1,\\ R_s = 0,\\ R_{{sh}} = \\infty$')
        iv_sq.det_perfparam()
        return iv_sq

    @staticmethod
    def iv_rad(Vocrad: Any, Jsc: float, light_int: float = 100, x_max: Any | None = None) -> Any:
        """
        Iv rad.
        
        Parameters
        ----------
        Vocrad : Any
            Vocrad.
        Jsc : float
            Jsc, in ma/cm².
        light_int : float
            Light int, in mw/cm².
        x_max : Any | None
            X max.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.iv_rad()
        """
        if (x_max is None) or (x_max < Vocrad):
            x_max = Vocrad
        step = 0.001
        new_x = np.arange(0, x_max+2*step, step= step, dtype= np.float64)
        Jrad = np.array([IVData.i_of_v(new_x[i], Jsc, Voc = Vocrad, nid = 1, Rs = 0, Rsh = 1e15, T = T_RT) for i in range(len(new_x))])
        iv_rad = IVData(new_x, Jrad, light_int = light_int, name = f'Radiative limit: $V_{{oc,rad}} = {Vocrad:.3f} \\ V, \\ J_{{sc,rad}} = {Jsc:.2f} \\ mA/cm^2, \\ n_{{id}} = 1,\\ R_s = 0,\\ R_{{sh}} = \\infty$')
        iv_rad.det_perfparam()
        return iv_rad

    @staticmethod
    def iv_trans(V_arr: np.ndarray, Voc: float, Jsc: float, nid_rec: Any, light_int: float = 100) -> Any:
        """
        Iv trans.
        
        Parameters
        ----------
        V_arr : np.ndarray
            V arr.
        Voc : float
            Voc, in v.
        Jsc : float
            Jsc, in ma/cm².
        nid_rec : Any
            Nid rec.
        light_int : float
            Light int, in mw/cm².
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.iv_trans()
        """
        Jtrans = np.array([IVData.i_of_v(V_arr[i], Jsc, Voc = Voc, nid = nid_rec, Rs = 0, Rsh = 1e10, T = T_RT) for i in range(len(V_arr))])
        iv_trans = IVData(V_arr, Jtrans, light_int = light_int, name = f'Transport limit: $V_{{oc}} = {Voc:.3f} \\ V, \\ J_{{sc,trans}} = {Jsc:.2f} \\ mA/cm^2, \\ n_{{id}} = {nid_rec:.2f},\\ R_s = 0,\\ R_{{sh}} = \\infty$')
        iv_trans.det_perfparam()
        return iv_trans


    def loss_plot(self, bg: float | None = None, Vocrad: Any | None = None, nid_rec: Any | None = None, x_max: Any | None = None, iv_sq: Any | None = None, iv_rad: Any | None = None, iv_trans: Any | None = None, IVfit: Any | None = None, title: str | None = None, xscale: str = 'linear', yscale: str = 'linear',
                  left: float | None = None, right: float | None = None, bottom: float | None = None, top: float | None = None, what_to_show: Any = ['measurement', 'fit', 'SQ limit', 'rad. limit', 'transp. limit'],
                  show_legend_details: bool= False, plot_table: bool = False, figsize: Any=(12,8), save: bool = False, save_dir: str = '', bbox: Any = [0.15, 0.25, 0.5, 0.3], **kwargs) -> Any:
        """
        Plot the IV curve, the fit with the internal 5 parameters and the fit with the ini_fp parameters.
        """
        def print5param(Jsc: float, Voc: float, nid: float, Rs: float, Rsh: float) -> Any:
            #text = f'Voc = {Voc:.3f} V, Jsc = {Jsc:.2f} mA/cm2' '\n' f'nid = {nid:.2f}, Rs = {Rs:.2e} Ohm cm2, Rsh = {Rsh:.2e} Ohm cm2'
            text = f'Voc = {Voc:.3f} V, Jsc = {Jsc:.2f} mA/cm$^2$, $n_{{id}} = {nid:.2f}, \\ R_s = {Rs*1e3:.2e} \\ \\Omega \\cdot cm^2, \\ R_{{sh}} = {Rsh*1e3:.2e} \\ \\Omega \\cdot cm^2$'
            return text

        #fp_sq = IVData.sq_limit(bg)

        if x_max is None:
            x_max = max(self.x)

        new_x = np.arange(0, x_max, step = 0.001, dtype = np.float64)

        linewidth = 5
        if 'linewidth' in kwargs:
            linewidth = kwargs['linewidth']

        if iv_sq is None:
            if self.light_int == 100:
                iv_sq = IVData.iv_sq(bg, x_max = x_max)  # type: ignore
            else:
                iv_sq = IVData.iv_sq(bg, x_max = x_max, light_int = self.light_int)  # type: ignore
                print(f'Attention: light intensity {self.light_int:.1e} mW/cm2 is used to calculate Shockley-Queisser limit!')
        iv_sq.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'black')

        if iv_rad is None:
            iv_rad = IVData.iv_rad(Vocrad, self.Jsc, self.light_int, x_max = Vocrad + 0.01)  # type: ignore
        iv_rad.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'green')

        if iv_trans is None:
            iv_trans = IVData.iv_trans(self.x, self.Voc, self.Jsc, nid_rec, light_int = self.light_int)  # type: ignore
        iv_trans.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'cyan')

        if IVfit is None and 'fit' in what_to_show:
            Jfit = np.array([IVData.i_of_v(new_x[i], self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh, T = T_RT) for i in range(len(new_x))])  # type: ignore
            IVfit = IVData(new_x, Jfit, light_int = self.light_int, name = 'Fit: ' + print5param(self.Jsc, self.Voc, self.nid, self.Rs, self.Rsh))  # type: ignore
            IVfit.det_perfparam()
            IVfit.plotstyle = dict(linestyle = '-', linewidth = linewidth, color = 'red')

        #iv_sq.det_perfparam()
        #iv_rad.det_perfparam()
        #iv_trans.det_perfparam()

        show_this = []
        lab = []
        if 'measurement' in what_to_show:
            show_this.append(self)
            lab.append('Measurement')
        if 'fit' in what_to_show:
            show_this.append(IVfit)  # type: ignore
            if show_legend_details:
                lab.append(IVfit.name)  # type: ignore
            else:
                lab.append('fit')
        if 'SQ limit' in what_to_show:
            show_this.append(iv_sq)
            if show_legend_details:
                lab.append(iv_sq.name)
            else:
                lab.append('SQ limit')
        if 'rad. limit' in what_to_show:
            show_this.append(iv_rad)
            if show_legend_details:
                lab.append(iv_rad.name)
            else:
                lab.append('Radiative limit')
        if 'transp. limit' in what_to_show:
            show_this.append(iv_trans)
            if show_legend_details:
                lab.append(iv_trans.name)
            else:
                lab.append('Transport limit')

        mIV = MIVData(show_this)
        #mIV.names_to_label()
        mIV.label(lab)
        if 'legend' in kwargs:
            if kwargs['legend'] == False:
                mIV.no_label()


        row_labels = ['SQ lim.', 'Rad. lim.', 'Transp. lim.', 'Meas.']
        col_labels = ['Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'PCE (%)']

        if save:

            filepath = f'{title} - performance_data.csv'

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

            IVData.save_perf_data(mIV.sa, row_labels, col_labels, save_dir, filepath)
            mIV.save(save_dir, title, row_labels)  # type: ignore

        if plot_table:

            cell_text = []
            row_labels = []

            if 'SQ limit' in what_to_show:
                cell_text.append([f'{iv_sq.Voc:.3f}', f'{iv_sq.Jsc:.2f}', f'{iv_sq.pd.FF:2.1f}', f'{iv_sq.pd.PCE:2.1f}'])
                row_labels.append('SQ lim.')
            if 'rad. limit' in what_to_show:
                cell_text.append([f'{iv_rad.Voc:.3f}', f'{iv_rad.Jsc:.2f}', f'{iv_rad.pd.FF:2.1f}', f'{iv_rad.pd.PCE:2.1f}'])
                row_labels.append('Rad. lim.')
            if 'transp. limit' in what_to_show:
                cell_text.append([f'{iv_trans.Voc:.3f}', f'{iv_trans.Jsc:.2f}', f'{iv_trans.pd.FF:2.1f}', f'{iv_trans.pd.PCE:2.1f}'])
                row_labels.append('Transp. lim.')
            if 'measurement' in what_to_show:
                cell_text.append([f'{self.Voc:.3f}', f'{self.Jsc:.2f}', f'{self.pd.FF:2.1f}', f'{self.pd.PCE:2.1f}'])
                row_labels.append('Meas.')


        else:
            cell_text = None
            row_labels = None  # type: ignore
            col_labels = None  # type: ignore

        if title is None:
            title = self.name

        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, bottom = bottom, left = left, right = right, top = top,
                 title = title, plot_table = plot_table, cell_text = cell_text, row_labels = row_labels,
                 col_labels = col_labels, bbox = bbox, figsize = figsize, **kwargs)

        return iv_sq, iv_rad, iv_trans


    @staticmethod
    def loss_barplot(IV1: Any, IVtrans1: Any, IVrad1: Any, IVsq1: Any, IV2: Any, IVtrans2: Any, IVrad2: Any, IVsq2: Any, sample_names: Any = ['Control', 'Treated'], save: bool = False, save_dir: str | None = None, filepath: str | None = None) -> Any:
        """
        Loss barplot.
        
        Parameters
        ----------
        IV1 : Any
            Iv1.
        IVtrans1 : Any
            Ivtrans1.
        IVrad1 : Any
            Ivrad1.
        IVsq1 : Any
            Ivsq1.
        IV2 : Any
            Iv2.
        IVtrans2 : Any
            Ivtrans2.
        IVrad2 : Any
            Ivrad2.
        IVsq2 : Any
            Ivsq2.
        sample_names : Any
            Sample names.
        save : bool
            Save.
        save_dir : str | None
            Save dir.
        filepath : str | None
            Filepath.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.loss_barplot()
        """

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
        labels = ['$V_{{OC,SQ}} \\rightarrow V_{{OC,rad}}$', '$J_{SC,SQ} \\rightarrow J_{SC,rad}$', '$V_{OC,rad} \\rightarrow V_{OC,TL}$', '$FF_{rad} \\rightarrow FF_{TL}$', '$FF_{TL} \\rightarrow FF_{meas.}$']


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

            TFN = join(save_dir, filepath)  # type: ignore
            if save_ok(TFN):
                df.to_csv(join(save_dir, filepath), header = True, index = True)  # type: ignore


    def plot_fp(self, fp: Any, title: str | None = None, xscale: str = 'linear', yscale: str = 'linear', bottom: float | None = None, top: float | None = None) -> Any:
        """
        Plot the IV curve and the fit with the fp 5 parameters.
        """
        J = np.array([IVData.i_of_v(self.x[i], fp.Jsc, fp.Voc, fp.nid, fp.Rs, fp.Rsh, T = T_RT) for i in range(len(self.x))])
        IVp = IVData(self.x, J, name=IVData._format_nid_rs_rsh_label(fp.nid, fp.Rs, fp.Rsh))  # type: ignore
        IVp.plotstyle = dict(linestyle = '-', color = 'red')
        mIV = MIVData([self, IVp])
        #mIV.names_to_label()
        mIV.label(['Measurement', IVp.name])
        mIV.plot(plotstyle = 'individual', xscale = xscale, yscale = yscale, bottom = bottom, top = top, title = self.name)


    @staticmethod
    def from_fp(x_arr: np.ndarray, fp: Any, name: str = '', light_int: float = 100, T: float = T_RT, perfparam: Any = True) -> Any:
        """
        Returns an instance of IVData from the five parameters in fp.

        Parameters
        ----------
        x_arr : numpy array
            array of Voltage in V.
        fp : instance of FiveParam.
            contains the five parameters Voc, Jsc, nid, Rs, Rsh (and cell_area).
        name: string
            Name of the curve.
        light_int : float, optional
            Light intensity in mW/cm2. The default is 100.
        T : float, optional
            Cell temperature in K. The default is T_RT.

        Returns
        -------
        IV : Instance of IVData
            
        """
        J = np.array([IVData.i_of_v(x_arr[i], fp.Jsc, fp.Voc, fp.nid, fp.Rs, fp.Rsh, T = T) for i in range(len(x_arr))])
        IV = IVData(x_arr, J, light_int = light_int, name = name)
        if perfparam:
            IV.det_ini_5param()
            IV.det_perfparam()
        if fp.cell_area is not None:
            IV.cell_area = fp.cell_area
        return IV


    @staticmethod
    def i_of_v(V: float, Isc: Any, Voc: float, nid: float, Rs: float, Rsh: float, T: float = T_RT) -> Any:
        """
        Computes the I-V curve using an analytical LambertW-based model.
        See: Zhang et al., J. of Appl. Phys. 110, 064504 (2011) --> Eq. 8

            V   : terminal voltage  [V]
            Isc : short-circuit current density [mA/cm2 or mA if cell_area is None]
            Voc : open-circuit voltage [V]
            nid : ideality factor   [-]
            Rs  : series resistance [kOhm cm2]
            Rsh : shunt resistance  [kOhm cm2]  (use np.inf if absent)
            T   : temperature       [K]

        Returns
        -------
        I : array
            Current [mA/cm2] (NumPy array, same shape as V).
        """
        if Rs * Isc > 700:
            print(f'i_of_v error: The factor Rs * Isc is {Rs*Isc:.1e}, i.e. > 700, np.exp(Rs*Isc) is too large, decrease Rs (in kOhm cm2) to below {700/Isc:.0f}!')
        if Rs == 0:
            Rs = 1e-10
        if Rsh == np.inf:
            Rsh = 1e20
        argW = q * Rs / (nid * k * T) * (Isc - Voc / (Rs + Rsh)) * math.exp(- q * Voc / (nid * k * T)) * math.exp(q / (nid * k * T) * (Rs * Isc +  Rsh * V / (Rsh + Rs)))
        I = nid * k * T / (q * Rs) * lambertw(argW) + V / Rs - Isc - Rsh * V / (Rs * (Rsh + Rs))
        return I.real


    @staticmethod
    def i_of_v_safe(V: float, Isc: Any, Voc: float, nid: float, Rs: float, Rsh: float, T: float=T_RT) -> Any:
        """
        Computes the I-V curve using an analytical LambertW-based model.
        See: Zhang et al., J. Appl. Phys. 110, 064504 (2011), Eq. 8
        This is a safer version to the previous one but much slower (factor ~8!).

        """
        V = np.asarray(V, dtype=np.float64)  # type: ignore

        EPS = 1e-12                                # numerical floor
        Rs  = np.clip(Rs,  EPS,  np.inf)           # avoid 0
        Rsh = np.clip(Rsh, EPS, 1e20)              # ∞ ↦ big number

        # ===== helper factors =====================================================
        beta = q / (nid * k * T)                   # q/(n k T)
        exp1 = np.exp(np.clip(-beta * Voc, -700, 700))

        exp2_arg = beta * (Rs * Isc + Rsh * V / (Rsh + Rs))
        exp2      = np.exp(np.clip(exp2_arg, -700, 700))

        # ===== Lambert‑W argument (clip to real domain) ===========================
        argW = (q * Rs / (nid * k * T)) * (Isc - Voc / (Rs + Rsh)) * exp1 * exp2
        argW = np.maximum(argW, -1/np.e)           # keep inside real branch

        W = lambertw(argW)                         # SciPy’s vectorised version

        # ===== final current ======================================================
        I = (nid * k * T / (q * Rs)) * W + V / Rs - Isc \
            - Rsh * V / (Rs * (Rsh + Rs))

        # replace NaN / ±Inf by large finite numbers so least_squares stays happy
        return np.nan_to_num(I.real, nan=1e10, posinf=1e10, neginf=-1e10)



    @staticmethod
    def sq_limit_voc(bg: float, illumspec_PF_eV: Any | None = None, light_int: float | None = None, from_file: str = True) -> Any:  # type: ignore
        """
        Sq limit open-circuit voltage.
        
        Parameters
        ----------
        bg : float
            Bg.
        illumspec_PF_eV : Any | None
            Illumspec pf ev, in ev.
        light_int : float | None
            Light int, in mw/cm².
        from_file : str
            From file.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.sq_limit_voc()
        """

        if from_file:
            filepath = 'Vocsq.csv'
            d = XYData.load(system_dir, filepath, take_quants_and_units_from_file = True)
            Vocsq = d.y_of(bg, interpolate = True)
        else:
            if illumspec_PF_eV is None:
                am15_ev = DiffSpectrum.am15_ev()
                if light_int is not None:
                    am15_ev.y = am15_ev.y * light_int/100
                illumspec_PF_eV = am15_ev

            left = min(illumspec_PF_eV.x)
            right = max(illumspec_PF_eV.x)
            eV_arr = np.arange(left, right, 0.001)
            #Absorptance Spectrum
            ab = AbsSpectrum(eV_arr, np.array([0 if eV < bg else 1 for eV in eV_arr]))
            ab.calc_jradlim(illumspec_PF_eV, start_eV = bg, handover_eV = bg+0.1)
            ab.calc_vocrad(E_start = bg-0.3, E_stop = bg+0.3, T = T_RT, show_table = False)
            Vocsq = ab.Vocrad

        return Vocsq

    @staticmethod
    def sq_limit_jsc(bg: float, illumspec_PF_eV: Any | None = None, light_int: float | None = None) -> Any:
        """
        Sq limit short-circuit current density.
        
        Parameters
        ----------
        bg : float
            Bg.
        illumspec_PF_eV : Any | None
            Illumspec pf ev, in ev.
        light_int : float | None
            Light int, in mw/cm².
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.sq_limit_jsc()
        """

        if illumspec_PF_eV is None:
            AM15 = DiffSpectrum.am15_ev()
            if light_int is not None:
                AM15.y = AM15.y * light_int/100
            illumspec_PF_eV = AM15

        # Calculation of above bandgap photon flux
        abpf = illumspec_PF_eV.photonflux(start = bg, stop = illumspec_PF_eV.x[-1])
        # Calculation of Shockley-Queisser photocurrent
        return abpf * q * 1e3/1e4


    @staticmethod
    def sq_limit(bg: float, illumspec_PF_eV: Any | None = None, light_int: float | None = None) -> Any:
        """
        Returns the SQ-limit five parameters. 
    
        Parameters
        ----------
        bg : FLOAT
            Bandgap in eV.
        illumspec_PF_eV: DiffSpectrum
            Illumination Spectrum (spectral photon flux as a function of photon energy in eV) other than AM1.5GT        
            If None then AM1.5GT Spectrum is taken
        light_int: FLOAT
            If AM1.5GT Spectrum as illumination Spectrum used, then with light_int the light intensity in mW/cm2 can be chosen.
            If None than 100 mW/cm2 (1 sun) is used.
    
        Returns
        -------
        fp : instance of FiveParam.
    
        """

        if illumspec_PF_eV is None:
            am15_ev = DiffSpectrum.am15_ev(left = 0.310, right = 4.428, delta = 0.001, y_unit = 'Spectral photon flux')
            if light_int is not None:
                am15_ev.y = am15_ev.y * light_int/100
            illumspec_PF_eV = am15_ev

        v_sq = IVData.sq_limit_voc(bg, illumspec_PF_eV, light_int = light_int, from_file = False)  # type: ignore
        Jsq = IVData.sq_limit_jsc(bg, illumspec_PF_eV, light_int = light_int)

        return FiveParam(cell_area = 1, Voc = v_sq, Jsc = Jsq, nid = 1, Rs = 0, Rsh = np.inf)


    def calc_resistance_curve(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Calculate resistance curve.
        
        Parameters
        ----------
        left : float | None
            Left.
        right : float | None
            Right.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.calc_resistance_curve()
        """
        #Calculates a resistance from the derivative of the J-V curve
        if left is None:
            left = min(self.x)
        if right is None:
            right = max(self.x)
        ra = range(self.x_idx_of(left), self.x_idx_of(right)+1)
        curve = self.copy()
        curve.x = curve.x[ra]
        curve.y = curve.y[ra]
        dIV = curve.diff()
        #dIV.plot()
        resist_arr = 1 / dIV.y * 1000 # Ohm * cm2
        R = IVData(dIV.x, resist_arr, quants = dict(x = 'Voltage', y = 'Resistance'), units = dict(x = 'V', y = 'Ohm cm2'))
        return R

    def resistance_plot(self, V_rel: Any, left: float | None = None, right: float | None = None, bottom: float | None = None, top: float | None = None, vline: Any | None = None, hline: Any | None=None, yscale: str = 'linear', title: str | None = None, noshow: Any=False) -> Any:
        """
        Resistance plot.
        
        Parameters
        ----------
        V_rel : Any
            V rel.
        left : float | None
            Left.
        right : float | None
            Right.
        bottom : float | None
            Bottom.
        top : float | None
            Top.
        vline : Any | None
            Vline.
        hline : Any | None
            Hline.
        yscale : str
            Yscale.
        title : str | None
            Title.
        noshow : Any
            Noshow.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.resistance_plot()
        """

        if left is None:
            left = V_rel - 0.3
        if right is None:
            right = V_rel+0.3
        R = self.calc_resistance_curve(left = left, right = right)
        if bottom is None:
            bottom = R.min_within(left = left, right = right) * 0.9
        if top is None:
            top = R.max_within(left = left, right = right) * 1.1
            #top = 10*self.Rs*1000
        if vline is None:
            vline = V_rel

        R_rel = R.y_of(V_rel)
        if not noshow:
            R.plot(yscale = yscale, left = left, right = right, bottom = bottom, top = top, vline = vline, hline=hline, title = title)
            print(f'The resistance at {V_rel:.3f} V is: {R_rel:.2f} Ohm cm2')

        return R_rel


class MIVData(MXYData):
    """
    Container class for MIVData data and operations.
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

