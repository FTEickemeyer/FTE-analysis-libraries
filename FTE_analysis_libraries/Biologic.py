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
from os import getcwd, remove, listdir
import math
import matplotlib.pyplot as plt
from importlib import reload
from IPython import embed

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
import matplotlib.pyplot as plt
from impedance.visualization import plot_nyquist, plot_bode


from .General import linfit, findind, save_ok, plx, q, k, T_RT, str_round_sig, colors, idx_range
from .XYdata import xy_data, mxy_data
from .IV import IV_data


def load_Biologic_CV(dir, FN, cell_area, light_int = 100, J_1sun = None, raw_data = False, both_scans = False, reverse_scan = True, warning = True, encoding = "ISO-8859-1"):
    """
    Loads CV data measured with a Biologic potentiostat. It works if the CV was measured first forward and then backward.

    Parameters
    ----------
    :param dir: string
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
    :returns: an instance of IV_data if both_scans is False or raw_data is True, otherwise a tuple of two instances of IV_data

    """
    if warning:
        print('Function IV_data.load_Biologic_CV: The CV scan has to be first in forward then in reverse sweep direction!')
        print('To switch off this message, set argument warning = False!')

    TFN = join(dir,FN)

    # Returns the number of header lines in the .mpt file
    def header_lines(TFN):

        with open(TFN, encoding = encoding) as mpt_file:
        #with open(Dir_FN) as mpt_file:

            for line in mpt_file:

                if 'Nb header lines' in line:
                    header_lines = int(line[17:].strip())
                    #print(f'Number of header_lines = {header_lines}')
                    break

        return header_lines


    raw = pd.read_csv(TFN, delimiter='\t', header = header_lines(TFN), encoding = encoding)
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



def load_Biologic_CA(dir, FN, uA = False, cell_area = None):
    """
    Load chrono-amperometry files
    """

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
    raw_t = np.array(raw)[:,7]
    raw_t -= raw_t[0]
    raw_I = np.array(raw)[:,10]
    if uA:
        raw_I *= 1000
        if cell_area == None:
            I = xy_data(raw_t, raw_I, quants = dict(x = 'Time', y = 'Current'), units = dict(x = 's', y = 'uA'), name = FN.split('.mpt')[0])
        else:
            I = xy_data(raw_t, raw_I/cell_area, quants = dict(x = 'Time', y = 'Current density'), units = dict(x = 's', y = 'uA/cm2'), name = FN.split('.mpt')[0])            
    else:
        if cell_area == None:
            I = xy_data(raw_t, raw_I, quants = dict(x = 'Time', y = 'Current'), units = dict(x = 's', y = 'mA'), name = FN.split('.mpt')[0])
        else:
            I = xy_data(raw_t, raw_I/cell_area, quants = dict(x = 'Time', y = 'Current density'), units = dict(x = 's', y = 'mA/cm2'), name = FN.split('.mpt')[0])

    raw_V = np.array(raw)[:,9]
    V = xy_data(raw_t, raw_V, quants = dict(x = 'Time', y = 'Voltage'), units = dict(x = 's', y = 'V'), name = FN.split('.mpt')[0])

    return I, V
    
    
    
def load_Biologic_CstC(dir, FN):
    """

    """

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
    raw_t = np.array(raw)[:,6]
    raw_t -= raw_t[0]
    raw_V = np.array(raw)[:,8]    
    raw_I = np.array(raw)[:,9]

    V = xy_data(raw_t, raw_V, quants = dict(x = 'Time', y = 'Voltage'), units = dict(x = 's', y = 'V'), name = FN.split('.mpt')[0])
    I = xy_data(raw_t, raw_I, quants = dict(x = 'Time', y = 'Current'), units = dict(x = 's', y = 'mA'), name = FN.split('.mpt')[0])

    return V, I


def EIS_convert_mpt_to_csv(data_dir, save_dir, tolerance = 0.005, show_details = True):
    # Converts EIS data as a Biologic mpt file into csv
    # tolerance: Voltage tolerance for the evaluation
    # Get filenames of all *.mpt files
    filenames = listdir(data_dir)
    SPEIS_in_FNs = any(['EIS' in file for file in filenames])
    if SPEIS_in_FNs:
        in_filename = 'EIS'
    else:
        in_filename = '.mpt'
    i = 0
    while i < len(filenames):
        if ('.mpt' in filenames[i]) and (in_filename in filenames[i]):
            i += 1
        else:
            filenames = np.delete(filenames,i)

    if show_details:
        print(filenames)
        
        
    # Function readfile reads in the data and returns the nonzero data, number of header lines, starting voltage Ei, 
    # final voltage Ef and number of voltages measured

    # Note: The number of voltages is N + 1

    def readfile(FN):

        data = []
        count = 0
        header_lines = 0
        number_of_voltages = 0
        Ei = 0
        Ef = 0

        with open(FN, encoding = "ISO-8859-1") as z:
            isdata = False
            for line in z:

                #print(str(count) + ':' + line)
                count += 1
                if isdata:
                    #data.append(line.strip('\n'))
                    #print(str(count) + ':' + line)
                    data.append(line)

                if 'freq/Hz' in line:
                    isdata = True

                if 'Nb header lines' in line:
                    header_lines = int(line[17:].strip())

                if 'Ei (V)' in line:
                    Ei = float(line[6:].strip())
                    #print(line)

                if 'Ef (V)' in line:
                    Ef = float(line[6:].strip())
                    #print((line[6:].strip()))

                if line[0:2] == 'N ':
                    number_of_voltages = int(line[1:].strip()) + 1
                    
            if number_of_voltages == 0:
                number_of_voltages = 1

        return(data, header_lines, Ei, Ef, number_of_voltages)
    
    
    # Split the data into numpy arrays, for each voltage one column, delete 0 frequency entries

    def get_data(fn):

        data, header_lines, Ei, Ef, number_of_voltages = readfile(join(data_dir, fn))
        #print(header_lines)
        #print(Ei)
        #print(Ef)
        #print(number_of_voltages)
        #print(data)

        voltages = np.linspace(Ei, Ef, number_of_voltages)
        #print(voltages)

        with open(join(data_dir, 'temp.txt'), 'w') as f:    
            for count, line in enumerate(data):
                f.write(line)

        data_raw = pd.read_csv(join(data_dir, 'temp.txt'), delimiter='\t', header = 0)
        remove(join(data_dir,'temp.txt'))

        raw_data_len = len(data_raw)
        #print(data_raw)

        raw_data_freq = np.array(data_raw)[:,0] # frequency in Hz
        raw_data_Re = np.array(data_raw)[:,1] # Re(Z) in Ohm
        raw_data_Im = np.array(data_raw)[:,2] # -Im(Z) in Ohm
        raw_data_V = np.array(data_raw)[:,6] # Ewe in V
        #print(len(raw_data_freq))

        #Initialize data arrays
        data_freq = np.zeros((raw_data_len, number_of_voltages), dtype=float) # frequency in Hz
        data_Re = np.zeros((raw_data_len, number_of_voltages), dtype=float) # Re(Z) in Ohm
        data_Im = np.zeros((raw_data_len, number_of_voltages), dtype=float) # -Im(Z) in Ohm
        #data_V = np.zeros((raw_data_len, number_of_voltages), dtype=float) # Ewe in V

        real_data_count = np.zeros(len(voltages), dtype=int)

        #For debuggin only: test voltage index
        #test_voltage_index = 2
        #print(voltages[test_voltage_index])

        for count in range(raw_data_len):
            if raw_data_freq[count] != 0:
                for i, V in enumerate(voltages):

                    if (round(raw_data_V[count], 3) < round(V, 3) + tolerance) and (round(raw_data_V[count], 3) > round(V, 3) - tolerance):
                        data_freq[real_data_count[i], i] = raw_data_freq[count]
                        data_Re[real_data_count[i], i] = raw_data_Re[count]
                        data_Im[real_data_count[i], i] = raw_data_Im[count]

                        #For debugging only
                        #if (round(raw_data_V[count], 3) < round(voltages[test_voltage_index], 3) + tolerance) and (round(raw_data_V[count], 3) > round(voltages[test_voltage_index], 3) - tolerance):
                        #    print(raw_data_V[count])
                        #    print(raw_data_freq[count])

                        #test[real_data_count] = float(raw_data_freq[count])
                        real_data_count[i] += 1

        index = np.arange(real_data_count[0], raw_data_len)
        data_freq = np.delete(data_freq, index,0)
        data_Re = np.delete(data_Re, index,0)
        data_Im = np.delete(data_Im, index,0)

        return voltages, data_freq, data_Re, data_Im
    
    # Save the data in csv format

    def save_data(voltages, data_freq, data_Re, data_Im, fn):
        if show_details:
            print(fn[:-4])

        for i, V in enumerate(voltages):
            newfn = fn[:-4] + '_' + str_round_sig(V,4) + 'V.csv'
            dataset = pd.DataFrame({'Freqency' : data_freq[:,i], 'Re(Z)' : data_Re[:,i], 'Im(Z)' : -data_Im[:,i]}) # in the original file -Im(Z) was saved
            dataset.to_csv(join(save_dir,newfn), header = False, index = False)
            if show_details:
                print(newfn)
            
    for fn in filenames:
        voltages, data_freq, data_Re, data_Im = get_data(fn)
        save_data(voltages, data_freq, data_Re, data_Im, fn)


def EIS_get_data(TFNs, f_range = None, Z_4th_quadrant = True, cell_area = 1, V_idx_list = None, show_details = True, title = '', show_title = True):
    #Get EIS data of one sample for all voltages given in V_idx_list
    #TFNs: list of file names (including directory) with each element corresponding to one voltage according to V_idx_list. If only one voltage used then use [TFN]
    #Attention: in a previous version title was the first argument, now it is changed to a keyword argument
    
    #f_range: frequency range to be displayed
    
    #title = data_label[cell_idx]
    #print(all_TFNs[cell_idx])
    #cell_area = cell_area_list[cell_idx]

    fs = []
    Zs = []
    Vs = []
    
    if V_idx_list is None:
        V_idx_list = [i for i in range(len(TFNs))]
    
    for idx in V_idx_list:
        file = TFNs[idx]
        #print(file)
        frequencies, Z = preprocessing.readCSV(file)

        # keep only the impedance data in the first quandrant
        if Z_4th_quadrant:
        
            def Z_in_4th_quadrant(frequencies, Z):
                """
                Trim out all data points outside 4th quadrant
            
            
                Parameters
                ----------
                frequencies : np.ndarray
                    Array of frequencies
                Z : np.ndarray of complex numbers
                    Array of complex impedances
            
                Returns
                -------
                frequencies : np.ndarray
                    Array of frequencies after filtering
                Z : np.ndarray of complex numbers
                    Array of complex impedances after filtering
                """
            
                frequencies = frequencies[np.imag(Z) < 0]
                Z = Z[np.imag(Z) < 0]

                frequencies = frequencies[np.real(Z) > 0]
                Z = Z[np.real(Z) > 0]
                
                return frequencies, Z
            
            frequencies, Z = Z_in_4th_quadrant(frequencies, Z)

            
        Z *= cell_area

        if not (f_range is None):
            ra = idx_range(frequencies, left = f_range[0], right = f_range[1])
            frequencies = frequencies[ra]
            Z = Z[ra]

        voltage = float(file.split('.csv')[0].split('_')[-1].split('V')[0])

        Zs.append(Z)
        fs.append(frequencies)
        Vs.append(voltage)

    if show_details:

        #print(data_dir_list[cell_idx])
        print(title)
        print(f'Cell area: {cell_area:.1f} cm2')
        
        #Plot Nyquist plot
        print('_________')
        print('')
        print('Nyquist plot')
        print('_________')
        print('')

        fig, ax = plt.subplots(figsize=(10,10))
        for Z, voltage in zip(Zs, Vs):
            label = f'{voltage:.3f} V'
            plot_nyquist(ax, Z, fmt='o-', label = label)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(False)
        plt.legend()
        
        if show_title:
            plt.title(title)
        plt.xlabel('$Z^{\\prime}(\\omega) \, [\\Omega \, cm^2]$')
        plt.ylabel('$-Z^{\\prime \\prime}(\\omega) \, [\\Omega \, cm^2]$')
        plt.show()
        
        #Plot Bode plot
        print('_________')
        print('')
        print('Bode plot')
        print('_________')
        print('')
        fig, ax = plt.subplots(nrows = 2, figsize=(10,10))
        for f, Z, voltage in zip(fs, Zs, Vs):
            label = f'{voltage:.3f} V'
            #plot_bode(ax[0], ax[1], f, Z, fmt='o', label = label)
            plot_bode([ax[0], ax[1]], f, Z, fmt='o-', label = label)

        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].grid(False)
        ax[0].set_ylabel('$\\vert Z(\\omega) \\vert \, [\\Omega \, cm^2]$')
        if show_title:
            ax[0].set_title(title)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[1].grid(False)
        #ax[1].set_title(title)

        plt.legend()
        fig.tight_layout()
        plt.show()
            
    return fs, Zs, Vs


def EIS_get_data_old(title, TFNs, f_idx_end, pos_Z_only = True, cell_area = 1, V_idx_list = None, show_details = True, show_title = True):
    #f_idx_end changed to f_range in the new version
    #f_idx_end: index until which the frequencies will be sliced. If -1, then the whole list will be taken and not sliced until -1.
    
    #title = data_label[cell_idx]
    #print(all_TFNs[cell_idx])
    #cell_area = cell_area_list[cell_idx]

    fs = []
    Zs = []
    Vs = []
    
    if V_idx_list is None:
        V_idx_list = [i for i in range(len(TFNs))]
    
    for idx in V_idx_list:
        file = TFNs[idx]
        #print(file)
        frequencies, Z = preprocessing.readCSV(file)

        # keep only the impedance data in the first quandrant
        if pos_Z_only:
            frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)
        Z *= cell_area

        if f_idx_end == -1:
            f_idx_end = len(frequencies)
        frequencies = frequencies[0:f_idx_end]
        Z = Z[0:f_idx_end]

        voltage = float(file.split('.csv')[0].split('SPEIS_')[1].split('V')[0])

        Zs.append(Z)
        fs.append(frequencies)
        Vs.append(voltage)

    if show_details:

        #print(data_dir_list[cell_idx])
        print(title)
        print(f'Cell area: {cell_area:.1f} cm2')
        
        #Plot Nyquist plot
        print('_________')
        print('')
        print('Nyquist plot')
        print('_________')
        print('')

        fig, ax = plt.subplots(figsize=(10,10))
        for Z, voltage in zip(Zs, Vs):
            label = f'{voltage:.3f} V'
            plot_nyquist(ax, Z, fmt='o-', label = label)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(False)
        plt.legend()
        
        if show_title:
            plt.title(title)
        plt.xlabel('$Z^{\\prime}(\\omega) \, [\\Omega \, cm^2]$')
        plt.ylabel('$-Z^{\\prime \\prime}(\\omega) \, [\\Omega \, cm^2]$')
        plt.show()
        
        #Plot Bode plot
        print('_________')
        print('')
        print('Bode plot')
        print('_________')
        print('')
        fig, ax = plt.subplots(nrows = 2, figsize=(10,10))
        for f, Z, voltage in zip(fs, Zs, Vs):
            label = f'{voltage:.3f} V'
            #plot_bode(ax[0], ax[1], f, Z, fmt='o', label = label)
            plot_bode([ax[0], ax[1]], f, Z, fmt='o-', label = label)

        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].grid(False)
        ax[0].set_ylabel('$\\vert Z(\\omega) \\vert \, [\\Omega \, cm^2]$')
        if show_title:
            ax[0].set_title(title)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[1].grid(False)
        #ax[1].set_title(title)

        plt.legend()
        fig.tight_layout()
        plt.show()
            
    return fs, Zs, Vs


def EIS_predict(f, circuit_str, params):
    
    circuit = CustomCircuit(circuit_str, initial_guess = params)

    Z_pred = circuit.predict(f, use_initial = True)
    
    return Z_pred

def EIS_fit(f, Z, circuit_str, initial_guess, f_range = None, bounds = None, show_details = True, data_label = ''):


    circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)
    
    if not f_range is None:
        ra = idx_range(f, left = f_range[0], right = f_range[1])
        f = f[ra]
        Z = Z[ra]

    circuit.fit(f, Z, bounds = bounds)

    Z_fit = circuit.predict(f)

    if show_details:
        print('____________________________________')
        print('')
        print(data_label)
        print('____________________________________')
        print(circuit)
    
    return Z_fit, circuit

    
def multiple_Nyquist_and_Bode_plot(f_list, Z_list, f_range = None, Z_fit_list = None, label_list = None, what_to_show = ['nyquist', 'bode'], figsize = (10, 10), textsize = 16, colors = colors, fmts = None):

    if isinstance(figsize, int):
        figsize = (figsize, figsize)
        
    # Check if the data is a list of lists:
    if not(isinstance(f_list[0], np.ndarray)):
       f_list = [f_list]
    if not(isinstance(Z_list[0], np.ndarray)):
        Z_list = [Z_list]
    if not(Z_fit_list is None):
        if not(isinstance(Z_fit_list[0], np.ndarray)):
            Z_fit_list = [Z_fit_list]
    if not(label_list == None):
        if not(isinstance(label_list, list)):
            label_list = [label_list]
            
    if fmts is None:
        if Z_fit_list is None:
            fmts = ['o--' for fs in f_list]
        else:
            fmts = ['o' for fs in f_list]
            
    #Plot Nyquist plot

    if 'nyquist' in what_to_show:
        #print('_________')
        #print('')
        #print('Nyquist plot')
        #print('_________')
        #print('')
    
        fig, ax = plt.subplots(figsize=figsize)
        
        if label_list is None:
            label_list = [f'{i}' for i in range(len(Z_list))]
        if Z_fit_list is None:
            for f, Z, label, color, fmt in zip(f_list, Z_list, label_list, colors, fmts):
                if not f_range is None:
                    ra = idx_range(f, left = f_range[0], right = f_range[1])
                    plot_nyquist(ax, Z[ra], fmt=fmt, color = color, label = label)
                else:
                    plot_nyquist(ax, Z, fmt=fmt, color = color, label = label)

        else:
            for f, Z, Z_fit, label, color, fmt in zip(f_list, Z_list, Z_fit_list, label_list, colors, fmts):
                if not f_range is None:
                    ra = idx_range(f, left = f_range[0], right = f_range[1])
                    plot_nyquist(ax, Z[ra], fmt=fmt, color = color, label = label)
                    plot_nyquist(ax, Z_fit[ra], fmt='-', color = color, label = label+' (fit)')
                else:
                    plot_nyquist(ax, Z, fmt=fmt, color = color, label = label)
                    plot_nyquist(ax, Z_fit, fmt='-', color = color, label = label+' (fit)')
    
        ax.tick_params(axis='both', which='major', labelsize=textsize)
        ax.grid(False)
        plt.xlabel('$Z^{\\prime}(\\omega) \, [\\Omega \, cm^2]$', fontsize = textsize)
        plt.ylabel('$-Z^{\\prime \\prime}(\\omega) \, [\\Omega \, cm^2]$', fontsize = textsize)
        plt.legend(fontsize = textsize, bbox_to_anchor=(1,1), loc="upper left")
        plt.show()

    if 'bode' in what_to_show:
        #Plot Bode plot
        #print('_________')
        #print('')
        #print('Bode plot')
        #print('_________')
        #print('')
        fig, ax = plt.subplots(nrows = 2, figsize=figsize)
        
        if Z_fit_list is None:
            for f, Z, label, color, fmt in zip(f_list, Z_list, label_list, colors, fmts):
                if not f_range is None:
                    ra = idx_range(f, left = f_range[0], right = f_range[1])
                    plot_bode([ax[0], ax[1]], f[ra], Z[ra], fmt=fmt, color = color, label = label)
                else:
                    plot_bode([ax[0], ax[1]], f, Z, fmt=fmt, color = color, label = label)
    
        else:
            for f, Z, Z_fit, label, color, fmt in zip(f_list, Z_list, Z_fit_list, label_list, colors, fmts):
                if not f_range is None:
                    ra = idx_range(f, left = f_range[0], right = f_range[1])
                    plot_bode([ax[0], ax[1]], f[ra], Z[ra], fmt=fmt, color = color, label = label)
                    plot_bode([ax[0], ax[1]], f[ra], Z_fit[ra], fmt='-', label = label+' (fit)')
                else:
                    plot_bode([ax[0], ax[1]], f, Z, fmt=fmt, color = color, label = label)
                    plot_bode([ax[0], ax[1]], f, Z_fit, fmt='-', label = label+' (fit)')

    
        ax[0].tick_params(axis='both', which='major', labelsize=textsize)
        ax[0].grid(False)
        ax[0].set_xlabel('f [Hz]', fontsize = textsize)
        ax[0].set_ylabel('$\\vert Z(\\omega) \\vert \, [\\Omega \, cm^2]$', fontsize = textsize)
        ax[1].tick_params(axis='both', which='major', labelsize=textsize)
        ax[1].set_xlabel('f [Hz]', fontsize = textsize)
        ax[1].set_ylabel('$-\\phi_Z(\\omega)$ [°]', fontsize = textsize)
        ax[1].grid(False)
    
        plt.legend(fontsize = textsize, bbox_to_anchor=(1,1), loc="upper left")
        fig.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    
    data_dir = r'C:\Users\dreickem\switchdrive\Work\Projects\Exeger\211005 Dummy cell EIS\data\dummy cells'
    #data_dir = r'C:\Users\dreickem\switchdrive\Work\Projects\Exeger\211005 Dummy cell EIS\data\test'
    cell_area = 4.5*1.4 #cm2
    print(cell_area)
    FNs = listdir(data_dir)
    for idx, fn in enumerate(FNs):
        print(f'{idx:2}: {fn}')
        
    #Use selected data and change order
    FNs = listdir(data_dir)
    order = [1, 3, 5, 7, 9, 11]
    FNs = [FNs[idx] for idx in order]
    print(FNs)
    for idx, fn in enumerate(FNs):
        print(f'{idx:2}: {fn}')
    TFNs = [join(data_dir, label) for label in FNs]
    #print(TFNs)
    data_label = [label.split('.csv')[0] for label in FNs]
    #print(data_label)
    
    cell_area = cell_area
    print(f'The cell area is {cell_area:.1f} cm2')
    fs_list = []
    Zs_list = []
    for TFN in TFNs:
        title = TFN
        #print(title)
        fs, Zs, Vs = EIS_get_data(title, [TFN], f_range = None, Z_4th_quadrant = True, V_idx_list = [0], cell_area = cell_area, show_details = False, show_title = True)
        fs_list.append(fs[0])
        Zs_list.append(Zs[0])
        
    f_range = [1e5, 1e6]
    f_range = None