# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:01:16 2020

@author: dreickem
"""

# Import standard libraries and modules
import os
from os.path import join
import math
import platform
import pkg_resources
import warnings
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import butter,filtfilt, savgol_filter
from scipy.interpolate import interp1d

from .General import findind, findind_exact, int_arr, save_ok, q, k, T_RT, linfit, idx_range
from . import General as gen


system_dir = pkg_resources.resource_filename( 'FTE_analysis_libraries', 'System_data' )

class xy_data:
    
    def __init__(self, x, y, quants = {"x": "x", "y": "y"}, units = {"x": None, "y": None}, name = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3), check_data = True):
        """
        x is a numpy array e.g. the wavelengths or photon energies
        y is a numpy array e.g. cts, cps, photon flux, spectral flux
        quants is a dict with the type of the data, e.g. {"x": "Wavelength", "y": "Intensity"}
        units is a dict with the units of the data, e.g. {"x": "nm", "y": "cps"}
        name is the name of the data, e.g. the file name
        check_data: Checks if the x-values are in ascending order and if there are nan values.
        """
        self.x = x
        self.y = y
        if type(quants) == list:
            self.qx = quants[0]
            self.qy = quants[1]
        else:
            self.qx = quants["x"]
            self.qy = quants["y"]
        if type(units) == list:
            self.ux = units[0]
            self.uy = units[1]
        else:
            self.ux = units["x"]
            self.uy = units["y"]
        self.name = name
        self.plotstyle = plotstyle
        if check_data:
            ok = self.data_check()
            if not(ok):
                print('To switch off this message use check_data = False')

        
    def __mul__(self, other):        
        s = self.copy()
        if type(self).mro()[-2] == type(other).mro()[-2]:
            o = other.copy()
            x_min = max(min(self.x), min(other.x))
            x_max = min(max(self.x), max(other.x))
            delta = min(self.x[1]-self.x[0], other.x[1]-other.x[0])
            s.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            o.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            s.y = s.y * o.y
        else:
            s.y = s.y * other
        return s
    
    def __add__(self, other):        
        s = self.copy()
        if type(self).mro()[-2] == type(other).mro()[-2]:
            o = other.copy()
            x_min = max(min(self.x), min(other.x))
            x_max = min(max(self.x), max(other.x))
            delta = min(self.x[1]-self.x[0], other.x[1]-other.x[0])
            s.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            o.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            s.y = s.y + o.y
        else:
            s.y = s.y + other
        return s
    
    def __sub__(self, other):        
        s = self.copy()
        if type(self).mro()[-2] == type(other).mro()[-2]:
            o = other.copy()
            x_min = max(min(self.x), min(other.x))
            x_max = min(max(self.x), max(other.x))
            delta = min(self.x[1]-self.x[0], other.x[1]-other.x[0])
            s.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            o.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            s.y = s.y - o.y
        else:
            s.y = s.y - other
        return s
    
    def __truediv__(self, other):
        s = self.copy()
        if type(self).mro()[-2] == type(other).mro()[-2]:
            o = other.copy()    
            x_min = max(min(self.x), min(other.x))
            x_max = min(max(self.x), max(other.x))
            delta = min(self.x[1]-self.x[0], other.x[1]-other.x[0])
            s.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            o.equidist(left = x_min, right = x_max, delta = delta, kind = 'cubic')
            s.y = s.y / o.y
        else:
            s.y = s.y / other
        return s
            
    @classmethod
    def from_df(cls, df, y_col= 0, take_quants_and_units_from_df= True, **kwargs):
        #The index is taken as
        #y_col can be either an integer to denote the y_colth column or a column name
    
        x = df.index.values
    
        if isinstance(y_col, int):
            y_col = df.columns[y_col]
            y = df[y_col].values
        elif y_col in df.columns:
            y = df[y_col]
        else:
            y = x*0
            print('Attention (df_to_xy): y_col is not a valid column!')
    
        if take_quants_and_units_from_df:
            qx_ux = df.index.name.split(' (')
            qx = qx_ux[0]
            if len(qx_ux) == 2:
                ux = qx_ux[1].split(')')[0]
            else:
                ux = None
            qy_uy = y_col.split(' (')
            qy = qy_uy[0]
            if len(qy_uy) == 2:
                uy = qy_uy[1].split(')')[0]
            else:
                uy = None
                 
        return cls(x, y, quants= [qx, qy], units= [ux, uy], **kwargs)
                
    def to_df(self):
        col_name = self.qy
        if self.uy is not None and self.uy != '':
            col_name += f' ({self.uy})'
        index_name = self.qx
        if self.ux is not None and self.ux != '':
            index_name += f' ({self.ux})'
        return pd.DataFrame({index_name: self.x, col_name: self.y}).set_index(index_name)
    
    def data_check(self):
        #Checks if the x-values are in ascending order and if there are nan values.
        #Returns True if data is ok and False if not.
        #Recommended usage: 
        #ok = object.data_check()
        #if not(ok): ...
        ok = True
        x = self.x
        y = self.y
        # Check if x-values are ascending
        if False in (np.sort(x) == x):
            ok = False
            print(f'Attention (data {self.name}): The x-values are not in ascending order!')
            print('To reverse the order, use function reverse().')
        # Check if there are np.nan values in x or y
        if (True in np.isnan(x)) or (True in np.isnan(y)):
            ok = False
            print(f'Attention (data {self.name}): There are nan values in array x or y!')
            print('To remove nan values use function remove_nan().')
        return ok
        
        
    @classmethod
    def generate_empty(cls):
        x = np.array([])
        y = np.array([])
        return cls(x, y)
        
    def copy(self):
        x = self.x.copy()
        y = self.y.copy()
        qx = self.qx
        qy = self.qy
        ux = self.ux
        uy = self.uy
        name = self.name[:]
        plotstyle = self.plotstyle.copy()
        return type(self)(x, y, quants = dict(x = qx, y = qy), units = dict(x = ux, y = uy), name = name, plotstyle = plotstyle, check_data = False)
        
    def y_of(self, x_value, interpolate = False):
        if x_value < min(self.x):
            print('Attention: x_value is smaller than minimum, y of minimum x used!')
            y = self.y_of(min(self.x))
        elif x_value > max(self.x):
            print('Attention: x_value is larger than maximum, y of maximum x used!')
            y = self.y_of(max(self.x))
        else:
            
            if interpolate:
                f_interp = interp1d(self.x, self.y, 'cubic', bounds_error=False, fill_value=0)
                y = f_interp(x_value)
            else:
                idx = findind(self.x, x_value)
                y = self.y[idx]

        return float(y)
    
    
    def x_idx_of(self, x_value):
        """
        Works only for ascending arrays!
        """
        idx = findind(self.x, x_value)
        return idx
    
    def x_of(self, y_value, start = None,  interpolate = False):
        """
        Find the first x_value > start  where self.y = y_value
        """

        if start == None:
            idx_start = 0
        else:
            idx_start = findind(self.x, start)
        
        if interpolate:
            # It is important that self.y is monotonous.
            # Make sure that there are no duplicate y-values:
            x_arr = self.x[idx_start:]
            y_arr = self.y[idx_start:]
            seen = set()
            idx_arr = [idx for idx in range(len(y_arr)) if y_arr[idx] not in seen and not seen.add(y_arr[idx])]
            #idx_arr = [0]+[idx+1 for idx in range(len(y_arr)-1) if (y_arr[idx+1] != y_arr[idx])]
            x_arr_new = np.array([x_arr[idx_arr[idx]] for idx in range(len(idx_arr))])
            y_arr_new = np.array([y_arr[idx_arr[idx]] for idx in range(len(idx_arr))], dtype = np.float64)
     
            f_interp = interp1d(y_arr_new, x_arr_new, 'cubic', bounds_error=False, fill_value=0)
            x = f_interp(y_value)                
        else:
            x = self.x[idx_start + findind_exact(self.y[idx_start:], y_value)]

        return float(x)

    
    def normalize(self, x_lim = None, norm_val = 1):
        if x_lim == None:
            r = range(0, len(self.x))
        else:
            idx_min = self.x_idx_of(x_lim[0])
            idx_max = self.x_idx_of(x_lim[1])
            r = range(idx_min, idx_max+1)
            
        if r != range(0,0):
            self.y = self.y / max(self.y[r]) * norm_val
        
    def equidist(self, left = None, right = None, delta = 0.1, kind = 'cubic'):
        """
        Change x values so that they are equidistant with a delta of delta. x ranges from left to right.
        If left (right) = None then the new left (right) value is the old one.
        """
        
        if left == None:
            #min_x = math.ceil(min(self.x))
            min_x = min(self.x)
        else:
            min_x = left
        if right == None:
            max_x = max(self.x)
        else: 
            max_x = right
        
        
        #new_x = np.arange(min_x, max_x, delta) # does not give exactly evenly spaced x-values
        new_x = np.linspace(min_x, max_x, int(round((max_x-min_x)/delta)+1))
        self.y = int_arr(self.x, self.y, new_x, kind = kind)
        self.x = new_x
    
    def quants(self):
        return dict(x=self.qx, y=self.qy)
    
    def units(self):
        return dict(x=self.ux, y=self.uy)
    
    def qy_uy(self, qy, uy):
        self.qy = qy
        self.uy = uy
        
    def plot(self, title = 'self.name', ax= None, xscale = 'linear', yscale = 'linear', 
             left = None, right = None, bottom = None, divisor = None, top = None,
             plot_table = False, cell_text = None, row_labels = None, 
             hline = None, hline_colors = None, vline = None, vline_colors = None, figsize=(9,6), return_fig = False, show_plot = True, create_image_stream= False):
        """
        Plots the x and y data.
        examples for plotstyle:
            plotstyle = dict(linestyle = 'None', marker = 'o', color = 'green', markersize = 20)
            plotstyle = dict(linestyle = '-', color = 'green', linewidth = 5)
        return_fig: if True than the figure is returned as an object of type matplotlib.figure.Figure. This figure can be then saved with matplotlib.figure.Figure.savefig(filename).
                    To show this returned figure one can use the function matplotlib.figure.Figure.show(); this works however only if a GUI backend is chosen, 
                    e.g. by %matplotlib qt in jupyterlab (%matplotlib inline doesn't work').
                    if show_plot == False and retrun_fig == True: It is important to close the figure with matplotlib.pyplot.plt.close(Figure), otherwise it will be shown.
        """
        
        if ax is not None:
            show_plot = False

        if ax is None:
            plt.rcParams.update({'font.size': 12})
            #plt.rc('text', usetex=True) Latex needs to be installed first
            fig, ax = plt.subplots(figsize=figsize)
            #if plot_table:
            #    ax = fig.add_subplot(111)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if left != None:
            ax.set_xlim(left = left)
        if right != None:
            ax.set_xlim(right = right)
            
        if bottom != None:
            if top == None:
                bottom_, top = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #top = max(self.y) + 0.1 * abs(max(self.y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if top != None:
            if bottom == None:
                bottom, top_ = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #bottom = min(self.y) - 0.1 * abs(min(self.y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if (bottom == None) and (top == None):
            bottom, top = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
            ax.set_ylim(bottom = bottom, top = top)   

        ax.plot(self.x, self.y, **self.plotstyle)

        if self.ux is not None and self.ux != '':
            ax.set_xlabel(f'{self.qx} ({self.ux})')
        else:
            ax.set_xlabel(f'{self.qx}')
            
        if self.uy is not None and self.uy != '':
            ax.set_ylabel(f'{self.qy} ({self.uy})')
        else:
            ax.set_ylabel(f'{self.qy}')
            
        if title == 'self.name':
            ax.set_title(self.name)
        else:
            ax.set_title(title)

        if plot_table:
            the_table = ax.table(cellText = cell_text, rowLabels = row_labels, loc='bottom',
                                  bbox = [0.4, 0.2, 0.1, 0.7])  #bbox = [x0, y0, width, height])
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(12)
            the_table.scale(1, 5)
            
        if hline != None:
            if type(hline) == list:
                for idx, n in enumerate(hline):
                    if hline_colors is not None:
                        color = hline_colors[idx]
                    ax.axhline(y = n, color=color, linestyle='-')    
            else:
                if hline_colors is not None:
                    color = hline_colors
                else:
                    color = 'b'
                ax.axhline(y = hline, color=color, linestyle='-')

        if vline != None:
            if type(vline) == list:
                for idx, n in enumerate(vline):
                    if vline_colors is not None:
                        color = vline_colors[idx]
                    else:
                        color = 'r'
                    ax.axvline(x = n, color=color, linestyle='-')    
            else:
                if vline_colors is not None:
                    color = vline_colors
                else:
                    color = 'r'
                ax.axvline(x = vline, color=color, linestyle='-')
            
        if create_image_stream:
            image_stream = io.BytesIO()
            plt.savefig(image_stream)
            if show_plot:
                plt.show()
            return image_stream

        #plt.legend()
        if show_plot:
            plt.show()
            
        if return_fig:
            return fig


    def plot_new(self, title = 'self.name', xscale = 'linear', yscale = 'linear', 
             left = None, right = None, bottom = None, top = None,
             plot_table = False, cell_text = None, row_labels = None, 
             hline = None, vline = None, figsize=(9,6)):
        """
        Plots the x and y data.
        examples for plotstyle:
            plotstyle = dict(linestyle = 'None', marker = 'o', color = 'green', markersize = 20)
            plotstyle = dict(linestyle = '-', color = 'green', linewidth = 5)
        """
        
        #plt.rcParams.update({'font.size': 12})
        #fig = plt.figure(figsize=figsize)
        fig = Figure(figsize = figsize, dpi = 100) 
        if plot_table:
            ax = fig.add_subplot(111)

        graph = fig.add_subplot(111) 
        graph.set_xscale(xscale)
        graph.set_yscale(yscale)

        if left != None:
            graph.set_xlim(left=left)
        if right != None:
            graph.set_xlim(right=right)
        if bottom != None:
            graph.set_ylim(bottom = bottom)
        if top != None:
            graph.set_ylim(top = top)
        plt.plot(self.x, self.y, **self.plotstyle)

        plt.xlabel(f'{self.qx} ({self.ux})')
        if self.uy == '':
            plt.ylabel(f'{self.qy}')
        else:
            plt.ylabel(f'{self.qy} ({self.uy})')
        if title == 'self.name':
            plt.title(self.name)
        else:
            plt.title(title)

        if plot_table:
            the_table = plt.table(cellText = cell_text, rowLabels = row_labels, loc='bottom',
                                  bbox = [0.4, 0.2, 0.1, 0.7])  #bbox = [x0, y0, width, height])
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(12)
            the_table.scale(1, 5)
            
        if hline != None:
            plt.axhline(y = hline, color='black', linestyle='-')
        if vline != None:
            plt.axvline(x = vline, color='r', linestyle='-')

        #plt.legend()
        plt.show()

        
    def plot_linfit(self, von = None, bis = None, residue = False, return_data = False):
        
        if von == None:
            von = min(self.x)
        if bis == None:
            bis = max(self.x)
        m, b = linfit(self.x, self.y, von, bis)
        self.m = m
        self.b = b
        fit = xy_data(self.x, m * self.x + b)
        if residue:
            res = xy_data(self.x, (self.y - fit.y)/self.y * 100)
            res.qx = self.qx
            res.qy = 'Delta'
            res.ux = self.ux
            res.uy = '%'
            m_min = res.min_within(von, bis)
            m_max = res.max_within(von, bis)
            res.plot(left = von, right = bis, bottom = m_min * 1.1, top = m_max * 1.1, title = f'({self.name} - linear fit) / {self.name}')
        else:
            both = mxy_data([self, fit])
            both.label(['self.name', f'linear fit: m = {m:.2e}, b = {b:.2e}'])
            both.plot()
        if return_data:
            return both
            
        
    def load_old(directory, FN = '', delimiter = ',', header = 'infer', quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False):

        """
        This is the old version. The new one is defined as classmethod.
        Loads a single xy data. If a filename is given it will be used, if not the first file in the directory will be used.
        """
        
        if FN == '':
            FN = os.listdir(directory)[0]
        dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)
        
        x = np.array(dat, dtype = np.float64)[:,0]
        y = np.array(dat, dtype = np.float64)[:,1]
        
        sp = xy_data(x, y, quants, units, FN)
        
        if take_quants_and_units_from_file:
            
            col0 = list(dat)[0]
            sp.qx = col0.split(' (')[0]
            if ' (' in col0:
                sp.ux = col0.split(' (')[1].split(')')[0]

            col1 = list(dat)[1]
            sp.qy = col1.split(' (')[0]
            if ' (' in col1:
                sp.uy = col1.split(' (')[1].split(')')[0]
                
        return sp 
    
    @classmethod
    def load(cls, filepath_or_directory, FN = '', delimiter = ',', columns=(0, 1), header = 'infer', 
             quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False,  check_data = True, name=None):

        """
        Loads a single xy data. If a filename is given it will be used, if not the first file in the directory will be used.
        colunns: tuple which indicates which column is x and which is y.
        """
        
        if (FN == '') and os.path.isdir(filepath_or_directory):
            FN = os.listdir(filepath_or_directory)[0]
            file = join(filepath_or_directory, FN)
        elif (filepath_or_directory == '') and os.path.isfile(FN):
            file = FN
        elif os.path.isfile(filepath_or_directory):
            file = filepath_or_directory
        elif os.path.isfile(join(filepath_or_directory, FN)):
            file = join(filepath_or_directory, FN)
        else:
            warnings.warn("Attention: Not a valid file or directory!")
            return

            
        #print(file)
        windows_long_file_prefix = '\\\\?\\'
        
        if (
            ( platform.system() == 'Windows' ) and
            ( len( file ) > 255 ) and
            ( not file.startswith( windows_long_file_prefix ) )
        ):
            file = windows_long_file_prefix + file
        
        dat = pd.read_csv(file, delimiter = delimiter, header = header)
        
        #The conversion to np.float64 should be done after the conversion into an should be done
        #after the np.array() function. Then it is possible to also read in data where some columns
        #contain text.
        x = np.array(dat)[:,columns[0]].astype(np.float64)
        y = np.array(dat)[:,columns[1]].astype(np.float64)
        
        qx = quants["x"]
        qy = quants["y"]
        ux = units["x"]
        uy = units["y"]
                
        if take_quants_and_units_from_file:
            
            col0 = list(dat)[0]
            qx = col0.split(' (')[0]
            if ' (' in col0:
                ux = col0.split(' (')[1].split(')')[0]

            col1 = list(dat)[1]
            qy = col1.split(' (')[0]
            if ' (' in col1:
                uy = col1.split(' (')[1].split(')')[0]
                
        if name is None:
            name = FN
        
        return cls(x, y, quants = dict(x = qx, y = qy), units = dict(x = ux, y = uy), name = name,  check_data = check_data)
    
    def save(self, save_dir, FN, check_existing = True):
        
        x_col_name = self.qx
        y_col_name = self.qy
        
        if self.ux != "":
            x_col_name = x_col_name + f' ({self.ux})'

        if self.uy != "":
            y_col_name = y_col_name + f' ({self.uy})'

        df = pd.DataFrame({x_col_name : self.x, y_col_name : self.y})
        TFN = join(save_dir, FN)
        if check_existing:
            if save_ok(TFN):
                df.to_csv(TFN, header = True, index = False)
        else:
            df.to_csv(TFN, header = True, index = False)
            
    def idfac_fit_old(self):
        
        m, b = np.polyfit(np.log10(self.x), self.y, 1)
        nid = q/(k * T_RT * math.log(10)) * m
        
        fit = xy_data(self.x, m*np.log10(self.x) + b, quants = {"x": "Light intensity", "y": "Voc"}, units = {"x": "mW/cm2", "y": "V"}, name = 'fit')
        fit.nid = nid
        
        return fit 
    
    def lowpass_filter(self, test = False, yscale = 'log', left = None, right = None, T = 5.0, fs = 30.0, cutoff = 0.7, order = 2, filter_only_from_left_to_right = False):
    
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 0.7      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
    
        def butter_lowpass_filter(data, cutoff, fs, order):
            normal_cutoff = cutoff / nyq
            # Get the filter coefficients 
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y
    
        # Filter the data, and plot both the original and filtered signals.
        data = self.y.copy()
    
        if filter_only_from_left_to_right:
            r = range(findind(self.x, left), findind(self.x, right)+1)
            y = self.y.copy()
            y[r] = butter_lowpass_filter(data, cutoff, fs, order)[r]

        else:
            y = butter_lowpass_filter(data, cutoff, fs, order)
        
        if test == True:
            xy_filt = self.copy()
            xy_filt.y = y
            mxy = mxy_data([self, xy_filt])
            mxy.label = ['original', 'filtered']
            if left != None and right != None:
                m_max = mxy.max_within(left = left, right = right)
                m_min = mxy.min_within(left = left, right = right)
                if m_min < 0:
                    m_min = m_max/100
                mxy.plot(yscale = yscale, left = left, right = right, bottom = m_min*0.9, top = m_max*1.1, title = 'Check if filter is ok')
            else:
                mxy.plot(yscale = yscale, title = 'Check if filter is ok')
        elif test == False:
            self.y = y
            
    def savgol(self, n1 = 51, n2 = 1, name = None):
            
        sgf = self.copy()
        sgf.y = savgol_filter(self.y, n1, n2)

        if name != None:
            sgf.name = name
                
        return sgf 
    
    def residual(self, other, left = None, right = None, relative = False):
        #Attention: selv and other have to have the same x_values!
        
        d = self.copy()
        
        if left == None:
            left = min(self.x)
        if right == None:
            right = max(self.x)
    
        le = findind(self.x, left)
        ri = findind(self.x, right)
    
        ra = range(le, ri+1)
        x = self.x[ra]
    
        d.x = x
        #d.y = self.y[ra]/other.y[ra] - 1
        d.y = self.y[ra]-other.y[ra]
        d.qy = 'Residual (self-other)'
        if relative:
            d.y = d.y/self.y[ra]
            d.qy = 'Relative residual (self-other)/self'

        #print(d.y)
        #d.qy = 'self/other - 1'
        d.uy = ''
        return d
    
    @staticmethod
    def chisquare(data, fit, left = None, right = None):
        # data and fit must have the same x values
        if left == None:
            left = min(data.x)
        if right == None:
            right = max(data.x)
    
        le = findind(data.x, left)
        ri = findind(data.x, right)
    
        ra = range(le, ri+1)

        res = data.residual(fit, left = left, right = right)
        #return np.sum(np.array([res.y[i]**2/fit.y[i] for i in range(len(data.y))]))/len(data.y)
        return np.sum(res.y**2/ np.abs(fit.y[ra]))/len(data.y[ra])
            
    def diff(self, left = None, right = None):
        
        self_asc = self.copy()
        self_asc.strictly_ascending()
        
        if left == None:
            left = min(self_asc.x)
        if right == None:
            right = max(self_asc.x)

        le = findind(self_asc.x, left)
        ri = findind(self_asc.x, right)

        ra = range(le, ri+1)
        x = self_asc.x[ra]
        
        dydx = np.gradient(self_asc.y[ra], x)
        name = f'First derivative of: {self.name}'
        quants = dict(x = self_asc.qx, y = f'd({self.qy})/d({self.qx})')
        units = dict(x = self_asc.ux, y = f'{self.uy}/{self.ux}')
                
        return type(self_asc)(x, dydx, quants = quants, units = units, name = name)
    
           
    def max_within(self, left = None, right = None):
        """
        Returns the maximum y-value within left < x < right.
        left: left x boundary
        right: right y boundary
        If no values for left or right are given then the global maximum is returned.
        """

        l = left
        r = right

        if (left == None) or (left < min(self.x)): 
            l = min(self.x)
        if (right == None) or (right > max(self.x)):
            r = max(self.x)

        ra = range(findind(self.x, l), findind(self.x, r)+1)
        
        return max(self.y[ra])    
    
    
    def min_within(self, left = None, right = None, absolute = False):
        """
        Returns the minimum y-value within left < x < right.
        left: left x boundary
        right: right y boundary
        If no values for left or right are given then the global maximum is returned.
        """

        l = left
        r = right

        if (left == None) or (left < min(self.x)): 
            l = min(self.x)
        if (right == None) or (right > max(self.x)):
            r = max(self.x)

        ra = range(findind(self.x, l), findind(self.x, r)+1)
        
        if absolute:
            return min(np.absolute(self.y[ra]))
        else:
            return min(self.y[ra])
        
    
    def bottom_top_for_plot(self, left = None, right = None, yscale = 'log', divisor = None):
        """
        Returns the minimum and maximum y-value within left < x < right.
        This is important for plotting a graph.
        If yscale == 'log' then values <= 0 for bottom lead to an error, this is accounted for.
        divisor can be used to define the bottom as top/divisor.
        """
        top = self.max_within(left = left, right = right)
        
        if yscale == 'log':
            top *= 1.1
            bottom = self.min_within(left = left, right = right, absolute = True)
            if bottom == 0:
                if divisor == None:
                    print('Attention: bottom = 0, use a divisor to self-define the bottom = top/divisor, here divisor = 1e8 is used as standard!')
                    bottom = top/1e8
                else:
                    bottom = top/divisor

        else:
            bottom = self.min_within(left = left, right = right)
            delta = top - bottom
            top = top + delta/10
            bottom = bottom - delta/10
        
        return bottom, top

    
    def zero_data(self, left = None, right = None):
        """
        Sets the y-values to zero from x = left to x = right.
        """
    
        l = left
        r = right
    
        if (left == None) or (left < min(self.x)): 
            l = min(self.x)
        if (right == None) or (right > max(self.x)):
            r = max(self.x)
    
        ra = range(findind(self.x, l), findind(self.x, r)+1)
        new = self.copy()
        new.y[ra] = 0
    
        return new
    
    def cut_data_outside(self, left = None, right = None):
        """
        Cuts the data outside x = [left, right].
        """
    
        l = left
        r = right
    
        if (left == None) or (left < min(self.x)): 
            l = min(self.x)
        if (right == None) or (right > max(self.x)):
            r = max(self.x)
    
        ra = range(findind(self.x, l), findind(self.x, r)+1)
        new = self.copy()
        new.x = new.x[ra]
        new.y = new.y[ra]
    
        return new
    
    def reverse(self):
        self.x = self.x[::-1]
        self.y = self.y[::-1]
        
    def swap_axes(self):
        self_copy = self.copy()
        new_x = self.y
        new_y = self.x
        new_qx = self.qy
        new_qy = self.qx
        new_ux = self.uy
        new_uy = self.ux
        
        self_copy.x = new_x
        self_copy.y = new_y
        self_copy.qx = new_qx
        self_copy.qy = new_qy
        self_copy.ux = new_ux
        self_copy.uy = new_uy

        return self_copy
        
    def remove_nan(self):
        # Removes all numpy.nan values in self.x and self.y (only gives sensible result if there is a nan in both x[i] and y[i])
        x_raw = self.x
        y_raw = self.y
        
        x_list = np.logical_not(np.isnan(x_raw))
        y_list = np.logical_not(np.isnan(y_raw))
        logic_list = [x_list[idx] and y_list[idx] for idx in range(len(x_list))]
        self.x = x_raw[logic_list]
        self.y = y_raw[logic_list]
        #if len(self.x) != len(self.y):
        #    print('Attention: xy_data.remove_nan() gave an x-array and a y-array with different sizes. The reason could be that, e.g. x[i] = nan but y[i] = number')

        
    def keep_interval(self, intval):
        """
        Legacy, use cut_data_outside!
        Cuts the data outside x = [left, right]. 
        """
        left = findind(self.x, intval[0])
        right = findind(self.x, intval[1])
        ra = range(left, right+1)
        self.x = self.x[ra]
        self.y = self.y[ra]
        
    def idfac_fit(self, left = None, right = None, plot = False, plotrange = [None, None], return_fit = True):
        
        if (left == None) or (left < min(self.x)):
            left = min(self.x)
        if (right == None) or (right > max(self.x)):
            right = max(self.x)
        
        ra = range(findind(self.x, left), findind(self.x, right)+1)
            
        m, b = np.polyfit(np.log10(self.x[ra]), self.y[ra], 1)
        nid = q/(k * T_RT * math.log(10)) * m
        
        fit = xy_data(self.x, m*np.log10(self.x) + b, quants = {"x": "Light intensity", "y": "Voc"}, units = {"x": "mW/cm2", "y": "V"}, name = 'fit')
        fit.nid = nid
        
        if plot:
            if (plotrange[0] == None) or (plotrange[0] < min(self.x)):
                plot_left = min(self.x) * 0.5
            else:
                plot_left = plotrange[0]
                
            if (plotrange[1] == None) or (plotrange[1] > max(self.x)):
                plot_right = max(self.x) * 1.1
            else:
                plot_right = plotrange[1]
                
            self.plotstyle = dict(linestyle = 'None', marker = 'o', color = 'green', markersize = 20)
            fit.plotstyle = dict(linestyle = '-', color = 'green', linewidth = 5)
            da = mxy_data([self, fit])
            da.label([self.name, f'm = ln(10) $\cdot$ kT/q $\cdot$ {fit.nid:.2f}'])
            da.plot(xscale = 'log', left = plot_left, right = plot_right, bottom = 0.9 * self.min_within(left=plot_left, right=plot_right), top = 1.1 * self.max_within(left=plot_left, right=plot_right), plotstyle = 'individual')
        
        if return_fit:    
            return fit

        
    def product(self, s2, qy = None, uy = None, delta = 1):
        # Calculates the product self.y*s2.y at the right wavelengths and returns a new instance of class of the same class as self
    
        #Determine the global minimum x-value 
        x1_min = min(self.x)
        x2_min = min(s2.x)
        if x1_min < x2_min:
            x_min = x2_min
        else:
            x_min = x1_min
        
        #Determine the global maximum x-value 
        x1_max = max(self.x)
        x2_max = max(s2.x)
        if x1_max < x2_max:
            x_max = x1_max
        else:
            x_max = x2_max
        
        d1 = self.copy()
        d2 = s2.copy()
            
        d1.equidist(left = x_min, right = x_max, delta = delta)
        d2.equidist(left = x_min, right = x_max, delta = delta)
        
        result = d1.copy()
        result.y = d1.y * d2.y
        
        if qy != None:
            result.qy = qy
        if uy != None:
            result.uy = uy
        
        return result
    
    def all_values_greater_min(self, min = None):
        """
        Looks for values < min and sets them to min.
        """
        self.y = np.array([self.y[i] if (self.y[i] > min) else min for i in range(len(self.y))], dtype = np.float64)
    
    def shift_x(self, x):
        """
        Shifts the x-values by x
        """
        self.x = self.x + x
        
    def shift_y(self, y):
        """
        Shifts the y-values by y
        """
        self.y = self.y + y

        
    def idx_range(self, left = None, right = None):
        # Returns the x-index range which goes from x = left to x = right
        return idx_range(self.x, left = left, right = right)
    
    def polyfit(self, order = 1, left = None, right = None, new_x_arr = None, new_meshsize = None):
        # Fits a polynomial of order to self and returns the data as an object of the same class as self.
        # new_x_arr: The x values of the fit.
        # new_meshsize: If no new_x_arr is provided, new_meshsize is a number of evenly spaced x-values between left and right.
        ra = self.idx_range(left = left, right = right)
        p = np.poly1d(np.polyfit(self.x[ra], self.y[ra], order))
        fit = self.copy()
        if new_x_arr == None:
            if new_meshsize != 0:
                fit.x = np.linspace(self.x[ra[0]], self.x[ra[-1]], new_meshsize)
            else:
                fit.x = self.x[ra]
        else:
            fit.x = new_x_arr
        fit.y = p(fit.x)
        fit.name = 'fit of ' + self.name
        return fit

    def del_first_and_last_n_data_points(self, n=1):
        r = range(n, len(self.x)-n)
        self.x = self.x[r]
        self.y = self.y[r]
        
    def del_edge_zero_data(self):
        start_idx = 0
        while self.y[start_idx] == 0:
            start_idx += 1
        stop_idx = len(self.y) - 1
        while self.y[stop_idx] == 0:
            stop_idx -= 1
        r = range(start_idx, stop_idx + 1)
        self.x = self.x[r]
        self.y = self.y[r]

    def rm_cosray(self, m = 3, threshold = 5):
        """
        Removes cosmic rays from the spectrum.
        m: 2 m + 1 points around the spike are selected
        threshold: The threshold value from which on a spike is detected as such
        """
        sp = self.copy()
    
        def modified_z_score(intensity):
            median_int = np.median(intensity)
            mad_int = np.median([np.abs(intensity - median_int)])
            modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
            return modified_z_scores
    
        spikes = abs(np.array(modified_z_score(np.diff(sp.y)))) > threshold
    
        y_out = sp.y.copy() # So we don’t overwrite self 
        l_spikes = len(spikes)
        for i in np.arange(l_spikes):
            if spikes[i] != 0: # If we have an spike in position i
                # Make sure that interval does not exceed the boundaries
                if i >= l_spikes - m:
                    m_right = l_spikes - i - 1
                else:
                    m_right = m 
                if i < m:
                    m_left = i
                else:
                    m_left = m
                w = np.arange(i-m_left,i+1+m_right) # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                if len(w2) != 0:
                    y_out[i] = np.mean(sp.y[w2]) # and we average their values
    
        sp.y = y_out
        
        return sp    
    
    def monotoneous_ascending(self):
        # Orders the data, so that x is monotoneous ascending
        s = self.x.argsort()
        self.x = self.x[s]
        self.y = self.y[s]


    def strictly_ascending(self):
    # Transforms the data self, so that self.x is strictly ascending
    # Important is that the data is monotoneous ascending. To accomplish this use
    # the method monotoneous_ascending.
        idx = 0
        while True:
            if idx == len(self.x)-1:
                break
            else:
                if self.x[idx+1] > self.x[idx]:
                    idx += 1
                else:
                    self.x = np.delete(self.x, idx+1)
                    self.y = np.delete(self.y, idx+1)
    
    
class mxy_data:
    
    def __init__(self, sa):
        self.sa = sa
        self.label_defined = False
        self.n_y = len(sa)
        if self.n_y != 0:
            self.n_x = len(sa[0].x)
        else:
            self.n_x = 0
        
    def __mul__(self, other):
        new_sa = []
        for i, sp in enumerate(self.sa):
            new_sa.append(sp * other)
        return type(self)(new_sa)
    
    def __iter__(self):
        return iter(self.sa)
        
    def qx_ux(self, qx, ux):
        for i, sp in enumerate(self.sa):
            sp.qx = qx
            sp.ux = ux

    def qy_uy(self, qy, uy):
        for i, sp in enumerate(self.sa):
            sp.qy = qy
            sp.uy = uy
            
    def copy(self):
        sa_new = []
        for i, sp in enumerate(self.sa):
            sa_new.append(sp.copy())
        ms = type(self)(sa_new)
        if self.label_defined:
            ms.label(self.lab)
        ms.n_y = self.n_y
        ms.n_x = self.n_x
        return ms
        
    def add(self, data): #depreciated, use append
        self.sa.append(data)
        self.label_defined = False
        
    def append(self, data):
        self.sa.append(data)
        self.label_defined = False
        
    @staticmethod
    def combine(mxy1, mxy2):
        """
        Returns an instance new_mxy of type(mxy1) with new_mxy.sa = mxy1.sa + mxy2.sa.
        """
        if type(mxy1) != type(mxy2):
            print('mxy_data.combine(mxy1, mxy2): mxy1 and mxy2 are of different type!')
        #Make sure that there are no dependencies of new_mxy on mxy1 and mxy2 
        mxy1_ = mxy1.copy()
        mxy2_ = mxy2.copy()
        sa = mxy1_.sa + mxy2_.sa
        new_mxy = type(mxy1)(sa)
        if mxy1.label_defined and mxy2.label_defined:
            new_mxy.label(mxy1_.lab + mxy2_.lab)
            new_mxy.label_defined = True
        return new_mxy
        
    def delete(self, data):
        i = 0
        while i < len(self.sa):
            if self.sa[i].name == data.name:
                self.sa.pop(i)
            else:
                i += 1
                
    @classmethod
    def generate_empty(cls):
        return cls([])

        
    def label(self, lab):
        self.lab = lab
        self.label_defined = True
        
    def no_label(self):
        self.label_defined = False
                
    def replace(self, idx, sp_new):
        self.sa[idx] = sp_new
        
    def remain(self, idx_list):
        """
        Return all spectra with indices in list idx_list.
        """
        sa = []
        lab = []
        for i, idx in enumerate(idx_list):
            new = self.sa[idx].copy()
            sa.append(new)
            if self.label_defined:
                lab.append(self.lab[idx])
        #rem_sa = mxy_data(sa)
        rem_sa = type(self)(sa)
        rem_sa.label(lab)
        return rem_sa
    
    def set_plotstyle(self, linestyle = None, marker = None, color = None, markersize = None, linewidth = None):
        for idx, sp in enumerate(self.sa):
            
            if linestyle != None:
                sp.plotstyle['linestyle'] = linestyle
            
            if marker != None:
                sp.plotstyle['marker'] = marker
            
            if color != None:
                sp.plotstyle['color'] = color
            
            if markersize != None:
                sp.plotstyle['markersize'] = markersize
            
            if linewidth != None:
                sp.plotstyle['linewidth'] = linewidth

    def names_to_label(self, split_ch = None):
        lab = []
        for i, sp in enumerate(self.sa):
            if split_ch == None:
                lab.append(sp.name)
            else:
                lab.append(sp.name.split(split_ch)[0])
        self.label(lab)
        self.label_defined = True
        
        
    def print_all_names(self, split_ch = None, unique_only = False, print_all = True, print_idx = True, return_list = False):
        all_names = []
        idx = 0
        for i, sp in enumerate(self.sa):
            if split_ch == None:
                next_name = sp.name
            else:
                next_name = sp.name.split(split_ch)[0]
            if unique_only:
                if not(next_name in all_names):
                    all_names.append(next_name)
                    if print_all:
                        if print_idx:
                            print(f'{idx}: {next_name}')
                        else:
                            print(next_name)
                    idx += 1
            else:
                all_names.append(next_name)
                if print_all:
                    if print_idx:
                        print(f'{idx}: {next_name}')
                    else:
                        print(next_name)
                idx += 1
        if return_list:
            return all_names
    
    def print_names_containing(self, name):
        for i, sp in enumerate(self.sa):
            if name in sp.name:
                print(sp.name)
        
    
    def plot(self, title = '', ax= None, xscale = 'linear', yscale = 'linear', left = None, right = None, 
             bottom = None, divisor = None, top = None, plotstyle = 'auto', showindex = False, in_name = [], not_in_name = None,
             plot_table = False, cell_text = None, row_labels = None, col_labels = None,
             bbox = [0.3, 0.25, 0.1, 0.5], figsize=(9,6), hline = None, hline_colors = None, vline = None, vline_colors = None, nolabel = False, 
             return_fig = False, generate_image_stream=False, show_plot = True, ylabel=None, xlabel=None, create_image_stream= False, **kwargs):

        """
        Plots multiple xy-data of type xy_data. The axis title are taken from the first spectrum.
        showindex: If True then the index of the sa list will be shown before the regular label. 
        This is helpful when certain curves have to be selected e.g. for PLQY. 
        in_name: e.g ['laser'], List with strings that have to be in the name to be plotted. If [] then everything is plotted.
        If individual xy_data has the attribute plotrange (list of begin and end value), than only this plotrange is plotted.
        return_fig: if True than the figure is returned as an object of type matplotlib.figure.Figure. This figure can be then saved with matplotlib.figure.Figure.savefig(filename).
                    To show this returned figure one can use the function matplotlib.figure.Figure.show(); this works however only if a GUI backend is chosen, 
                    e.g. by %matplotlib qt in jupyterlab (%matplotlib inline doesn't work').
                    if show_plot == False and retrun_fig == True: It is important to close the figure with matplotlib.pyplot.plt.close(Figure), otherwise it will be shown.
        example for kwargs:
        kwargs = dict(fontsize = 24, legend = False, save_plot = True, plot_save_dir = save_dir, plot_FN = 'IV5 - transp. limit.png')
        """
        
        if ax is not None:
            show_plot = False

        self_old = self.copy()
        if in_name != []:

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
            
            idx_list = []
            for idx, sp in enumerate(self.sa):                
                if in_name_in_spec(in_name, sp):
                    idx_list.append(idx)
    
            self = self.remain(idx_list)
            
        if not(not_in_name is None):
            
            def not_in_name_in_spec(not_in_name, spec):
                
                result = False
    
                for i, name in enumerate(not_in_name):
                    
                    if name in spec.name:
                        result = True
                        break
    
                return result
            
            idx_list = []
            for idx, sp in enumerate(self.sa):                
                if not(not_in_name_in_spec(not_in_name, sp)):
                    idx_list.append(idx)
    
            self = self.remain(idx_list)

        if ax is None:
            plt.rcParams.update({'font.size': 12})
            #plt.rc('text', usetex=True) Latex needs to be installed first
            fig, ax = plt.subplots(figsize=figsize)
            #if plot_table:
            #    ax = fig.add_subplot(111)

        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'fontsize':
                    plt.rcParams.update({'font.size': value})
        
        #if plot_table:
        #    ax = fig.add_subplot(111)
        
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
        if left is None:
            left = self.sa[0].x[0]
            for sp in self.sa:
                if sp.x[0] < left:
                    left = sp.x[0]
        if right is None:
            right = self.sa[0].x[-1]
            for sp in self.sa:
                if sp.x[-1] > right:
                    right = sp.x[-1]
        ax.set_xlim(left, right)
        
        if bottom != None:
            if top == None:
                bottom_, top = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #top = max(self.sa[0].y) + 0.1 * abs(max(self.sa[0].y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if top != None:
            if bottom == None:
                bottom, top_ = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #bottom = min(self.sa[0].y) - 0.1 * abs(min(self.sa[0].y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if (bottom == None) and (top == None):
            bottom, top = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
            ax.set_ylim(bottom = bottom, top = top)        
            
            
        for i, spec in enumerate(self.sa):
            x = spec.x.copy()
            if xscale == 'log':
                x = np.abs(x)
            y = spec.y.copy()
            if yscale == 'log':
                y = np.abs(y)
                                            
            if hasattr(spec, 'plotrange'):
                r = range(spec.x_idx_of(spec.plotrange[0]), spec.x_idx_of(spec.plotrange[1])+1)
            else:
                r = range(0,len(x))    
        
            if self.label_defined and not(nolabel):
                if plotstyle == 'auto':
                    if showindex == True:
                        ax.plot(x[r], y[r], label = f'{i}: {self.lab[i]}')
                    else:
                        ax.plot(x[r], y[r], label = self.lab[i])
                else:
                    if showindex == True:
                        ax.plot(x[r], y[r], **spec.plotstyle, label = f'{i}: {self.lab[i]}')
                    else:
                        ax.plot(x[r], y[r], **spec.plotstyle, label = self.lab[i])
                ax.legend()
            else:
                if plotstyle == 'auto':
                    ax.plot(x[r], y[r])
                else:
                    ax.plot(x[r], y[r], **spec.plotstyle)
    
        sp = self.sa[0]
        
        if xlabel is None:
            if sp.ux is not None and sp.ux != '':
                ax.set_xlabel(f'{sp.qx} ({sp.ux})')
            else:
                ax.set_xlabel(f'{sp.qx}')
        else:
            ax.set_xlabel(xlabel)
            
        if ylabel is None:

            if sp.uy is not None and sp.uy != '':
                ax.set_ylabel(f'{sp.qy} ({sp.uy})')
            else:
                ax.set_ylabel(f'{sp.qy}')
        else:
            ax.set_ylabel(ylabel)
        
        if title != '':
            ax.set_title(title)
            
        if plot_table:
            the_table = ax.table(cellText = cell_text, rowLabels = row_labels, colLabels = col_labels, loc='bottom',
                                  bbox = bbox)  #bbox = [x0, y0, width, height])
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1, 5)
            
        if 'save_plot' in kwargs:
            if kwargs['save_plot']:
                save_dir = kwargs['plot_save_dir']
                FN = kwargs['plot_FN']
                TFN = join(save_dir, FN)
                if save_ok(TFN):
                    plt.savefig(TFN)
                    
        if hline != None:
            if type(hline) == list:
                for idx, n in enumerate(hline):
                    if hline_colors is not None:
                        color = hline_colors[idx]
                    ax.axhline(y = n, color=color, linestyle='-')    
            else:
                if hline_colors is not None:
                    color = hline_colors
                else:
                    color = 'b'
                ax.axhline(y = hline, color=color, linestyle='-')

        if vline != None:
            if type(vline) == list:
                for idx, n in enumerate(vline):
                    if vline_colors is not None:
                        color = vline_colors[idx]
                    else:
                        color = 'r'
                    ax.axvline(x = n, color=color, linestyle='-')    
            else:
                if vline_colors is not None:
                    color = vline_colors
                else:
                    color = 'r'
                ax.axvline(x = vline, color=color, linestyle='-')
        
        if create_image_stream:
            image_stream = io.BytesIO()
            plt.savefig(image_stream)
            if show_plot:
                plt.show()
            return image_stream

        if show_plot:
            plt.show()
        
        self = self_old
        
        if generate_image_stream:
            image_stream = io.BytesIO()
            plt.savefig(image_stream)
            return image_stream
        
        if return_fig:
            return fig

    
    def save(self, save_dir, title, label = None):
        if label == None:
            label = self.lab
        for i, xy_dat in enumerate(self.sa):
            FN = title + ' - ' + label[i] + '.csv'
            xy_dat.save(save_dir, FN)
            
    def save_in_one_file(self, fp):
        #check if all files have the same wavelengths
        all_have_same_x = True
        x_arr = self.sa[0].x
        df = self.sa[0].to_df()
        for sp in self:
            if not np.all(sp.x == x_arr):
                all_have_same_x = False
                break
        if all_have_same_x:
            if not self.label_defined:
                self.label([sp.name for sp in self])
            for sp, label in zip(self, self.lab):
                col_name = label+', '+sp.qy
                if sp.uy is not None and sp.uy != '':
                    col_name += f' ({sp.uy})'
                df[col_name] = sp.y
            df.drop(df.columns[0], axis=1, inplace=True)
            df.to_csv(fp)
        else:
            print('Not saved: Not all xy_data have the same x!')
            
    def save_individual(self, save_dir = None, FNs = None, check_existing = True, check_FN_extension = True):
        
        quitted = False
        if save_dir == None:
            save_dir = os.getcwd()
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
            df = pd.DataFrame({x_col_name : sp.x, y_col_name : sp.y})
            #FN = FN.split('.'+FN.split('.')[-1])[0] + '.csv'
            if check_FN_extension:
                FN = os.path.splitext(FN)[0] + '.csv'
            else:
                FN = FN + '.csv'

            if check_existing:
            
                ok_to_save, quitted = save_ok(TFN, quitted)
                if ok_to_save and not(quitted):

                    df.to_csv(join(save_dir, FN), header = True, index = False)
                    
            else:
                    df.to_csv(join(save_dir, FN), header = True, index = False)
            
    @classmethod
    def load_individual(cls, directory, delimiter = ',', header = 'infer', quants = {"x": "x", "y": "y"}, units = {"x": "", "y": ""}, take_quants_and_units_from_file = False):

        """
        Loads all xy data in individual files in directory.
        """
        FNs = os.listdir(directory)
        sa = []
        
        for i, FN in enumerate(FNs):
            
            dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)
                        
            x = np.array(dat)[:,0].astype(np.float64)
            y = np.array(dat)[:,1].astype(np.float64)

            if cls.__name__ == 'mxy_data':
                sp = xy_data(x, y, quants, units, FN)
       
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
            
    def max_within(self, left = None, right = None):
        """
        Returns the maximum y-value within left < x < right.
        left: left x boundary
        right: right y boundary
        If no values for left or right are given then the global maximum is returned.
        """
        # Take starting value
        #maximum = self.sa[0].y[0]
        maximum = 0

        for i, sp in enumerate(self.sa):
            l = left
            r = right
                
            if (left == None) or (left < min(sp.x)): 
                l = min(sp.x)
            if (right == None) or (right > max(sp.x)):
                r = max(sp.x)

            ra = range(findind(sp.x, l), findind(sp.x, r)+1)

            if ra != range(0,0):
                m = max(sp.y[ra])
                if m > maximum:
                    maximum = m

        return maximum
    

    def min_within(self, left = None, right = None, absolute = False):
        """
        Returns the minimum y-value within left < x < right.
        left: left x boundary
        right: right y boundary
        If no values for left or right are given then the global minimum is returned.
        """
        # Take starting value
        minimum = abs(max(self.sa[0].y))

        for i, sp in enumerate(self.sa):

            l = left
            r = right

                
            if (left == None) or (left < min(sp.x)): 
                l = min(sp.x)
            if (right == None) or (right > max(sp.x)):
                r = max(sp.x)

            ra = range(findind(sp.x, l), findind(sp.x, r)+1)
            
            if ra != range(0,0):

                if absolute:
                    m = min(np.absolute(sp.y[ra]))
                else:
                    m = min(sp.y[ra])
                if m < minimum:
                    minimum = m

        return minimum
    
    def bottom_top_for_plot(self, left = None, right = None, yscale = 'log', divisor = None):
        """
        Returns the minimum and maximum y-value within left < x < right.
        This is important for plotting a graph.
        If yscale == 'log' then values <= 0 for bottom lead to an error, this is accounted for.
        divisor can be used to define the bottom as top/divisor.
        """

        top = self.max_within(left = left, right = right)
        
        if yscale == 'log':
            top *= 1.1
            bottom = self.min_within(left = left, right = right, absolute = True)
            if bottom == 0:
                if divisor == None:
                    print('Attention: bottom = 0, use a divisor to self-define the bottom = top/divisor, here divisor = 1e8 is used as standard!')
                    bottom = top/1e8
                else:
                    bottom = top/divisor

        else:
            bottom = self.min_within(left = left, right = right)
            delta = top - bottom
            top = top + delta/10
            bottom = bottom - delta/10
        
        return bottom, top


    def normalize(self, x_lim = None, norm_val = 1):    

        for i, sp in enumerate(self.sa):
            sp.normalize(x_lim = x_lim, norm_val = norm_val)
            
    def equidist(self, left = None, right = None, delta = 0.1, kind = 'cubic'):

        for i, sp in enumerate(self.sa):
            sp.equidist(left = left, right = right, delta = delta, kind = kind)
        
            
    def all_values_greater_min(self, min = None):
        """
        Looks for values < min and sets them to min.
        """
        for idx, sp in enumerate(self.sa):
            sp.all_values_greater_min(min = min)
            
    def cut_data_outside(self, left = None, right = None):
        """
        Cuts the data outside x = [left, right].
        """
  
        new_sa = []          
        for idx, sp in enumerate(self.sa):
            new_xy = sp.cut_data_outside(left = left, right = right)
            new_sa.append(new_xy)
        
        new_mxy = type(self)(new_sa)
        
        return new_mxy
    
    def reverse(self):
        """
        Reverse the x values in all spectra.
        Returns the reversed spectra.
        """
        new = self.copy()
        for idx, sp in enumerate(new.sa):
            new.sa[idx].reverse()
            
        return new
    
    def del_first_and_last_n_data_points(self, n=1):
        for idx, sp in enumerate(self.sa):
            sp.del_first_and_last_n_data_points(n=n)
            
    def del_edge_zero_data(self):
        for idx, sp in enumerate(self.sa):
            sp.del_edge_zero_data()
            
            
    def rm_cosray(self, m = 3, threshold = 5):
        new = self.copy()
        for idx, sp in enumerate(new.sa):
            sp_new = sp.rm_cosray(m = m, threshold = threshold)
            new.replace(idx, sp_new)
        return new
    
    
    def idfac_fit(self, left = None, right = None, plot = True, plotrange = [None, None], return_all = False):
    
        all_data = []
        all_data_label = []
        save_names = []
        for i, sp in enumerate(self.sa):
    
            sp.plotstyle = dict(linestyle = 'None', marker = 'o', color = gen.colors[i], markersize = 20)
            all_data.append(sp)
            label = sp.name.split('.csv')[0]
            all_data_label.append(label)
            save_names.append(label)
    
            all_data.append(all_data[-1].idfac_fit(left = left, right = right))
            #fit[i].plotstyle = dict(linestyle = '-', color = colors[i], linewidth = 5)
            all_data[-1].plotstyle = dict(linestyle = '-', color = gen.colors[i], linewidth = 5)
            all_data_label.append(f'm = ln(10) $\cdot$ kT/q $\cdot$ {all_data[-1].nid:.2f}')
            save_names.append(label+'_fit')
    
        da = mxy_data(all_data)
        #da.label([FNs[0], FNs[1], f'm = ln(10) $\cdot$ kT/q $\cdot$ {fit[0].nid:.2f}', f'm = ln(10) $\cdot$ kT/q $\cdot$ {fit[1].nid:.2f}'])
    
        da.label(all_data_label)
        if plot:
            da.plot(xscale = 'log', plotstyle = 'individual', figsize = (10,7))
        if return_all:
            return da
        
    def shift_x(self, x):
        """
        Shifts the x-values by x
        """
        for sp in self.sa:
            sp.shift_x(x)
            
    def shift_y(self, y):
        """
        Shifts the y-values by y
        """
        for sp in self.sa:
            sp.shift_y(y)
            
    def average(self):
        # Averages over all spectra and returns the averaged spectrum
        av = self.sa[0].copy()
        av.y *= 0
        for sp in self.sa:
            av += sp
        av /= len(self.sa)
        return av
    
    def diff(self, left = None, right = None):
        new = self.copy()
        sa = []
        for sp in new.sa:
            sa.append(sp.diff(left=left, right=right))
        new.sa = sa
        return new
    
    def remove_nan(self):
        for sp in self.sa:
            sp.remove_nan()
            
    def strictly_ascending(self):
        for sp in self.sa:
            sp.strictly_ascending()

                
class xyz_data:
    
    def __init__(self, x, y, z , quants = {"x": "x", "y": "y", "z": "z"}, units = {"x": "", "y": "", "z": ""}, name = '', plotstyle = dict(linestyle = '-', color = 'black', linewidth = 3)):
        """
        x is a numpy array e.g. the wavelengths or photon energies
        y is a numpy array e.g. cts, cps, photon flux, spectral flux
        z is a numpy array e.g. cts, cps, photon flux, spectral flux
        quants is a dict with the type of the data, e.g. {"x": "Wavelength", "y": "np.trantensity", "z": ""}
        units is a dict with the units of the data, e.g. {"x": "nm", "y": "cps", "z": ""}
        name is the name of the data, e.g. the file name
        """
        self.x = x
        self.y = y
        self.z = z
        self.qx = quants["x"]
        self.qy = quants["y"]
        self.qz = quants["z"]
        self.ux = units["x"]
        self.uy = units["y"]
        self.uz = units["z"]
        self.name = name
        self.plotstyle = plotstyle
        
    @classmethod
    def load(cls, directory, FN = '', delimiter = ',', header = 'infer', 
              quants = {"x": "x", "y": "y", "z": "z"}, units = {"x": "", "y": "", "z": ""}, take_quants_and_units_from_file = False):

        """
        Loads a single xyz data. If a filename is given it will be used, if not the first file in the directory will be used.
        """
        
        if FN == '':
            FN = os.listdir(directory)[0]
        dat = pd.read_csv(join(directory, FN), delimiter = delimiter, header = header)
        
        x = np.array(dat)[:,0].astype(np.float64)
        y = np.array(dat)[:,1].astype(np.float64)
        z = np.array(dat)[:,2].astype(np.float64)
        
        qx = quants["x"]
        qy = quants["y"]
        qz = quants["z"]
        ux = units["x"]
        uy = units["y"]
        uz = units["z"]
                
        if take_quants_and_units_from_file:
            
            col0 = list(dat)[0]
            qx = col0.split(' (')[0]
            if ' (' in col0:
                ux = col0.split(' (')[1].split(')')[0]

            col1 = list(dat)[1]
            qy = col1.split(' (')[0]
            if ' (' in col1:
                uy = col1.split(' (')[1].split(')')[0]
                
            col2 = list(dat)[2]
            qz = col2.split(' (')[0]
            if ' (' in col2:
                uz = col2.split(' (')[1].split(')')[0]
        
        return cls(x, y, z, quants = dict(x = qx, y = qy, z = qz), units = dict(x = ux, y = uy, z = uz), name = FN)
