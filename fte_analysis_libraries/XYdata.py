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
from importlib.resources import files as _resource_files
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
from typing import Any


system_dir = str(_resource_files('fte_analysis_libraries').joinpath('System_data'))


def _draw_hlines_vlines(ax: Any, hline: Any, hline_colors: Any, vline: Any, vline_colors: Any) -> None:
    """Render horizontal and vertical reference lines onto a matplotlib Axes."""
    if hline is not None:
        if isinstance(hline, list):
            for idx, n in enumerate(hline):
                color = hline_colors[idx] if hline_colors is not None else 'b'
                ax.axhline(y=n, color=color, linestyle='-')
        else:
            color = hline_colors if hline_colors is not None else 'b'
            ax.axhline(y=hline, color=color, linestyle='-')
    if vline is not None:
        if isinstance(vline, list):
            for idx, n in enumerate(vline):
                color = vline_colors[idx] if vline_colors is not None else 'r'
                ax.axvline(x=n, color=color, linestyle='-')
        else:
            color = vline_colors if vline_colors is not None else 'r'
            ax.axvline(x=vline, color=color, linestyle='-')


def _bottom_top_for_plot(obj: Any, left: float | None=None, right: float | None=None, yscale: str='log', divisor: float | None=None) -> None:
    """Return (bottom, top) y-limits for plotting, handling log-scale edge cases."""
    top = obj.max_within(left=left, right=right)
    if yscale == 'log':
        top *= 1.1
        bottom = obj.min_within(left=left, right=right, absolute=True)
        if bottom == 0:
            if divisor is None:
                print('Attention: bottom = 0, use a divisor to self-define the bottom = top/divisor, here divisor = 1e8 is used as standard!')
                bottom = top / 1e8
            else:
                bottom = top / divisor
    else:
        bottom = obj.min_within(left=left, right=right)
        delta = top - bottom
        top = top + delta / 10
        bottom = bottom - delta / 10
    return bottom, top  # type: ignore


class XYData:
    """
    Container for a single (x, y) data set with units and plotting helpers.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": None, "y": None}, name: str = '', plotstyle: Any = None, check_data: bool = True) -> None:
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
        if plotstyle is None:
            plotstyle = dict(linestyle='-', color='black', linewidth=3)
        self.plotstyle = plotstyle
        if check_data:
            ok = self.data_check()
            if not(ok):
                print('To switch off this message use check_data = False')

        
    def _align_with(self, other: Any) -> tuple[Any, Any]:
        """Return grid-aligned copies of self and other over their overlapping x range."""
        s, o = self.copy(), other.copy()
        x_min = max(min(self.x), min(other.x))
        x_max = min(max(self.x), max(other.x))
        delta = min(self.x[1] - self.x[0], other.x[1] - other.x[0])
        s.equidist(left=x_min, right=x_max, delta=delta, kind='cubic')
        o.equidist(left=x_min, right=x_max, delta=delta, kind='cubic')
        return s, o

    def __mul__(self, other: Any) -> None:
        """
        Multiply element-wise with another object or scalar.
        
        Parameters
        ----------
        other : Any
            Other.
        
        Examples
        --------
        >>> obj.__mul__()
        """
        if type(self).mro()[-2] == type(other).mro()[-2]:
            s, o = self._align_with(other)
            s.y = s.y * o.y
        else:
            s = self.copy()
            s.y = s.y * other
        return s

    def __add__(self, other: Any) -> None:
        """
        Add another object or scalar element-wise.
        
        Parameters
        ----------
        other : Any
            Other.
        
        Examples
        --------
        >>> obj.__add__()
        """
        if type(self).mro()[-2] == type(other).mro()[-2]:
            s, o = self._align_with(other)
            s.y = s.y + o.y
        else:
            s = self.copy()
            s.y = s.y + other
        return s

    def __sub__(self, other: Any) -> None:
        """
        Subtract another object or scalar element-wise.
        
        Parameters
        ----------
        other : Any
            Other.
        
        Examples
        --------
        >>> obj.__sub__()
        """
        if type(self).mro()[-2] == type(other).mro()[-2]:
            s, o = self._align_with(other)
            s.y = s.y - o.y
        else:
            s = self.copy()
            s.y = s.y - other
        return s

    def __truediv__(self, other: Any) -> None:
        """
        Divide element-wise by another object or scalar.
        
        Parameters
        ----------
        other : Any
            Other.
        
        Examples
        --------
        >>> obj.__truediv__()
        """
        if type(self).mro()[-2] == type(other).mro()[-2]:
            s, o = self._align_with(other)
            s.y = s.y / o.y
        else:
            s = self.copy()
            s.y = s.y / other
        return s
            
    @classmethod
    def from_df(cls, df: Any, y_col: int= 0, take_quants_and_units_from_df: bool= True, **kwargs) -> Any:
        """
        Construct an XYData object from a pandas DataFrame column.
        
        Parameters
        ----------
        df : Any
            Df.
        y_col : int
            Y col.
        take_quants_and_units_from_df : bool
            Take quants and units from df.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.from_df()
        """
        #The index is taken as
        #y_col can be either an integer to denote the y_colth column or a column name
    
        if not isinstance(df, pd.core.frame.DataFrame):
            if isinstance(df, pd.core.series.Series):
                df = df.to_frame()
            else:
                print(f'Attention df has to be a DataFrame or a Series! However, it is of type {type(df)}')
                return
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
            qy_uy = y_col.split(' (')  # type: ignore
            qy = qy_uy[0]
            if len(qy_uy) == 2:
                uy = qy_uy[1].split(')')[0]
            else:
                uy = None
            return cls(x, y, quants= [qx, qy], units= [ux, uy], **kwargs)
            
        else:
            return cls(x, y, **kwargs)
                
    def to_df(self) -> Any:
        """
        Export the XYData as a pandas DataFrame with labelled columns.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.to_df()
        """
        col_name = self.qy
        if self.uy is not None and self.uy != '':
            col_name += f' ({self.uy})'
        index_name = self.qx
        if self.ux is not None and self.ux != '':
            index_name += f' ({self.ux})'
        return pd.DataFrame({index_name: self.x, col_name: self.y}).set_index(index_name)
    
    def data_check(self) -> Any:
        """
        Validate x/y arrays and optionally sort or trim the data.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.data_check()
        """
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
    def generate_empty(cls) -> Any:
        """
        Generate an XYData with all-zero y values on a given x grid.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.generate_empty()
        """
        x = np.array([])
        y = np.array([])
        return cls(x, y)
        
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
        x = self.x.copy()
        y = self.y.copy()
        qx = self.qx
        qy = self.qy
        ux = self.ux
        uy = self.uy
        name = self.name[:]
        plotstyle = self.plotstyle.copy()
        return type(self)(x, y, quants = dict(x = qx, y = qy), units = dict(x = ux, y = uy), name = name, plotstyle = plotstyle, check_data = False)
        
    def y_of(self, x_value: Any, interpolate: Any = False) -> Any:
        """
        Interpolate and return the y value at a given x position.
        
        Parameters
        ----------
        x_value : Any
            X value.
        interpolate : Any
            Interpolate.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.y_of()
        """
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
    
    
    def x_idx_of(self, x_value: Any) -> Any:
        """
        Works only for ascending arrays!
        """
        idx = findind(self.x, x_value)
        return idx
    
    def x_of(self, y_value: Any, start: float | None = None,  interpolate: Any = False) -> Any:
        """
        Find the first x_value > start  where self.y = y_value
        """

        if start is None:
            idx_start = 0
        else:
            idx_start = findind(self.x, start)
        
        if interpolate:
            # It is important that self.y is monotonous.
            # Make sure that there are no duplicate y-values:
            x_arr = self.x[idx_start:]
            y_arr = self.y[idx_start:]
            seen = set()
            idx_arr = [idx for idx in range(len(y_arr)) if y_arr[idx] not in seen and not seen.add(y_arr[idx])]  # type: ignore
            #idx_arr = [0]+[idx+1 for idx in range(len(y_arr)-1) if (y_arr[idx+1] != y_arr[idx])]
            x_arr_new = np.array([x_arr[idx_arr[idx]] for idx in range(len(idx_arr))])
            y_arr_new = np.array([y_arr[idx_arr[idx]] for idx in range(len(idx_arr))], dtype = np.float64)
     
            f_interp = interp1d(y_arr_new, x_arr_new, 'cubic', bounds_error=False, fill_value=0)
            x = f_interp(y_value)                
        else:
            x = self.x[idx_start + findind_exact(self.y[idx_start:], y_value)]

        return float(x)

    
    def normalize(self, x_lim: Any | None = None, norm_val: float = 1) -> None:
        """
        Normalise y so that its maximum (or a chosen point) equals norm_val.
        
        Parameters
        ----------
        x_lim : Any | None
            X lim.
        norm_val : float
            Norm val.
        
        Examples
        --------
        >>> obj.normalize()
        """
        if x_lim is None:
            r = range(0, len(self.x))
        else:
            idx_min = self.x_idx_of(x_lim[0])
            idx_max = self.x_idx_of(x_lim[1])
            r = range(idx_min, idx_max+1)
            
        if r != range(0,0):
            self.y = self.y / max(self.y[r]) * norm_val
        
    def equidist(self, left: float | None = None, right: float | None = None, delta: float = 0.1, kind: str = 'cubic') -> None:
        """
        Resample y onto an equidistant x grid.
        If left (right) = None then the new left (right) value is the old one.
        """
        
        if left is None:
            #min_x = math.ceil(min(self.x))
            min_x = min(self.x)
        else:
            min_x = left
        if right is None:
            max_x = max(self.x)
        else: 
            max_x = right
        
        
        #new_x = np.arange(min_x, max_x, delta) # does not give exactly evenly spaced x-values
        new_x = np.linspace(min_x, max_x, int(round((max_x-min_x)/delta)+1))
        self.y = int_arr(self.x, self.y, new_x, kind = kind)
        self.x = new_x
    
    def quants(self) -> Any:
        """
        Return the (x_quantity, y_quantity) axis label strings.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.quants()
        """
        return dict(x=self.qx, y=self.qy)
    
    def units(self) -> Any:
        """
        Return the (x_unit, y_unit) string tuple.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.units()
        """
        return dict(x=self.ux, y=self.uy)
    
    def qy_uy(self, qy: Any, uy: Any) -> None:
        """
        Return the (y_quantity, y_unit) strings for the y axis.
        
        Parameters
        ----------
        qy : Any
            Qy.
        uy : Any
            Uy.
        
        Examples
        --------
        >>> obj.qy_uy()
        """
        self.qy = qy
        self.uy = uy
        
    def plot(self, title: str = 'self.name', ax: Any | None= None, xscale: str = 'linear', yscale: str = 'linear', 
             left: float | None = None, right: float | None = None, bottom: float | None = None, divisor: float | None = None, top: float | None = None,
             plot_table: bool = False, cell_text: Any | None = None, row_labels: Any | None = None, 
             hline: Any | None = None, hline_colors: Any | None = None, vline: Any | None = None, vline_colors: Any | None = None, figsize: Any=(9,6), return_fig: bool = False, show_plot: bool = True, create_image_stream: bool= False) -> None:
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

        if left is not None:
            ax.set_xlim(left = left)
        if right is not None:
            ax.set_xlim(right = right)
            
        if bottom is not None:
            if top is None:
                bottom_, top = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #top = max(self.y) + 0.1 * abs(max(self.y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if top is not None:
            if bottom is None:
                bottom, top_ = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #bottom = min(self.y) - 0.1 * abs(min(self.y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if (bottom is None) and (top is None):
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
                                  bbox = [0.4, 0.2, 0.1, 0.7])  # type: ignore
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(12)
            the_table.scale(1, 5)
            
        _draw_hlines_vlines(ax, hline, hline_colors, vline, vline_colors)
            
        if create_image_stream:
            image_stream = io.BytesIO()
            plt.savefig(image_stream)
            if show_plot:
                plt.show()
            return image_stream  # type: ignore

        #plt.legend()
        if show_plot:
            plt.show()
            
        if return_fig:
            return fig  # type: ignore

        
    def plot_linfit(self, von: Any | None = None, bis: Any | None = None, residue: Any = False, return_data: bool = False) -> Any:
        """
        Plot data with a linear fit overlaid and residuals optionally shown.
        
        Parameters
        ----------
        von : Any | None
            Von.
        bis : Any | None
            Bis.
        residue : Any
            Residue.
        return_data : bool
            Return data.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.plot_linfit()
        """
        
        if von is None:
            von = min(self.x)
        if bis is None:
            bis = max(self.x)
        m, b = linfit(self.x, self.y, von, bis)
        self.m = m
        self.b = b
        fit = XYData(self.x, m * self.x + b)
        if residue:
            res = XYData(self.x, (self.y - fit.y)/self.y * 100)
            res.qx = self.qx
            res.qy = 'Delta'
            res.ux = self.ux
            res.uy = '%'
            m_min = res.min_within(von, bis)
            m_max = res.max_within(von, bis)
            res.plot(left = von, right = bis, bottom = m_min * 1.1, top = m_max * 1.1, title = f'({self.name} - linear fit) / {self.name}')
            if return_data:
                return res
        else:
            both = MXYData([self, fit])
            both.label(['self.name', f'linear fit: m = {m:.2e}, b = {b:.2e}'])
            both.plot()
            if return_data:
                return both
    
    @classmethod
    def load(cls, filepath_or_directory: str, filepath: str = '', delimiter: str = ',', columns: Any=(0, 1), header: Any = 'infer', 
             quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, take_quants_and_units_from_file: bool = False,  check_data: bool = True, name: str | None=None) -> Any:

        """
        Loads a single xy data. If a filename is given it will be used, if not the first file in the directory will be used.
        colunns: tuple which indicates which column is x and which is y.
        """
        
        if (filepath == '') and os.path.isdir(filepath_or_directory):
            filepath = os.listdir(filepath_or_directory)[0]
            file = join(filepath_or_directory, filepath)
        elif (filepath_or_directory == '') and os.path.isfile(filepath):
            file = filepath
        elif os.path.isfile(filepath_or_directory):
            file = filepath_or_directory
        elif os.path.isfile(join(filepath_or_directory, filepath)):
            file = join(filepath_or_directory, filepath)
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
            name = filepath
        
        return cls(x, y, quants = dict(x = qx, y = qy), units = dict(x = ux, y = uy), name = name,  check_data = check_data)
    
    def save(self, save_dir: str, filepath: str, check_existing: Any = True) -> None:
        """
        Save the data to a CSV file, prompting before overwriting.
        
        Parameters
        ----------
        save_dir : str
            Save dir.
        filepath : str
            Filepath.
        check_existing : Any
            Check existing.
        
        Examples
        --------
        >>> obj.save()
        """
        
        x_col_name = self.qx
        y_col_name = self.qy
        
        if self.ux != "":
            x_col_name = x_col_name + f' ({self.ux})'

        if self.uy != "":
            y_col_name = y_col_name + f' ({self.uy})'

        df = pd.DataFrame({x_col_name : self.x, y_col_name : self.y})
        TFN = join(save_dir, filepath)
        if check_existing:
            if save_ok(TFN):
                df.to_csv(TFN, header = True, index = False)
        else:
            df.to_csv(TFN, header = True, index = False)
    
    def lowpass_filter(self, test: Any = False, yscale: str = 'log', left: float | None = None, right: float | None = None, T: float = 5.0, fs: Any = 30.0, cutoff: Any = 0.7, order: Any = 2, filter_only_from_left_to_right: Any = False) -> Any:
        """
        Apply a Butterworth low-pass filter to remove high-frequency noise.
        
        Parameters
        ----------
        test : Any
            Test.
        yscale : str
            Yscale.
        left : float | None
            Left.
        right : float | None
            Right.
        T : float
            T.
        fs : Any
            Fs.
        cutoff : Any
            Cutoff.
        order : Any
            Order.
        filter_only_from_left_to_right : Any
            Filter only from left to right.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.lowpass_filter()
        """
    
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 0.7      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
    
        def butter_lowpass_filter(data: Any, cutoff: Any, fs: Any, order: Any) -> Any:
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
            mxy = MXYData([self, xy_filt])
            mxy.label = ['original', 'filtered']  # type: ignore
            if left is not None and right is not None:
                m_max = mxy.max_within(left = left, right = right)
                m_min = mxy.min_within(left = left, right = right)
                if m_min < 0:
                    m_min = m_max/100
                mxy.plot(yscale = yscale, left = left, right = right, bottom = m_min*0.9, top = m_max*1.1, title = 'Check if filter is ok')
            else:
                mxy.plot(yscale = yscale, title = 'Check if filter is ok')
        elif test == False:
            self.y = y
            
    def savgol(self, n1: int = 51, n2: int = 1, name: str | None = None) -> Any:
        """
        Smooth y data with a Savitzky-Golay filter.
        
        Parameters
        ----------
        n1 : int
            N1.
        n2 : int
            N2.
        name : str | None
            Name.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.savgol()
        """
            
        sgf = self.copy()
        sgf.y = savgol_filter(self.y, n1, n2)

        if name is not None:
            sgf.name = name
                
        return sgf 
    
    def residual(self, other: Any, left: float | None = None, right: float | None = None, relative: Any = False) -> Any:
        """
        Return the (weighted) residual between data and a reference array.
        
        Parameters
        ----------
        other : Any
            Other.
        left : float | None
            Left.
        right : float | None
            Right.
        relative : Any
            Relative.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.residual()
        """
        #Attention: selv and other have to have the same x_values!
        
        d = self.copy()
        
        if left is None:
            left = min(self.x)
        if right is None:
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
    def chisquare(data: Any, fit: Any, left: float | None = None, right: float | None = None) -> Any:
        """
        Compute the chi-square statistic between data and a model.
        
        Parameters
        ----------
        data : Any
            Data.
        fit : Any
            Fit.
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
        >>> obj.chisquare()
        """
        # data and fit must have the same x values
        if left is None:
            left = min(data.x)
        if right is None:
            right = max(data.x)
    
        le = findind(data.x, left)
        ri = findind(data.x, right)
    
        ra = range(le, ri+1)

        res = data.residual(fit, left = left, right = right)
        #return np.sum(np.array([res.y[i]**2/fit.y[i] for i in range(len(data.y))]))/len(data.y)
        return np.sum(res.y**2/ np.abs(fit.y[ra]))/len(data.y[ra])
            
    def diff(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Compute the numerical derivative dy/dx.
        
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
        >>> obj.diff()
        """
        
        self_asc = self.copy()
        self_asc.strictly_ascending()
        
        if left is None:
            left = min(self_asc.x)
        if right is None:
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
    
           
    def max_within(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Return the maximum y value within [left, right].
        left: left x boundary
        right: right y boundary
        If no values for left or right are given then the global maximum is returned.
        """

        l = left
        r = right

        if (left is None) or (left < min(self.x)): 
            l = min(self.x)
        if (right is None) or (right > max(self.x)):
            r = max(self.x)

        ra = range(findind(self.x, l), findind(self.x, r)+1)
        
        return max(self.y[ra])    
    
    
    def min_within(self, left: float | None = None, right: float | None = None, absolute: Any = False) -> Any:
        """
        Return the minimum y value within [left, right].
        left: left x boundary
        right: right y boundary
        If no values for left or right are given then the global maximum is returned.
        """

        l = left
        r = right

        if (left is None) or (left < min(self.x)): 
            l = min(self.x)
        if (right is None) or (right > max(self.x)):
            r = max(self.x)

        ra = range(findind(self.x, l), findind(self.x, r)+1)
        
        if absolute:
            return min(np.absolute(self.y[ra]))
        else:
            return min(self.y[ra])
        
    
    def bottom_top_for_plot(self, left: float | None=None, right: float | None=None, yscale: str='log', divisor: float | None=None) -> Any:
        """
        Returns the minimum and maximum y-value within left < x < right.
        This is important for plotting a graph.
        If yscale == 'log' then values <= 0 for bottom lead to an error, this is accounted for.
        divisor can be used to define the bottom as top/divisor.
        """
        return _bottom_top_for_plot(self, left, right, yscale, divisor)

    
    def zero_data(self, left: float | None = None, right: float | None = None) -> None:
        """
        Sets the y-values to zero from x = left to x = right.
        """
    
        l = left
        r = right
    
        if (left is None) or (left < min(self.x)): 
            l = min(self.x)
        if (right is None) or (right > max(self.x)):
            r = max(self.x)
    
        ra = range(findind(self.x, l), findind(self.x, r)+1)
        new = self.copy()
        new.y[ra] = 0
    
        return new
    
    def cut_data_outside(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Remove data points outside [left, right].
        """
    
        l = left
        r = right
    
        if (left is None) or (left < min(self.x)): 
            l = min(self.x)
        if (right is None) or (right > max(self.x)):
            r = max(self.x)
    
        ra = range(findind(self.x, l), findind(self.x, r)+1)
        new = self.copy()
        new.x = new.x[ra]
        new.y = new.y[ra]
    
        return new
    
    def reverse(self) -> Any:
        """
        Reverse.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.reverse()
        """
        self.x = self.x[::-1]
        self.y = self.y[::-1]
        
    def swap_axes(self) -> Any:
        """
        Swap axes.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.swap_axes()
        """
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
        
    def remove_nan(self) -> Any:
        """
        Remove nan.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.remove_nan()
        """
        # Removes all numpy.nan values in self.x and self.y (only gives sensible result if there is a nan in both x[i] and y[i])
        x_raw = self.x
        y_raw = self.y
        
        x_list = np.logical_not(np.isnan(x_raw))
        y_list = np.logical_not(np.isnan(y_raw))
        logic_list = [x_list[idx] and y_list[idx] for idx in range(len(x_list))]
        self.x = x_raw[logic_list]
        self.y = y_raw[logic_list]
        #if len(self.x) != len(self.y):
        #    print('Attention: XYData.remove_nan() gave an x-array and a y-array with different sizes. The reason could be that, e.g. x[i] = nan but y[i] = number')
        
    def idfac_fit(self, left: float | None = None, right: float | None = None, plot: bool = False, plotrange: list | None = [None, None], return_fit: bool = True) -> Any:
        """
        Fit the diode ideality factor from a semi-log JV curve.
        
        Parameters
        ----------
        left : float | None
            Left.
        right : float | None
            Right.
        plot : bool
            Plot.
        plotrange : list | None
            Plotrange.
        return_fit : bool
            Return fit.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.idfac_fit()
        """
        
        if (left is None) or (left < min(self.x)):
            left = min(self.x)
        if (right is None) or (right > max(self.x)):
            right = max(self.x)
        
        ra = range(findind(self.x, left), findind(self.x, right)+1)
            
        m, b = np.polyfit(np.log10(self.x[ra]), self.y[ra], 1)
        nid = q/(k * T_RT * math.log(10)) * m
        
        fit = XYData(self.x, m*np.log10(self.x) + b, quants = {"x": "Light intensity", "y": "Voc"}, units = {"x": "mW/cm2", "y": "V"}, name = 'fit')
        fit.nid = nid  # type: ignore
        
        if plot:
            if (plotrange[0] is None) or (plotrange[0] < min(self.x)):  # type: ignore
                plot_left = min(self.x) * 0.5
            else:
                plot_left = plotrange[0]  # type: ignore
                
            if (plotrange[1] is None) or (plotrange[1] > max(self.x)):  # type: ignore
                plot_right = max(self.x) * 1.1
            else:
                plot_right = plotrange[1]  # type: ignore
                
            self.plotstyle = dict(linestyle = 'None', marker = 'o', color = 'green', markersize = 20)
            fit.plotstyle = dict(linestyle = '-', color = 'green', linewidth = 5)
            da = MXYData([self, fit])
            da.label([self.name, f'm = ln(10) $\cdot$ kT/q $\cdot$ {fit.nid:.2f}'])  # type: ignore
            da.plot(xscale = 'log', left = plot_left, right = plot_right, bottom = 0.9 * self.min_within(left=plot_left, right=plot_right), top = 1.1 * self.max_within(left=plot_left, right=plot_right), plotstyle = 'individual')
        
        if return_fit:    
            return fit

        
    def product(self, s2: Any, qy: Any | None = None, uy: Any | None = None, delta: float = 1) -> Any:
        """
        Product.
        
        Parameters
        ----------
        s2 : Any
            S2.
        qy : Any | None
            Qy.
        uy : Any | None
            Uy.
        delta : float
            Delta.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.product()
        """
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
        
        if qy is not None:
            result.qy = qy
        if uy is not None:
            result.uy = uy
        
        return result
    
    def all_values_greater_min(self, min_val: Any | None=None) -> Any:
        """
        Return True if all y values exceed the given minimum.
        """
        self.y = np.array([self.y[i] if (self.y[i] > min_val) else min_val for i in range(len(self.y))], dtype = np.float64)
    
    def shift_x(self, x: np.ndarray) -> Any:
        """
        Shifts the x-values by x
        """
        self.x = self.x + x
        
    def shift_y(self, y: np.ndarray) -> Any:
        """
        Shifts the y-values by y
        """
        self.y = self.y + y

        
    def idx_range(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Return the index range [i_left, i_right] for a value interval in a sorted array.
        
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
        >>> obj.idx_range()
        """
        # Returns the x-index range which goes from x = left to x = right
        return idx_range(self.x, left = left, right = right)
    
    def polyfit(self, order: Any = 1, left: float | None = None, right: float | None = None, new_x_arr: np.ndarray | None = None, new_meshsize: np.ndarray | None = None) -> Any:
        """
        Fit a polynomial of specified degree to the data.
        
        Parameters
        ----------
        order : Any
            Order.
        left : float | None
            Left.
        right : float | None
            Right.
        new_x_arr : np.ndarray | None
            New x arr.
        new_meshsize : np.ndarray | None
            New meshsize.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.polyfit()
        """
        # Fits a polynomial of order to self and returns the data as an object of the same class as self.
        # new_x_arr: The x values of the fit.
        # new_meshsize: If no new_x_arr is provided, new_meshsize is a number of evenly spaced x-values between left and right.
        ra = self.idx_range(left = left, right = right)
        p = np.poly1d(np.polyfit(self.x[ra], self.y[ra], order))
        fit = self.copy()
        if new_x_arr is None:
            if new_meshsize != 0:
                fit.x = np.linspace(self.x[ra[0]], self.x[ra[-1]], new_meshsize)  # type: ignore
            else:
                fit.x = self.x[ra]
        else:
            fit.x = new_x_arr
        fit.y = p(fit.x)
        fit.name = 'fit of ' + self.name
        return fit

    def del_first_and_last_n_data_points(self, n: float=1) -> Any:
        """
        Del first and last n data points.
        
        Parameters
        ----------
        n : float
            N.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.del_first_and_last_n_data_points()
        """
        r = range(n, len(self.x)-n)  # type: ignore
        self.x = self.x[r]
        self.y = self.y[r]
        
    def del_edge_zero_data(self) -> Any:
        """
        Del edge zero data.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.del_edge_zero_data()
        """
        start_idx = 0
        while self.y[start_idx] == 0:
            start_idx += 1
        stop_idx = len(self.y) - 1
        while self.y[stop_idx] == 0:
            stop_idx -= 1
        r = range(start_idx, stop_idx + 1)
        self.x = self.x[r]
        self.y = self.y[r]

    def rm_cosray(self, m: float = 3, threshold: Any = 5) -> Any:
        """
        Removes cosmic rays from the Spectrum.
        m: 2 m + 1 points around the spike are selected
        threshold: The threshold value from which on a spike is detected as such
        """
        sp = self.copy()
    
        def modified_z_score(intensity: Any) -> Any:
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
                    m_right = m   # type: ignore
                if i < m:
                    m_left = i
                else:
                    m_left = m  # type: ignore
                w = np.arange(i-m_left,i+1+m_right) # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                if len(w2) != 0:
                    y_out[i] = np.mean(sp.y[w2]) # and we average their values
    
        sp.y = y_out
        
        return sp    
    
    def monotoneous_ascending(self) -> Any:
        """
        Monotoneous ascending.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.monotoneous_ascending()
        """
        # Orders the data, so that x is monotoneous ascending
        s = self.x.argsort()
        self.x = self.x[s]
        self.y = self.y[s]


    def strictly_ascending(self) -> Any:
        """
        Strictly ascending.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.strictly_ascending()
        """
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
    
    
class MXYData:
    """
    Container for a collection of XYData objects (multi-spectrum).
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
        self.sa = sa
        self.label_defined = False
        self.n_y = len(sa)
        if self.n_y != 0:
            self.n_x = len(sa[0].x)
        else:
            self.n_x = 0
        
    def __mul__(self, other: Any) -> None:
        """
        Multiply element-wise with another object or scalar.
        
        Parameters
        ----------
        other : Any
            Other.
        
        Examples
        --------
        >>> obj.__mul__()
        """
        new_sa = []
        for i, sp in enumerate(self.sa):
            new_sa.append(sp * other)
        return type(self)(new_sa)  # type: ignore
    
    def __iter__(self) -> None:
        """
        Iterate over elements.
        
        Examples
        --------
        >>> obj.__iter__()
        """
        return iter(self.sa)
        
    def qx_ux(self, qx: Any, ux: Any) -> None:
        """
        Qx ux.
        
        Parameters
        ----------
        qx : Any
            Qx.
        ux : Any
            Ux.
        
        Examples
        --------
        >>> obj.qx_ux()
        """
        for i, sp in enumerate(self.sa):
            sp.qx = qx
            sp.ux = ux

    def qy_uy(self, qy: Any, uy: Any) -> None:
        """
        Qy uy.
        
        Parameters
        ----------
        qy : Any
            Qy.
        uy : Any
            Uy.
        
        Examples
        --------
        >>> obj.qy_uy()
        """
        for i, sp in enumerate(self.sa):
            sp.qy = qy
            sp.uy = uy
            
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
        sa_new = []
        for i, sp in enumerate(self.sa):
            sa_new.append(sp.copy())
        ms = type(self)(sa_new)
        if self.label_defined:
            ms.label(self.lab)
        ms.n_y = self.n_y
        ms.n_x = self.n_x
        return ms
        
    def append(self, data: Any) -> Any:
        """
        Append another XYData to this collection.
        
        Parameters
        ----------
        data : Any
            Data.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.append()
        """
        self.sa.append(data)
        self.label_defined = False
        
    @staticmethod
    def combine(mxy1: Any, mxy2: Any) -> Any:
        """
        Returns an instance new_mxy of type(mxy1) with new_mxy.sa = mxy1.sa + mxy2.sa.
        """
        if type(mxy1) != type(mxy2):
            print('MXYData.combine(mxy1, mxy2): mxy1 and mxy2 are of different type!')
        #Make sure that there are no dependencies of new_mxy on mxy1 and mxy2 
        mxy1_ = mxy1.copy()
        mxy2_ = mxy2.copy()
        sa = mxy1_.sa + mxy2_.sa
        new_mxy = type(mxy1)(sa)
        if mxy1.label_defined and mxy2.label_defined:
            new_mxy.label(mxy1_.lab + mxy2_.lab)
            new_mxy.label_defined = True
        return new_mxy
        
    def delete(self, data: Any) -> Any:
        """
        Delete.
        
        Parameters
        ----------
        data : Any
            Data.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.delete()
        """
        i = 0
        while i < len(self.sa):
            if self.sa[i].name == data.name:
                self.sa.pop(i)
            else:
                i += 1
                
    @classmethod
    def generate_empty(cls) -> Any:
        """
        Generate empty.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.generate_empty()
        """
        return cls([])

        
    def label(self, lab: Any) -> None:
        """
        Label.
        
        Parameters
        ----------
        lab : Any
            Lab.
        
        Examples
        --------
        >>> obj.label()
        """
        self.lab = lab
        self.label_defined = True
        
    def no_label(self) -> Any:
        """
        No label.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.no_label()
        """
        self.label_defined = False
                
    def replace(self, idx: int, sp_new: Any) -> Any:
        """
        Replace.
        
        Parameters
        ----------
        idx : int
            Idx.
        sp_new : Any
            Sp new.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.replace()
        """
        self.sa[idx] = sp_new
        
    def remain(self, idx_list: Any) -> Any:
        """
        Return a copy trimmed to the interval [left, right].
        """
        sa = []
        lab = []
        for i, idx in enumerate(idx_list):
            new = self.sa[idx].copy()
            sa.append(new)
            if self.label_defined:
                lab.append(self.lab[idx])
        #rem_sa = MXYData(sa)
        rem_sa = type(self)(sa)
        rem_sa.label(lab)
        return rem_sa
    
    def set_plotstyle(self, linestyle: Any | None = None, marker: Any | None = None, color: Any | None = None, markersize: Any | None = None, linewidth: Any | None = None) -> Any:
        """
        Set plotstyle.
        
        Parameters
        ----------
        linestyle : Any | None
            Linestyle.
        marker : Any | None
            Marker.
        color : Any | None
            Color.
        markersize : Any | None
            Markersize.
        linewidth : Any | None
            Linewidth.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.set_plotstyle()
        """
        for idx, sp in enumerate(self.sa):
            
            if linestyle is not None:
                sp.plotstyle['linestyle'] = linestyle
            
            if marker is not None:
                sp.plotstyle['marker'] = marker
            
            if color is not None:
                sp.plotstyle['color'] = color
            
            if markersize is not None:
                sp.plotstyle['markersize'] = markersize
            
            if linewidth is not None:
                sp.plotstyle['linewidth'] = linewidth

    def names_to_label(self, split_ch: str | None = None) -> Any:
        """
        Names to label.
        
        Parameters
        ----------
        split_ch : str | None
            Split ch.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.names_to_label()
        """
        lab = []
        for i, sp in enumerate(self.sa):
            if split_ch is None:
                lab.append(sp.name)
            else:
                lab.append(sp.name.split(split_ch)[0])
        self.label(lab)
        self.label_defined = True
        
        
    def print_all_names(self, split_ch: str | None = None, unique_only: Any = False, print_all: Any = True, print_idx: Any = True, return_list: bool = False) -> Any:
        """
        Print all names.
        
        Parameters
        ----------
        split_ch : str | None
            Split ch.
        unique_only : Any
            Unique only.
        print_all : Any
            Print all.
        print_idx : Any
            Print idx.
        return_list : bool
            Return list.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.print_all_names()
        """
        all_names = []
        idx = 0
        for i, sp in enumerate(self.sa):
            if split_ch is None:
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
    
    def print_names_containing(self, name: str) -> Any:
        """
        Print names containing.
        
        Parameters
        ----------
        name : str
            Name.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.print_names_containing()
        """
        for i, sp in enumerate(self.sa):
            if name in sp.name:
                print(sp.name)
        
    
    def plot(self, title: str = '', ax: Any | None= None, xscale: str = 'linear', yscale: str = 'linear', left: float | None = None, right: float | None = None, 
             bottom: float | None = None, divisor: float | None = None, top: float | None = None, plotstyle: str = 'auto', showindex: bool = False, in_name: Any = [], not_in_name: Any | None = None,
             plot_table: bool = False, cell_text: Any | None = None, row_labels: Any | None = None, col_labels: Any | None = None,
             bbox: Any = [0.3, 0.25, 0.1, 0.5], figsize: Any=(9,6), hline: Any | None = None, hline_colors: Any | None = None, vline: Any | None = None, vline_colors: Any | None = None, nolabel: Any = False, 
             return_fig: bool = False, generate_image_stream: bool=False, show_plot: bool = True, ylabel: Any | None=None, xlabel: Any | None=None, create_image_stream: bool= False, **kwargs) -> None:

        """
        Plots multiple xy-data of type XYData. The axis title are taken from the first Spectrum.
        showindex: If True then the index of the sa list will be shown before the regular label. 
        This is helpful when certain curves have to be selected e.g. for PLQY. 
        in_name: e.g ['laser'], List with strings that have to be in the name to be plotted. If [] then everything is plotted.
        If individual XYData has the attribute plotrange (list of begin and end value), than only this plotrange is plotted.
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

            def in_name_in_spec(in_name: Any, spec: Any) -> Any:
                
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
            
            def not_in_name_in_spec(not_in_name: Any, spec: Any) -> Any:
                
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
        
        if bottom is not None:
            if top is None:
                bottom_, top = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #top = max(self.sa[0].y) + 0.1 * abs(max(self.sa[0].y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if top is not None:
            if bottom is None:
                bottom, top_ = self.bottom_top_for_plot(left = left, right = right, yscale = yscale, divisor = divisor)
                #bottom = min(self.sa[0].y) - 0.1 * abs(min(self.sa[0].y))
            ax.set_ylim(bottom = bottom, top = top)
            
        if (bottom is None) and (top is None):
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
                filepath = kwargs['plot_FN']
                TFN = join(save_dir, filepath)
                if save_ok(TFN):
                    plt.savefig(TFN)
                    
        _draw_hlines_vlines(ax, hline, hline_colors, vline, vline_colors)
        
        if create_image_stream:
            image_stream = io.BytesIO()
            plt.savefig(image_stream)
            if show_plot:
                plt.show()
            return image_stream  # type: ignore

        if show_plot:
            plt.show()
        
        
        if return_fig:
            return fig  # type: ignore

    
    def save(self, save_dir: str, title: str, label: Any | None = None) -> None:
        """
        Save.
        
        Parameters
        ----------
        save_dir : str
            Save dir.
        title : str
            Title.
        label : Any | None
            Label.
        
        Examples
        --------
        >>> obj.save()
        """
        if label is None:
            label = self.lab
        for i, xy_dat in enumerate(self.sa):
            filepath = title + ' - ' + label[i] + '.csv'
            xy_dat.save(save_dir, filepath)
            
    def save_in_one_file(self, fp: Any) -> None:
        """
        Save all datasets to a single CSV file.
        
        Parameters
        ----------
        fp : Any
            Fp.
        
        Examples
        --------
        >>> obj.save_in_one_file()
        """
        #check if all files have the same wavelengths
        all_have_same_x = True
        x_arr = self.sa[0].x
        df = self.sa[0].to_df()
        for sp in self:  # type: ignore
            if not np.all(sp.x == x_arr):
                all_have_same_x = False
                break
        if all_have_same_x:
            if not self.label_defined:
                self.label([sp.name for sp in self])  # type: ignore
            for sp, label in zip(self, self.lab):  # type: ignore
                col_name = label+', '+sp.qy
                if sp.uy is not None and sp.uy != '':
                    col_name += f' ({sp.uy})'
                df[col_name] = sp.y
            df.drop(df.columns[0], axis=1, inplace=True)
            df.to_csv(fp)
        else:
            print('Not saved: Not all XYData have the same x!')
            
    def save_individual(self, save_dir: str | None = None, FNs: str | None = None, check_existing: Any = True, check_FN_extension: Any = True) -> None:
        """
        Save each dataset to its own CSV file.
        
        Parameters
        ----------
        save_dir : str | None
            Save dir.
        FNs : str | None
            Fns.
        check_existing : Any
            Check existing.
        check_FN_extension : Any
            Check fn extension.
        
        Examples
        --------
        >>> obj.save_individual()
        """
        
        quitted = False
        if save_dir is None:
            save_dir = os.getcwd()
        for i, sp in enumerate(self.sa):
                    
            x_col_name = sp.qx
            if sp.ux != "":
                x_col_name = x_col_name + f' ({sp.ux})'
            y_col_name = sp.qy
            if sp.uy != "":
                y_col_name = y_col_name + f' ({sp.uy})'
                                    
            if FNs is None:
                filepath = sp.name
            else:
                filepath = FNs[i]
                
            TFN = join(save_dir, filepath)
            df = pd.DataFrame({x_col_name : sp.x, y_col_name : sp.y})
            #filepath = filepath.split('.'+filepath.split('.')[-1])[0] + '.csv'
            if check_FN_extension:
                filepath = os.path.splitext(filepath)[0] + '.csv'
            else:
                filepath = filepath + '.csv'

            if check_existing:
            
                ok_to_save, quitted = save_ok(TFN, quitted)
                if ok_to_save and not(quitted):

                    df.to_csv(join(save_dir, filepath), header = True, index = False)
                    
            else:
                    df.to_csv(join(save_dir, filepath), header = True, index = False)
            
    @classmethod
    def load_individual(cls, directory: str, delimiter: str = ',', header: Any = 'infer', quants: Any = {"x": "x", "y": "y"}, units: Any = {"x": "", "y": ""}, take_quants_and_units_from_file: bool = False) -> Any:

        """
        Load multiple individual files into a collection.
        """
        FNs = os.listdir(directory)
        sa = []
        
        for i, filepath in enumerate(FNs):
            
            dat = pd.read_csv(join(directory, filepath), delimiter = delimiter, header = header)
                        
            x = np.array(dat)[:,0].astype(np.float64)
            y = np.array(dat)[:,1].astype(np.float64)

            if cls.__name__ == 'MXYData':
                sp = XYData(x, y, quants, units, filepath)
       
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
            
    def max_within(self, left: float | None = None, right: float | None = None) -> Any:
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
                
            if (left is None) or (left < min(sp.x)): 
                l = min(sp.x)
            if (right is None) or (right > max(sp.x)):
                r = max(sp.x)

            ra = range(findind(sp.x, l), findind(sp.x, r)+1)

            if ra != range(0,0):
                m = max(sp.y[ra])
                if m > maximum:
                    maximum = m

        return maximum
    

    def min_within(self, left: float | None = None, right: float | None = None, absolute: Any = False) -> Any:
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

                
            if (left is None) or (left < min(sp.x)): 
                l = min(sp.x)
            if (right is None) or (right > max(sp.x)):
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
    
    def bottom_top_for_plot(self, left: float | None = None, right: float | None = None, yscale: str = 'log', divisor: float | None = None) -> Any:
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
                if divisor is None:
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


    def normalize(self, x_lim: Any | None = None, norm_val: float = 1) -> None:    
        """
        Normalize.
        
        Parameters
        ----------
        x_lim : Any | None
            X lim.
        norm_val : float
            Norm val.
        
        Examples
        --------
        >>> obj.normalize()
        """

        for i, sp in enumerate(self.sa):
            sp.normalize(x_lim = x_lim, norm_val = norm_val)
            
    def equidist(self, left: float | None = None, right: float | None = None, delta: float = 0.1, kind: str = 'cubic') -> None:
        """
        Equidist.
        
        Parameters
        ----------
        left : float | None
            Left.
        right : float | None
            Right.
        delta : float
            Delta.
        kind : str
            Kind.
        
        Examples
        --------
        >>> obj.equidist()
        """

        for i, sp in enumerate(self.sa):
            sp.equidist(left = left, right = right, delta = delta, kind = kind)
        
            
    def all_values_greater_min(self, min_val: Any | None=None) -> Any:
        """
        Looks for values < min_val and sets them to min_val.
        """
        for idx, sp in enumerate(self.sa):
            sp.all_values_greater_min(min_val = min_val)
            
    def cut_data_outside(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Cuts the data outside x = [left, right].
        """
  
        new_sa = []          
        for idx, sp in enumerate(self.sa):
            new_xy = sp.cut_data_outside(left = left, right = right)
            new_sa.append(new_xy)
        
        new_mxy = type(self)(new_sa)
        
        return new_mxy
    
    def reverse(self) -> Any:
        """
        Reverse the x values in all Spectra.
        Returns the reversed Spectra.
        """
        new = self.copy()
        for idx, sp in enumerate(new.sa):
            new.sa[idx].reverse()
            
        return new
    
    def del_first_and_last_n_data_points(self, n: float=1) -> Any:
        """
        Del first and last n data points.
        
        Parameters
        ----------
        n : float
            N.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.del_first_and_last_n_data_points()
        """
        for idx, sp in enumerate(self.sa):
            sp.del_first_and_last_n_data_points(n=n)
            
    def del_edge_zero_data(self) -> Any:
        """
        Del edge zero data.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.del_edge_zero_data()
        """
        for idx, sp in enumerate(self.sa):
            sp.del_edge_zero_data()
            
            
    def rm_cosray(self, m: float = 3, threshold: Any = 5) -> Any:
        """
        Rm cosray.
        
        Parameters
        ----------
        m : float
            M.
        threshold : Any
            Threshold.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.rm_cosray()
        """
        new = self.copy()
        for idx, sp in enumerate(new.sa):
            sp_new = sp.rm_cosray(m = m, threshold = threshold)
            new.replace(idx, sp_new)
        return new
    
    
    def idfac_fit(self, left: float | None = None, right: float | None = None, plot: bool = True, plotrange: list | None = [None, None], return_all: bool = False) -> Any:
        """
        Idfac fit.
        
        Parameters
        ----------
        left : float | None
            Left.
        right : float | None
            Right.
        plot : bool
            Plot.
        plotrange : list | None
            Plotrange.
        return_all : bool
            Return all.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.idfac_fit()
        """
    
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
    
        da = MXYData(all_data)
        #da.label([FNs[0], FNs[1], f'm = ln(10) $\cdot$ kT/q $\cdot$ {fit[0].nid:.2f}', f'm = ln(10) $\cdot$ kT/q $\cdot$ {fit[1].nid:.2f}'])
    
        da.label(all_data_label)
        if plot:
            da.plot(xscale = 'log', plotstyle = 'individual', figsize = (10,7))
        if return_all:
            return da
        
    def shift_x(self, x: np.ndarray) -> Any:
        """
        Shifts the x-values by x
        """
        for sp in self.sa:
            sp.shift_x(x)
            
    def shift_y(self, y: np.ndarray) -> Any:
        """
        Shifts the y-values by y
        """
        for sp in self.sa:
            sp.shift_y(y)
            
    def average(self) -> Any:
        """
        Average.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.average()
        """
        # Averages over all Spectra and returns the averaged Spectrum
        av = self.sa[0].copy()
        av.y *= 0
        for sp in self.sa:
            av += sp
        av /= len(self.sa)
        return av
    
    def diff(self, left: float | None = None, right: float | None = None) -> Any:
        """
        Diff.
        
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
        >>> obj.diff()
        """
        new = self.copy()
        sa = []
        for sp in new.sa:
            sa.append(sp.diff(left=left, right=right))
        new.sa = sa
        return new
    
    def remove_nan(self) -> Any:
        """
        Remove nan.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.remove_nan()
        """
        for sp in self.sa:
            sp.remove_nan()
            
    def strictly_ascending(self) -> Any:
        """
        Strictly ascending.
        
        Returns
        -------
        Any
            Computed result.
        
        Examples
        --------
        >>> obj.strictly_ascending()
        """
        for sp in self.sa:
            sp.strictly_ascending()

                
class XYZData:
    """
    Container class for XYZData data and operations.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray , quants: Any = {"x": "x", "y": "y", "z": "z"}, units: Any = {"x": "", "y": "", "z": ""}, name: str = '', plotstyle: Any = None) -> None:
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
        if plotstyle is None:
            plotstyle = dict(linestyle='-', color='black', linewidth=3)
        self.plotstyle = plotstyle

    @classmethod
    def load(cls, directory: str, filepath: str = '', delimiter: str = ',', header: Any = 'infer',
              quants: Any = {"x": "x", "y": "y", "z": "z"}, units: Any = {"x": "", "y": "", "z": ""}, take_quants_and_units_from_file: bool = False) -> Any:

        """
        Load data from a CSV or text file.
        """
        
        if filepath == '':
            filepath = os.listdir(directory)[0]
        dat = pd.read_csv(join(directory, filepath), delimiter = delimiter, header = header)
        
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
        
        return cls(x, y, z, quants = dict(x = qx, y = qy, z = qz), units = dict(x = ux, y = uy, z = uz), name = filepath)
