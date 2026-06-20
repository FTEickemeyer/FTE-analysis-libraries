# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import platform
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from os.path import join
import warnings
from numbers import Number
from typing import Any

if sys.platform == 'win32':
    import winsound

# Constants
pi = math.pi
heV = 4.14e-15 #eV*s
c = 2.99792458e8 #m/s (spped of light in vacuum)
keV = 8.6173e-5 #eV/K
h = 6.62607015e-34 #Js (Planck constant)
hbar = h / (2*pi) #Js (reduced Planck constant)
k = 1.380649e-23 #J/K (Boltzmann constant)
q = 1.602176634e-19 #C (elementary charge)
T_RT = 273.15+25 #K (room temperature)
epsilon_0 = 8.8541878128e-12 #F/m = C /(Vm) = As / Vm (vacuum electric permittivity)
F =  96485.3321233 #C·mol−1 (Faraday constant)
R = 8.314462618  # J/(mol·K) (molar gas constant)
f1240 = h * c / q / 1e-9
m_e = 9.1093837015e-31 #kg (electron mass)
N_A = 6.02214076e23 # 1/mol Avogadro constant


class color:
   """
    Container class for color data and operations.
    """
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BLACK = 'black'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   
def color_list(n: float) -> Any:
    """
    Return a list of matplotlib colours for cycling through multiple plots.
    
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
    >>> color_list()
    """
    
    #cl = [color.BLACK, color.RED, color.BLUE, color.GREEN, color.CYAN, color.PURPLE, color.DARKCYAN, color.YELLOW]
    colors = mcolors.TABLEAU_COLORS
    cl = list(colors)

    c_list = []
    for i in range(n):  # type: ignore
        c_list.append(cl[i])

    return c_list
   
   
#colors = mcolors.TABLEAU_COLORS
#colors = mcolors.CSS4_COLORS
#col_names = list(colors)
colors = color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10)+color_list(10) # Will give 120 colors

CSS_colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for name, color in CSS_colors.items())
CSS_col_names = [name for hsv, name in by_hsv]


# This function rounds a float to 'sig' significant figures

def round_sig(x: np.ndarray, sig: Any=2) -> Any:
    """
    Round a number to a given number of significant figures.
    
    Parameters
    ----------
    x : np.ndarray
        X.
    sig : Any
        Sig.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> round_sig()
    """
    if x != 0:
        result = round(x, sig-int(math.floor(math.log10(abs(x))))-1)  # type: ignore
    else:
        result = 0
    return result


# This function returns a string with 'sig' significant figures

def str_round_sig(x: np.ndarray, sig: Any=2) -> Any:
    """
    Format a number as a string rounded to significant figures.
    
    Parameters
    ----------
    x : np.ndarray
        X.
    sig : Any
        Sig.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> str_round_sig()
    """
    if x != 0:
        num_string = str(round(x, sig-int(math.floor(math.log10(abs(x))))-1))  # type: ignore
    else:
        num_string = '0.0'
    while len(num_string.replace('.','')) < sig:
        num_string = num_string + '0'
    return num_string

# This function returs a 1-dim array which represents the interpolated values of arr_y at the x-values new_x
    
def int_arr(arr_x: np.ndarray, arr_y: np.ndarray, newarr_x: np.ndarray, kind: str='cubic') -> Any:
    """
    Interpolate y values onto a new x grid via scipy interp1d.
    
    Parameters
    ----------
    arr_x : np.ndarray
        Arr x.
    arr_y : np.ndarray
        Arr y.
    newarr_x : np.ndarray
        Newarr x.
    kind : str
        Kind.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> int_arr()
    """
    arr_interp = interp1d(arr_x, arr_y, kind, bounds_error=False, fill_value=0)
    arr_int_y = np.zeros(len(newarr_x))
    for i, new_x in enumerate(newarr_x):
        arr_int_y[i] = arr_interp(new_x)
    return arr_int_y

def interpolated_array(arr_nm: np.ndarray, arr_y: np.ndarray, wavelengths: np.ndarray) -> Any:
    """
    Interpolate an array onto new x positions using linear interpolation.
    
    Parameters
    ----------
    arr_nm : np.ndarray
        Arr nm, in nm.
    arr_y : np.ndarray
        Arr y.
    wavelengths : np.ndarray
        Wavelengths.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> interpolated_array()
    """
    return int_arr(arr_nm, arr_y, wavelengths, kind='linear')

def df_interpolate(df: Any, new_index_arr: np.ndarray) -> Any:
    """
    Reindex a DataFrame to a new index array using quadratic interpolation.

    Parameters
    ----------
    df : pandas dataframe
        The dataframe to be interpolated.
    new_index_arr : numpy array
        The new index.

    Returns
    -------
    pandas dataframe
        with new index.

    """
    lower_limit = min(df.index.values)
    upper_limit = max(df.index.values)
    new_index_arr = new_index_arr[new_index_arr >= lower_limit]
    new_index_arr = new_index_arr[new_index_arr <= upper_limit]
    r = pd.Index(new_index_arr)
    t = df.index
    #df_new = df.reindex(t.union(r)).interpolate('index').loc[r] #changed for the next line 11.4.2025
    df_new = df.reindex(t.union(r)).interpolate(method='quadratic', axis='index').loc[r]
    df_new.index.name = df.index.name
    df_new.index = df_new.index.astype(type(new_index_arr[0]))
    return df_new

def findind(arr: Any, value: Any, show_warnings: bool = False) -> Any:
    """
    Return the index of the value in arr closest to the given target.
    """
    if value < min(arr):
        value = min(arr)
        if show_warnings:
            print('Attention (function findind(arr, value)): Value < minimum of arr! value set to minimum.')
    elif value > max(arr):
        value = max(arr)
        if show_warnings:
            print('Attention (function findind(arr, value)): Value > maximum of arr! Value set to maximum')
    
    if arr[-1] > arr[0]:
        ind, = next((idx for idx, val in np.ndenumerate(arr) if (val >= value)))
    else:
        arr_rev = arr[::-1]
        ind, = next((idx for idx, val in np.ndenumerate(arr_rev) if (val >= value))) 
        ind = len(arr) - ind - 1
    return ind


def findind_exact(array: Any, value: Any) -> Any:
    """
    Returns the first index idx where the array[idx] = value 
    """
    return np.where(array == value)[0][0]

def linfit(array_x: np.ndarray, array_y: np.ndarray, von: Any | None=None, bis: Any | None=None) -> Any:
    """
    Fit a linear function to data and return slope, intercept, and R².
    
    Parameters
    ----------
    array_x : np.ndarray
        Array x.
    array_y : np.ndarray
        Array y.
    von : Any | None
        Von.
    bis : Any | None
        Bis.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> linfit()
    """
    if von is None:
        von = array_x[0]
    if bis is None:
        bis = array_x[-1]
    m, b = np.polyfit(array_x[findind(array_x,von):findind(array_x,bis)], array_y[findind(array_x,von):findind(array_x,bis)], 1)
    return m, b

def plx(text: Any, size: Any = 14) -> Any:
    """
    Print a LaTeX-formatted string to the console.
    Parameters
    ----------
    text : STRING
        Latex in $$.
    size : INTEGER
        text size. The default is 14.

    Returns
    -------
    None.

    """
    
    rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']
    ax = plt.axes([0,0,2,0.1])  # type: ignore
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.text(0.0,0.0, text, size = size, horizontalalignment='left', 
             verticalalignment='center', wrap = True)
    plt.show()
    
def v_sq(Eg: float) -> Any:
    """
    This is a simple approximation of the SQ Voc limit. The precise calculation can be found in IV.py, IVData.sq_limit_voc.

    Parameters
    ----------
    Eg : float
        Band gap in eV.

    Returns
    -------
    float
        Shockley-Queisser limit of Voc in V.

    """
    return 0.932*Eg - 0.167

def v_loss(PLQY: Any, T: float = T_RT) -> Any:
    """
    Calculate the voltage loss due to non-radiative recombination in eV.
    
    Parameters
    ----------
    PLQY : Any
        Plqy.
    T : float
        T.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> v_loss()
    """
    return k * T / q * np.log(PLQY)

def qfls(Eg: float, PLQY: Any) -> Any:
    """
    Calculate the quasi-Fermi level splitting from Voc and bandgap in eV.
    
    Parameters
    ----------
    Eg : float
        Eg.
    PLQY : Any
        Plqy.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> qfls()
    """
    return v_sq(Eg) + v_loss(PLQY)

def diff_coeff(mu: Any) -> Any:
    """
    Calculate the diffusion coefficient from carrier mobility (Einstein relation).
    
    Parameters
    ----------
    mu : Any
        Mu.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> diff_coeff()
    """
    # Calculates the diffusion coefficient from the mobility
    # If the mobility is given in cm2/(Vs) then the diffusion coefficient is in units cm2/s.
    return k * T_RT / q * mu
  
def mobility(D: float) -> Any:
    """
    Calculate charge-carrier mobility from the diffusion coefficient.
    
    Parameters
    ----------
    D : float
        D.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> mobility()
    """
    # Caluclates the mobility form the diffusion coefficient.
    # If the diff. coefficient is given in cm2/s then the mobility is in units cm2/(Vs).
    return (k * T_RT / q)**(-1) * D

def save_ok(TFN: str, quitted: Any | None = None) -> Any:
    """
    Prompt the user before overwriting an existing file.
    Returns boolean indicating if it is ok to save (override) the file.
    TFN: Total filename including file path.
    To be compatible with older code, if quitted is None then just return save_ok, otherwise also return the variable quitted.
    Example how to use:
        quitted = False
        for i, sp in enumerate(self.sa):
            TFN = join(save_dir, filepath)
            ok_to_save, quitted = save_ok(TFN, quitted)
            if ok_to_save and not(quitted):
                df = pd.DataFrame({x_col_name : sp.x, y_col_name : sp.y})
                df.to_csv(join(save_dir, filepath), header = True, index = False)
    """

    if (quitted == False) or (quitted is None):
        my_file = Path(TFN)
        if my_file.is_file() or my_file.is_dir():
            print(f'Warning: "{my_file}" exists!')
            
            execute_loop = True
            while execute_loop:
                if quitted is None:
                    input_var = input("Override? (yes: y, no: n): ")
                else:
                    input_var = input("Override? (yes: y, no: n, quit: q): ")
                if input_var == 'y':
                    ok = True
                    print('File/path overwritten!')
                    execute_loop = False
                elif input_var == 'n':
                    ok = False
                    print('File/path not saved!')
                    execute_loop = False
                elif input_var == 'q':
                    ok = False
                    quitted = True
                    print('File/path not saved, saving process quitted!')
                    execute_loop = False
                else:
                    print('Input not valid!')
        else:
            ok = True
    else:
        ok = False

    if quitted is None:
        return ok
    else:
        return ok, quitted

    
def how_long(process: Any, arr: Any | None=None) -> Any:
    """
    Estimate and print the remaining time for a loop iteration.
    
    Parameters
    ----------
    process : Any
        Process.
    arr : Any | None
        Arr.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> how_long()
    """
    if arr is None:
        arr = np.arange(1, 2, 1)
    """
    Calculates how long a process takes if it is repeated according to the values of arr.
    if no arr is provided, then the process is evaluated once.

    Parameters
    ----------
    process : funciton
        Process to be evaluated.
    arr: numpy.ndarray, optional
        array. The default is array([1]).

    Returns
    -------
    total_time : float
        total elapsed time of the process.

    """
    import time

    t = time.process_time()
    process()
    elapsed_time = time.process_time() - t
    no_steps = len(arr)
    
    total_time = elapsed_time * no_steps
    return total_time

def beep(freq: Any = 600, duration: Any = 1000) -> Any:
    """
    Play a short audio beep to signal completion (Windows only).
    
    Parameters
    ----------
    freq : Any
        Freq, in hz.
    duration : Any
        Duration.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> beep()
    """
    if sys.platform == 'win32':
        winsound.beep( freq, duration )  # type: ignore
    

def plot_first_n_lines(directory: str, filepath: str, n: float=20, encoding: str = "ISO-8859-1") -> Any:
    """
    Plot the first n data lines from a text file for inspection.
    
    Parameters
    ----------
    directory : str
        Directory.
    filepath : str
        Filepath.
    n : float
        N.
    encoding : str
        Encoding.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> plot_first_n_lines()
    """
    
    TFN = join(directory,filepath)
        
    count = 0
    with open(TFN, encoding = encoding) as z:
        
        for line in z:
            
            if count < n:
                print(f'{count}: {line}', end = '\r')        
                count += 1

            else:
                break

def fullprint(*args, **kwargs) -> Any:
    """
    Print a numpy array without truncation.
    """
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)  # type: ignore
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)
    
def is_even(num: Any) -> Any:
    """
    Return True if n is even.
    
    Parameters
    ----------
    num : Any
        Num.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> is_even()
    """
    return num % 2 == 0

def is_odd(num: Any) -> Any:
    """
    Return True if n is odd.
    
    Parameters
    ----------
    num : Any
        Num.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> is_odd()
    """
    return num % 2 != 0


def idx_range(arr: Any, left: float | None = None, right: float | None = None) -> Any:
    """
    Return the index range [i_left, i_right] for a value interval in a sorted array.
    
    Parameters
    ----------
    arr : Any
        Arr.
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
    >>> idx_range()
    """
    # Returns the index range from x = left to x = right
    # Only works for monotoneous ascending or descending np.arrays
    l = left
    r = right
    
    #Ascending array
    if arr[-1] > arr[0]:
        if (l is None) or (l < min(arr)): 
            l = min(arr)
        if (r is None) or (r > max(arr)):
            r = max(arr)

        if l > r:
            tmp = l
            l = r
            r = tmp
                        
    #Descending array
    else:
        if (r is None) or (r < min(arr)): 
            r = min(arr)
        if (l is None) or (l > max(arr)):
            l = max(arr)

        if l < r:
            tmp = l
            l = r
            r = tmp
    
    ra = range(findind(arr, l), findind(arr, r)+1)    
    
    return ra

def scattered_boxplot(ax: Any, x: np.ndarray, notch: Any | None=None, sym: Any | None=None, vert: Any | None=None, whis: Any | None=None, positions: Any | None=None, widths: Any | None=None, patch_artist: Any | None=None, bootstrap: Any | None=None, usermedians: Any | None=None, conf_intervals: Any | None=None, meanline: Any | None=None, showmeans: Any=None, showcaps: Any=None, showbox: Any=None,
                      showfliers: Any="unif",
                      hide_points_within_whiskers: Any=False,
                      boxprops: Any | None=None, labels: Any | None=None, flierprops: Any | None=None, medianprops: Any | None=None, meanprops: Any | None=None, capprops: Any | None=None, whiskerprops: Any | None=None, manage_ticks: Any=True, autorange: Any=False, zorder: Any | None=None, *, data: Any | None=None,
                      alpha: Any=0.2,marker: Any="o", facecolors: Any='none', edgecolors: Any="k") -> Any:
    """
    Draw a boxplot with individual data points overlaid as a scatter.
    
    Parameters
    ----------
    ax : Any
        Ax.
    x : np.ndarray
        X.
    notch : Any | None
        Notch.
    sym : Any | None
        Sym.
    vert : Any | None
        Vert.
    whis : Any | None
        Whis.
    positions : Any | None
        Positions.
    widths : Any | None
        Widths.
    patch_artist : Any | None
        Patch artist.
    bootstrap : Any | None
        Bootstrap.
    usermedians : Any | None
        Usermedians.
    conf_intervals : Any | None
        Conf intervals.
    meanline : Any | None
        Meanline.
    showmeans : Any
        Showmeans.
    showcaps : Any
        Showcaps.
    showbox : Any
        Showbox.
    showfliers : Any
        Showfliers.
    hide_points_within_whiskers : Any
        Hide points within whiskers.
    boxprops : Any | None
        Boxprops.
    labels : Any | None
        Labels.
    flierprops : Any | None
        Flierprops.
    medianprops : Any | None
        Medianprops.
    meanprops : Any | None
        Meanprops.
    capprops : Any | None
        Capprops.
    whiskerprops : Any | None
        Whiskerprops.
    manage_ticks : Any
        Manage ticks.
    autorange : Any
        Autorange.
    zorder : Any | None
        Zorder.
    data : Any | None
        Data.
    alpha : Any
        Alpha.
    marker : Any
        Marker.
    facecolors : Any
        Facecolors.
    edgecolors : Any
        Edgecolors.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> scattered_boxplot()
    """
    if showfliers=="classic":
        classic_fliers=True
    else:
        classic_fliers=False
    ax.boxplot(x, notch=notch, sym=sym, vert=vert, whis=whis, positions=positions, widths=widths, patch_artist=patch_artist, bootstrap=bootstrap, usermedians=usermedians, conf_intervals=conf_intervals, meanline=meanline, showmeans=showmeans, showcaps=showcaps, showbox=showbox,
               showfliers=classic_fliers,
               boxprops=boxprops, labels=labels, flierprops=flierprops, medianprops=medianprops, meanprops=meanprops, capprops=capprops, whiskerprops=whiskerprops, manage_ticks=manage_ticks, autorange=autorange, zorder=zorder,data=data)
    N=len(x)
    datashape_message = ("List of boxplot statistics and `{0}` "
                             "values must have same the length")
    # check position
    if positions is None:
        positions = list(range(1, N + 1))
    elif len(positions) != N:
        raise ValueError(datashape_message.format("positions"))

    positions = np.array(positions)
    if len(positions) > 0 and not isinstance(positions[0], Number):
        raise TypeError("positions should be an iterable of numbers")

    # width
    if widths is None:
        widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
    elif np.isscalar(widths):
        widths = [widths] * N
    elif len(widths) != N:
        raise ValueError(datashape_message.format("widths"))

    if hide_points_within_whiskers:
        import matplotlib.cbook as cbook
        from matplotlib import rcParams
        if whis is None:
            whis = rcParams['boxplot.whiskers']
        if bootstrap is None:
            bootstrap = rcParams['boxplot.bootstrap']
        bxpstats = cbook.boxplot_stats(x, whis=whis, bootstrap=bootstrap,
                                       labels=labels, autorange=autorange)
    for i in range(N):
        if hide_points_within_whiskers:
            xi=bxpstats[i]['fliers']
        else:
            xi=x[i]
        if showfliers=="unif":
            jitter=np.random.uniform(-widths[i]*0.5,widths[i]*0.5,size=np.size(xi))
        elif showfliers=="normal":
            jitter=np.random.normal(loc=0.0, scale=widths[i]*0.1,size=np.size(xi))
        elif showfliers==False or showfliers=="classic":
            return
        else:
            raise NotImplementedError("showfliers='"+str(showfliers)+"' is not implemented. You can choose from 'unif', 'normal', 'classic' and False")

        plt.scatter(positions[i]+jitter,xi,alpha=alpha,marker=marker, facecolors=facecolors, edgecolors=edgecolors)

setattr(plt.Axes, "scattered_boxplot", scattered_boxplot)


def ignore_warnings(func: Any, *args, enable_warnings: Any = False, **kwargs) -> Any:
    """
    Decorator that suppresses Python warnings from the wrapped function.
    
    Parameters
    ----------
    func : Any
        Func.
    enable_warnings : Any
        Enable warnings.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> ignore_warnings()
    """
    #ignore all warnings for the function func
    if not enable_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return func(*args, **kwargs)

def copy_to_clipboard(text: Any) -> Any:
    """
    Copies text to the clipboard. The text can be copied by ctrl-v e.g. to an excel sheet. If commas are used, then the text between the commas is organized in different columns.
    Uses the package pyperclip (https://pypi.org/project/pyperclip/). To be installed via pip install pyperclip.
    Example:
        >>>text = 'Hello'
        >>>General.copy_to_clipboard(text)

    Parameters
    ----------
    text : str
        Text to be copied to clipboard.

    Returns
    -------
    None.

    """

    import pyperclip
    pyperclip.copy(text)
    
def win_long_fp(fp: Any) -> Any:
    """
    Prepend \\?\ to a path to enable Windows long-path support.
    
    Parameters
    ----------
    fp : Any
        Fp.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> win_long_fp()
    """
    # If the os is windows: Transforms a filepath fp that is too long for windows into a windows readable filepath
    # in all other cases it just returns fp
    windows_long_file_prefix = '\\\\?\\'
    if ((platform.system() == 'Windows') and (len(fp) > 255) and (not fp.startswith(windows_long_file_prefix))):
        return windows_long_file_prefix + fp
    else:
        return fp

def max_len(list_of_strings: Any) -> Any:
    """
    Return the length of the longest sequence in a list.
    
    Parameters
    ----------
    list_of_strings : Any
        List of strings.
    
    Returns
    -------
    Any
        Computed result.
    
    Examples
    --------
    >>> max_len()
    """
    #finds the max length of the strings in a list
    #This can be used e.g. in f-strings: print(f'{conditions[idx].ljust(max_len(conditions))}: text') T
    n = 0
    for string in list_of_strings:
        if len(string) > n:
            n = len(string)
    return n
