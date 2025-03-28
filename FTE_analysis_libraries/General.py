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
from IPython import embed
from pathlib import Path
import sys
from os.path import join
import warnings
from numbers import Number

if sys.platform == 'Win32':
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
R = 0.99999999965e-3 #kg/mol (molar gas constant)
f1240 = h * c / q / 1e-9
m_e = 9.1093837015e-31 #kg (electron mass)
N_A = 6.02214076e23 # 1/mol Avogadro constant


class color:
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
   
def color_list(n):
    
    #cl = [color.BLACK, color.RED, color.BLUE, color.GREEN, color.CYAN, color.PURPLE, color.DARKCYAN, color.YELLOW]
    colors = mcolors.TABLEAU_COLORS
    cl = list(colors)

    c_list = []
    for i in range(n):
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

def round_sig(x, sig=2):
    if x != 0:
        result = round(x, sig-int(math.floor(math.log10(abs(x))))-1)
    else:
        result = 0
    return result


# This function returns a string with 'sig' significant figures

def str_round_sig(x, sig=2):
    if x != 0:
        num_string = str(round(x, sig-int(math.floor(math.log10(abs(x))))-1))
    else:
        num_string = '0.0'
    while len(num_string.replace('.','')) < sig:
        num_string = num_string + '0'
    return num_string

# This function returs a 1-dim array which represents the interpolated values of arr_y at the x-values new_x
    
def interpolated_array(arr_nm, arr_y, wavelengths):
    arr_interp = interp1d(arr_nm, arr_y, kind = 'linear', bounds_error=False, fill_value=0)
    #arr_interp = interp1d(arr_nm, arr_y, kind = 'cubic', bounds_error=False, fill_value=0)
    arr_int_y = np.zeros(len(wavelengths)) # dummy array
    for i,nm in enumerate(wavelengths):
        arr_int_y[i] = arr_interp(nm)
    return arr_int_y  

def int_arr(arr_x, arr_y, newarr_x, kind = 'cubic'):
    arr_interp = interp1d(arr_x, arr_y, kind, bounds_error=False, fill_value=0)
    arr_int_y = np.zeros(len(newarr_x)) # dummy array
    for i,new_x in enumerate(newarr_x):
        arr_int_y[i] = arr_interp(new_x)
    return arr_int_y

def df_interpolate(df, new_index_arr):
    """
    Generates an interpolated dataframe, where the index of df is replaced by new_index_arr.

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
    df_new = df.reindex(t.union(r)).interpolate('index').loc[r]
    df_new.index.name = df.index.name
    df_new.index = df_new.index.astype(type(new_index_arr[0]))
    return df_new

def findind(arr, value, show_warnings = False):
    """
    Works only for ascending or descending arrays!
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


def findind_exact(array, value):
    """
    Returns the first index idx where the array[idx] = value 
    """
    return np.where(array == value)[0][0]

def linfit(array_x, array_y, von=None, bis=None):
    if von == None:
        von = array_x[0]
    if bis == None:
        bis = array_y[-1]
    m, b = np.polyfit(array_x[findind(array_x,von):findind(array_x,bis)], array_y[findind(array_x,von):findind(array_x,bis)], 1)
    return m, b

def plx(text, size = 14):
    """
    Print text with latex.
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
    ax = plt.axes([0,0,2,0.1]) #left,bottom,width,height
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.text(0.0,0.0, text, size = size, horizontalalignment='left', 
             verticalalignment='center', wrap = True)
    plt.show()
    
def Vsq(Eg):
    """
    This is a simple approximation of the SQ Voc limit. The precise calculation can be found in IV.py, IV_data.SQ_limit_Voc.

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

def V_loss(PLQY, T = T_RT):
    return k * T_RT / q * np.log(PLQY)

def QFLS(Eg, PLQY):
    return Vsq(Eg) + V_loss(PLQY)

def Diff_coeff(mu):
    # Calculates the diffusion coefficient from the mobility
    # If the mobility is given in cm2/(Vs) then the diffusion coefficient is in units cm2/s.
    return k * T_RT / q * mu
  
def Mobility(D):
    # Caluclates the mobility form the diffusion coefficient.
    # If the diff. coefficient is given in cm2/s then the mobility is in units cm2/(Vs).
    return (k * T_RT / q)**(-1) * D

def save_ok(TFN, quitted = None):
    """
    Check if file exists. If yes, ask to override.
    Returns boolean indicating if it is ok to save (override) the file.
    TFN: Total filename including file path.
    To be compatible with older code, if quitted == None then just return save_ok, otherwise also return the variable quitted.
    Example how to use:
        quitted = False
        for i, sp in enumerate(self.sa):
            TFN = join(save_dir, FN)
            ok_to_save, quitted = save_ok(TFN, quitted)
            if ok_to_save and not(quitted):
                df = pd.DataFrame({x_col_name : sp.x, y_col_name : sp.y})
                df.to_csv(join(save_dir, FN), header = True, index = False)
    """

    if (quitted == False) or (quitted == None):
        my_file = Path(TFN)
        if my_file.is_file() or my_file.is_dir():
            print(f'Warning: "{my_file}" exists!')
            
            execute_loop = True
            while execute_loop:
                if quitted == None:
                    input_var = input("Override? (yes: y, no: n): ")
                else:
                    input_var = input("Override? (yes: y, no: n, quit: q): ")
                if input_var == 'y':
                    save_ok = True
                    print('File/path overwritten!')
                    execute_loop = False
                elif input_var == 'n':
                    save_ok = False
                    print('File/path not saved!')
                    execute_loop = False
                elif input_var == 'q':
                    save_ok = False
                    quitted = True
                    print('File/path not saved, saving process quitted!')
                    execute_loop = False
                else:
                    print('Input not valid!')            
        else:
            save_ok = True
    else:
        save_ok = False
            
    if quitted == None:
        return save_ok
    else:
        return save_ok, quitted

    
def how_long(process, arr = np.arange(1, 2, 1)):
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

def beep(freq = 600, duration = 1000):
    if sys.platform == 'Win32':
        winsound.beep( freq, duration )
    

def plot_first_n_lines(dir, FN, n=20, encoding = "ISO-8859-1"):
    
    TFN = join(dir,FN)
        
    count = 0
    with open(TFN, encoding = encoding) as z:
        
        for line in z:
            
            if count < n:
                print(f'{count}: {line}', end = '\r')        
                count += 1

            else:
                break

def fullprint(*args, **kwargs):
    """
    Print the full np.array.
    """
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)
    
def is_even(num):
    return num % 2 == 0

def is_odd(num):
    return num % 2 != 0


def idx_range(arr, left = None, right = None):
    # Returns the index range from x = left to x = right
    # Only works for monotoneous ascending or descending np.arrays
    l = left
    r = right
    
    #Ascending array
    if arr[-1] > arr[0]:
        if (l == None) or (l < min(arr)): 
            l = min(arr)
        if (r == None) or (r > max(arr)):
            r = max(arr)

        if l > r:
            l = r
            r = l
                        
    #Descending array
    else:
        if (r == None) or (r < min(arr)): 
            r = min(arr)
        if (l == None) or (l > max(arr)):
            l = max(arr)

        if l < r:
            l = r
            r = l
    
    ra = range(findind(arr, l), findind(arr, r)+1)    
    
    return ra

def scattered_boxplot(ax, x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None,
                      showfliers="unif",
                      hide_points_within_whiskers=False,
                      boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, *, data=None,
                      alpha=0.2,marker="o", facecolors='none', edgecolors="k"):
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


if __name__ == "__main__":
    
    #print(str_round_sig(1123423.0234))
    
    # test interpolated_array
    arr_x = np.arange(1,10,5)
    arr_y = arr_x**2
    newarr_x = np.arange(0, 13, 0.5)
    plt.plot(arr_x, arr_y, '-')
    newy = int_arr(arr_x, arr_y, newarr_x, kind = 'linear')
    plt.plot(newarr_x, newy, 'o')
    plt.show()
    
def ignore_warnings(func, *args, enable_warnings = False, **kwargs):
    #ignore all warnings for the function func
    if not enable_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    return(func(*args, **kwargs))

def copy_to_clipboard(text):
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
    
def win_long_fp(fp):
    # If the os is windows: Transforms a filepath fp that is too long for windows into a windows readable filepath
    # in all other cases it just returns fp
    windows_long_file_prefix = '\\\\?\\'
    if ((platform.system() == 'Windows') and (len(fp) > 255) and (not fp.startswith(windows_long_file_prefix))):
        return windows_long_file_prefix + fp
    else:
        return fp

def max_len(list_of_strings):
    #finds the max length of the strings in a list
    #This can be used e.g. in f-strings: print(f'{conditions[idx].ljust(max_len(conditions))}: text') T
    n = 0
    for string in list_of_strings:
        if len(string) > n:
            n = len(string)
    return n
