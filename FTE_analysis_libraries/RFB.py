# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:32:46 2024

@author: dreickem
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
import scipy.constants as const

if __name__ != "__main__":
    from . import General as gen
    from . import XYdata as xyd
    from . import Electrochemistry as ech
else:
    from FTE_analysis_libraries import General as gen
    from FTE_analysis_libraries import XYdata as xyd
    from FTE_analysis_libraries import Electrochemistry as ech

R = const.physical_constants['molar gas constant'][0]
F = const.physical_constants['Faraday constant'][0]
a = const.Avogadro
q = const.elementary_charge

s_per_h = 3600 # seconds per hour
#Molecular weight of oxalic acid
MW_oxalicacid = 90.034 #g/mol for anhydrous oxalic acid, 126.065 g/mol for dihydrate


#%% Electrolyte

def conc_V_SO4(weight_pc_V = 6.0, weight_pc_SO4 = 28, density = 1.35, show_details=False):

    """
    **Information from AMG**
    weight_pc_V = 6.0 # Weight percent of Vanadium
    weight_pc_SO4 = 28 # Weight percent of SO42-
    density = 1.35 #kg/L
    """
    
    #**Concentration (M=mol/L) of V and SO4**
    MW_V = 50.942 # g/mol, Molar mass of Vanadium
    MW_SO4 = 96.06 # g/mol, Molar mass of SO4^2-
    
    concentration = lambda weight_pc, MW: density*1000 * weight_pc/100 / MW 
    c_V = concentration(weight_pc_V, MW_V)
    c_SO4 = concentration(weight_pc_SO4, MW_SO4)
    if show_details:
        print(f'Total vanadium concentration: {c_V:.2f} M')
        print(f'Total SO4 concentration: {c_SO4:.2f} M')

    return c_V, c_SO4

def pH_c_H_plus(c_H_plus):
    return -np.log10(c_H_plus)

def calc_conc_functions_without_Ka1(c_V, c_SO4):

    """
    This function calculates the H+ and HSO4- concentration as a function fo the average vanadium oxidation state from 2 to 5.

    
    We have the following four equations:  
    
    $\begin{equation}
    \ce{ H2O <=> HO- + H+ , \, K_w = \frac{[OH-] \ [H+]}{[H2O]} =  10^{-14} , \, [H2O] = 1 } \tag{1}
    \end{equation}$
    
    $\begin{equation}
    \ce{ HSO4- <=> SO4^{2-} + H+ , \, K_{a2} = \frac{[SO4^{2-}] \ [H+]}{[HSO4-]} = 10^{-1.99} } \tag{2}
    \end{equation}$
    
    $\begin{equation}
    \ce{ [HSO4-] + [SO4^{2-}] = [SO4] } \tag{3}
    \end{equation}$
    
    Charge balance:
    
    $\begin{equation}
    \ce{ c_{pos_V} + [H+] = [HSO4-] + 2 * [SO4^{2-}] + [OH-] } \tag{4}
    \end{equation}$
    
    $\begin{equation}
    \ce{ c_{pos_V} \equiv 2 \cdot [V^{2+}] + 3 \cdot [V^{3+}] + 2 \cdot [VO^{2+}] + 1 \cdot [VO_2^{+}]  \tag{5}}
    \end{equation}$

    Knowns: $\ce{K_a_2, K_w, c_{pos_V}, [SO4] }$  
    Unknowns: $\ce{ [HSO4-], [SO4^{2-}], [H+], [OH-]}$  
    So, 4 equations and 4 unknowns!

    Once we have the proton concentration we can calculate the sulfate and bisulfate concentrations using equations (2) and (3):  
    
    $\begin{equation}
    \ce{ [HSO4-] = \frac{[H+] [SO4]}{[H+] + K_{a2}} } \tag{6}
    \end{equation}$
    
    $\begin{equation}
    \ce{ [SO4^{2-}] = [SO4] - [HSO4-] } \tag{7}
    \end{equation}$

    Vanadium: Concentrations  
    -$\ce{V^{2+}}$: c_2  
    -$\ce{V^{3+}}$: c_3  
    -$\ce{(V^{IV}O)^{2+}}$: c_4    
    -$\ce{(V^{V}O_2)^{+}}$: c_5  

    """
    
    
    #Calculate concentration of positive charges from Vanadium
    c_pos_V = lambda conc: 2*conc['c_2'] + 3*conc['c_3'] + 2*conc['c_4'] + 1*conc['c_5']
    
    Ka2 = 10**(-1.99) #https://en.wikipedia.org/wiki/Sulfuric_acid
    Kw = 10**(-14)
    
    # h is the proton concentration, this is the quantity we want to know
    s = c_SO4
    func = lambda h, c_pos_V: h**3 + (c_pos_V+Ka2-s)*h**2 + (c_pos_V*Ka2-2*s*Ka2-Kw)*h - Kw*Ka2
    #func = lambda h: h**2 + (c+Ka2-s)*h + (c*Ka2-2*s*Ka2) # if Kw is zero
    
    h_initial_guess = 2.0
    c_H_plus = lambda conc: fsolve(func, h_initial_guess, args=(c_pos_V(conc)))[0]
    
    c_HSO4_minus = lambda c_H_plus: c_H_plus*c_SO4/(c_H_plus + Ka2)
    c_SO4_2minus = lambda c_HSO4_minus: c_SO4 - c_HSO4_minus 
    
    #x is the average oxidation state (from 2 to 5)
    c_2 = lambda x: c_V*(3-x) if (2 <= x) and (x < 3) else 0
    c_3 = lambda x: c_V*(x-2) if (2 <= x) and (x < 3) else (c_V*(4-x) if (3 <= x) and (x < 4) else 0)
    c_4 = lambda x: c_V*(x-3) if (3 <= x) and (x < 4) else (c_V*(5-x) if (4 <= x) and (x < 5) else 0)
    c_5 = lambda x: c_V*(x-4) if (4 <= x) and (x <= 5) else 0
    conc_x = lambda x: {'c_2': c_2(x), 'c_3': c_3(x), 'c_4': c_4(x), 'c_5': c_5(x)}

    return conc_x, c_H_plus, c_HSO4_minus, c_SO4_2minus


def calc_conc_functions(c_V, c_SO4):

    """
    This function calculates the H+ and HSO4- concentration as a function fo the average vanadium oxidation state from 2 to 5.

    We have the following five equations:  

    $\begin{equation}
    \ce{ H2O <=> HO- + H+ , \, K_w = \frac{[OH-] \ [H+]}{[H2O]} = [OH-] \ [H+] =  10^{-14} , \, [H2O] = 1 } \tag{1}
    \end{equation}$
    
    $\begin{equation}
    \ce{ H2SO4 <=> HSO4- + H+ , \, K_{a1} = \frac{[HSO4-] \ [H+]}{[H2SO4]} = 10^{2.8} } \tag{2}
    \end{equation}$
    
    $\begin{equation}
    \ce{ HSO4- <=> SO4^{2-} + H+ , \, K_{a2} = \frac{[SO4^{2-}] \ [H+]}{[HSO4-]} = 10^{-1.99} } \tag{3}
    \end{equation}$
    
    $\begin{equation}
    \ce{ [H2SO4] + [HSO4-] + [SO4^{2-}] = [SO4] } \tag{4}
    \end{equation}$
    
    Charge balance:
    
    $\begin{equation}
    \ce{ c_{pos_V} + [H+] = [HSO4-] + 2 * [SO4^{2-}] + [OH-] } \tag{5}
    \end{equation}$
    
    $\begin{equation}
    \ce{ c_{pos_V} \equiv 2 \cdot [V^{2+}] + 3 \cdot [V^{3+}] + 2 \cdot [VO^{2+}] + 1 \cdot [VO_2^{+}]  \tag{6}}
    \end{equation}$
    
    Knowns: $\ce{K_a_1, K_a_2, K_w, c_{pos_V}, [SO4] }$  
    Unknowns: $\ce{ [H2SO4], [HSO4-], [SO4^{2-}], [H+], [OH-]}$  
    So, 5 equations and 5 unknowns!
    
    Once we have the proton concentration we can calculate the sulfate and bisulfate concentrations using equations (2) and (3):  
    
    $\begin{equation}
    \ce{ [HSO4-] = \frac{[H+] [SO4]}{[H+] + K_{a2}} } \tag{7}
    \end{equation}$
    
    $\begin{equation}
    \ce{ [SO4^{2-}] = [SO4] - [HSO4-] } \tag{8}
    \end{equation}$
    
    Vanadium: Concentrations  
    -$\ce{V^{2+}}$: c_2  
    -$\ce{V^{3+}}$: c_3  
    -$\ce{(V^{IV}O)^{2+}}$: c_4    
    -$\ce{(V^{V}O_2)^{+}}$: c_5  
    
    """
    
    #Calculate concentration of positive charges from Vanadium
    c_pos_V_conc = lambda conc: 2*conc['c_2'] + 3*conc['c_3'] + 2*conc['c_4'] + 1*conc['c_5']
    
    Ka1 = 10**(2.8) #https://en.wikipedia.org/wiki/Sulfuric_acid
    Ka2 = 10**(-1.99) #https://en.wikipedia.org/wiki/Sulfuric_acid
    Kw = 10**(-14)
    
    #These are the knowns:
    a0 = Ka1
    a1 = Ka2
    a2 = Kw
    a3 = c_SO4
    
    #These are the unknowns:
    #x[0] = [H2SO4]
    #x[1] = [HSO4-]
    #x[2] = [SO4^2-]
    #x[3] = [H+]
    #x[4] = [OH-]
    
    
    func = lambda x, c_pos_V: [a2-x[3]*x[4], 
                      a0*x[0]-x[1]*x[3], 
                      a1*x[1]-x[2]*x[3], 
                      x[0]+x[1]+x[2]-a3,
                      c_pos_V+x[3]-x[1]-2*x[2]-x[4] ]
    
    #It is important to take a good initial guess, e.g. this doesn't work: x_initial_guess = [0.01, 3.0, 1.0, 0.2, 0.001]
    x_initial_guess = [0.01, 3.0, 1.0, 1.0, 0.001]
    
    Xconc_Vconc_fn = lambda conc: fsolve(func, x_initial_guess, args=(c_pos_V_conc(conc)))
    
    #x is the average oxidation state (from 2 to 5)
    c_2 = lambda x: c_V*(3-x) if (2 <= x) and (x < 3) else 0
    c_3 = lambda x: c_V*(x-2) if (2 <= x) and (x < 3) else (c_V*(4-x) if (3 <= x) and (x < 4) else 0)
    c_4 = lambda x: c_V*(x-3) if (3 <= x) and (x < 4) else (c_V*(5-x) if (4 <= x) and (x < 5) else 0)
    c_5 = lambda x: c_V*(x-4) if (4 <= x) and (x <= 5) else 0
    Vconc_x_fn = lambda x: {'c_2': c_2(x), 'c_3': c_3(x), 'c_4': c_4(x), 'c_5': c_5(x)}

    return Vconc_x_fn, Xconc_Vconc_fn


def calc_df_conc(c_V, c_SO4, show=False):
    
    #This function calculates the H+ and HSO4- concentration as a function of the average vanadium oxidation state from 2 to 5.

    Vconc_x_fn, Xconc_Vconc_fn = calc_conc_functions(c_V, c_SO4)
    #x is the average oxidation state (from 2 to 5)
    x_arr = np.arange(200, 501)/100
    c_H_plus_list = []
    c_H2SO4_list = []
    c_HSO4_minus_list = []
    c_SO4_2minus_list = []
    for x in x_arr:
        #x is the average oxidation state (from 2 to 5)
        Vconc = Vconc_x_fn(x)
        Xconc_Vconc = Xconc_Vconc_fn(Vconc)
        c_H_plus = Xconc_Vconc[3]
        c_H2SO4 = Xconc_Vconc[0]
        c_HSO4_minus = Xconc_Vconc[1]
        c_SO4_2minus = Xconc_Vconc[2]
        #if x>2.8 and x<3.2:
        #    print(f'{x}: {conc["c_2"]:.2f} {conc["c_3"]:.2f} {conc["c_4"]:.2f} {conc["c_5"]:.2f} - c_H: {c_H_plus}')
        c_H_plus_list.append(c_H_plus)
        c_H2SO4_list.append(c_H2SO4)
        c_HSO4_minus_list.append(c_HSO4_minus)
        c_SO4_2minus_list.append(c_SO4_2minus)

    df_conc = pd.DataFrame({'Avg. ox. state': x_arr, 'H+': c_H_plus_list, 'H2SO4': c_H2SO4_list, 'HSO4-': c_HSO4_minus_list, 'SO4[2-]': c_SO4_2minus_list}).set_index('Avg. ox. state')

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        ax = axes[0]
        df_conc.plot(ax=ax)
        ax.set_ylim(0, 1.1*c_SO4)
        ax.set_xlim(2.0, 5.0)
        ax.set_xlabel('Average oxidation state')
        ax.set_ylabel('Concentration (mol/L)')
        ax.legend(['$H^+$', '$H_2SO_4$', '$HSO_4^-$', '$SO_4^{2-}$'])
        
        ax = axes[1]
        ax.plot(df_conc.index, pH_c_H_plus(df_conc['H+']))
        #ax.set_ylim()
        ax.set_xlim(2.0, 5.0)
        ax.set_xlabel('Average oxidation state')
        ax.set_ylabel('pH')
        
        plt.show()

    return df_conc

def fit_proton_concentration(dfH, show=False):
    """
    Calculates c_Hplus_fit_till3, c_Hplus_fit_from3, the fit function (polynomial) which fits the proton concentration function (x-values are average oxidation state)
    The proton concentration can be calculated by: c_Hplus_fit(av_ox_state)
    """
    dfH_1 = dfH.loc[:3.0]
    dfH_2 = dfH.loc[3.0:]
    
    def polynomial(df, order):
        x = df.index
        y = np.log(df.values)
        z = np.polyfit(x, y, order)
        p = np.poly1d(z)
        return p
        
    p_fit_till3 = polynomial(dfH_1, 3)
    p_fit_from3 = polynomial(dfH_2, 5)

    c_Hplus_fit_till3 = lambda av_ox_state: np.exp( p_fit_till3(av_ox_state) )
    c_Hplus_fit_from3 = lambda av_ox_state: np.exp( p_fit_from3(av_ox_state) )
    
    if show:
        
        y_fit_till3 = c_Hplus_fit_till3(dfH_1.index)
        y_fit_from3 = c_Hplus_fit_from3(dfH_2.index)

        fig, ax = plt.subplots()
        
        def plot_data_and_fit(ax, ser, y_fit):
            ax.plot(ser.index, ser.values, color='black')
            ax.plot(ser.index, y_fit, color='red')
            return y_fit
        
        plot_data_and_fit(ax, dfH_1, y_fit_till3)
        plot_data_and_fit(ax, dfH_2, y_fit_from3)
        ax.set_yscale('log')
        #ax.set_ylim(0, c_SO4)
        #ax.set_xlim(2.0, 5.0)
        ax.set_xlabel('Average oxidation state')
        ax.set_ylabel('Concentration (mol/L)')
        ax.legend(['data', 'fit'])
        
        plt.show()

    return c_Hplus_fit_till3, c_Hplus_fit_from3

#%% Charging protocol

def upload_charging_protocol(fp):
    
    def time_to_seconds(time_str):
        # Split the input string by the colon ':'
        try:
            hours, minutes, seconds = map(int, time_str.split('.')[0].split(':'))
        except:
            try:
                hours, minutes, seconds = map(int, time_str.split('.')[0].split(':'))
            except: ValueError
        # Convert time to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds    
        return total_seconds    

    df_charging = pd.read_excel(fp, sheet_name=3, engine="openpyxl")
    #Rename important columns to be used with different csv files
    if 'Total Time' in df_charging.columns:
        df_charging = df_charging.rename(columns={'Total Time':'Time (s)'})
    elif 'Relative Time(h:min:s.ms)' in df_charging.columns:
        df_charging = df_charging.rename(columns={'Relative Time(h:min:s.ms)':'Time (s)'})
    else:
        raise ValueError
    df_charging.set_index('Time (s)', inplace=True)
    df_charging.index = df_charging.index.map(time_to_seconds)
    
    if 'Step Index' in df_charging.columns:
        df_charging = df_charging.rename(columns={'Step Index':'Step'})
    elif 'Steps' in df_charging.columns:
        df_charging = df_charging.rename(columns={'Steps':'Step'})
    else:
        raise ValueError
    
    # Delete resting period
    df_charging = df_charging[df_charging['Step']>1]
    
    #Correct time (time starts with 0 with each new step)
    df_charging_step1 = df_charging[df_charging['Step']==2]
    #add the charge from previous step, delete first value, since time is 0 and capacity is also 0
    df_charging_step2 = df_charging[df_charging['Step']==3].iloc[1:,:]
    df_charging_step2.index += df_charging_step1.index[-1]
    df_charging = pd.concat([df_charging_step1, df_charging_step2])
    
    #display(df_charging['Step'].to_string())
    #print(df_charging[['Step', 'Capacity(Ah)']].to_markdown())
    
    return df_charging, df_charging_step1, df_charging_step2


def split_dfcharging(df_charging, df_charging_step1, df_charging_step2):
    
    # Split into voltage, current and total charge
    
    df_voltage = df_charging['Voltage(V)']
    try:
        #Current in A
        df_current = df_charging['Current(mA)']*1000
    except:
        try:
            df_current = df_charging['Current(A)']
        except: ValueError
    #split df_charge into the two charging steps
    if 'Capacity(mAh)' in df_charging.columns:
        #Charge in Ah£
        #df_charge_step1 = df_charging[df_charging['Step']==2]['Capacity(mAh)'] * 1000
        df_charge_step1 = df_charging_step1['Capacity(mAh)'] * 1000
        #add the charge from previous step
        #df_charge_step2 = df_charging[df_charging['Step']==3].iloc[1:,:]['Capacity(mAh)'] * 1000 + df_charge_step1.iloc[-1]
        df_charge_step2 = df_charging_step2['Capacity(mAh)'] * 1000 + df_charge_step1.iloc[-1]
        
    elif 'Capacity(Ah)' in df_charging.columns:
        #df_charge_step1 = df_charging[df_charging['Step']==2]['Capacity(Ah)']
        df_charge_step1 = df_charging_step1['Capacity(Ah)']
        #add the charge from previous step
        df_charge_step2 = df_charging_step2['Capacity(Ah)'] + df_charge_step1.iloc[-1]
        #df_charge_step2 = df_charging[df_charging['Step']==3].iloc[1:,:]['Capacity(Ah)'] + df_charge_step1.iloc[-1]
    else:
        raise ValueError
    
    first_charging_step_end_s = df_charge_step1.index[-1]
    if len(df_charge_step1.index) != len(df_charge_step1.index.unique()):
        if df_charge_step1.index[df_charge_step1.index.duplicated()] == df_charge_step1.index[-1]:
            print(f'First step end-1: {df_charge_step1.index[-2]:.0f} s')
            print(f'First step end: {first_charging_step_end_s:.0f} s')
            print(f'Second step begin: {df_charge_step2.index[0]:.0f} s')
            df_charge_step1 = df_charge_step1.iloc[0:-1]
        else:
            print('Attention: df_charge_step1 indices are not unique! Last time deleted.')

    return df_charge_step1, df_charge_step2, df_voltage, df_current, first_charging_step_end_s
    
    
def combine_both_steps(df_charge_step1, df_charge_step2, df_voltage, df_current, show=False):
    
    #Combine the charge of both steps
    df_charge = pd.concat([df_charge_step1, df_charge_step2])
    df_charge.name = 'Charge (Ah)'
    
    if show:
        print('Combine the charge of both steps.')
        # Plot data
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        axl = ax[0]
        curve1,  = axl.plot(df_voltage.index, df_voltage, color='tab:blue', linewidth=5, label='Voltage')
        axr = axl.twinx()
        curve2,  = axr.plot(df_current.index, df_current, color='tab:orange', linewidth=5, label='Current')
        #axl.set_xlim(250, 1100)
        axl.set_ylim(1.0, 1.7)
        axr.set_ylim(0.0, 2.5)
        # Combine legends from both axes
        curves = [curve1, curve2]
        labels = [curve.get_label() for curve in curves]
        axl.legend(curves, labels, loc='center right')
        
        axl.set_xlabel('Time (s)')
        axl.set_ylabel('Voltage (V)')
        axr.set_ylabel('Current (A)')
        
        #axl.set_ylim((1.3, 1.65))
        #axl.set_xlim((0, 17500))
        
        plt.title('Charging protocol')
        
        ax = ax[1]
        ax.plot(df_charge.index, df_charge, color='tab:blue', linewidth=5, label='Charge')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Charge (Ah)')
        
        #axl.set_ylim((1.3, 1.65))
        #axl.set_xlim((0, 17500))
        
        plt.title('Charging protocol')
        
        
        plt.tight_layout()
        plt.show()
    
    return df_charge


def check_unique_time(df_charge):
    if len(df_charge.index) != len(df_charge.index.unique()):
        print('Attention: df_charge indices are not unique!')
        print(df_charge.index[df_charge.index.duplicated()])

        df_time_charge = df_charge.reset_index().set_index('Charge (Ah)').squeeze()
        df_time_charge.plot()
        plt.ylabel('Time (s)')
        plt.xlabel('Charge (Ah)')
        plt.show()
        
        
def det_time_and_charge_SOC0_and_SOC1_from_charging_protocol(df_charge, conc, Vol, show=False):
    # Determine time when V3.5 is oxidized/reduced to V4+/V3+ (SOC 0%) and to SOC 100%
    # Charge in mAh for it to happen
    charge = lambda el_per_species: conc * Vol * F / s_per_h * el_per_species #Ah
    
    el_per_species = 0.5 # number of electrons per species
    charge_SOC0_calc = charge(el_per_species)
    if show:
        print(f'Charge necessary to oxidize V3.5 to V4: {charge_SOC0_calc:.3f} Ah')
    
    # Calculate the time that this charge is reached
    idx_SOC0 = np.argmin(np.abs(df_charge-charge_SOC0_calc))
    time_to_reach_SOC0 = df_charge.index[idx_SOC0]
    charge_SOC0 = df_charge.iloc[idx_SOC0]
    if show:
        print(f'Charge ({charge_SOC0:.3f} mAh) reached after {time_to_reach_SOC0} s')
    
    #Total charge to go from SOC 0% to 100%
    
    el_per_species = 1 # number of electrons per species
    charge_SOC1_calc = charge(el_per_species)
    if show:
        print(f'Charge necessary to charge from SOC 0% to 100%: {charge_SOC1_calc:.3f} Ah')
    
    # Calculate the time that this charge is reached
    idx_SOC1 = -1
    time_to_reach_SOC1 = df_charge.index[idx_SOC1] - df_charge.index[idx_SOC0]
    charge_SOC1 = df_charge.iloc[idx_SOC1] - df_charge.iloc[idx_SOC0]
    if show:
        print(f'Charge ({charge_SOC1:.3f} Ah) reached after {time_to_reach_SOC1} s')
    return time_to_reach_SOC0, time_to_reach_SOC1, charge_SOC0, charge_SOC1
    

def det_no_equidist_times(df_charge, time_to_reach_SOC0, time_to_reach_SOC1, charge_SOC0, charge_SOC1, no=10, show=False):
    #Determine 10 equidistant times until V3.5 is oxidized/reduced to V4/V3
    times_until_SOC0 = np.array([int(round(time_to_reach_SOC0*i/no)) for i in range(no+1)])
    if show:
        print(f'{no} equidistant points until SOC 0: {times_until_SOC0}')
    #Determine further 10 times with equidistant delta charge until fully charged
    charge_steps = [charge_SOC1*i/no for i in range(no+1)]
    def determine_time(charge):
        idx = np.argmin(np.abs(df_charge-charge))
        return df_charge.index[idx]
    times_until_SOC1 = np.array([determine_time(charge+charge_SOC0) for charge in charge_steps])
    if show:
        print(f'{no} equidistant points until SOC 1: {times_until_SOC1}')
    return times_until_SOC0, times_until_SOC1


#%% Potential measurement

def upload_potential_measurement(potential_measurement_system,
                                 FN_negolyte_potential,
                                 FN_posolyte_potential,
                                 data_dir,
                                 refCellCalibration,
                                 no_times=1000,
                                 show=False):
    #Load measurement
    #no_times: Reduce the size to n_times time steps
    
    if potential_measurement_system == 'biologic':
        FN_neg = FN_negolyte_potential
        FN_pos = FN_posolyte_potential
        fp_neg = os.path.join(data_dir, FN_neg)
        fp_pos = os.path.join(data_dir, FN_pos)
        #extension = os.path.splitext(FN_neg)[1]
        
        exp_pos = ech.import_biologic_mpt_data(fp_pos, ['time/s', '<Ewe/V>'])
        exp_pos.columns = ['Time (s)', 'Potential (V)']
        exp_pos.set_index('Time (s)', inplace=True)
        exp_neg = ech.import_biologic_mpt_data(fp_neg, ['time/s', '<Ewe/V>'])
        exp_neg.columns = ['Time (s)', 'Potential (V)']
        exp_neg.set_index('Time (s)', inplace=True)
        
        new_times_s = np.linspace(exp_pos.index[0], exp_pos.index[-1], no_times)
        exp_pos = gen.df_interpolate(exp_pos, new_times_s)
        exp_neg = gen.df_interpolate(exp_neg, new_times_s)
        
        # Calibrate
        Ppos_raw = exp_pos + refCellCalibration
        Pneg_raw = exp_neg + refCellCalibration
        if show:
            # Plot
            fig, ax = plt.subplots()
            Ppos_raw.plot(ax=ax)
            Pneg_raw.plot(ax=ax)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Potential (V vs. SHE)')
            plt.legend(['Posolyte', 'Negolyte'])
            plt.show()
    
    elif potential_measurement_system == 'NI':
        FN_V = 'Potential.csv'
        fp_V = os.path.join(data_dir, FN_V)
        Praw = pd.read_csv(fp_V)
        Praw.set_index('Time (s)', inplace=True)
        Praw = Praw[['V1 (V)', 'V2 (V)']]
        #Reduce the size to n_times time steps
        new_times_s = np.linspace(0, Praw.index[-1], no_times)
        Praw = gen.df_interpolate(Praw, new_times_s)
        #print(Praw.describe())
        # Calibrate
        Praw += refCellCalibration
        Ppos_raw = Praw['V2 (V)']
        Pneg_raw = Praw['V1 (V)']
        if show:
            # Plot
            fig, ax = plt.subplots()
            Ppos_raw.plot(ax=ax)
            Pneg_raw.plot(ax=ax)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Potential (V vs. SHE)')
            plt.legend(['Posolyte', 'Negolyte'])
            plt.show()
    
    return Ppos_raw, Pneg_raw


def restrict_P_to_xlim(df_charge, Ppos_raw, Pneg_raw, xlim = (0, 2800), show=False):

    #Choose range: Make sure that time non-ambiguous
    def restrict_to_xlim(ser, xlim):
        x_idx_min = np.argmin(np.abs(ser.index-xlim[0]))
        x_idx_max = np.argmin(np.abs(ser.index-xlim[1]))
        return ser.iloc[x_idx_min:x_idx_max]

    #Ppos = restrict_to_xlim(Ppos_savgol, xlim)
    Ppos_raw = restrict_to_xlim(Ppos_raw, xlim)
    #Pneg = restrict_to_xlim(Pneg_savgol, xlim)
    Pneg_raw = restrict_to_xlim(Pneg_raw, xlim)
    df_charge = restrict_to_xlim(df_charge, xlim)
    
    if show:
        # Plot
        fig, ax = plt.subplots()
        Ppos_raw.plot(ax=ax)
        Pneg_raw.plot(ax=ax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Potential (V vs. SHE)')
        plt.legend(['Posolyte', 'Negolyte'])
        plt.show()

    return Ppos_raw, Pneg_raw, df_charge


def plot_first_n_seconds(Ppos_raw, Pneg_raw, n=1000):
    # Plot around t=0
    print('Plot around t=0.')
    xlim = (0, n)
    #xlim = (2210, 2870)
    fig, ax = plt.subplots()
    Ppos_raw.plot(ax=ax)
    Pneg_raw.plot(ax=ax)
    ax.set_xlim(xlim)
    idx_xlim = (np.argmin(np.abs(Ppos_raw.index - xlim[0])), np.argmin(np.abs(Ppos_raw.index - xlim[1])))
    ax.set_ylim(Pneg_raw.iloc[idx_xlim[0]:idx_xlim[1], :].min().min(), Ppos_raw.iloc[idx_xlim[0]:idx_xlim[1], :].max().max())
    ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Potential (V vs. Hg/HgSO4)')
    ax.set_ylabel('Potential (V vs. SHE)')
    plt.legend(['Posolyte', 'Negolyte'])
    plt.show()
    

def shift_time(Ppos_raw, Pneg_raw, t_shift=40, n=1000, show=False):
    #Charging switched on after the spike, therefore delete first seconds and shift time
    #Set the potentials equal at new t=0
    #Plot first n seconds
    time_shift = t_shift
    Ppos = Ppos_raw[Ppos_raw.index > time_shift].squeeze()
    Ppos.index -= Ppos.index[0]
    Pneg = Pneg_raw[Pneg_raw.index > time_shift].squeeze()
    Pneg.index -= Pneg.index[0]
    #set start at same value in-between pos and neg
    y_shift = (Ppos.iloc[0] - Pneg.iloc[0])/2 
    Pneg += y_shift
    Ppos -= y_shift
    if show:
        print('Charging switched on after the spike, therefore delete first seconds and shift time.')
        fig, ax = plt.subplots()
        xlim = (0, n)
        ax.set_xlim(xlim)
        idx_xlim = (np.argmin(np.abs(Ppos.index - xlim[0])), np.argmin(np.abs(Ppos.index - xlim[1])))
        ax.set_ylim(Pneg.iloc[idx_xlim[0]:idx_xlim[1]].min(), Ppos.iloc[idx_xlim[0]:idx_xlim[1]].max())
        ax.plot(Ppos.index, Ppos.values, linewidth=5, label='Posolyte')
        ax.plot(Pneg.index, Pneg.values, linewidth=5, label='Negolyte')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Potential (V vs. SHE)')
        plt.legend()
        plt.show()
    return Ppos, Pneg


def plot_Ppos_Pneg(Ppos, Pneg, time_to_reach_SOC0, xlim=None):
    fig, ax = plt.subplots()
    ax.plot(Ppos.index, Ppos.values, linewidth=5, label='Posolyte')
    ax.plot(Pneg.index, Pneg.values, linewidth=5, label='Negolyte')
    #ax.plot(Ppos.index, Ppos.values-Pneg.values, label='Delta')
    ax.vlines(time_to_reach_SOC0, min(min(Ppos), min(Pneg)), max(max(Ppos), max(Pneg)), color='black', label='Time to reach SOC0')
    ax.hlines([-0.25, 1.0], min(Ppos.index), max(Ppos.index))
    ax.set_xlim(xlim)
    #ax.set_ylim(0, 2.5)
    ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Potential (V vs. Hg/HgSO4)')
    ax.set_ylabel('Potential (V vs. SHE)')
    plt.legend()
    plt.show()
    

def det_times_and_charge_SOC0_from_potential(df_charge, Ppos, Pneg, xlim=None, show=False):

    if xlim is None:
        xlim = (0, Ppos.index[-1])
    # Take derivative and calculate the difference in time between the maxima
    dPpos = Ppos.diff()
    dPneg = Pneg.diff()

    if show:
        print('Take derivative and calculate the difference in time between the maxima.')

        fig, ax = plt.subplots()
        ax.plot(dPpos.index, dPpos.values, label='Delta')
        ax.plot(dPneg.index, dPneg.values, label='Delta')
        ax.set_xlim(xlim)
        #ax.set_ylim(0, 2.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Potential (V vs. Hg/HgSO4)')
        plt.legend()
        plt.show()
        
    def time_max_diff(P, max_min, xlim):
        x_idx_min = np.argmin(np.abs(P.index-xlim[0]))
        x_idx_max = np.argmin(np.abs(P.index-xlim[1]))
        dP = P.iloc[x_idx_min:x_idx_max].diff()
        if max_min == 'max':
            return dP[dP==dP.max()].index.to_list()[0]
        elif max_min == 'min':
            return dP[dP==dP.min()].index.to_list()[0]
    
    tV4_pos = time_max_diff(Ppos, 'max', xlim)
    if show:
        print(f'Time when all V3 is oxidized to V4 in positive electrolyte: {tV4_pos:.0f} s')
    tV3_neg = time_max_diff(Pneg, 'min', xlim)
    if show:
        print(f'Time when all V4 is reduced to V3 in negative electrolyte: {tV3_neg:.0f} s')
    t_diff = tV3_neg-tV4_pos
    if show:
        print(f'Time difference: {t_diff:.0f} s')
        
    if show:
        print('This time difference the negative electrolyte has to be charged.')
    charge_tV4_pos = df_charge.iloc[np.argmin(np.abs(df_charge.index-tV4_pos))]
    if show:
        print(f'Charge posolyte until all V3 is oxidized to V4: {charge_tV4_pos:.3f} Ah')
    charge_tV3_neg = df_charge.iloc[np.argmin(np.abs(df_charge.index-tV3_neg))]
    if show:
        print(f'Charge negolyte until all V4 is reduced to V3: {charge_tV3_neg:.3f} Ah')
    delta_charge = charge_tV3_neg - charge_tV4_pos
    if show:
        print(f'Delta charge: {delta_charge:.3f} Ah')
    
    return tV4_pos, tV3_neg, charge_tV4_pos, charge_tV3_neg


def amount_oxalic_acid_necessary_to_rebalance(charge_tV3_neg, charge_tV4_pos, show=False):
    # Amount of oxalic acid necessary to rebalance.
    delta_charge = charge_tV3_neg - charge_tV4_pos
    #Delta_charge in mol
    delta_charge_mol = delta_charge*s_per_h/F #mol
    #Each oxalic acid molecule reduces two V(V) species to V(IV)
    m_oxalicacid = MW_oxalicacid * delta_charge_mol/2
    if show:
        print(f'Amount of oxalic acid (to be added to V(V) posolyte) necessary to rebalance electrolyte: {m_oxalicacid:.3f} g')
    
    return m_oxalicacid


def P_vs_charge(Ppos, Pneg, df_charge, show=False):
    
    def P_charge(P):
        lower_limit = min(P.index.values)
        upper_limit = max(P.index.values)
        new_index_arr = df_charge.index.values
        new_index_arr = new_index_arr[new_index_arr >= lower_limit]
        new_index_arr = new_index_arr[new_index_arr <= upper_limit]
        P_charge = gen.df_interpolate(P, new_index_arr).to_frame()
        P_charge['Charge'] = df_charge.loc[new_index_arr].values
        return P_charge.set_index('Charge')

    #Check that all index values are unique
    if len(Ppos.index) != len(Ppos.index.unique()):
        print('Ppos indices are not unique!')
    elif len(df_charge.index) != len(df_charge.index.unique()):
        print('df_charge indices are not unique!')
    else:
        Ppos_charge = P_charge(Ppos)
        Pneg_charge = P_charge(Pneg)
        E_start_exp = Ppos_charge.iloc[0, 0]
        
        if show:
            fig, ax = plt.subplots()
            Ppos_charge.plot(ax=ax)
            Pneg_charge.plot(ax=ax)
            #ax.set_xlim(0, 0.1)
            #ax.set_ylim(0.4, 0.41)
            ax.set_xlabel('Charge (Ah)')
            ax.set_ylabel('Potential (V vs. SHE)')
            ax.set_title('P vs. charge')
            plt.show()
            print(f'E_start_exp = {E_start_exp:.3f}')
        
    return Ppos_charge, Pneg_charge, E_start_exp


#%% Simulation potentials


def fit_potentials(c_V, c_SO4, Ppos, Pneg, df_charge, Ppos_charge, Pneg_charge, E_start_exp, Vol, use_proton_concentration, T=gen.T_RT, show_details=True):
    #Fit function

    # Proton concentration as a function of ratio between V4 and V3 and between V5 and V4

    # Proton and bisulfate concentration as function of state of charge?
    #conc_x, c_H_plus, pH_c_H_plus, c_HSO4_minus, c_SO4_2minus = rfb.calc_conc_functions(c_V, c_SO4)
    df_conc = calc_df_conc(c_V, c_SO4, show=show_details)
    dfH = df_conc['H+']
    c_Hplus_fit_till3, c_Hplus_fit_from3= fit_proton_concentration(dfH, show= show_details)
    
    # The following functions are used in the Nernst equation
    
    def cH43(R43):
        avg_ox_state = (4*R43+3) / (1+R43)
        return c_Hplus_fit_from3(avg_ox_state)
    
    def cH54(R54):
        avg_ox_state = (5*R54+4) / (1+R54)
        return c_Hplus_fit_from3(avg_ox_state)
    
    #print(cH43(0.1))
    
    # Nernst equations
    # Ror: Ratio concentration oxidized / concentration reduced species
    R32 = lambda E, E0, T: np.exp( (E-E0)/ (R*T/F) )
    #E_Ror = lambda Ror, E0, T: E0 + R*T/F * np.log( Ror )
    def R43(E, E0, T, use_proton_concentration):
        func = lambda R43_, E, E0, T: R43_ - 1/cH43(R43_)**2 * np.exp( (E-E0)/ (R*T/F) )
        R43_initial = np.exp( (E-E0)/ (R*T/F) )
        ratio = fsolve(func, R43_initial, args=(E, E0, T), xtol=1e-4)[0]
        #print(f'E = {E:.3f} V, E0 = {E0:.3f}, R43 = {ratio:.2e}, R43_initial = {R43_initial:.2e}')
        if use_proton_concentration:
            return ratio
        else:
            return R43_initial # Nernst equation without proton concentration
    
    def R54(E, E0, T, use_proton_concentration):
        func = lambda R54_, E, E0, T: R54_ - 1/cH54(R54_)**2 * np.exp( (E-E0)/ (R*T/F) )
        R54_initial = np.exp( (E-E0)/ (R*T/F) )
        ratio =  fsolve(func, R54_initial, args=(E, E0, T), xtol=1e-4)[0]
        #print(f'E = {E:.3f} V, E0 = {E0:.3f}, R54 = {ratio:.2e}, R54_initial = {R54_initial:.2e}')
        if use_proton_concentration:
            return ratio
        else:
            return R54_initial # Nernst equation without proton concentration
    
    # Normalized concentrations, derived from the condition that the sum of all concentrations = 1
    def norm_conc(E, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration):
        maxE_only_23 = E0_V2X3 + (E0_V3X4-E0_V2X3)/2
        maxE_only_34 = E0_V3X4 + (E0_V4X5-E0_V3X4)/2
        if E <= maxE_only_23:
            R32_ = R32(E, E0_V2X3, T)
            cV2 = 1 / ( R32_ + 1 )
            cV3 = (1-cV2)
            return np.array([cV2, cV3, 0.0, 0.0])
        elif E<= maxE_only_34:
            R43_ = R43(E, E0_V3X4, T, use_proton_concentration)
            cV3 = 1 / ( R43_ + 1 )
            cV4 = (1-cV3)
            return np.array([0.0, cV3, cV4, 0.0])
        else:
            R54_ = R54(E, E0_V4X5, T, use_proton_concentration)
            cV4 = 1 / ( R54_ + 1 )
            cV5 = (1-cV4)
            return np.array([0.0, 0.0, cV4, cV5])
    
    # Accumulated charge as a function of E for posolyte and negolyte starting from completely discharged, i.e. all is in V+2 or V+5 state respectively
    # The scalar product is taken because e.g. every V5 had to undergo 3 oxidations... 
    # cVol is the product of concentration and volume in mol/L * L
    Qpos_E = lambda E, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration: F/3600 * cVol * np.dot(norm_conc(E, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration), np.array([0, 1, 2, 3]))
    Qneg_E = lambda E, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration: F/3600 * cVol * np.dot(norm_conc(E, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration), np.array([3, 2, 1, 0]))
    
    def Q_Earr(Earr, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, dot_array, use_proton_concentration):
        Q_list = []
        for E in Earr:
            Q_list.append(F/3600 * cVol * np.dot(norm_conc(E, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration), dot_array))
        return np.asarray(Q_list)
    
    Qpos_Earr = lambda Earr, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration: Q_Earr(Earr, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, np.array([0, 1, 2, 3]), use_proton_concentration)
    Qneg_Earr = lambda Earr, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration: Q_Earr(Earr, cVol, E0_V2X3, E0_V3X4, E0_V4X5, T, np.array([3, 2, 1, 0]), use_proton_concentration)
    
    # Reverse axes and create single function
    def swap_charge_potential(Ppos_charge, Pneg_charge):
        exppos = xyd.xy_data.from_df(Ppos_charge).swap_axes()
        expneg = xyd.xy_data.from_df(Pneg_charge).swap_axes()
        exp = exppos.copy()
        exp.x = np.concatenate((expneg.x[::-1], exppos.x))
        exp.y = np.concatenate((expneg.y[::-1], exppos.y))
        exp.qx = 'Potential'
        exp.ux = 'V vs. SHE'
        exp.monotoneous_ascending()
        exp_equidist = exp.copy()
        exp_equidist.equidist(delta=0.001, kind='linear')
        return exp, exp_equidist
    
    exp, exp_equidist = swap_charge_potential(Ppos_charge, Pneg_charge)
    if show_details:
        both = xyd.mxy_data([exp, exp_equidist])
        both.label(['original', 'equidistance'])
        both.plot()
    exp = exp_equidist
    
    def Qboth_Earr(Earr, c_V, E0_V2X3, E0_V3X4, E0_V4X5):
        E_pos = Earr[Earr >= E_start]
        E_neg = Earr[Earr < E_start]
    
        cVol_neg = c_V * Vol_neg 
        cVol_pos = c_V * Vol_pos
    
        Q_E_pos = Qpos_Earr(E_pos, cVol_pos, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration) - Qpos_E(E_start, cVol_pos, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration)
        Q_E_neg = Qneg_Earr(E_neg, cVol_neg, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration) - Qneg_E(E_start, cVol_neg, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration)
        return np.concatenate((Q_E_neg, Q_E_pos))
        
    def plot_fit_vs_potential(p0, popt):
        #fit0 = xyd.xy_data(exp.x, Qboth_Earr(exp.x, *p0)) 
        fit = xyd.xy_data(exp.x, Qboth_Earr(exp.x, *popt))
        both = xyd.mxy_data([exp, fit])
        both.label(['exp', 'fit'])
        exp.plotstyle = dict(color='tab:blue', linewidth=5, linestyle='--')
        fit.plotstyle = dict(color='tab:orange', linewidth=3)
        fig, ax = plt.subplots()
        both.plot(plotstyle='individual', ax=ax)
        #ax.set_xlim(0.39, 0.410)
        #ax.set_ylim(0.0, 0.2)
        plt.show()

    if show_details:
        print(f'Volume of posolyte and negolyte: {Vol:.3f} L')
    Vol_neg = Vol
    Vol_pos = Vol
    #Vol_neg = 0.090 #L
    #Vol_pos = 0.090 #L
    
    #Initial guesses
    E0_V2X3 = -0.25 # V vs. SHE
    E0_V3X4 = 0.394 # V vs. SHE
    E0_V4X5 = 1.00 # V vs. SHE
    c_V = c_V #mol/L
    if show_details:
        print(f'Initial guess for c_V: {c_V:.2f} M')
    
    E_start = E_start_exp
    if show_details:
        print(f'E_start = {E_start:.3f} V')
    #start = -0.400
    #stop = 1.100
    #stepsize = 0.01
    #n_points = int((stop-start)/stepsize) + 1
    #E_arr = np.linspace(start, stop, num=n_points)
    #dE = (E_arr[-1]- E_arr[0])/(n_points-1)
    
    p0 = [c_V, E0_V2X3, E0_V3X4, E0_V4X5]
    bounds = (
              [c_V*0.8, E0_V2X3-0.2, E0_V3X4-0.2, E0_V4X5-0.2],
              [c_V*1.2, E0_V2X3+0.2, E0_V3X4+0.2, E0_V4X5+0.2]
              )
    
    p = curve_fit(Qboth_Earr, exp.x, exp.y, p0=p0, bounds=bounds)
    popt = p[0]
    if show_details:
        print(popt)
    c_V = popt[0]
    E0_V2X3 = popt[1]
    E0_V3X4 = popt[2]
    E0_V4X5 = popt[3]
    if show_details:
        print(f'c_V = {c_V:.2f} M')
        print(f'E0_V2X3 = {E0_V2X3:.3f} V vs. SHE')
        print(f'E0_V3X4 = {E0_V3X4:.3f} V vs. SHE')
        print(f'E0_V4X5 = {E0_V4X5:.3f} V vs. SHE')
        plot_fit_vs_potential(p0, popt)
    
    def interpolate_df_fit(df_fit):        
        # Swap index (time) with charge, so that new index is charge
        df_charge_new = df_charge.to_frame().reset_index()
        df_charge_new = df_charge_new.set_index('Charge (Ah)').squeeze()
        #Make sure that interpolation is done within the correct limits
        lower_limit = min(df_fit.index)
        upper_limit = max(df_fit.index)
        new_index_arr = df_charge_new.index.values
        new_index_arr = new_index_arr[new_index_arr >= lower_limit]        
        new_index_arr = new_index_arr[new_index_arr <= upper_limit]
        df_fit = gen.df_interpolate(df_fit, new_index_arr)
        df_fit['Time (s)'] = df_charge_new.loc[new_index_arr].values
        df_fit.set_index('Time (s)', inplace=True)
        return df_fit
    
    #fit curve as function of time
    E_arr_pos = np.linspace(E_start, Ppos.values[-1], 1000)
    Qpos_E_arr = Qpos_Earr(E_arr_pos, Vol*c_V, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration) - Qpos_E(E_start, Vol*c_V, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration)
    df_fit_pos = pd.DataFrame({'Potential (V vs. SHE)': E_arr_pos,'Charge (Ah)': Qpos_E_arr}).set_index('Charge (Ah)')
    df_fit_pos = interpolate_df_fit(df_fit_pos)
    #df_fit_pos = gen.df_interpolate(df_fit_pos, df_charge.values) #xxx here df_charge should be extrapolated
    #df_fit_pos['Time (s)'] = df_charge.index
    #df_fit_pos.set_index('Time (s)', inplace=True)
    
    E_arr_neg = np.linspace(E_start, Pneg.values[-1], 1000)
    Qneg_E_arr = Qneg_Earr(E_arr_neg, Vol*c_V, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration) - Qneg_E(E_start, Vol*c_V, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration)
    df_fit_neg = pd.DataFrame({'Potential (V vs. SHE)': E_arr_neg,'Charge (Ah)': Qneg_E_arr}).set_index('Charge (Ah)')
    df_fit_neg = interpolate_df_fit(df_fit_neg)
    #df_fit_neg = gen.df_interpolate(df_fit_neg, df_charge.values)
    #df_fit_neg['Time (s)'] = df_charge.index
    #df_fit_neg.set_index('Time (s)', inplace=True)

    # Plot potential vs. time and concentrations
    conc_V2_pos_list = []
    conc_V3_pos_list = []
    conc_V4_pos_list = []
    conc_V5_pos_list = []
    
    conc_V2_neg_list = []
    conc_V3_neg_list = []
    conc_V4_neg_list = []
    conc_V5_neg_list = []
    
    def conc_append(P, time, c_V, conc_V2_list, conc_V3_list, conc_V4_list, conc_V5_list):
        E = P.loc[time]
        conc_arr = c_V * norm_conc(E, E0_V2X3, E0_V3X4, E0_V4X5, T, use_proton_concentration)
        #conc_arr = all_conc(E, c_V, E0_V2X3, E0_V3X4, E0_V4X5, T)
        conc_V2_list.append(conc_arr[0])
        conc_V3_list.append(conc_arr[1])
        conc_V4_list.append(conc_arr[2])
        conc_V5_list.append(conc_arr[3])
    
    for time in df_fit_pos.index.values:
        conc_append(df_fit_pos['Potential (V vs. SHE)'], time, c_V, conc_V2_pos_list, conc_V3_pos_list, conc_V4_pos_list, conc_V5_pos_list)

    for time in df_fit_neg.index.values:
        conc_append(df_fit_neg['Potential (V vs. SHE)'], time, c_V, conc_V2_neg_list, conc_V3_neg_list, conc_V4_neg_list, conc_V5_neg_list)
    
    #conc_V2_pos = xyd.xy_data(df_fit_pos.index.values, np.asarray(conc_V2_pos_list), name='V2 (posolyte)', check_data=False)
    conc_V3_pos = xyd.xy_data(df_fit_pos.index.values, np.asarray(conc_V3_pos_list), name='V3 (posolyte)', check_data=False)
    conc_V4_pos = xyd.xy_data(df_fit_pos.index.values, np.asarray(conc_V4_pos_list), name='V4 (posolyte)', check_data=False)
    conc_V5_pos = xyd.xy_data(df_fit_pos.index.values, np.asarray(conc_V5_pos_list), name='V5 (posolyte)', check_data=False)
    
    conc_V2_neg = xyd.xy_data(df_fit_neg.index.values, np.asarray(conc_V2_neg_list), name='V2 (negolyte)', check_data=False)
    conc_V3_neg = xyd.xy_data(df_fit_neg.index.values, np.asarray(conc_V3_neg_list), name='V3 (negolyte)', check_data=False)
    conc_V4_neg = xyd.xy_data(df_fit_neg.index.values, np.asarray(conc_V4_neg_list), name='V4 (negolyte)', check_data=False)
    #conc_V5_neg = xyd.xy_data(df_fit_neg.index.values, np.asarray(conc_V5_neg_list), name='V5 (negolyte)', check_data=False)

    return c_V, (E0_V2X3, E0_V3X4, E0_V4X5), (df_fit_pos, df_fit_neg), (conc_V3_pos, conc_V4_pos, conc_V5_pos), (conc_V4_neg, conc_V3_neg, conc_V2_neg)


def calculate_fit(c_V_start, c_SO4_start, Ppos, Pneg, Ppos_charge, df_charge, Pneg_charge, E_start_exp, Vol, use_proton_concentration=True, T=gen.T_RT, show_details=False, do_recalculation=True):
    c_V, E0s, df_fits, conc_pos, conc_neg = fit_potentials(c_V_start, c_SO4_start, Ppos, Pneg, df_charge, Ppos_charge, Pneg_charge, E_start_exp, Vol, use_proton_concentration, T=T, show_details=show_details)
    if show_details:
        print(f'First calculation finished (c_V = {c_V:.2f} M).')
        if not do_recalculation:
            print('No second calculation done. Set kwarg "do_recalculation" to True if you want to recalculate automatically!')
    if do_recalculation:
        c_SO4 = c_V/c_V_start * c_SO4_start
        c_V, E0s, df_fits, conc_pos, conc_neg = fit_potentials(c_V, c_SO4, Ppos, Pneg, df_charge, Ppos_charge, Pneg_charge, E_start_exp, Vol, use_proton_concentration, T=T, show_details=show_details)
        if show_details:
            print(f'Second calculation finished (c_V = {c_V:.2f} M).')
    return {'c_V': c_V, 'E0s': E0s, 'df_fits': df_fits, 'conc_pos': conc_pos, 'conc_neg': conc_neg}
    

def plot_complete_dataset(fit_data, Ppos, Pneg, time_to_reach_SOC0, time_to_reach_SOC1, first_charging_step_end_s):
    c_V = fit_data['c_V']
    (E0_V2X3, E0_V3X4, E0_V4X5) = fit_data['E0s']
    (df_fit_pos, df_fit_neg) = fit_data['df_fits']
    (conc_V3_pos, conc_V4_pos, conc_V5_pos) = fit_data['conc_pos']
    (conc_V4_neg, conc_V3_neg, conc_V2_neg) = fit_data['conc_neg']
    xlim = (min(Ppos.index), max(Ppos.index))
    #xlim = (0, 8000)
    #xlim = (2210, 2870)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.plot(Ppos.index, Ppos.values, linewidth=5, color='tab:blue', label='Posolyte')
    ax.plot(Pneg.index, Pneg.values, linewidth=5, color='tab:orange', label='Negolyte')
    ax.plot(df_fit_pos.index, df_fit_pos.values, color='black', label='fit')
    ax.plot(df_fit_neg.index, df_fit_neg.values, color='black', label='')
    
    ax.vlines(time_to_reach_SOC0,min(min(Ppos), min(Pneg)), max(max(Ppos), max(Pneg)), color='black', linestyle='--', linewidth=0.5, label='Time to reach SOC0')
    ax.hlines([E0_V2X3, E0_V3X4, E0_V4X5], min(Ppos.index), max(Ppos.index), linewidth=0.5, linestyle='--')
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Potential (V vs. Hg/HgSO4)')
    ax.set_ylabel('Potential (V vs. SHE)')
    ax.set_title('Potential measurements')
    ax.legend()
    
    ax = axes[1]
    conc_V3_pos.plotstyle = {'linewidth': 3, 'linestyle': '--', 'color': 'tab:blue'}
    conc_V4_pos.plotstyle = {'linewidth': 5, 'linestyle': '-.', 'color': 'tab:blue'}
    conc_V5_pos.plotstyle = {'linewidth': 5, 'linestyle': '-', 'color': 'tab:blue'}
    
    conc_V4_neg.plotstyle = {'linewidth': 3, 'linestyle': '--', 'color': 'tab:orange'}
    conc_V3_neg.plotstyle = {'linewidth': 5, 'linestyle': '-.', 'color': 'tab:orange'}
    conc_V2_neg.plotstyle = {'linewidth': 5, 'linestyle': '-', 'color': 'tab:orange'}
    
    all_cV = xyd.mxy_data([conc_V3_pos, conc_V4_pos, conc_V5_pos, conc_V4_neg, conc_V3_neg, conc_V2_neg])
    all_cV.names_to_label()
    all_cV.plot(ax=ax, bottom=0, top=c_V, plotstyle='individual')
    
    ax.set_xlim(xlim)
    ax.set_ylim(0, c_V*1.1)
    ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Potential (V vs. Hg/HgSO4)')
    ax.set_ylabel('Concentration (M)')
    ax.set_title('Concentrations of different oxidation states')
    ax.legend()
    
    ax = axes[2]
    SOC_pos = conc_V5_pos/c_V*100
    SOC_pos.plotstyle = {'linewidth': 5, 'color': 'tab:blue'}
    SOC_pos.plot(ax=ax)
    
    SOC_neg = conc_V2_neg/c_V*100
    SOC_neg.plotstyle = {'linewidth': 5, 'color': 'tab:orange'}
    SOC_neg.plot(ax=ax)
    
    # Calculate when SOC90% is reached
    x_idx = np.argmin(np.abs(SOC_pos.y-90))
    t_SOC90 = SOC_pos.x[x_idx]
    print(f'Posolyte: SOC 90% reached after {t_SOC90:.0f} s')
    print(f'Negolyte: SOC at this time is {SOC_neg.y[x_idx]:.0f} %')
    ax.vlines(t_SOC90, 0, 90, color='black', linestyle='--', linewidth=0.5, label=f'Posolyte SOC 90%: {t_SOC90:.0f} s')
    ax.vlines(first_charging_step_end_s, 0, 100, color='black', linestyle='-', linewidth=0.5, label=f'Posolyte SOC 90%: {t_SOC90:.0f} s')
    ax.hlines(90, 0, t_SOC90, linestyle='--', linewidth=0.5)
    ax.hlines(SOC_neg.y[x_idx], 0, t_SOC90, linestyle='--', linewidth=0.5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SOC (%)')
    ax.set_title('State-of-charge (SOC)')
    ax.legend(['Posolyte', 'Negolyte'])
    
    plt.show()
    
    print(f'c_V = {c_V:.2f} M')
    print(f'E0_V2X3 = {E0_V2X3:.3f} V vs. SHE')
    print(f'E0_V3X4 = {E0_V3X4:.3f} V vs. SHE')
    print(f'E0_V4X5 = {E0_V4X5:.3f} V vs. SHE')
    

def fitted_times(fit_data, show=False):
    
    #**Time when posolyte consists of pure V(IV) and negolyte consists of pure V(III)**
    c_V = fit_data['c_V']
    (E0_V2X3, E0_V3X4, E0_V4X5) = fit_data['E0s']
    (df_fit_pos, df_fit_neg) = fit_data['df_fits']
    (conc_V3_pos, conc_V4_pos, conc_V5_pos) = fit_data['conc_pos']
    (conc_V4_neg, conc_V3_neg, conc_V2_neg) = fit_data['conc_neg']
    idx_V4_pos = np.argmin(np.abs(conc_V4_pos.y - c_V))
    t_V4_pos = conc_V4_pos.x[idx_V4_pos]
    idx_V3_neg = np.argmin(np.abs(conc_V3_neg.y - c_V))
    t_V3_neg = conc_V3_neg.x[idx_V3_neg]
    t_diff = t_V3_neg-t_V4_pos

    if show:
        print(f'Time when all V3 is oxidized to V4 in positive electrolyte: {t_V4_pos:.0f} s')
        print(f'Time when all V4 is reduced to V3 in negative electrolyte: {t_V3_neg:.0f} s')
        print(f'Time difference: {t_diff:.0f} s')

    return t_V4_pos, t_V3_neg, t_diff
    
    
def fitted_average_ox_state(fit_data, show=False):

    c_V = fit_data['c_V']
    (E0_V2X3, E0_V3X4, E0_V4X5) = fit_data['E0s']
    (df_fit_pos, df_fit_neg) = fit_data['df_fits']
    (conc_V3_pos, conc_V4_pos, conc_V5_pos) = fit_data['conc_pos']
    (conc_V4_neg, conc_V3_neg, conc_V2_neg) = fit_data['conc_neg']

    #**Average oxidation state at beginning of charging process**
    #Difference in concentration of V4 and V3 at t=0
    delta_conc_V4_V3 = conc_V4_pos.y[0] - conc_V3_pos.y[0]
    if show:
        print(f'Difference in concentration of V4 and V3 at t=0: {delta_conc_V4_V3:.3f} mol/L')
    av_ox_state = (4*conc_V4_pos.y[0]+3*conc_V3_pos.y[0])/c_V
    if show:
        print(f'Average oxidation state at beginning: {av_ox_state:.3f}')

    return av_ox_state

def fitted_m_oxalic_acid(fit_data, Vol, show=False):

    (E0_V2X3, E0_V3X4, E0_V4X5) = fit_data['E0s']
    (df_fit_pos, df_fit_neg) = fit_data['df_fits']
    (conc_V3_pos, conc_V4_pos, conc_V5_pos) = fit_data['conc_pos']
    (conc_V4_neg, conc_V3_neg, conc_V2_neg) = fit_data['conc_neg']

    delta_conc_V4_V3 = conc_V4_pos.y[0] - conc_V3_pos.y[0]
    #**Amount of oxalic acid necessary to rebalance electrolyte**
    #Each oxalic acid molecule reduces two V(V) species to V(IV)
    #Factor 1/2 because only half of the concentration difference has to be reduced
    #Factor 2*Vol because Vol is the volume of posolyte (negolyte) only
    m_oxalicacid_fit = 1/2*delta_conc_V4_V3*2*Vol*MW_oxalicacid/2
    if show:
        print(f'Amount of oxalic acid (to be added to V(V) posolyte) necessary to rebalance electrolyte: {m_oxalicacid_fit:.3f} g')

    return m_oxalicacid_fit


#%% UV-VIS

def load_ref_spec(ref_spec_dir):
    
    def df_to_series(df):
        return df[df.columns[0]]
    
    ref_spec_V2 = df_to_series(pd.read_csv(os.path.join(ref_spec_dir, 'spec_V2.csv'), index_col=0))
    ref_spec_V3 = df_to_series(pd.read_csv(os.path.join(ref_spec_dir, 'spec_V3.csv'), index_col=0))
    ref_spec_V4 = df_to_series(pd.read_csv(os.path.join(ref_spec_dir, 'spec_V4.csv'), index_col=0))
    ref_spec_V5 = df_to_series(pd.read_csv(os.path.join(ref_spec_dir, 'spec_V5.csv'), index_col=0))

    return [ref_spec_V2, ref_spec_V3, ref_spec_V4, ref_spec_V5]


def plot_ref_spectra(ref_spec_V2, ref_spec_V3, ref_spec_V4, ref_spec_V5, xlim= (250, 1050), ylim= None):
    
    fig, ax = plt.subplots()
    
    def plot_df(ax, df, color, label):
        ax.plot(df.index, df.values, color=color, label=label)
    
    plot_df(ax, ref_spec_V2, color='violet', label='V(II)')
    plot_df(ax, ref_spec_V3, color='green', label='V(III)')
    plot_df(ax, ref_spec_V4, color='blue', label='V(IV)')
    plot_df(ax, ref_spec_V5, color='orange', label='V(V)')
        
    ax.hlines(0, *xlim, colors='black')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Molar extinction coefficient (1/[M cm])')
    ax.legend()
    ax.set_title('Reference spectra')
    
    plt.show()


def load_absorbance_data(fp_cuv_pos, fp_cuv_neg, xlim = (250, 1050)):
    
    def read_and_int_col(fp, xlim):
        df = pd.read_csv(fp)
        df.set_index('Wavelength (nm)', inplace=True)
        if 'times1000' in fp:
            df /= 1000
        df.columns = np.int16(np.float64(df.columns.values))
        return df.loc[xlim[0]:xlim[1]]
        
    df_cuv_pos = read_and_int_col(fp_cuv_pos, xlim)
    df_cuv_neg = read_and_int_col(fp_cuv_neg, xlim)

    return df_cuv_pos, df_cuv_neg


def plot_UVVIS_data(df_cuv_pos, df_cuv_neg, ylabel='Absorbance', time_array_pos= None, time_array_neg= None, no_curves=20, xlim_cuv_pos=(380, 400), xlim_cuv_neg=(280, 1100), ylim_cuv_pos=(0, 2), ylim_cuv_neg=(-0.5, 2)):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax1 = ax[0]
    ax2 = ax[1]
        
    def plot_df(df, ax, xlim, ylim, time_array):
        
        if time_array is None:
            time_array = np.array([int(round(df.columns[-1]*i/no_curves)) for i in range(no_curves)])
        nm = df.index.values
        #cmap = mpl.colormaps['Reds']
        cmap = mpl.colormaps['viridis']
        norm = mpl.colors.Normalize(vmin= 0, vmax= len(time_array))
        ncmap = lambda num: cmap(norm(num))

        for i, time in enumerate(time_array):
            #df[col].plot()
            idx = np.argmin(np.abs(df.columns-time))
            ax.plot(nm, df.iloc[:,idx], label=time, color=ncmap(i))
        ax.hlines(0, *xlim, colors='black')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(ylabel)
        ax.legend()
    
    plot_df(df_cuv_pos, ax1, xlim_cuv_pos, ylim_cuv_pos, time_array_pos)
    plot_df(df_cuv_neg, ax2, xlim_cuv_neg, ylim_cuv_neg, time_array_neg)
    ax1.set_title('Positive electrode')
    ax2.set_title('Negative electrode')
    plt.show()


def UVVIS_get_offset_and_scaling_factor_cuv_pos(first_spectra=None, show=False):

    if first_spectra is not None:
        #This works only if negolyte and posolyte are the same at t=0
        def fit_scalingfactor_and_offset(target_spectrum, spec, initial_guess = [-0.04, 12]):
        
            def fit_model(wavelengths, offset, scaling_factor):
                #offset, scaling_factor = p[0], p[1]
                return (spec + offset) * scaling_factor
        
            # Extract wavelength and absorbance data from target spectrum
            wavelengths = target_spectrum.index
        
            # Initial guess for coefficients and offsets [c1, c2, offset1, offset2]
            #bounds = ([0, 0], [np.inf, np.inf])
        
            # Perform curve fitting
            #popt, _ = curve_fit(fit_model, wavelengths, absorbance, p0=initial_guess, bounds=bounds)
            popt, _ = curve_fit(fit_model, wavelengths, target_spectrum, p0=initial_guess)
        
            # Extract fitted coefficients and offsets
            offset = popt[0]
            scaling_factor = popt[1]
        
            # Compute the fitted spectrum
            fitted_spectrum = fit_model(wavelengths, offset, scaling_factor)
            #df_fit = pd.Series(fitted_spectrum, index=target_spectrum.index)
        
            # Return the fit results and coefficients
            return fitted_spectrum, offset, scaling_factor
        
        cuv_pos_t0 = first_spectra[0]
        cuv_neg_t0 = first_spectra[1]
        scaling_factor_cuv_pos= 12
        offset_cuv_pos = -0.04
        cuv_pos_t0_scaled, offset_cuv_pos, scaling_factor_cuv_pos = fit_scalingfactor_and_offset(cuv_neg_t0, cuv_pos_t0, initial_guess = [scaling_factor_cuv_pos, offset_cuv_pos])
        if show:
            print(f'Offset = {offset_cuv_pos}, Scaling factor = {scaling_factor_cuv_pos}')
            fig, ax = plt.subplots()
            xlim=(280, 1100)
            #ylim=(0, )
            cuv_pos_t0_scaled.plot(ax=ax)
            cuv_neg_t0.plot(ax=ax)
            ax.hlines(0, *xlim, colors='black')
            ax.set_xlim(xlim)
            #ax.set_ylim(ylim)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Absorbance')
            ax.legend(['Cuvette 1', 'Cuvette 2'])
            ax.set_title('Spectra at t=0')
            plt.show()
    else:
        #Otherwise use fixed values
        offset_cuv_pos = -0.02532#-0.03971111423464058 
        scaling_factor_cuv_pos = 12.300663352444664
    return offset_cuv_pos, scaling_factor_cuv_pos


def UVVIS_to_molar_extinction_coefficient(df_cuv_pos_raw, df_cuv_neg_raw, conc, pathlength_cuv_neg, scaling_factor_cuv_pos, offset_cuv_pos):

    df_cuv_pos = df_cuv_pos_raw.copy()
    df_cuv_neg = df_cuv_neg_raw.copy()
    df_cuv_neg /= (conc*pathlength_cuv_neg)
    pathlength_cuv_pos = pathlength_cuv_neg/scaling_factor_cuv_pos #cm (Cuvette 1 thickness)
    #pathlength_V4V5_cuv_pos = pathlength_cuv_pos
    df_cuv_pos += offset_cuv_pos
    df_cuv_pos /= conc*pathlength_cuv_pos

    return df_cuv_pos, df_cuv_neg

# Fitting and plot functions

def UVVIS_new_range(df1, df2, df3, xlim):
    von = xlim[0]
    bis = xlim[1]
    return df1.loc[von:bis], df2.loc[von:bis], df3.loc[von:bis]


def UVVIS_plot_fitted_single(target_spectrum, df_fit, popt, spec_Vr, spec_Vo, xlim, ylim):
    fig, ax = plt.subplots(figsize = (6,4))

    def plot_df(ax, df, linewidth, color, label):
        ax.plot(df.index, df.values, linewidth=linewidth, color=color, label=label)
    
    plot_df(ax, spec_Vr, linewidth=0.1, color='violet', label='Vx')
    plot_df(ax, spec_Vo, linewidth=0.1, color='green', label='Vy')
    plot_df(ax, target_spectrum, linewidth=1, color='black', label='target spectrum')
    plot_df(ax, df_fit, linewidth=1, color='red', label=f'fitted (ar = {popt[0]:.2f}, ao = {popt[1]:.2f})')
    
    #ax.plot(sp.index, sp.values, label='Original Spectrum (sp)')
    #ax.plot(fitted_spectrum.index, fitted_spectrum.values, label='Fitted Spectrum', linestyle='--')
    
    ax.hlines(0, *xlim, colors='black')
    #ax.vlines([280, 300, 360, 1000], *ylim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Molar extinction coefficient (1/[M cm])')
    ax.legend()
    ax.set_title('Spectra')
    
    plt.show()

def UVVIS_plot_fitted_multiple(time_list, target_spectrum_list, df_fit_list, ar_cuv_list, ao_cuv_list, xlim, ylim, title='Spectra'):
    fig, axes = plt.subplots(1, 2, figsize = (12,4))

    ax = axes[0]
    #cmap = mpl.colormaps['Reds']
    cmap = mpl.colormaps['viridis']
    norm = mpl.colors.Normalize(vmin= 0, vmax= len(target_spectrum_list))
    ncmap = lambda num: cmap(norm(num))

    def plot_df(ax, df, linewidth, linestyle, color, label=None):
        ax.plot(df.index, df.values, linewidth=linewidth, linestyle=linestyle, color=color, label=label)
    
    for i, (time, target_spectrum, df_fit) in enumerate(zip(time_list, target_spectrum_list, df_fit_list)):        
        color = ncmap(i)
        plot_df(ax, target_spectrum, linewidth=0.1, linestyle='-', color='black')
        plot_df(ax, df_fit, linewidth=1, linestyle='-', color=color, label=time)
    
    #ax.plot(sp.index, sp.values, label='Original Spectrum (sp)')
    #ax.plot(fitted_spectrum.index, fitted_spectrum.values, label='Fitted Spectrum', linestyle='--')
    
    ax.hlines(0, *xlim, colors='black')
    #ax.vlines([280, 300, 360, 1000], *ylim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Molar extinction coefficient (1/[M cm])')
    ax.legend()
    ax.set_title(title)

    ax = axes[1]
    ar_cuv_arr = np.array(ar_cuv_list) #* 1.6
    ao_cuv_arr = np.array(ao_cuv_list)
    ax.plot(time_list, ar_cuv_arr, marker='x', label='reduced species')
    ax.plot(time_list, ao_cuv_arr, marker='x', label='oxidized species')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration (mol/L)')
    ax.set_title(title)
    ax.legend()
    
    plt.show()
    

def UVVIS_fit_spectrum_old(target_spectrum, spec_Vr, spec_Vo):

    def fit_model(wavelengths, ar, ao):
        #offset1 = 0
        #offset2 = 0
        #return (ar * spec_Vr.values + offset1 +
        #        ao * spec_Vo.values + offset2)
        return (ar * spec_Vr.values + ao * spec_Vo.values)

    # Extract wavelength and absorbance data from target spectrum
    wavelengths = target_spectrum.index
    absorbance = target_spectrum.values

    # Initial guess for coefficients and offsets [c1, c2, offset1, offset2]
    initial_guess = [0.5, 0.5]
    bounds = ([0, 0], [np.inf, np.inf])

    # Perform curve fitting
    popt, _ = curve_fit(fit_model, wavelengths, absorbance, p0=initial_guess, bounds=bounds)

    # Extract fitted coefficients and offsets
    ar = popt[0]
    ao = popt[1]

    # Compute the fitted spectrum
    fitted_spectrum = fit_model(wavelengths, ar, ao)
    df_fit = pd.Series(fitted_spectrum, index=target_spectrum.index)

    # Return the fit results and coefficients
    return df_fit, ar, ao

def UVVIS_fit_spectrum(target_spectrum, spec_Vr, spec_Vo):

    def fit_model(wavelengths, ar, ao, offset_r, offset_o):
        #offset1 = 0
        #offset2 = 0
        return (ar * (spec_Vr.values + offset_r) + ao * (spec_Vo.values + offset_o))
        #return (ar * spec_Vr.values + ao * spec_Vo.values)

    # Extract wavelength and absorbance data from target spectrum
    wavelengths = target_spectrum.index
    absorbance = target_spectrum.values

    # Initial guess for coefficients and offsets [c1, c2, offset1, offset2]
    initial_guess = [0.5, 0.5, 0, 0]
    bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    # Perform curve fitting
    popt, _ = curve_fit(fit_model, wavelengths, absorbance, p0=initial_guess, bounds=bounds)

    # Extract fitted coefficients and offsets
    ar = popt[0]
    ao = popt[1]
    offset_r = popt[2]
    offset_o = popt[3]

    # Compute the fitted spectrum
    fitted_spectrum = fit_model(wavelengths, ar, ao, offset_r, offset_o)
    df_fit = pd.Series(fitted_spectrum, index=target_spectrum.index)

    # Return the fit results and coefficients
    return df_fit, ar, ao


def UVVIS_calculate_fit_for_all(df, time_switch_fitspectra, fit_spectra, time_from, time_to, xlim_fit):
    ar_list = []
    ao_list = []
    target_list = []
    fit_list = []
    fit_spectrum_Vr_list = []
    fit_spectrum_Vo_list = []
    idx_time_stop = None
    time_list = []
    x_von = xlim_fit[0]
    x_bis = xlim_fit[1]
    if time_to == -1:
        time_to = df.columns.values[-1]
    for idx, time in enumerate(df.columns):
        if time >= time_from and time <= time_to:

            if time < time_switch_fitspectra:
                fit_spectrum_Vr = fit_spectra[0].loc[x_von:x_bis]
                fit_spectrum_Vo = fit_spectra[1].loc[x_von:x_bis]
            else:
                fit_spectrum_Vr = fit_spectra[2].loc[x_von:x_bis]
                fit_spectrum_Vo = fit_spectra[3].loc[x_von:x_bis]
            fit_spectrum_Vr_list.append(fit_spectrum_Vr.values)
            fit_spectrum_Vo_list.append(fit_spectrum_Vo.values)

            target_spectrum = df.loc[x_von:x_bis, time]
            df_fit, ar, ao = UVVIS_fit_spectrum(target_spectrum, fit_spectrum_Vr, fit_spectrum_Vo)
            ar_list.append(ar)
            ao_list.append(ao)
            target_list.append(target_spectrum.values)
            fit_list.append(df_fit.values)
            time_list.append(time)
        elif idx_time_stop is None:
            idx_time_stop = idx
    
    wavelengths = df.loc[x_von:x_bis].index
    df_target = pd.DataFrame(np.array(target_list).transpose(), columns=time_list, index=wavelengths)
    df_fit = pd.DataFrame(np.array(fit_list).transpose(), columns=time_list, index=wavelengths)
    df_fit_spectra_Vr = pd.DataFrame(np.array(fit_spectrum_Vr_list).transpose(), columns=time_list, index=wavelengths)
    df_fit_spectra_Vo = pd.DataFrame(np.array(fit_spectrum_Vo_list).transpose(), columns=time_list, index=wavelengths)
    ar_array = np.asarray(ar_list)
    ao_array = np.asarray(ao_list)
    data = {'Measurement': df_target, 'Fit': df_fit, 'ar': ar_array, 'ao': ao_array, 'Time (s)': time_list, 'Spectra_Vr': df_fit_spectra_Vr, 'Spectra_Vo': df_fit_spectra_Vo}
    return data


def UVVIS_reduce_number_of_spectra(df_cuv_pos, df_cuv_neg, one_out_of):
    df_cuv_pos_reduced = df_cuv_pos.iloc[:, ::one_out_of]
    df_cuv_neg_reduced = df_cuv_neg.iloc[:, ::one_out_of]
    return df_cuv_pos_reduced, df_cuv_neg_reduced


def UVVIS_fit(df_cuv_pos, df_cuv_neg, ref_spec_V2, ref_spec_V3, ref_spec_V4, ref_spec_V5,
              one_out_of=1, time_cuv_pos_limits=(0,-1), time_cuv_neg_limits=(0,-1), xlim_fit=(380, 1000),
              tV4_pos=None, tV3_neg=None):

    if one_out_of > 1:
        df_cuv_pos_reduced, df_cuv_neg_reduced = UVVIS_reduce_number_of_spectra(df_cuv_pos, df_cuv_neg, one_out_of)
    else:
        df_cuv_pos_reduced = df_cuv_pos
        df_cuv_neg_reduced = df_cuv_neg
    
    df = df_cuv_pos_reduced
    time_from = time_cuv_pos_limits[0]
    time_to = time_cuv_pos_limits[1]
    xlim_fit = xlim_fit
    if tV4_pos is not None:
        time_switch_fitspectra = tV4_pos
    else:
        time_switch_fitspectra = 0 
    fit_spectra = [ref_spec_V3, ref_spec_V4, ref_spec_V4, ref_spec_V5]
    data_cuv_pos = UVVIS_calculate_fit_for_all(df, time_switch_fitspectra, fit_spectra, time_from, time_to, xlim_fit)
    
    df = df_cuv_neg_reduced
    time_from = time_cuv_neg_limits[0]
    time_to = time_cuv_neg_limits[1]
    xlim_fit = xlim_fit
    if tV3_neg is not None:
        time_switch_fitspectra = tV3_neg
    else:
        time_switch_fitspectra = 0
    fit_spectra = [ref_spec_V3, ref_spec_V4, ref_spec_V2, ref_spec_V3]
    data_cuv_neg = UVVIS_calculate_fit_for_all(df, time_switch_fitspectra, fit_spectra, time_from, time_to, xlim_fit)

    return data_cuv_pos, data_cuv_neg

def UVVIS_reduce_data(data_cuv, one_out_of):
    (df_target, df_fit, ar_array, ao_array, time_list, df_fit_spectra_Vr, df_fit_spectra_Vo) = tuple(data_cuv.values())

    df_target = df_target.iloc[:, ::one_out_of]
    df_fit = df_fit.iloc[:, ::one_out_of]
    ar_array = ar_array[::one_out_of]
    ao_array = ao_array[::one_out_of]
    time_list = time_list[::one_out_of]
    df_fit_spectra_Vr = df_fit_spectra_Vr.iloc[:, ::one_out_of]
    df_fit_spectra_Vo = df_fit_spectra_Vo.iloc[:, ::one_out_of]
    return {'Measurement': df_target, 'Fit': df_fit, 'ar': ar_array, 'ao': ao_array, 'Time (s)': time_list, 'Spectra_Vr': df_fit_spectra_Vr, 'Spectra_Vo': df_fit_spectra_Vo}

    

def animate_UVVIS(data_cuv_pos, data_cuv_neg, conc, 
                  ylim_spec_cuv_pos=(0, 80), ylim_spec_cuv_neg=(0, 80),  
                  ylim_conc_cuv_pos=None, ylim_conc_cuv_neg=None, figsize=(12, 8), save_FN=None):

    if ylim_conc_cuv_pos is None:
        ylim_conc_cuv_pos=(0, conc*1.1)
    if ylim_conc_cuv_neg is None:
        ylim_conc_cuv_neg=(0, conc*1.1)
    mpl.rcParams['animation.embed_limit'] = 2**30
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    plt.ioff()

    def initialize_fig_spectra(ax, data_cuv, ylim, title):
        (df_target, df_fit, ar_array, ao_array, time_list, df_fit_spectra_Vr, df_fit_spectra_Vo) = tuple(data_cuv.values())
        # Create figure and axis    
        line1, = ax.plot([], [], label='Vr', linewidth=0.5, c='tab:cyan')
        line2, = ax.plot([], [], label='Vo', linewidth=0.5, c='tab:pink')
        line3, = ax.plot([], [], label='meas', linewidth=0.5, c='tab:green')
        line4, = ax.plot([], [], label='fit', c='tab:red')
        time_label = ax.text(0.05, 0.90, '', transform=ax.transAxes) # initialize the time label for the graph
        ax.set_xlim((380,1000))
        ax.set_ylim(ylim)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Molar extinction coefficient (1/[M cm])')
        ax.set_title(title)
        ax.legend()
        # Initialization function
        return line1, line2, line3, line4, time_label

    def initialize_fig_conc(ax, data_cuv, ylim, title):
        (df_target, df_fit, ar_array, ao_array, time_list, df_fit_spectra_Vr, df_fit_spectra_Vo) = tuple(data_cuv.values())
        # Create figure and axis    
        line1, = ax.plot([], [], label='cr', linewidth=1.0, c='tab:green')
        line2, = ax.plot([], [], label='co', linewidth=1.0, c='tab:red')
        SOC_label = ax.text(0.05, 0.90, '', transform=ax.transAxes) # initialize the time label for the graph
        ax.set_xlim((time_list[0], time_list[-1]))
        ax.set_ylim(ylim)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration (M)')
        ax.set_title(title)
        ax.legend()
        # Initialization function
        return line1, line2, SOC_label

    ax00 = axes[0][0]
    ax01 = axes[0][1]
    ax10 = axes[1][0]
    ax11 = axes[1][1]
    line00_1, line00_2, line00_3, line00_4, time_label00 = initialize_fig_spectra(ax00, data_cuv_neg, ylim_spec_cuv_neg, 'Cuvette 2')
    line01_1, line01_2, line01_3, line01_4, time_label01 = initialize_fig_spectra(ax01, data_cuv_pos, ylim_spec_cuv_pos, 'Cuvette 1')
    line10_1, line10_2, SOC_label10 = initialize_fig_conc(ax10, data_cuv_neg, ylim_conc_cuv_neg, 'Cuvette 2')
    line11_1, line11_2, SOC_label11 = initialize_fig_conc(ax11, data_cuv_pos, ylim_conc_cuv_pos, 'Cuvette 1')
    plt.tight_layout()

    def init():
        
        time_label00.set_text('')
        time_label01.set_text('')
        line00_1.set_data([], [])
        line00_2.set_data([], [])
        line00_3.set_data([], [])
        line00_4.set_data([], [])
        line01_1.set_data([], [])
        line01_2.set_data([], [])
        line01_3.set_data([], [])
        line01_4.set_data([], [])
    
        SOC_label10.set_text('')
        SOC_label11.set_text('')
        line10_1.set_data([], [])
        line10_2.set_data([], [])
        line11_1.set_data([], [])
        line11_2.set_data([], [])
        return [line00_1, line00_2, line00_3, line00_4, line01_1, line01_2, line01_3, line01_4,line10_1, line10_2, line11_1, line11_2],

    # Animation function
    def animate(i):
        def set_data_spectra(line1, line2, line3, line4, time_label, data_cuv):
            (df_target, df_fit, ar_array, ao_array, time_list, df_fit_spectra_Vr, df_fit_spectra_Vo) = tuple(data_cuv.values())
            wavelengths = df_target.index.values
            idx = i
            if idx >= len(time_list)-1:
                idx = len(time_list)-1
            time = time_list[idx]
            line1.set_data(wavelengths, df_fit_spectra_Vr[time].values)
            line2.set_data(wavelengths, df_fit_spectra_Vo[time].values)
            line3.set_data(wavelengths, df_target[time].values)
            line4.set_data(wavelengths, df_fit[time].values)
            time_label.set_text(f'{time}s')
            #return line1, line2, line3, line4

        def set_data_conc(line1, line2, SOC_label, data_cuv):
            (df_target, df_fit, ar_array, ao_array, time_list, df_fit_spectra_Vr, df_fit_spectra_Vo) = tuple(data_cuv.values())
            idx = i
            if idx >= len(time_list)-1:
                idx = len(time_list)-1
            time = time_list[idx]
            line1.set_data(time_list[:idx], ar_array[:idx])
            line2.set_data(time_list[:idx], ao_array[:idx])
            #SOC = cr_array[idx]
            SOC_label.set_text(f'{time}s; cr = {ar_array[idx]:.2f} M, co = {ao_array[idx]:.2f} M')
            #return line1, line2
            
        set_data_spectra(line00_1, line00_2, line00_3, line00_4,time_label00, data_cuv_neg)
        set_data_spectra(line01_1, line01_2, line01_3, line01_4,time_label01, data_cuv_pos)        
        set_data_conc(line10_1, line10_2, SOC_label10, data_cuv_neg)
        set_data_conc(line11_1, line11_2, SOC_label11, data_cuv_pos)        
        return [line00_1, line00_2, line00_3, line00_4, line01_1, line01_2, line01_3, line01_4, line10_1, line10_2, line11_1, line11_2],
    
    # Create animation
    time_list1 = data_cuv_pos['Time (s)']
    time_list2 = data_cuv_neg['Time (s)']
    frames = max(len(time_list1), len(time_list2))
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=100)
    # saving to m4 using ffmpeg writer 
    if save_FN is not None:
        writervideo = animation.FFMpegWriter(fps=60) 
        anim.save(save_FN, writer=writervideo) 
    return anim

#%%
if __name__ == "__main__":

    # Vanadium electrolyte
    c_V, c_SO4 = conc_V_SO4(weight_pc_V = 6.0, weight_pc_SO4 = 28, density = 1.35, show_details=True)
    
    # Proton and bisulfate concentration as function of state of charge?
    #conc_x, c_H_plus, pH_c_H_plus, c_H2SO4, c_HSO4_minus, c_SO4_2minus = rfb.calc_conc_functions(c_V, c_SO4)
    df_conc = calc_df_conc(c_V, c_SO4, show=True)

