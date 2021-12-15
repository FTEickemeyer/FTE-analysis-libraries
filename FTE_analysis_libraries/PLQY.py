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
import re
import thot
from scipy.optimize import least_squares
import math
import numpy as np
from os.path import join

system_dir = pkg_resources.resource_filename( 'FTE_analysis_libraries', 'System_data' )


from . import Spectrum as spc
from .General import findind, int_arr, linfit, save_ok, q, k, T_RT, h, c, f1240, pi
from .XYdata import xy_data, mxy_data


from FTE_analysis_libraries.General import f1240, Vsq, V_loss, QFLS


def get_Andor_metadata(f, showall = False):
    """
    Extracts all metadata from the filename f for measurements with the Andor spectrometer.
    """

    # name
    name = f.split('--')[0]
    metadata = dict(name = name)
    if showall:
        print(name)
        
    # add original filename
    metadata['orig_fn'] = f

    #f = f.split('--')[1]

    # fs or ip
    fsip = f.split('--')[1].split('_')[0]
    metadata['fsip'] = fsip
    if showall:
        print(fsip)

    # calibration lamp
    if name.lower() == 'calibration':
        cl = f.split('--')[2]
        metadata['calib_lamp'] = cl
        if showall:
            print(cl)

    else:
        # inbeam or outofbeam
        if fsip == 'ip':
            inout = f.split('--')[1].split('_')[1]
            metadata['inboob'] = inout
            if showall:
                print(inout)

        # laser wavelength
        lw = f.split('laser_')[1].split('nm')[0]
        lw = int(lw)
        metadata['laser_nm'] = lw
        if showall:
            print(lw)

        # laser power
        lp = f.split('laser_')[1].split('_')[1].split('mW')[0]
        lp = float(lp)
        metadata['laser_mW'] = lp
        if showall:
            print(lp)

        # OD filter
        OD = f.split('_OD')[1].split('--')[0]
        OD = float(OD)
        metadata['OD_filter'] = OD
        if showall:
            print(OD)


    # integration time
    it = f.split('Andor_')[1].split('s_')[0]
    it = float(it)
    metadata['int_s'] = it
    if showall:
        print(it)

    # accumulations
    acc_pattern = '(\d+)acc'
    acc_match = re.search(acc_pattern, f.lower())
    acc = int(acc_match.group(1))
    metadata['acc'] = acc
    if showall:
        print(acc)

    # grating
    lmm_pattern = '(\d+)lmm'
    lmm_match = re.search(lmm_pattern, f.lower())
    lmm = int(lmm_match.group(1))
    metadata['grating'] = lmm
    if showall:
        print(lmm)

    center_pattern = 'center(\d+)'
    center_match = re.search(center_pattern, f.lower())
    center = int(center_match.group(1))
    metadata['grating_center_nm'] = center
    if showall:
        print(center)
    
    # slit (is only used for the new Andor system)
    if 'slit' in f.lower():
        slit_pattern = '(\d+)'+ 'umslit'
        slit_match = re.search(slit_pattern, f.lower())
        sl = int(slit_match.group(1))
        metadata['slit_um'] = sl
        #sl = f.split('--')[3].split('_')[-2]
        if showall:
            print(f'slit size = {sl} um')

    # emission filter
    ef = f.split('--')[3].split('_')[-1].split('.')[0]
    metadata['em_filter'] = ef
    
    if showall:
        print(ef)
        
    return metadata

def raw_to_asset_with_metadata(container, asset_type, db, show_FN = False, show_new_asset = False):
    # Generate new asset with metadata from raw measurements
    
    raw = db.find_assets( { 'parent' : container._id, 'type': asset_type } )
    
    # Generate new calibration asset with metadata
    for idx, asset in enumerate(raw):
        """
        Creates a new asset with metadata.
        """
        f = asset.file
        f = f.replace( '\\', '/' )
        f = os.path.basename(f)
        if show_FN:
            print(f)
        metadata = get_Andor_metadata(f, showall = False)
        #print(metadata)
        if container.name.lower() == 'calibration':
            asset_prop = dict(name = f'{idx}_raw calibration.csv', type = 'raw calibration', metadata = metadata, file = asset.file)
        elif container.name.lower() == 'samples':
            name = metadata['name']
            asset_prop = dict(name = f'{idx}_{name}_raw PL spectrum.csv', type = 'raw PL spectrum', metadata = metadata, file = asset.file)
        else:
            print('Attention: No known container_name!')
        asset = db.add_asset(asset_prop)
        if show_new_asset:
            print(asset)    

def add_graph(db, fn, graph):
    """
    Adds a graph as an asset and saves it

    Parameters
    ----------
    db : ThotProject
        DESCRIPTION.
    fn : STRING
        filename (without path but with extension).
    graph : matplotlib.figure.Figure
        The graph to be saved.

    Returns
    -------
    None.

    """
    
    asset_prop = dict(name = 'plt_'+fn, type = 'graph', file = fn)
    asset_filepath = db.add_asset(asset_prop)
    graph.savefig(asset_filepath)


def find(dic, assets, show_details = False):
    asts = thot.filter(dic, assets)
    if len(asts) == 0:
        print(f'Error: {dic} in assets not found!')
    elif len(asts) > 1:
        print(f'Error: {dic} found more than one instance!')
    else:
        if show_details:
            print(asts[0].metadata['orig_fn'])
        return asts[0]
    
    
class exp_param:
    
    def __init__(self, which_sample = None, excitation_laser = None, PL_left = None, PL_right = None, PL_peak = None, corr_offs_left = 40, corr_offs_right = 50, PL_peak_auto = False, eval_Pb = False):

        # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 

        self.excitation_laser = excitation_laser
        self.PL_left = PL_left
        self.PL_right = PL_right
        self.PL_peak = PL_peak        
        self.corr_offs_left = corr_offs_left
        self.corr_offs_right = corr_offs_right
        self.eval_Pb = eval_Pb
        self.PL_peak_auto = PL_peak_auto

        if which_sample == 'FAPbI3':
            self.excitation_laser = 657 #nm
            self.PL_left = 700 #nm
            self.PL_right = 950 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = 40 # nm
            self.corr_offs_right = 50 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = False
        elif which_sample == 'Haizhou-FAPbI3': # Very high quality FAPbI3 samples
            self.excitation_laser = 657 #nm
            self.PL_left = 700 #nm
            self.PL_right = 950 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = 60 # nm
            self.corr_offs_right = 70 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = True
        elif which_sample == 'Yameng DSC':
            self.excitation_laser = 421 #nm
            self.PL_left = 600 #nm
            self.PL_right = 1000 #nm
            self.PL_peak = 700 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = -10 # nm
            self.corr_offs_right = +10 # nm
            self.eval_Pb = False
        elif which_sample == 'dye on TiO2':
            self.excitation_laser = 421 #nm
            self.PL_left = 600 #nm
            self.PL_right = 900 #nm
            #PL_peak = 700 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = -10 # nm
            self.corr_offs_right = +10 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = False
        elif which_sample == 'dye on Al2O3':
            self.excitation_laser = 421 #nm
            self.PL_left = 600 #nm
            self.PL_right = 1000 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = -10 # nm
            self.corr_offs_right = +10 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = False
        elif which_sample == 'Coumarin 153':
            self.excitation_laser = 421 #nm
            self.PL_left = 450 #nm
            self.PL_right = 830 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = 20 # nm
            self.corr_offs_right = 30 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = True
        elif which_sample == 'XY1b':
            self.excitation_laser = 421 #nm
            self.PL_left = 590 #nm
            self.PL_right = 1000 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = 20 # nm
            self.corr_offs_right = 30 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = False
        elif which_sample == 'MS5':
            self.excitation_laser = 421 #nm
            self.PL_left = 500 #nm
            self.PL_right = 920 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = 20 # nm
            self.corr_offs_right = 30 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = False
        elif which_sample == 'Cs2AgBiBr6':
            self.excitation_laser = 421 #nm
            self.PL_left = 500 #nm
            self.PL_right = 920 #nm
            # The ip PL signal will be corrected by the fs measurement. The fit is carried out from PL_peak+corr_offs_left to PL_peak+corr_offs_right 
            self.corr_offs_left = 20 # nm
            self.corr_offs_right = 30 # nm
            self.PL_peak_auto = True # determine automatically
            self.eval_Pb = False
        else:
            if excitation_laser is None:
                #self.excitation_laser = 403 #nm
                #self.excitation_laser = 419 #nm
                self.excitation_laser = 421 #nm
                #self.excitation_laser = 422 #nm old system
                #self.excitation_laser = 657 #nm
            else:
                self.excitation_laser = excitation_laser

            if PL_left is None:
                # PL signal will be evaluated from left to right
                #self.PL_left = 500 #nm
                self.PL_left = 600 #nm
                #self.PL_left = 760 #nm
                #self.PL_left = 735 #nm
                #self.PL_left = 700 #nm
                #self.PL_left = 750 #nm
            else:
                self.PL_left = PL_left
                    
            if PL_right is None:
                #self.PL_right = 600 #nm
                #self.PL_right = 900 #nm
                #self.PL_right = 920 #nm
                #self.PL_right = 950 #nm
                self.PL_right = 1000 #nm
                #self.PL_right = 1030 #nm
            else:
                self.PL_right = PL_right

            # PL peak (for the readjustment of the inbeam PL with fs PL)
            if PL_peak == None:
                #self.PL_peak = 540
                self.PL_peak = 700 #nm
                #self.PL_peak = 750 #nm
                #self.PL_peak = 790 #nm
                #self.PL_peak = 800 #nm
                #self.PL_peak = 920 #nm
            else:
                self.PL_peak = PL_peak

        # These laser limits will be used to evaluate PL photon flux
        if self.excitation_laser == 403:
            self.laser_left = 390
            self.laser_right = 420
        elif self.excitation_laser == 419:
            self.laser_left = 410
            self.laser_right = 428
        elif self.excitation_laser == 421:
            self.laser_left = 414
            self.laser_right = 428
        elif self.excitation_laser == 422:
            self.laser_left = 415
            self.laser_right = 428
        elif self.excitation_laser == 657:
            self.laser_left = 640
            self.laser_right = 670
        else:
            print('No valid laser wavelength! Valid wavelengths are: 403, 419, 421, 422, 657')

        # Laser and PL marker
        if self.excitation_laser == 403:
            self.laser_marker = '405BPF'
            self.PL_marker = '450LPF'
        elif self.excitation_laser == 419:
            self.laser_marker = '420BPF'
            self.PL_marker = '450LPF'
        elif self.excitation_laser == 421:
            self.laser_marker = '420BPF'
            self.PL_marker = '450LPF'
        elif self.excitation_laser == 422:
            self.laser_marker = '420BPF'
            self.PL_marker = '450LPF'
        elif self.excitation_laser == 657:
            self.laser_marker = '650BPF'
            #self.laser_marker = '660BPF'
            #self.PL_marker = '685LPF'
            self.PL_marker = '700LPF'
        else:
            print('No valid laser wavelength! Valid wavelengths are: 403, 419, 421, 657')


class PLQY_dataset:
    
    def __init__(self, db, La, Lb, Lc, Pa, Pb, Pc, fs, sample_name, param):
        
        def load_spectrum(LP):
            return spc.PEL_spectrum.load(os.path.dirname(LP.file), FN = os.path.basename(LP.file), take_quants_and_units_from_file = True)
        
        def create_PELspectra_obj(sa):
            PEL = spc.PEL_spectra(sa)
            PEL.label([])
        
        self.db = db
        self.sample_name = sample_name
        self.param = param
        self.La_asset = La
        self.La = load_spectrum(La)
        self.Lb_asset = Lb
        self.Lb = load_spectrum(Lb)
        self.Lc_asset = Lc
        self.Lc = load_spectrum(Lc)
        self.Pa_asset = Pa
        self.Pa = load_spectrum(Pa)
        self.Pb_asset = Pb
        self.Pb = load_spectrum(Pb)
        self.Pc_asset = Pc
        self.Pc = load_spectrum(Pc)
        self.fs_asset = fs
        self.fs = load_spectrum(fs)
        self.PL_peak = param.PL_peak

        self.all = spc.PEL_spectra([self.La, self.Lb, self.Lc, self.Pa, self.Pb, self.Pc, self.fs])
        self.all.label(['La', 'Lb', 'Lc', 'Pa', 'Pb', 'Pc', 'fs'])
        
        self.P = spc.PEL_spectra([self.Pa, self.Pb, self.Pc])
        self.P.label(['Pa', 'Pb', 'Pc'])
        self.L = spc.PEL_spectra([self.La, self.Lb, self.Lc])
        self.L.label(['La', 'Lb', 'Lc'])

    def plot(self, *args, **kwargs):
        self.all.plot(*args, **kwargs)
        all_graph = self.all.plot(*args, return_fig = True, show_plot = False, **kwargs)
        add_graph(self.db, self.sample_name+'_all.png', all_graph)
        plt.close( all_graph )
        

    def find_PL_peak(self):
        if self.param.PL_peak_auto:
            #Finds the PL peak of the free space spectrum between PL_left and PL_right
            if self.PL_peak == None:
                if self.param.PL_peak_auto:
                    ra = self.fs.idx_range(left = self.param.PL_left, right = self.param.PL_right)
                    PL_peak = self.fs.x_of(max(self.fs.y[ra]), start = self.param.PL_left)
                self.PL_peak = PL_peak
        self.Eg = f1240/self.PL_peak #eV
        self.Vsq = Vsq(self.Eg) #V
    
        
    def inb_oob_adjust(self, what = 'inb', adj_factor = None, show_adjust_factor = False, save_plots = False, divisor = 1e3):
        # adj_factor: manual adjustment factor. It is advisable to run this routine first with show_adjust_factor = True and then take this as a basis for the adj_factor
        # automatically calculate the factor

        fs = self.fs
        if what == 'inb':
            sp = self.Pc
        elif what == 'oob':
            sp = self.Pb
            
        def guess_factor(left, right):
            """
            Returns the inbeam or outofbeam-free space adjustment factor.
            """

            fs_ = fs.copy()
            sp_ = sp.copy()
            
            delta = fs_.x[1] - fs_.x[0] 
            fs_.equidist(left = left, right = right, delta = delta)
            sp_.equidist(left = left, right = right, delta = delta)

            def f(fac): 
                diff = sp_.y - fac * fs_.y
                return math.sqrt(1/len(diff) * np.dot(diff, diff))

            result = least_squares(fun = f, x0 = [1])

            return result.x[0]

        left = self.PL_peak + self.param.corr_offs_left
        right = self.PL_peak + self.param.corr_offs_right

        if adj_factor == None:
            factor = guess_factor(left = left, right = right)
        else:
            factor = adj_factor
            
        self.adj_factor = factor
        
        if show_adjust_factor:
            print(f'The inbeam/outofbeam adjust factor is {factor:.2e}')
        
        sp_orig = sp.copy()
        if what == 'inb':
            #We'll need the original spectrum later
            self.Pc_orig = sp_orig
            #self.Pc.y = fs.y * factor 
            self.Pc_corrfac = factor
        elif what == 'oob':
            #We'll need the original spectrum later
            self.Pb_orig = sp_orig
            self.Pb_corrfac = factor
        sp.y = fs.y * factor  

        if save_plots:
            fssp = spc.PEL_spectra([sp_orig, sp])
            fssp.label([what, 'adjusted'])
            
            fssp_lin_graph = fssp.plot(yscale = 'linear', left = self.param.PL_left, right = self.param.PL_right, divisor = divisor, title = 'Correction for '+ what, figsize = (7,5), return_fig = True, show_plot = False)
            add_graph(self.db, f'{self.sample_name}_fs_{what}_correction(linear).png', fssp_lin_graph)
            plt.close( fssp_lin_graph )
            
            fssp_log_graph = fssp.plot(yscale = 'log', left = self.param.PL_left, right = self.param.PL_right, divisor = divisor, title = 'Correction for '+ what, figsize = (7,5), return_fig = True, show_plot = False)
            add_graph(self.db, f'{self.sample_name}_fs_{what}_correction(semilog).png', fssp_log_graph)
            plt.close( fssp_log_graph )
        
    def inb_adjust(self, adj_factor = None, show_adjust_factor = False, save_plots = False, divisor = 1e3):
            self.inb_oob_adjust(what = 'inb', adj_factor = adj_factor, show_adjust_factor = show_adjust_factor, save_plots = save_plots, divisor = divisor)
    def oob_adjust(self, adj_factor = None, show_adjust_factor = False, save_plots = False, divisor = 1e3):
            self.inb_oob_adjust(what = 'oob', adj_factor = adj_factor, show_adjust_factor = show_adjust_factor, save_plots = save_plots, divisor = divisor)

    def calc_abs(self, what = 'inb', save_plots = False):
        #Calculates the absorptance spectrum from the fs and inbeam or outofbeam PL spectrum
        if what == 'inb':
            sp_orig = self.Pc_orig
            sp = self.Pc
        elif what == 'oob':
            sp_orig = self.Pb_orig
            sp = self.Pb

        # I,fs -> ip -> I,ib ==> A = 1-T = 1-I,ib/I,fs
        left = self.param.PL_left
        right = self.param.PL_right
        r = range(sp.x_idx_of(left), sp.x_idx_of(right)-1)
        x = sp.x[r]
        y_ib = sp_orig.y[r]
        y_fs = sp.y[r]
        zero_mask = np.where( y_fs != 0 )
        A = 1-y_ib[zero_mask]/y_fs[zero_mask]
        s = spc.abs_spectrum(x[zero_mask], A)
        s.qy = 'A'
        if save_plots:
            abs_graph = s.plot(title = 'Absorptance spectrum', hline = 0, bottom = -0.2, top = 1, figsize = (8,5), return_fig = True, show_plot = False)
            add_graph(self.db, f'{self.sample_name}_absorptance_with_{what}.png', abs_graph)
            plt.close( abs_graph )

        
        
    def calc_PLQY(self, eval_Pa = False, show = False, save_plots = False, show_lum = 'log'):
        
        La = self.La.photonflux(start = self.param.laser_left, stop = self.param.laser_right)
        Lb = self.Lb.photonflux(start = self.param.laser_left, stop = self.param.laser_right)
        Lc = self.Lc.photonflux(start = self.param.laser_left, stop = self.param.laser_right)

        if eval_Pa:
            Pa = self.Pa.photonflux(start = self.param.PL_left, stop = self.param.PL_right)
        else:
            Pa = 0

        if self.param.eval_Pb:
            Pb = self.Pb.photonflux(start = self.param.PL_left, stop = self.param.PL_right)
        else:
            Pb = 0

        Pc = self.Pc.photonflux(start = self.param.PL_left, stop = self.param.PL_right)

        Pb = Pb - Pa
        Pc = Pc - Pa
        A = 1 - Lc/Lb
        PLQY = (Pc - (1 - A) * Pb) / (La * A)

        if save_plots:
            laser_graph = self.L.plot(yscale = 'linear', left = self.param.laser_left, right = self.param.laser_right, title = 'Laser signal', showindex = False, in_name = self.param.laser_marker, figsize = (7,5), hline = 0, return_fig = True, show_plot = False)
            add_graph(self.db, f'{self.sample_name}_L.png', laser_graph)
            plt.close( laser_graph )

            PL_graph = self.P.plot(yscale = show_lum, left = self.param.PL_left, right = self.param.PL_right, divisor = 1e7, title = 'Luminescence signal', showindex = False, in_name = self.param.PL_marker, figsize = (7,5), hline = 0, return_fig = True, show_plot = False)
            add_graph(self.db, f'{self.sample_name}_P.png', PL_graph)
            plt.close( PL_graph )

        if show:
            print(f'La = {La:.2e} 1/(s m2)')
            print(f'Lb = {Lb:.2e} 1/(s m2)')
            print(f'Lc = {Lc:.2e} 1/(s m2)')

            print(f'Pa = {Pa:.2e} 1/(s m2)')
            print(f'Pb = {Pb:.2e} 1/(s m2)')
            print(f'Pc = {Pc:.2e} 1/(s m2)')

            print(f'A = 1 - Lc/Lb = {A*100:.1f} %')

            print(f'PLQY = (Pc - (1 - A) * Pb) / (La * A) = {PLQY:.2e}')

        self.PLQY = PLQY
        self.A = A
        self.LaPF = La
        self.LbPF = Lb
        self.LcPF = Lc
        self.PaPF = Pa
        self.PbPF = Pb
        self.PcPF = Pc
        self.V_loss = V_loss(PLQY)
        self.QFLS = QFLS(self.Eg, PLQY)

        
    def abs_pf_spec(self, nsuns = 1):
        """
        Calculates the absolute photon flux spectrum for nsuns excitation and saves it as self.absPFspec
        :param nsuns: number of suns
        """
        PF = self.fs.photonflux(start = self.param.PL_left, stop = self.param.PL_right)

        # 1 sun photon flux
        #Bandgap in eV
        Eg = f1240/self.PL_peak #eV
        sun_PF = spc.above_bg_photon_flux(Eg)

        nsun_PF = nsuns * sun_PF
        # Factor fac with which relative spectral photon flux has to be multiplied to yield an absolute spectral photon flux
        fac =  sun_PF * self.PLQY / PF    
        self.fs_absint_factor = fac
        sp = self.fs.copy()
        sp.y = sp.y * fac
        sp = sp.cut_data_outside(left = self.param.PL_left, right = self.param.PL_right)
        
        PF_new = sp.photonflux(start = self.param.PL_left, stop = self.param.PL_right)
        #print(f'PF of absolute spectrum: {PF_new:.1e} 1/(s m2) (PLQY {self.PLQY:.1e})')
        #print(f'sun_PF = {sun_PF:.1e}, Eg = {Eg:.2f} eV, PL peak = {self.PL_peak:.0f} nm')
        self.absolutePFspec = sp
        
    def save_asset(self):
        
        metadata = dict(A = self.A, PLQY = self.PLQY, Peak = self.PL_peak, Eg = self.Eg, Vsq = self.Vsq, dV = self.V_loss, QFLS = self.QFLS, adj_fac = self.adj_factor, fs_absint_factor = self.fs_absint_factor)
        #print(metadata)
        asset_name = f'{self.sample_name}_absolute PL spectrum'
        asset_prop = dict(name = asset_name + '.csv', type = 'absolute PL spectrum', metadata = metadata)
        TFN = self.db.add_asset(asset_prop)
        fn = os.path.basename(TFN) 
        #print(fn)
        directory = os.path.dirname(TFN)
        self.absolutePFspec.save(directory, fn, check_existing = False)
