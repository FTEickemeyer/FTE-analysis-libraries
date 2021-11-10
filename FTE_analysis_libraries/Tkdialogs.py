# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:29:18 2020

@author: dreickem
"""


import tkinter as tk
from tkinter import filedialog
import os
from os import getcwd



def getFilenames(title, types=[], initialdir=None):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    filenames = filedialog.askopenfilenames(title=title, filetypes=types, initialdir = initialdir)
    root.destroy()
    var = root.tk.splitlist(filenames)
    filePaths = []
    for f in var:
        filePaths.append(f)
        
    cell_directory = os.path.dirname(filePaths[0])
    cell_filenames = [os.path.basename(FN) for FN in filePaths]
    return cell_directory, cell_filenames

def getFilename(title, types=[], initialdir=None):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    filepath = filedialog.askopenfilename(title=title, filetypes=types, initialdir = initialdir)
    root.destroy()
    cell_directory = os.path.dirname(filepath)
    cell_filename = os.path.basename(filepath)
    return cell_directory, cell_filename
    
def saveFile(title, types=[], initialdir=None):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    #f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
    f = filedialog.asksaveasfile(title = title, mode='w', defaultextension=".csv", filetypes = types, initialdir = initialdir)
    root.destroy()
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    #text2save = ''
    #f.write(text2save)
    f.close()
    return(f.name)




if __name__ == "__main__":

    from os import getcwd
    import sys
    
    # Import my own libraries and modules
    mod_dir = r'C:\Users\dreickem\switchdrive\Work\Python\My modules' 
    sys.path.append(mod_dir)
    import Tkdialogs as tk
    
    save_FN = tk.saveFile('Save file', initialdir = getcwd(), types = [('csv files','*.csv'), ('txt files', '*.txt'), ('all', '*.*')], )
    print(save_FN)
    
    load_FNs = tk.getFilenames('Load files', types = [('csv files','*.csv'), ('txt files', '*.txt'), ('all', '*.*')], initialdir = getcwd())
    print(load_FNs)