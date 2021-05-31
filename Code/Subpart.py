# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:58:56 2021

@author: admin
"""

import numpy as np
import sys 
import os
#import pandas as pd
from pyevtk.hl import gridToVTK


path_g = 'C:/Users/admin/Documents/Eloy/Real doc/EPFL/Master 4/PDS/'
path_data = path_g + 'data_Thabuis/'
path_code = path_g + 'Code/'


#important to import file that are not here
sys.path.append(os.path.abspath(path_code))
#sys.path.append( path_code ) 'works too, let it there in case'

from fct_utile import *

 
import skeleton_analysis_functions as skf

from matplotlib import pyplot as plt

from skimage.morphology import medial_axis, skeletonize_3d

from skan import Skeleton

from scipy.sparse import csr_matrix


def load_data_old(file_name, plot = True):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Function to load data 

    Input:  
        nodes
        density
        elements
    
    Output: 
        nodes_np
        density_np
        elements_np
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    nodes_np = np.loadtxt( path_data + 'nodes.dat' )
    if plot:
        print("nodes_load")
    
    density_np = np.loadtxt( path_data + 'density.dat' )
    if plot:
        print("density_load")
    
    elements_np = np.loadtxt( path_data + 'elements.dat' ).astype(int)
    if plot:
        print("elements_load \n")
    
    print("loading done \n")
    
    return nodes_np, density_np, elements_np


def binarization(data, inverse = False, plot = False):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Binarize a matrice with a half threshold
    
    Input:  np.matrice
    
    Output: np.matrice
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    if inverse == True:
        results = np.where(data > np.max(data)/2, 0, 255 )
    else:
        results = np.where(data > np.max(data)/2, 255, 0 )
    
    
    if plot:
        plt.matshow(results)
    
    return results



def skeletonization(grid, method = 0, plot = False):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Deduce the skeleton of a binarize image
    
    Input:  grid(np.array): string - path of the folder where to store file
            method(int):    0 - use skeletonize_3D from skimage.morphology
                            1 - use medial_axis from skimage.morphology
            plot(bool):     True (Default), plot all results / False, unplot all
            
    Output: img_skel(np.array)
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    distance = None
    dist = False
    
    if method == 0:
        img_skel = skeletonize_3d(grid)
        
        
    elif method == 1:
        img_skel, distance = medial_axis(grid, return_distance=True)
        
        dist = True
            
    
    if (plot):
        plt.imshow(img_skel)
        
    if (plot & dist):
        plt.imshow(distance)
        
    return img_skel.astype('uint8'), distance




############################################################################## effacer

############################################################################## 

def export_data_vtr(path_data, data, namespace):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Expport data from a numpy format to a vtr format
    
    Input:  path_data(string):  path of the folder where to store file
            data(np.array):
            namespace():        globals()
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # use for the name of the file
    name_file = [name for name in namespace if namespace[name] is data]
    
    #generate axis correxponding to the studied volume 
    x = np.arange(0, 72)
    y = np.arange(0, 143)
    z = np.arange(0, 1)
    
    # pyevtk.hl.pointsToVTK()
    gridToVTK( path_data + name_file[0], x, y, z, cellData = {name_file[0]: data.ravel()})
    #array_type=vtk.VTK_FLOAT
    print("data export ", name_file[0], " in vtk")
    

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#######################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# read nodes.dat to a list of lists
#nodes = [i.strip().split() for i in open(path_data + "nodes.dat").readlines()]
#print("done")
#nodes_pd = pd.read_table( path_data + "nodes.dat", sep="\s+", header = None)
#print("pd_done")
'''
nodes_np = np.loadtxt( path_data + 'nodes.dat' )
print("nodes_load")
density_np = np.loadtxt( path_data + 'density.dat' )
print("density_load")
elements_np = np.loadtxt( path_data + 'elements.dat' )
print("elements_load \n")

'''


##############################################################################
##test rapide
##############################################################################
'''
#grid = vite(density_np, elements_np)
#print_info(grid, globals())


#plt.imshow(grid)


#img_skel = skeletonize_3d(grid)

#plt.figure(2)
#plt.imshow(img_skel)
'''


##############################################################################
#add a third dimension but with always the same data 
'''
new_column = np.zeros((nodes_np.shape[0],1))
nodes_np = np.append(nodes_np, new_column, axis=1)

#add a fourth column for Idk
new_column = np.ones((nodes_np.shape[0],1))
nodes_np = np.append(nodes_np, new_column, axis=1)
'''


##############################################################################
# To export in CSV and in VTK format
'''
name = path_data + "nodes_from_py.csv"
np.savetxt( name, nodes_np, delimiter=",", header = '')
print("data export in csv")


x = np.arange(0, 143)
y = np.arange(0, 72)
z = np.arange(0, 1)

# pyevtk.hl.pointsToVTK()
gridToVTK( path_data+'truc', x, y, z, cellData = {'grid': grid.ravel()})
#array_type=vtk.VTK_FLOAT
print("data export in vtk")
'''
##############################################################################


#suppose to help me to debug
#os.getcwd()

if __name__ == '__main__':
    print('Subpart executed')