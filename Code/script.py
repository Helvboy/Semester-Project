# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 23:02:37 2021

@author: admin
"""

import numpy as np
import sys 
import os
#import matplotlib as plt
#import pandas as pd
#from pyevtk.hl import gridToVTK

path_code = os.path.dirname(__file__)
path_g =  os.path.abspath(os.path.join(path_code, os.pardir))

#path_g = 'C:/Users/admin/Documents/Eloy/Real doc/EPFL/Master 4/PDS/'
path_data = path_g + 'data/'
path_data_T = path_g + 'data_Thabuis/'
path_code = path_g + 'Code/'
path_results = path_g + 'Results/'


#important to import file that are not here
sys.path.append(os.path.abspath(path_code))
#sys.path.append( path_code ) 'works too, let it there in case'
from loading import load_data
from Subpart import load_data_old, voxels_2_grid, binarization, skeletonization,\
                    skeleton_analysis, export_data_vtr

#from matplotlib import pyplot as plt
#from skimage.morphology import skeletonize_3d
##############################################################################


nodes_np, density_np, elements_np = load_data( file_name,False)

grid = voxels_2_grid(density_np, elements_np)

bin_grid = binarization(grid)


method = 1
skeleton, dist = skeletonization(bin_grid, method, plot = False)

if method != 1:
    print("only medial axis transform give dist, without it, impossible to deduce"
           "the width of each segment")

print("###########################")
#plt.figure(2)

cmplx_coef = 4
coordinates, new_links, path_width = skeleton_analysis(skeleton, dist, cmplx_coef, True)

print("###########################")

# print_info( coordinates ,globals())
# print_info( end_pt ,globals())
# print_info( inter ,globals())


##############################################################################

if 0:
    export_data_vtr(path_results, grid,     globals())
    export_data_vtr(path_results, skeleton, globals())


#plt.scatter(coordinates[:,0], coordinates[:,1] )

#plt.imshow(results)


def script_steps(grid):

    bin_grid = binarization(grid)
    
    method = 1
    skeleton, dist = skeletonization(bin_grid, method, plot = False)
    
    if method != 1:
        print("only medial axis transform give dist, without it, impossible to deduce"
               "the width of each segment")
    
    print("###########################")
    
    
    cmplx_coef = 4
    coordinates, new_links, path_width = skeleton_analysis(skeleton, dist, cmplx_coef, True)
    
    print("###########################")
    


if __name__ == '__main__':
    print('script executed')
