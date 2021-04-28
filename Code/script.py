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


path_g = 'C:/Users/admin/Documents/Eloy/Real doc/EPFL/Master 4/PDS/'
path_data = path_g + 'data_Thabuis/'
path_code = path_g + 'Code/'
path_results = path_g + 'Results/'


#important to import file that are not here
sys.path.append(os.path.abspath(path_code))
#sys.path.append( path_code ) 'works too, let it there in case' 
from Subpart import *
from fct_utile import *

#from matplotlib import pyplot as plt
#from skimage.morphology import skeletonize_3d
##############################################################################


nodes_np, density_np, elements_np = load_data(False)

grid = voxels_2_grid(density_np, elements_np)

bin_grid = binarization(grid)

skeleton, dist = skeletonization(bin_grid, method = 1, plot = False)

print("###########################")
#plt.figure(2)
coordinates, end_pts, inter_pts = skeleton_analysis(skeleton, dist, 0, True)
#soortir graph aussi

#step
#1. trouver ends et jnts
#2. trouver indice end et joint
#3. trouver relation entre end et joint
#4. moyenne width segment

#5. chose several intermediate ptn


# links, intermediaire point  = complexity_control( graph, coordinate, end_pt, inter)



# plt.scatter(end_pt[:,0], end_pt[:,1] )
# plt.scatter(inter[:,0], inter[:,1] )


print("-----------------")

g0, c0, d0 = experimentation(skeleton, False)


print("###########################")

# print_info( coordinates ,globals())
# print_info( end_pt ,globals())
# print_info( inter ,globals())


##############################################################################

if 0:
    export_data_vtr(path_results, grid, globals())
    export_data_vtr(path_results, skeleton, globals())


#plt.scatter(coordinates[:,0], coordinates[:,1] )




#plt.imshow(results)