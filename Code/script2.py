# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 23:02:37 2021

@author: Eloi Schlegel
"""

import numpy as np
import sys 
import os

path_code    = os.path.dirname(__file__)
path_g       =  os.path.abspath(os.path.join(path_code, os.pardir))
path_data    = path_g + '\data'
path_results = path_g + '\Results'

#important to import file that are not here
sys.path.append(os.path.abspath(path_code))

from loading import load_data
from convertor import voxels_to_cloud_points, cloud_points_to_pixels, format_convertor
from Subpart import binarization, skeletonization, export_data_vtr # ,skeleton_analysis

from skeleton_analysis_functions import skeleton_analysis
from presentation import draw_segment, similarity, draw_segment_2


from joints_extension import joints_correction

##### /!\ better to right: import joints_extension as jext
##### avoid problem of double defintion ( function with the same name )


#from skimage.filters import threshold_otsu


from matplotlib import pyplot as plt
#from skimage.morphology import skeletonize_3d
##############################################################################

resolution = 0.09
#0.0704225
method = 1
cmplx_coef = 3

corr_fct = 0.1

#res: 0.08 / cc:30 -> 0.76 %

##############################################################################
# C:\Users\admin\Documents\Eloy\Real doc\EPFL\Master 4\PDS\data\prepared images
# path_spe = \
#"C:\\Users\\admin\\Documents\\Eloy\\Real doc\\EPFL\\Master 4\\PDS\\data\\prepared images\\struc.png"

# path = path_data + '\prepared images\\struc.png'
# grid = load_data( path, 'Img', plot=False)
# inv = True

###

nodes_np, density_np, elements_np = load_data( path_data, 'Dflt', plot=False)

points = voxels_to_cloud_points(elements_np, nodes_np)
#resolutionà 0.2 jolie visualisation
#error wth 0.27, 0.0704225
grid = cloud_points_to_pixels(points, density_np, resolution)
inv = False
plt.matshow(grid)                                                            #print

#############################################################################

bin_grid = binarization(grid, inv, plot = False)
# thresh = threshold_otsu(grid)
# bin_grid = grid > thresh

plt.matshow(bin_grid)                                                            #print

skeleton, dist = skeletonization(bin_grid, method)
if method != 1:
    print("only medial axis transform give dist, without it, impossible to deduce"
           "the width of each segment")

plt.matshow(skeleton)                                                            #print


coordinates, links, widths, skeleton_cl = skeleton_analysis(skeleton, dist,
                                                            cmplx_coef, corr_fct,
                                                            False)
plt.matshow(skeleton_cl)

# print(links.shape, widths.shape)

links_coor = format_convertor(coordinates, links)

links_coor_ext = joints_correction(links_coor, links, widths)



plt.figure(49)                                                                 #print
for i in range(len(links)):
    plt.plot([links_coor[i,0],links_coor[i,2]],[links_coor[i,1],links_coor[i,3]],
              linewidth = (widths[i]) )
plt.figure(50)                                                                 #print
for i in range(len(links)):
    plt.plot([links_coor_ext[i,0],links_coor_ext[i,2]],
              [links_coor_ext[i,1],links_coor_ext[i,3]],
              linewidth = (widths[i]) )



output = draw_segment_2( links_coor, widths, skeleton.shape, plot = False)
print('error', similarity(bin_grid/255, output), '%')
plt.matshow(output)                                                            #print
output = draw_segment_2( links_coor_ext, widths, skeleton.shape, plot = False)
plt.matshow(output)                                                            #print
print('error', similarity(bin_grid/255, output), '%')

# plt.matshow(skeleton)
# plt.matshow(skeleton_cl)
# plt.matshow(bin_grid-skeleton_cl*125)
# plt.matshow(output*2-skeleton_cl)

# # output = draw_segment( links, widths, coordinates, skeleton.shape)
# print('error', similarity(bin_grid/255, output), '%')



##############################################################################

# with open('coordinates.txt','wb') as f:
#     for line in coordinates:
#         np.savetxt(f, line, fmt='%.2f')

# f = path_results + 'coordinates.txt'
# np.savetxt(f, line, fmt='%.2f')
#coordinates.tofile(path_results, sep=" ", format="%s")
##############################################################################

if 0:
    export_data_vtr(path_results, grid,     globals())
    export_data_vtr(path_results, skeleton, globals())


#plt.scatter(coordinates[:,0], coordinates[:,1] )

#plt.imshow(results)


if __name__ == '__main__':
    print('script executed')
