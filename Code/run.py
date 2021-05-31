# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:00:35 2021

@author: admin
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
from convertor import voxels_to_cloud_points, cloud_points_to_pixels
from Subpart import binarization, skeletonization,\
                    skeleton_analysis, export_data_vtr
from presentation import draw_segment, similarity

##############################################################################


def features_extractor(path=path_data, resolution=0.0704225, method = 1,
                cmplx_coef = 1, plot = True):
    
    nodes_np, density_np, elements_np = load_data( path, 'Dflt', plot=False)   # mode a changer
    
    points = voxels_to_cloud_points(elements_np, nodes_np)
    #resolutionà 0.2 jolie visualisation
    #error wth 0.27, 0.0704225
    grid = cloud_points_to_pixels(points, density_np, resolution)
    
    
    bin_grid = binarization(grid, False)
    
    skeleton, dist = skeletonization(bin_grid, method, plot = False)
    
    coordinates, links, widths = skeleton_analysis(skeleton, dist, cmplx_coef, True)

    print(links.shape, widths.shape)
    
    
    output = draw_segment( links, widths, coordinates)
    print('error', similarity(bin_grid/255, output), '%')
    
    print("finished")
    


if __name__ == '__main__':
    print('run executed')