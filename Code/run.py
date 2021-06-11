# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:00:35 2021

@author: admin
"""

import numpy as np
import sys 
import os

path_code    = os.path.dirname(__file__)
path_g       = os.path.abspath(os.path.join(path_code, os.pardir))
# path_data    = path_g + '\data'
# path_results = path_g + '\Results'
path_data    = os.path.join( path_g, 'data')
path_results = os.path.join( path_g, 'Results')

#important to import file that are not here
sys.path.append(os.path.abspath(path_code))

from loading import load_data
from binarization import binarize
from skeletonization import skeletonize               
from skeleton_analysis_functions import skeleton_analysis
from convertor import format_convertor
from joints_extension import joints_correction

from presentation import draw_segment, similarity

##############################################################################

    #resolutionà 0.2 jolie visualisation
    #error wth 0.27, 0.0704225
    
path_input = path_data
#path_input = path_data + '\prepared images\\struc.png'
    



from matplotlib import pyplot as plt



def features_extractor(path = path_input, resolution=0.0704225, load_mode = "Dflt",
                       inv = False, skl_meth = 1,
                       cmplx_coef = 1, clean_coef = 0.1,
                       display = False, plot = False):
    """
    Extract the features of a geometry which could be defined in different way

    E - number of elements
    
    D - number of dimensions 

    Parameters
    ----------
    path : string, optional
        Path of the folder or file to load. The default is path_input.
    resolution : float, optional
        Give the resolution parameter to pass from Cloud points to an image.
        The default is 0.0704225.
    load_mode : string, optional
        Give the format of the data to import. The default is "Dflt".
    inv : Bool, optional
        Indicate if the intensity imported data must be inverted.
        The default is False.
    skl_meth : int, optional
        Select the skeltionization process to use. The default is 1.
    cmplx_coef : int, optional
        Define the complexity to apply to the skeleton. The default is 1.
    clean_coef : float, optional
        Define how much the skeleton should be cleanned. The default is 0.1.    
    display : Bool, optional
        Display messages to inform of the step process. The default is False.
    plot : Bool, optional
        Plot the different step of the process. The default is True.

    Returns
    -------
    links_coor_ext : np.ndarray of float
        [Ex(D*2)] - a version of link_coor with some points modified in the order
        to improve the connectivity
        
    widths: np.ndarray of float
        [Ex1] - list of the width of each segment of the simplied skeleton
    
    """

    grid = load_data( path, load_mode, resolution, display=display)                                                            #print

    bin_grid = binarize(grid, inverse=inv, plot = plot)                          #modify file
    
    skeleton, dist = skeletonize(bin_grid, skl_meth, plot = plot)          #modify file
    
    coordinates, links, widths, skeleton_cl \
        = skeleton_analysis(skeleton, dist, cmplx_coef, clean_coef, plot=plot)
    
    links_coor = format_convertor(coordinates, links)

    links_coor_ext = joints_correction(links_coor, links, widths,
                                       display=display, plot=plot)

    output = draw_segment( links_coor_ext, widths, skeleton.shape,
                          display=display, plot = plot)                           #devrai etre enlevé 
    #####
    # output1 = draw_segment( links_coor, widths, skeleton.shape,
    #                       display=display, plot = plot)
    # plt.matshow(output.transpose()[::-1])
    # plt.title("Result with extension\n")

    # plt.matshow(output1.transpose()[::-1])
    # plt.title("Result without extension\n")
    # similarity(grid, output1, display = display)

    ####
    
    similarity(grid, output, display = display)                               #modify file

    return links_coor_ext, widths*2

if __name__ == '__main__':
    print('run executed')