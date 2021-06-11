# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:55:06 2021

@author: Eloi Schlegel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from convertor import voxels_to_cloud_points, cloud_points_to_pixels

def load_data( path, data_format="Dflt", resolution = 0.1,
               display = False, plot_d = False):
    """
    Load data from a folder or a file given in path parameter and generate 
    a picture stored in an array. The folder must contain the elements in
    the following way:
        
        Voxels:
            3 files (.dat) - one with an array of the vertices coordinates [NxD],
            one with the densities of each voxel [Ex1], and one with
            an array of the indices of each element [ExV]
            in the folder -> nodes.dat / densities.dat / elements.dat
        
        Points Cloud:
            2 files (.dat) - one with an array of the points coordinates [NxD],
            one with the densities of each voxel [Nx1]
            
        Image:
            1 file (.png) - the path is the directory of the file
    
    N - number of points
    
    E - number of elements 
    
    D - number of dimensions 
    
    V - number of vertices by voxel 

    Parameters
    ----------
    path : string
        Path of the element.s to load 
        
    data_format : srting, optional
        Give the format of the data to import: Dflt (Default), Vxl (Voxels),
        PC (Points Cloud), Img (Image). The default is "Dflt" 
        
    resolution : float, optional
        Give the resolution parameter to pass from Cloud points to an image.
        The default is 0.1 
        
    plot : Bool, optional
        Plot a message to inform of the loading end. The default is False 
        
    plot_d : Bool, optional
        Plot a message to inform of the loading details. The default is False 

    Returns
    -------
    img : np.ndarray of float
        Array of the image. The dimension of the array can depend of the minimum
        and maximum values on each axes, of the resolution and of the input 
        data type.
    """
    
    if plot_d:
            print("Loading process:")
    
    if data_format == "Vxl":
        nodes = np.loadtxt( path + '\nodes.dat' )
        if plot_d:
            print("   nodes_load")
        densities = np.loadtxt( path + '\densities.dat' )
        if plot_d:
            print("   density_load")
        elements = np.loadtxt( path + '\elements.dat' )
        if plot_d:
            print("   elements_load \n")

        points = voxels_to_cloud_points(elements, nodes)
        img = cloud_points_to_pixels(points, densities, resolution)
        if display:
            print("Voxels loading done \n")
    
    elif data_format == "PC":
        points = np.loadtxt( path + '\nodes.dat' )
        if plot_d:
            print("   nodes_load")
        densities = np.loadtxt( path + '\densities.dat' )
        if plot_d:
            print("   density_load")
        
        img = cloud_points_to_pixels(points, densities, resolution)
        if display:
            print("Points Cloud loading done \n")
    
    elif data_format == "Img":
        image = plt.imread(path)
        # Load the picture with gray scale
        img = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        if display:
            print("Image loading done \n")

    elif data_format == "Dflt":
        path_Thab = os.path.join(path,'data_Thabuis')
        
        path_nodes = os.path.join(path_Thab,'nodes.dat')
        nodes = np.loadtxt( path_nodes )
        if plot_d:
            print("   nodes_load")
        path_densities = os.path.join(path_Thab,'density.dat')
        densities = np.loadtxt( path_densities )
        if plot_d:
            print("   density_load")
        path_elements = os.path.join(path_Thab,'elements.dat')
        elements = np.loadtxt( path_elements ).astype(int)
        if plot_d:
            print("   elements_load \n")

        points = voxels_to_cloud_points(elements, nodes)
        img = cloud_points_to_pixels(points, densities, resolution)
        if display:
            print("Default loading done \n")
    
    return img

if __name__ == '__main__':
    print('loading executed')