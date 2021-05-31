# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:55:06 2021

@author: Eloi Schlegel
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(path, data_format="Dflt", plot = False, plot_d = False):
    '''
    Load data
    N - number of points
    E - number of elements
    D - number of dimensions
    V - number of vertices by voxel

    input:
        path(string)
        data_format(srting) - Dflt(Default) - Vxl(Voxels) - PC(Points Cloud)
                            - Img(Image)
        plot(bool)
        
    output:
        nodes(np.ndarray) - [N x D]
        densities(np.ndarray) - [E x 1]
        elements(np.ndarray) - [E x V]
    '''
    
    if data_format == "Vxl":
        nodes = np.loadtxt( path + '\nodes.dat' )
        densities = np.loadtxt( path + '\densities.dat' )
        elements = np.loadtxt( path + '\elements.dat' )

        print("Voxels loading done \n")
        return nodes, densities, elements
    
    elif data_format == "PC":
        
        print("Points Cloud loading done \n")
        return nodes, densities
    
    elif data_format == "Img":
        image = plt.imread(path)
        
        img = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        print("Image loaded \n")

        return img
    
    elif data_format == "Dflt":
        nodes = np.loadtxt( path + '/data_Thabuis/nodes.dat' )
        if plot_d:
            print("nodes_load")
        
        densities = np.loadtxt( path + '/data_Thabuis/density.dat' )
        if plot_d:
            print("density_load")
        
        elements = np.loadtxt( path + '/data_Thabuis/elements.dat' ).astype(int)
        if plot_d:
            print("elements_load \n")
        
        print("Default loading done \n")
        return nodes, densities, elements
        

    return "coucou"


def load_default_data( path_data, plot = True):
    '''
    Load default data give by Thabuis
    N - number of points
    E - number of elements
    D - number of dimensions
    V - number of vertices by voxel
    
    input:
        file_name(string)
        
    output:
        nodes_np(np.ndarray) - [N x D]
        density_np(np.ndarray) - [E x 1]
        elements_np(np.ndarray) - [E x V]
    '''
    
    nodes_np = np.loadtxt( path_data + '/nodes.dat' )
    if plot:
        print("nodes_load")
    
    density_np = np.loadtxt( path_data + '/density.dat' )
    if plot:
        print("density_load")
    
    elements_np = np.loadtxt( path_data + '/elements.dat' ).astype(int)
    if plot:
        print("elements_load \n")
    
    print("loading done \n")
    
    return nodes_np, density_np, elements_np

if __name__ == '__main__':
    print('loading executed')