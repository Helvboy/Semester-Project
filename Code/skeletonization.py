# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:28:42 2021

@author: admin
"""

from matplotlib import pyplot as plt
from skimage.morphology import medial_axis, skeletonize_3d



def skeletonize(grid, method = 0, plot = False):
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
        plt.matshow(img_skel.transpose()[::-1])
        plt.title("Result of the skeletonization\n")
        
    if (plot & dist):
        plt.matshow(distance.transpose()[::-1])
        plt.title("Result of the skeletonization: distance\n")
        
    return img_skel.astype('uint8'), distance


if __name__ == '__main__':
    print( 'skeletonization executed')