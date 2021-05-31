# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:37:20 2021

@author: Eloi Schlegel
"""

import numpy as np


def cloud_points_to_pixels(points, densities, resolution=0.1):
    '''
    Transform the a cloud points into a pixelized image
    
    N - number of points \n
    D - number of dimensions \n
    L - Length of the image \n
    H - Height of the image \n
    Ni - number of graduation on the scale for the i-Dimension, extrema included

    Parameters
    ----------
    points : (np.ndarray)
        [N x D] - all the points of the cloud
    densities : (np.ndarray)
        [N x 1] - densities at each point
    resolution : float, optional
        The default is 0.1. Set the distance between 2 pixels

    Returns
    -------
    pixels : (np.ndarray)
        [L x H] - array containing the densities of each points in the space

    '''
    
    N, D = points.shape
    
    minima = np.min(points,0)
    maxima = np.max(points,0)
        
    scales = []
    
    #scale creation for each D-dimension
    for i in range(D):
        scales.append(np.arange(minima[i], maxima[i]+0.5*resolution, resolution) )
    
    pixels = np.zeros(tuple([np.shape(scales[i])[0]for i in range(D)]))
    
    # keep the last assigned value ( should make a mean of the assigned values)
    for i in range(N):
        idxs = locate_points_on_scale(points[i], scales)
        pixels[tuple(idxs)] = densities[i]
        
    return pixels


def locate_points_on_scale(pt, scales):
    '''
    Find the location of a point on a scale
    Give back the indexs of the nearest graduations from the point

    Parameters
    ----------
    pt : (np.ndarray)
        [1 x D] - coordinates of a point
    scales : (list of np.ndarray)
        [N1 x 1] .... [Nd x 1] - scale of each dimension

    Returns
    -------
    idxs : (np.ndarray)
        [D x 1] - index of where the point correspond

    '''
    
    D = len(pt)
    
    idxs = np.zeros(D).astype(int)
    
    for i in range(D):
        idxs[i] = np.argmin(abs(scales[i] - pt[i]) )
    
    return idxs


def voxels_to_cloud_points(elements, vertices):
    '''
    Transform the voxels into a cloud points in D-dimension.
    
    E - number of voxels \n
    V - number of vertices which define the voxel \n
    N - number of whole vertices \n
    D - number of dimensions


    Parameters
    ----------
    elements : (np.ndarray of int)
        [E x V] - indices of the points which define an element
    vertices : (np.ndarray of float)
        [N x D] - coordinates of the vertices

    Returns
    -------
    points : (np.ndarray)
        [N x D] - coordinates of each voxel center

    '''
    
    points = []
    
    for idxs_vertices in elements:
        points = np.append( points, voxel_middle( vertices[idxs_vertices-1]) )
    
    return points.reshape( -1, vertices.shape[1])


def voxel_middle( vertices):
    '''
    Calculate the coordinates of the middle of a voxel defined by N-vertices
    in D-dimension
    
    N - number of points \n
    D - number of dimensions
    
    Parameters
    ----------
    vertices : (np.ndarray)
        [N  x D] - vertices of the element

    Returns
    -------
    coord : (np.ndarray)
        [D x 1] - coordinates of the center

    '''
    
    coord = np.mean(vertices, 0)
    
    return coord


def format_convertor(coordinates, links):
    '''
    Convert the matrix with index of the points in a matrix with the coordinates
    of all the points.
    
    N - number of points
    
    D - number of dimensions
    
    P - number of links

    Parameters
    ----------
    coordinates : np.ndarray
        [N x D] - contains the coordinates of points
    links : np.ndarray
        [P x 2] - contains the indices of the end points of the path

    Returns
    -------
    links_matrix : np.ndarray
        [N x D] - contains the coordinates of the end points of the path
        
    '''
    
    nb_path = len(links)
    D = np.shape(coordinates)[1]
    
    links_coor = np.zeros((nb_path, D*2))
    
    for i in range(nb_path):
        id_ps = links[i]
        
        links_coor[i,0:D] = coordinates[id_ps[0]]
        links_coor[i,D:]  = coordinates[id_ps[1]]
        
    return links_coor


if __name__ == '__main__':
    print('convertor executed')