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
#from skeleton_analysis_functions import *
import skeleton_analysis_functions as skf

from matplotlib import pyplot as plt

from skimage.morphology import medial_axis, skeletonize_3d

from skan import skeleton_to_csgraph, draw, _testdata
from skan import Skeleton

from scipy.sparse import csr_matrix


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function to load data 

Input:  nodes
        density
        elements
    
Output: nodes_np
        density_np
        elements_np
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def load_data(plot = True):
    nodes_np = np.loadtxt( path_data + 'nodes.dat' )
    if plot:
        print("nodes_load")
    
    density_np = np.loadtxt( path_data + 'density.dat' )
    if plot:
        print("density_load")
    
    elements_np = np.loadtxt( path_data + 'elements.dat' )
    if plot:
        print("elements_load \n")
    
    print("loading done \n")
    
    return nodes_np, density_np, elements_np


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Transform voxels ( points ) into a space defined by a grid 

Input:  list of densities of the elements
        list of index of vertices that composed each voxels
        list of the voxels coordinates ()
        
Output: results(np.array)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def voxels_2_grid(density, elements):
    results = np.zeros((142,71))
    
    for i in range(142):
        for j in range(71):
            ind = i+j*142
            
            results[i,j] = density[ind]
            
    return results


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Binarize a matrice with a half threshold

Input:  np.matrice

Output: np.matrice
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def binarization(data):
    
    results = np.where(data > np.max(data)/2, 255, 0 )
    
    return results


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Deduce the skeleton of a binarize image

Input:  grid(np.array): string - path of the folder where to store file
        method(int):    0 - use skeletonize_3D from skimage.morphology
                        1 - use medial_axis from skimage.morphology
        plot(bool):     True (Default), plot all results / False, unplot all
        
Output: img_skel(np.array)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def skeletonization(grid, method = 0, plot = True):
    distance = None
    
    if method == 0:
        img_skel = skeletonize_3d(grid)
        
        img = True
        dist = False
        
    elif method == 1:
        img_skel, distance = medial_axis(grid, return_distance=True)
        
        img = True
        dist = True
            
    
    if (plot & img):
        plt.imshow(img_skel)
        
    if (plot & dist):   
        plt.imshow(distance)
        
    return img_skel.astype('uint8'), distance


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Binarize a matrice with a half threshold

Input:  path_data(string):  path of the folder where to store file
        data(np.array):
        namespace():        globals()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def export_data_vtr(path_data, data, namespace):
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


    

def skeleton_analysis(skeleton, dist, cmplx_coef, plot = True):    
    # extraction of informations
    graph, coordinates, degrees = skeleton_to_csgraph(skeleton)
    graph_mat  = csr_matrix.toarray(graph)
    
    #coordinates - coordinates in the matrix

    
    #create an object of class Skeleton to get all stuff cool
    data_skeleton = Skeleton(skeleton)
    paths = csr_matrix.toarray(data_skeleton.paths).astype(int)
    nb_path = len(paths)
    
    #1. Find end and joint points
    #2. Find index end and joint points    
    ky_pts, ky_pts_id = skf.key_points_deduction(coordinates, degrees)
    ky_pts, ky_pts_id, end_pts, inter_pts, id_end_pts, id_inter_pts = skf.key_points_deduction(coordinates, degrees, True)

    #3. Find relations between key points (end et joint points)   
    #give the index of the pt couple (first case = complexity 0)
    links = skf.key_points_relation(paths, ky_pts_id)

    ##############################################
#4. moyenne width 1 segment
    
    new_links, path_width = skf.complexity_control_2(paths, coordinates,
                                                     graph_mat, dist, cmplx_coef)




    dist_links = data_skeleton.path_lengths()
    


    '''
    
    distance
    
    '''

    links = new_links
    print(links)

    if plot:
        print("here is the analaysis\n")
        
        #plot skeleton
        plt.scatter(coordinates[:,0], coordinates[:,1] )
        
        #plot point
        plt.scatter(end_pts[:,0], end_pts[:,1] )
        plt.scatter(inter_pts[:,0], inter_pts[:,1] )
        
        #plot the each link
        for i in range(len(links)):
            print(i)
            
            plt.plot(coordinates[links[i],0],
                     coordinates[links[i],1],
                     linewidth = (path_width[i]) )

            

# plot the graph of the systeme
    if 0:
        plt.figure(3)
        pxl_g1, coordinates1, degrees1 = skeleton_to_csgraph(skeleton)
        draw.overlay_skeleton_networkx(pxl_g1, coordinates1, image=skeleton)
    
    
    return coordinates, end_pts, inter_pts



def experimentation(skeleton, plot = True):
    
    g0, c0, d0 = skeleton_to_csgraph(_testdata.skeleton0)
    g1, c1, _  = skeleton_to_csgraph(_testdata.skeleton1)
    
    if plot:
        fig, axes = plt.subplots(1, 2)
        draw.overlay_skeleton_networkx(g0, c0, image=_testdata.skeleton0,
                                        axis=axes[0])
        draw.overlay_skeleton_networkx(g1, c1, image=_testdata.skeleton1,
                                        axis=axes[1])
    
    g0 = csr_matrix.toarray(g0)
    return g0, c0, d0
    
    # g0, c0, _ = skeleton_to_csgraph(skeleton)
    # fig, axes = plt.subplots(1, 2)

    # draw.overlay_skeleton_networkx(g0, c0, image = skeleton, axis=axes[0])



##############################################################################

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