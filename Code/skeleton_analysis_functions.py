# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:51:42 2021

@author: admin
"""

import sys 
import os
import numpy as np

path_g = 'C:/Users/admin/Documents/Eloy/Real doc/EPFL/Master 4/PDS/'
path_data = path_g + 'data_Thabuis/'
path_code = path_g + 'Code/'

#important to import file that are not here
sys.path.append(os.path.abspath(path_code))

from convertor import coord_2_id

from skan import skeleton_to_csgraph
from skan import Skeleton

from matplotlib import pyplot as plt

from scipy.sparse import csr_matrix


def skeleton_analysis(skel, dist, cmplx_coef, clean_coef, plot = False):
    """
    Analyse the skeleton and generate a new simplified skeleton depending of
    the complexity factor

    L - Long of the image
    
    H - Height of the image
    
    N - number of points
    
    D - number of dimensions 
 
    E - number of elements
    
    Parameters
    ----------
    skel : np.ndarray of int
        [LxH] - Array of the skeleton image
    dist : np.ndarray of float
        [LxH] - Distance map from the boarder of the binarized image
    cmplx_coef : int
        Define the complexity to apply to the skeleton.
    clean_coef : float, optional
        Define how much the skeleton should be cleanned.
    plot : Bool, optional
        Plot the different step of the process. The default is False.

    Returns
    -------
    coordinates : np.ndarray of float
        [NxD] - coordinates of points        
    links : np.ndarray of int
        [Ex2] - indices of the end points of the path
    path_width : np.ndarray of float
        [Ex1] - Width of each element
    skeleton : np.ndarray of int
        [LxH] - Array of the cleaned skeleton image

    """
    
    skeleton = np.copy(skel)

    # extraction of informations
    graph, coordinates, degrees, graph_mat, data_skeleton, paths \
        = skeleton_information(skeleton)

    ##############################################
                ### clean skeleton ###
    ##############################################

    #make a first correction for all the little useless branches
    skeleton_corr = skeleton_cleaning(paths, coordinates, degrees,
                                      skeleton, clean_coef)

    graph, coordinates, degrees, graph_mat, data_skeleton, paths \
        = skeleton_information(skeleton_corr)
        
    #make a second correction for the pixels stacks
    if clean_coef > 0.03:
        skeleton_corr2 = skeleton_cleaning_compl(paths, coordinates,
                                                 degrees, skeleton_corr)

        skeleton = skeleton_corr2
        
        graph, coordinates, degrees, graph_mat, data_skeleton, paths \
            = skeleton_information(skeleton)

        skeleton_corr2 = skeleton_cleaning_compl(paths, coordinates,
                                                 degrees, skeleton)
        
        skeleton = skeleton_corr2
        
        graph, coordinates, degrees, graph_mat, data_skeleton, paths \
            = skeleton_information(skeleton)
    else:
        skeleton = skeleton_corr
    
    ##############################################
    
    if plot:
        plt.matshow(skeleton.transpose()[::-1])
        plt.title("Result of the skeleton cleaning\n")
    
    # Extract features
    links, path_width = complexity_control_2(paths, coordinates,
                                             graph_mat, dist, cmplx_coef)
    
    if plot:
        plt.figure(42)
        
        #plot the each element
        for i in range(len(links)):
            plt.plot(coordinates[links[i],0],
                     coordinates[links[i],1],
                     linewidth = (path_width[i]) )
            
    return coordinates, links, path_width, skeleton


def skeleton_information(skeleton):
    """
    Generate all the needed variables containing intersting informations

    P is the number of path ( link between 2 key points)
    
    N is the number of skeleton's points
    
    L - Length of the image
    
    H - Height of the image

    Parameters
    ----------
    skeleton : np.ndarray of int
        [H x L] - Array of the skeleton image

    Returns
    -------
    graph : sparse.csr.csrmatrix
        DESCRIPTION.
    coordinates : np.ndarray of int
        [Nx2] - List of all the coordinates 
    degrees : np.ndarray of int
        [HxL] - Array with the number of neighbour for each point
    graph_mat : np.ndarray of int
        [(N+1)x(N+1)] - Array with relation between each point
    data_skeleton : csr.Skeleton
        obj class Skeleton(skan)
    paths : np.ndarray of int
        [PxN] - array with all the belongs of each point to each main element

    """

    # extraction of informations
    graph, coordinates, degrees = skeleton_to_csgraph(skeleton)
    graph_mat  = csr_matrix.toarray(graph)

    #create an object of class Skeleton to get all stuff cool
    data_skeleton = Skeleton(skeleton)
    paths = csr_matrix.toarray(data_skeleton.paths).astype(int)

    return graph, coordinates.astype(int), degrees, graph_mat, data_skeleton, paths


def skeleton_cleaning(paths, coordinates, degrees, skeleton, corr_fct = 0.1):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Correct the skeleton of its little error
    P is the number of path ( link between 2 key points)
    N is the number of skeleton's points
    L - Length of the image
    H - Height of the image

    Input:
        paths(np.ndarray): [P x N] 
        coordinates(np.ndarray): [N x 2]
        degrees(np.ndarray): [H x L]
        skeleton(np.ndarray): [H x L]
        corr_fct(float):
 
    Output: 
        skeleton_corr(np.ndarray): [H x L]
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    skeleton_corr = np.copy(skeleton)
    lengths = np.sum(paths, 1)
    max_len = max(lengths)
    
    #obtain id of the path which are not enough long to be real path
    small_path = np.where(lengths < corr_fct*max_len )[0].tolist()
    
    for id_path in small_path:
        id_pxls = np.where(paths[id_path] != 0)[0]
        
        # save degrees of each point and see how many end-point there is
        # we only want to erase paths with at least one tip
        tuple_of_tuples = tuple(map(tuple, coordinates[id_pxls].astype(int)))
        nb_pt = len(tuple_of_tuples)
        deg = np.zeros(nb_pt)
        for i in range(nb_pt):
            deg[i] = degrees[tuple_of_tuples[i]]
        
        condit = np.sum(np.where(deg==1,1,0))
        if condit == 0:
            continue
        
        # modify the skeleton image to re-analyse it afterward only points with
        # with 1 or 2 neighbors are deleted to avoid disconnection in the skeleton 
        for idx in id_pxls:
            coord = tuple(coordinates[idx].astype(int))
            
            if degrees[coord] < 3:
                skeleton_corr[coord] = 0
                
    return skeleton_corr


def skeleton_cleaning_compl(paths, coordinates, degrees, skeleton):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Complete a first coorection of the skeleton and its little error.
    Here the correction is specific for the little connection with end-points 
    or points cluster (path composed of 3 points).
    
    P - number of path ( link between 2 key points)
    N - number of skeleton's points
    L - Length of the image
    H - Height of the image

    Input:
        paths(np.ndarray): [P x N] 
        coordinates(np.ndarray): [N x 2]
        degrees(np.ndarray): [H x L]
        skeleton(np.ndarray): [H x L]
    
    Output: 
        skeleton_corr(np.ndarray): [H x L]
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    skeleton_corr = np.copy(skeleton)
    lengths = np.sum(paths, 1)
    
    error_path = np.where(lengths < 4)[0].tolist()
    
    for id_path in error_path:
        id_pxls = np.where(paths[id_path] == 1)[0]
        
        #check if there is risk to break the connectivity
        ecart = np.zeros(len(id_pxls))
        for i in range(len(id_pxls)):
            ecart[i] = np.linalg.norm( coordinates[id_pxls[i-1]]
                                      - coordinates[id_pxls[i]])
        condit = max(ecart) < 1.7

        for idx in id_pxls:
            coord = tuple(coordinates[idx].astype(int))
            if (degrees[coord] < 3) and condit :
                skeleton_corr[coord] = 0
     
    return skeleton_corr       


def complexity_control_2( paths, coordinates, graph_mat, dist, cmplx_coef = 1):
    '''
    Control the complexity of the system by adding point between the key points
    
    P - number of paths ( link between 2 key points) 
    N - number of skeleton's points 
    L - Length of the image 
    H - Height of the image 
    Pn - number of new paths afterward 
    
    input:
        paths (np.ndarray): [P x N] list of the path pts coordinates
        
        coordinates (np.ndarray): [N x 2]
        
        graph_mat (np.ndarray): [N+1 x N+1]
        
        dist (np.ndarray): [L x H]
        
        cmplx_coef (int): [int/float] nb of new pt between the fixed one
        
    output:
        pts_id_new (np.ndarray): [Pn x 2]
        path_width (np.ndarray): [Pn x 1]
    '''
    
    nb_path = len(paths)
    
    path_width = np.zeros(nb_path*(cmplx_coef+1)).astype(int)
    pts_id_new = []
    path_max   = np.max(np.sum(paths, 1))
    path_width_test = []

    for n in range(nb_path):
        # define a list of the coordinates of all the points of a path
        path_pts_id = np.where(paths[n])
        path_pts    = coordinates[path_pts_id].astype(int)
    
        # define a little matrix of the relations  
        graph_mat_path = graph_mat.transpose()[path_pts_id].transpose()[path_pts_id]
        
        #ids of the little matrix (graph_mat_path) /!\
        sort_id          = sort_path_pt(graph_mat_path)
        path_pts_sort_id = np.asarray(path_pts_id[0])[sort_id]
        path_pts_sort    = path_pts[sort_id]
        

        ############################## fct segment #########################
        # give the list of the new key pts and the index it cooresponds in
        # the list "path_pts_sort_id"
        path_ky_pts_id_new, index = path_segmentation_2(path_max,
                                                        path_pts_sort_id,
                                                        cmplx_coef)
        
        pts_id_new = np.append(pts_id_new, path_ky_pts_id_new)
        ####################################################################
        
        for i in range(len(path_ky_pts_id_new)):
            width = medium_width( dist, path_pts_sort[index[i]:index[i+1]] )
            path_width_test = np.append(path_width_test, width)
            
        path_width = path_width_test
        
    return pts_id_new.reshape(-1,2).astype(int), path_width



def sort_path_pt(graph_mat_path):
    '''
    Sort the id to have them in the right order
    
    N - number of path's points
    
    input:
        graph_mat_path (np.ndarray): [N x N]
        
    output:
        pt_sort (np.ndarray): [N x 1]
    '''
    
    #troncate value to be able to count the nb of link by point
    graph_int = np.where( graph_mat_path != 0, 1, 0)

    #find the extremities
    extrem_id = np.where(np.sum(graph_int, axis = 0) == 1 )[0]
    
    # Need a start and an end
    if len(extrem_id) != 2:
        print("\n/!\ Data corrupted: there is more or less than 2 extremities to a path/!\ \n")
        sys.exit()
        return None
    
    pt_sort = np.zeros( len(graph_int) )
    
    old_pt = next_pt = extrem_id[0]
    for i in range( len(graph_int)-1 ):
        #find which point is adjacent to the studied point
        temp = np.where(graph_int[next_pt] == 1)[0]
        
        # take the point which is not equal to the previous one
        if (temp[0] == old_pt):
            pt_sort[i] = next_pt
            old_pt     = next_pt
            next_pt    = temp[1]
        else:
            pt_sort[i] = next_pt
            old_pt     = next_pt
            next_pt    = temp[0]
    
    pt_sort[i+1] = extrem_id[1]
    
    return pt_sort.astype(int)


def path_segmentation_2(path_max, sort_id, cmplx_coef):
    """
    Generate a list of the new key pts and the indices it cooresponds in the
    list "path_pts_sort_id"

    N - number of points which compose the path

    Parameters
    ----------
    path_max : int
        number maximum a path can be segmented
    sort_id : np.ndarray of int
        [N x 1] - matrix with the id of the path points sorted.
    cmplx_coef : int
        complexity coeficient which define the number of new points have
        to be generate on a path.

    Returns
    -------
    path_ky_pts_id_new : np.ndarray of int
        DESCRIPTION.

    """
        
    nb = len(sort_id)
    step = path_max/(cmplx_coef+1)
    
    if nb%step < step/2:
        step = step + nb%step

    path_ky_pts_id_new = []
    index = []
    
    i = 0
    while i + step < (nb-1):
        
        index = np.append(index, int(i))
        path_ky_pts_id_new = np.append(path_ky_pts_id_new,
                                       np.array([sort_id[int(i)],
                                                 sort_id[int(i+step)] ]) )
        
        i = i + step
        
    index = np.append(index, i)
    path_ky_pts_id_new = np.append(path_ky_pts_id_new,
                                   np.array([sort_id[int(i)],
                                             sort_id[-1] ]) )
    index = np.append(index, nb-1)

    return path_ky_pts_id_new.reshape(-1,2).astype(int), index.astype(int)


def medium_width( dist, data_path):
    """
    Calculate the mean of the path width
    
    L - Length of the image
    
    H - Height of the image
    
    N is the number of skeleton's points

    Parameters
    ----------
    dist : np.ndarray of float
        [LxH] - array of all the dist from the boudaries (like an image)
    data_path : np.ndarray of int
        [Nx2] - list of coordinates of all the point of a path

    Returns
    -------
    width_mean : float
        the mean of all the width along the studied segment

    """  
    width_mean = np.mean(dist[data_path[:,0], data_path[:,1]])
    
    return width_mean


if __name__ == '__main__':
    print('skeleton_analysis_functions executed')