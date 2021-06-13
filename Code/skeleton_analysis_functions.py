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
#sys.path.append( path_code ) 'works too, let it there in case' 

from convertor import coord_2_id

from skan import skeleton_to_csgraph
from skan import Skeleton

from matplotlib import pyplot as plt
from skan import draw

from scipy.sparse import csr_matrix

##############################################################################

def skeleton_analysis(skel, dist, cmplx_coef, clean_coef, plot = False):
    """
    Analyse the skeleton and generate a new simplified skeleton depending of
    the complexity factor

    Parameters
    ----------
    skel : np.ndarray of int
        Array of the skeleton image
    dist : np.ndarray of float
        Distance map from the boarder of the binarized image
    cmplx_coef : int
        Define the complexity to apply to the skeleton.
    clean_coef : float, optional
        Define how much the skeleton should be cleanned.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    '''
   
    
    Parameters
    ----------
    skel : np.ndarray
        DESCRIPTION.
    dist : np.ndarray
        DESCRIPTION.
    cmplx_coef : int
        DESCRIPTION.
    corr_fct : TYPE
        DESCRIPTION.
    plot : Bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    skeleton = np.copy(skel)
    # plt.matshow(skeleton)

   
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
    if clean_coef > 0.03:                                                        # coef to adjust
        skeleton_corr2 = skeleton_cleaning_compl(paths, coordinates,
                                                 degrees, skeleton_corr)

        skeleton = skeleton_corr2
        
        graph, coordinates, degrees, graph_mat, data_skeleton, paths \
            = skeleton_information(skeleton)

                                                        #  besoin de le faire plusieurs fois ?
        skeleton_corr2 = skeleton_cleaning_compl(paths, coordinates,
                                                 degrees, skeleton)
        
        skeleton = skeleton_corr2
        
        graph, coordinates, degrees, graph_mat, data_skeleton, paths \
            = skeleton_information(skeleton)
        
    else:
        skeleton = skeleton_corr
    
    if plot:
        plt.matshow(skeleton.transpose()[::-1])
        plt.title("Result of the skeleton cleaning\n")


    ##############################################
                    ### Next ###
    ##############################################
    #1. Find end and joint points
    #2. Find index end and joint points    
    #ky_pts, ky_pts_id = skf.key_points_deduction(coordinates, degrees)
    
    # ky_pts, ky_pts_id, end_pts, inter_pts, id_end_pts, id_inter_pts = key_points_deduction(coordinates, degrees, True)

    #3. Find relations between key points (end et joint points)   
    #give the index of the pt couple (first case = complexity 0)
    
    # links = key_points_relation(paths, ky_pts_id)

    ##############################################
#4. manage the complexity and the width of each segment
    
    new_links, path_width = complexity_control_2(paths, coordinates,
                                                 graph_mat, dist, cmplx_coef)

    links = new_links

    if plot:
        plt.figure(42)
        #plot skeleton
        #plt.scatter(coordinates[:,0], coordinates[:,1] )
        
        #plot key points
        # plt.scatter(end_pts[:,1], end_pts[:,0] )
        # plt.scatter(inter_pts[:,1], inter_pts[:,0] )
        

        
        #plot the each link
        for i in range(len(links)):

            plt.plot(coordinates[links[i],0],
                     coordinates[links[i],1],
                     linewidth = (path_width[i]) )
                
    # plt.scatter(end_pts[:,0], end_pts[:,1] )
    # plt.scatter(inter_pts[:,0], inter_pts[:,1] )

            

# plot the graph of the systeme, to delete
    if 0:
        plt.figure(3)
        pxl_g1, coordinates1, degrees1 = skeleton_to_csgraph(skeleton)
        draw.overlay_skeleton_networkx(pxl_g1, coordinates1, image=skeleton)
     
    return coordinates, new_links, path_width, skeleton


def skeleton_information(skeleton):
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Generate all the needed variable containing intersting informations
    P is the number of path ( link between 2 key points)
    N is the number of skeleton's points
    L - Length of the image
    H - Height of the image

    Input:
        skeleton(np.ndarray): [H x L]
    
    Output:
        graph(sparse.csr.csrmatrix)
        coordinates(np.ndarray): [N x 2]
        degrees(np.ndarray): [H x L]
        graph_mat(np.ndarray): [N+1 x N+1]
        data_skeleton(csr.Skeleton): obj class Skeleton(skan)
        paths(np.ndarray): [P x N]
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
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
    # skeleton_corr = np.copy(skeleton)
    # lengths = np.sum(paths, 1)
    
    # error_path = np.where(lengths < 4)[0].tolist()
    
    # for id_path in error_path:
    #     id_pxls = np.where(paths[id_path] == 1)[0]
        
    #     for idx in id_pxls:
            
    #         coord = tuple(coordinates[idx].astype(int))
            
    #         if degrees[coord] == 2:
    #             skeleton_corr[coord] = 0
                
    # return skeleton_corr

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



def key_points_deduction(coordinates, degrees, details = False):
    '''
    Define the key points of the skeleton which are end points and joints 
    N - number of skeleton's points 
    L - Length of the image 
    H - Height of the image 
    K - number of key points

    Input:  
        coordinates(np.ndarray): [N x 2]
        degrees(np.ndarray): [H x L]
        details(Bool): 
    
    Output: 
        ky_pts(np.ndarray): [K x 2]
        ky_pts_id(np.ndarray): [1 x K]
        
        end_pts, inter_pts, id_end_pts, id_inter_pts (in option)
    '''

    # Calculate the end-points and intersection points
    end_pts   = np.array(np.where(degrees == 1)).transpose()   
    inter_pts = np.array(np.where(degrees == 3)).transpose()
    
    ky_pts    = np.append(end_pts, inter_pts, axis = 0)

    # Find index end and joints points
    id_end_pts   = coord_2_id( end_pts, coordinates)
    id_inter_pts = coord_2_id( inter_pts, coordinates)

    ky_pts_id    = np.append(id_end_pts, id_inter_pts)
    
    if details:
        return ky_pts, ky_pts_id, end_pts, inter_pts, id_end_pts, id_inter_pts
    else:
        return ky_pts, ky_pts_id



def key_points_relation(paths, ky_pts_id):
    '''
    Define the couple of id points which defines a path
    
    P - number of path ( link between 2 key points) 
    N - number of skeleton's points 
    K - number of key points 
    
    Input:  
        paths(np.ndarray): [P x N] 
        ky_pts_id(np.ndarray): [K x 1]
    
    Output: 
        id_extr_path (np.ndarray): [P x 2]
    '''
    
    nb_path = len(paths)
    nb_ky_pts = len(ky_pts_id)
    
    #create the array where relation between key points will be stored
    ky_pts_link = np.zeros( ( nb_ky_pts, nb_ky_pts) ).astype(int)
    
    #create a list with the relation between 2 key points
    links = np.zeros((2,nb_path)).transpose().astype(int)
    
    for n in range(nb_path):
        #look for the 1st extremity
        for i in range(nb_ky_pts):
            if paths[n,ky_pts_id[i]] == 0:
                continue
            
            #look for the 2nd extremity
            for j in range(nb_ky_pts):
                if paths[n,ky_pts_id[j]] == 0:
                    continue
                
                if i != j:
                    ky_pts_link[i,j] = ky_pts_link[j,i] = 1
                    links[n] = np.array([i,j]).astype(int)
                    
    id_extr_path = ky_pts_id[links]     
    return id_extr_path



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
    '''
    Generate a list of the new key pts and the indices it cooresponds in the
    list "path_pts_sort_id"

    Parameters
    ----------
    path_max : int
        number maximum a path can be segmented.
    sort_id : (np.ndarray): [N x 1]
        matrix with the id of the path points sorted.
    cmplx_coef : int
        complexity coeficient which define the number of new points have
        to be generate on a path.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
        #input: graph_mat_path, sort_id, cmplx_coef
        
    #graph_mat_path sert surement à rien, à remplace ici part sort_id

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


def path_segmentation(graph_mat_path, sort_id, cmplx_coef):                     #should be deleted
            
        #graph_mat_path sert surement à rien, à remplace ici part sort_id
        nb = len(graph_mat_path)
    
        step = nb/(cmplx_coef+1)
    
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
# sort les id des extremités des sub-paths du main path


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