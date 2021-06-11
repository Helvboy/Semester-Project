# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:07:38 2021

@author: admin
"""

import numpy as np


def joints_correction(links_coor, links, widths, display= False, plot = False):
    """
    Extend the elements ends of a structure in order to improve the connection
    between them
    
    E - number of elements
    
    D - number of dimensions 

    Parameters
    ----------
    links_coor : np.ndarray of int
        [Ex(D*2)] - Array containing the coordinates of each element end 
    links : np.ndarray of int
        [Ex2] - Indices of the elements end points 
    widths : np.ndarray of float
        [Ex1] - Width of each element
    display: Bool, optionnal
        Display message or not. The default is False.
    plot: Bool, optionnal
        Plot the state of the process or not. The default is False.

    Returns
    -------
    links_coor_ext : np.ndarray of float
        [Ex(D*2)] - a version of link_coor with some points modified in the order
        to improve the connectivity

    """
    # gives indices of the points which have to be modified with the path
    # to which it belongs
    joint_pts_id, paths_invlv = intersection_finder(links)
    
    # determines how long the path should be extended with indices of the paths
    ext_dist, path_2_ext = width_comparator( paths_invlv, widths)
    
    # gives back the new link_coor with the modified points
    links_coor_ext = new_points_generator(links_coor, links, ext_dist,
                                          path_2_ext, joint_pts_id)    
    
    if display:
        print("Extension done\n")
    return links_coor_ext
    

def intersection_finder(links):
    """
    Deduce the points which belong to two or more paths and give also the paths
    involved for each points
    
    E  - number of element 
    Nj - number of joint points 
    
    Parameters
    ----------
    links : np.ndarray of int
        [Ex2] - Indices of the elements end points 

    Returns
    -------
    pts_inter : list of int
        [Nj] - Array with the indices of the joint points 
    paths_invlv : list of np.ndarray
        [Nj] - list containing the pats indices where the point belongs 

    """
    pts = np.unique(links)
    
    pts_inter = []
    paths_invlv = []
    
    for pt in pts:
        concern_paths = np.where(links == pt )[0]
        
        if len(concern_paths) > 1:
            pts_inter.append(pt)
            paths_invlv.append(concern_paths)
    
    return pts_inter, paths_invlv


def width_comparator( paths_invlv, widths):
    """
    Compare the widths of each paths and select the path with the bigger width
    to extend by the 2nd bigger width
    
    Nj - number of joint points 
    E  - number of elements 

    Parameters
    ----------
    paths_invlv : list of np.ndarray
        [Nj] - list containing the pats indices where the point belongs 
    widths : np.ndarray of float
        [Ex1] - Array with the distance of each path 

    Returns
    -------
    ext_dist : list of float
        [Nj] - distance extensions to apply to each concerned path 
    path_2_ext : list of int
        [Nj] - list containing the indices of the path to extend 

    """
    
    path_2_ext = []
    ext_dist = []
    
    for i in range(len(paths_invlv)):
        paths = paths_invlv[i]
        paths_w = widths[paths]
        
        # take the bigger path to extend it by the 2nd bigger width
        ext_dist.append(np.sort(paths_w)[-2])
        path_2_ext.append(paths[np.argmax(paths_w)])
    
    return ext_dist, path_2_ext


def new_points_generator(links_coor, links, ext_dist, path_2_ext, inter_pts_id):
    """
    From the given informations, modified links_coor by changing the coordinates
    of elements the end points which have to be extend by the ext_dist 
    
    Nj - number of joint points
    
    E  - number of elements
    
    D  - number of dimensions 

    Parameters
    ----------
    links_coor : np.ndarray of float
        [ExD*2] - Array containing the coordinates of each element end 
    links : np.ndarray of int
        [Ex2] - Indices of the elements end points 
    ext_dist : list of float
        [Nj] - distance extensions to apply to each concerned path 
    path_2_ext : list of int
        [Nj] - list containing the indices of the path to extend 
    inter_pts_id : list of int
        [Nj] - Array with the indices of the joint points 
    
    Returns
    -------
    links_coor_ext : np.ndarray of float
        [ExD*2] - a version of link_coor with some points modified 

    """
    links_coor_ext = np.copy(links_coor)
    
    D = int(links_coor.shape[1]/2)
    
    for i in range(len(path_2_ext)):
        path = path_2_ext[i]
        idx = np.where(links[path] == inter_pts_id[i])[0][0]

        links_coor_ext[path,idx*D:idx*D+2] = new_ext_coord(links_coor[path],
                                                           idx, ext_dist[i])
    
    return links_coor_ext


def new_ext_coord(link_coor, idx, ext_dist):
    """
    Calculate the new coordinates of a point after being extended 
    
    D - number of dimensions 

    Parameters
    ----------
    link_coor : np.ndarray of float
        [D*2x1] - Coordinates of the 2 end points of the element 
    idx : int
        indices telling which end points of the element to extend 
    ext_dist : float
        extension distance to apply to the point 

    Returns
    -------
    coord : np.ndarray of float
        [Dx1] - Coordinates of the new point 

    """
    D = int(link_coor.shape[0]/2)
    
    vect = link_coor[D:] - link_coor[0:D]
    ext = (vect/np.linalg.norm(vect))*ext_dist
    
    if idx == 1:
        coord = link_coor[D:]  + ext*0.5
    else:
        coord = link_coor[0:D] - ext*0.5
        
    return coord

        
if __name__ == '__main__':
    print( 'joints_extension executed')