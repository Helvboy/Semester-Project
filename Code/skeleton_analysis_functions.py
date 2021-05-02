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

from fct_utile import *


def key_points_deduction(coordinates, degrees, details = False):

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
    

'''
# find the 2 extremities of a path

#input:
    paths [P,N]
    ky_pts_id [?] list of the key points id
    
    '''
def key_points_relation(paths, ky_pts_id):
    
    nb_path = len(paths)
    nb_ky_pts = len(ky_pts_id)
    
    #create the array where relation between key points will be stored
    ky_pts_link = np.zeros( ( nb_ky_pts, nb_ky_pts) ).astype(int)
    
    #create a list with the relation between 2 key points
    links = np.zeros((2,nb_path)).transpose().astype(int)
    
    for n in range(nb_path):
        for i in range(nb_ky_pts):
            if paths[n,ky_pts_id[i]] == 0:
                continue
            
            for j in range(nb_ky_pts):
                if paths[n,ky_pts_id[j]] == 0:
                    continue
                
                if i != j:
                    ky_pts_link[i,j] = ky_pts_link[j,i] = 1
                    links[n] = np.array([i,j]).astype(int)
                    
    return ky_pts_id[links]


'''
#Control the complexity of the system by adding point between the key points

#input:
    
    
    path_pts - [Np] list of the path pts coordinates
    graph_mat_path - [array of relation between each pts of the path]
    cmplx_coef - [int/float] nb of new pt between the fixed one
'''
def complexity_control_2( paths, coordinates, graph_mat, dist, cmplx_coef = 1):
    
    nb_path = len(paths)
    
    path_width = np.zeros(nb_path).astype(int)
    path_width_test = np.zeros(nb_path*(cmplx_coef+1)).astype(int)
    pts_id_new = []

    for n in range(nb_path):
        #4.1 needed: matrice graph for 1 path, list coordonnée pt du path
        path_pts_id = np.where(paths[n])
        path_pts    = coordinates[path_pts_id].astype(int)
    
        # define a little matrix of the relation  
        graph_mat_path = graph_mat.transpose()[path_pts_id].transpose()[path_pts_id]
        
        #id du graph /!\
        sort_id          = sort_path_pt(graph_mat_path)
        path_pts_sort_id = np.asarray(path_pts_id[0])[sort_id]
        path_pts_sort    = path_pts[sort_id]
        

        ############################## fct segment #########################
        # give the list of the new key pts and the index it cooresponds in
        # the list "path_pts_sort_id"
        path_ky_pts_id_new, index = path_segmentation(graph_mat_path,
                                                      path_pts_sort_id,
                                                      cmplx_coef)
        pts_id_new = np.append(pts_id_new, path_ky_pts_id_new)

        ############################## fct segment#########################
        
        for i in range(cmplx_coef+1):
            path_width_test[n*(cmplx_coef+1)+i] \
                = medium_width( dist,path_pts_sort[index[i]:index[i+1]])
            
        
    return pts_id_new.reshape(-1,2).astype(int), path_width_test

def tempo_a_degager_apre():
        for ends in path_ky_pts_id_new:
            #path_pts_seg = coordinates[ends[0]:ends[1]]
            
            path_width[n] = skf.medium_width( dist, path_pts)
            
            #path_width[n] = medium_width( dist, path_pts)
        for i in range(cmplx_coef+1):
            #path_width_test[n*(cmplx_coef+1)+i] = path_width[n]
            path_width_test[n*(cmplx_coef+1)+i] = medium_width( dist,path_pts_sort[index[i]:index[i+1]])
          
            
            
            ky_pts_id_new
            
        
        #4.2 faire moyenne

        path_pts_id_new = skf.complexity_control(path_pts, graph_mat_path, 2)
        path_pts_new    = coordinates[path_pts_id].astype(int)
        
        #find the width of one path
        path_width[n] = skf.medium_width( dist, path_pts)
        #4.3 
        #4.4 


'''
#Segment path in several small one

#input:
    path_pts - [Np] list of the path pts coordinates
    graph_mat_path - [array of relation between each pts of the path]
    cmplx_coef - [int/float] nb of new pt between the fixed one
'''
def complexity_control( path_pts, graph_mat_path, cmplx_coef):
    
    pt_sort = sort_path_pt(graph_mat_path)
    
    nb = len(graph_mat_path)
    
    step = int( nb/(cmplx_coef+1) )
    
    links = []
    
    i = 0
    while i + step < (nb-1):
        
        links = np.append(links, np.array([pt_sort[i], pt_sort[i+step] ]) )
        
        i = i + step
        
    links = np.append(links, np.array([pt_sort[i], pt_sort[-1] ]) ).astype(int)
    
    return links.reshape(-1,2)


'''
Sort the id to have them in the right order

input:
    matrice dist with all the coordinates pts of the paths
    path_data [Nx2]
    
output:
    
'''
def sort_path_pt(graph_mat_path):
    #troncate value to be able to count the nb of link by point
    graph_int = graph_mat_path.astype(int)

    #find the extremities
    extrem_id = np.where(np.sum(graph_int, axis = 0) == 1 )[0]
    
    # Need a start and an end
    if len(extrem_id) != 2:
        print("Data corrupted: there is more or less than 2 edges")
        
        return None
    
    pt_sort = np.zeros(len(graph_int))
    
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



def path_segmentation(graph_mat_path, sort_id, cmplx_coef):
            #input: graph_mat_path, sort_id, cmplx_coef
            
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

if __name__ == '__main__':
    print('skeleton_analysis_functions executed')