# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:59:23 2021

@author: admin
"""

import numpy as np


def print_info(data, namespace ):    
    print( [name for name in namespace if namespace[name] is data])    
    print(type(data))
    print(np.shape(data),'\n')


''' pts et coordinates = array [Nx2]'''
def coord_2_id( pts, coordinates):
    
    pt_id = []
    
    for i in range(len(pts)):
        #find the id of the node i
        temp1 = np.array(np.where(coordinates[1:,0] == pts[i,0]))
        temp2 = np.array(np.where(coordinates[1:,1] == pts[i,1]))
        
        for x in range( temp1.shape[1]):
            for y in range( temp2.shape[1]):
                if ( temp1[0,x] == temp2[0,y]):
                    break
            if ( temp1[0,x] == temp2[0,y]):
                break
        
        #Import to add the "+1" to have the id and not the index
        pt_id = np.append(pt_id, temp1[0,x]+1 )
    
    
    return pt_id.astype(int)


'''
Calculate the mean of the distance 
input:
    matrice dist with all the dist from th boudaries (like an image)
    data_path [Nx2]: list of coordinates of all the point of a path
'''
def medium_width( dist, data_path):
    
    width_mean = np.mean(dist[data_path[:,0], data_path[:,1]])
    
    return width_mean








##############################################################################

''' Find all the pts of a path

input:
    rlt_mat [N+1,N+1] using index
    links [P,2] using index
    paths = list indice

 '''
def path_2_pts( rlt_mat, links, path = 0):
    
    path_pts = []
    next_pt = links[path,0]
    
    while next_pt != links[path,1]:
        
        next_pt
    
    return path_pts

def vite(density, elements):
    results = np.zeros((142,71))
    
    for i in range(142):
        for j in range(71):
            ind = i+j*142
            
            results[i,j] = density[ind]
            
    return results

###

def array_2_csv(data):
    #nodes = where
    print( "not finished yet")
    

 
if __name__ == '__main__':
    print('fct_utile executed')
    
    
        
''' Take mesh data to transform them into a pixelized image
    Nodes:
    Density:
    Elements:
    Resolution: values in ??

'''

'''
def mesh_2_pixel_2D(nodes, density, elements, resolution = 0.1):
    nodes    = np.asarray(nodes, dtype = np.float32)
    density  = np.asarray(density, dtype = np.float32)
    elements = np.asarray(elements, dtype = np.float32)
    
    nb_element = density.shape[0]
    
    # 1. check extremum
    max_val = np.amax(nodes, axis=0)
    min_val = np.amin(nodes, axis=0)
    print(max_val)
    print(min_val)
    
    # 2. Resolutions
    x_size = (max_val[0]- min_val[0])//0.1
    y_size = (max_val[1]- min_val[1])//0.1
    
    for i in range(nb_element):
        for j in range(4):
            #model_resol = nodes  
    
    # 3. Pixels calculation
    img_pixelized = np.zeros((x_size,y_size))

    for x in range(x_size):
        for y in range(y_size):
            
            
            #np.where( nodes[][])
            #tot = np.sum(  )
            img_pixelized[x,y] = 4
            
            
    ########
    the last element of each dimension will be smaller than the others
    if the range is not perfectly divided by the resolution.
    then, there is 2 options:
        - delete the last column
        - keep it and accept it will be not as correcte than the others
    '''

    #return img_pixelized