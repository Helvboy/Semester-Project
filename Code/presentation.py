# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:50:54 2021

@author: Eloi Schlegel
"""

import numpy as np
import matplotlib.pyplot as plt

def similarity(img1, img2):
    '''
    Give the purcentage of similarity between 2 pictures of D dimensions
    L - Long of the image
    H - Height of the image
    
    input:
        img1(np.ndarray) - [L x H]
        img2(np.ndarray) - [L x H]

    output:
        diff_perc(float)
    '''    
    S = img1.shape
    #L, H = img1.shape
    diff_perc = np.sum(abs(img1 - img2))/(np.prod(S))*100
    
    return diff_perc


def draw_segment(links, width, coordinates, dim, plot = True):
    '''
    Plot and give the matrix of the simplify skeleton
    S - number of segments
    N - number of points
    D - number of dimensions
    
    input:
        links(np.ndarray) - [S x 2]
        width(int/float)
        coordinates(np.ndarray) - [N+1 x D]
        dim(tuple): [2]
        plot(Bool)
        
    output:
        output(np.ndarray) - [L x H]
    '''
    # L, H = np.max(coordinates,0).astype(int)+1 
                                #modify and set dim to the previous matrix
    L, H = dim
    S = len(links)
    
    output = np.zeros((L,H))
    process = 0
    
    #put a minimum width
    
    print("__________")
    for i in range(S):
        pt_a = coordinates[links[i,0]]
        pt_b = coordinates[links[i,1]]
        
        vct_seg = pt_b - pt_a
        vct_seg_n = vct_seg/ np.linalg.norm(vct_seg)

        
        for x in range(L):
            for y in range(H):
                
                pxl_ctr = [x,y]
                vct_pxl_a = pxl_ctr - pt_a
                vct_pxl_b = pxl_ctr - pt_b

                dist = abs(np.cross(vct_seg_n,vct_pxl_a))
                prod1 = np.dot( vct_seg_n, vct_pxl_a )
                prod2 = np.dot( vct_seg_n, vct_pxl_b )

                check = ( (prod1 >= 0) & (prod2 <= 0) ) | ( (prod2 >= 0) & (prod1 <= 0) )

                if (dist <= width[i]) & (check == True):
                    output[x,y] = 1
        if int(i/S*10) >= process:
            process += 1
            print("|", end = '')
    
    print("\n \n")
    
    if plot:
        plt.matshow(output)
        
    return output


def draw_segment_2(links_coor, width, dim, plot = True):
    
    L, H = dim
    S = len(links_coor)
    
    output = np.zeros((L,H))
    process = 0
    
    #put a minimum width
    
    print("__________")
    for i in range(S):
        pt_a = links_coor[i,0:2]
        pt_b = links_coor[i,2:4]
        
        vct_seg = pt_b - pt_a
        vct_seg_n = vct_seg/ np.linalg.norm(vct_seg)

        
        for x in range(L):
            for y in range(H):
                
                pxl_ctr = [x,y]
                vct_pxl_a = pxl_ctr - pt_a
                vct_pxl_b = pxl_ctr - pt_b

                dist = abs(np.cross(vct_seg_n,vct_pxl_a))
                prod1 = np.dot( vct_seg_n, vct_pxl_a )
                prod2 = np.dot( vct_seg_n, vct_pxl_b )

                check = ( (prod1 >= 0) & (prod2 <= 0) ) | ( (prod2 >= 0) & (prod1 <= 0) )

                if (dist <= width[i]) & (check == True):
                    output[x,y] = 1
        if int(i/S*10) >= process:
            process += 1
            print("|", end = '')
    
    print("\n \n")
    
    if plot:
        plt.matshow(output)
    
    return output

if __name__ == '__main__':
    print( 'presentation executed')