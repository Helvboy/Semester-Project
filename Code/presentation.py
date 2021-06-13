# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:50:54 2021

@author: Eloi Schlegel
"""

import numpy as np
import matplotlib.pyplot as plt

def similarity(img1, img2, display = False, plot = False):
    """
    Give the purcentage of similarity between 2 pictures of D dimensions

    L - Long of the image
    
    H - Height of the image
    
    Parameters
    ----------
    img1 : np.ndarray of float
        [LxH] - First image to compare
    img2 : np.ndarray of float
        [LxH] - Second image to compare
    Display : Bool, optional
        Display message or not. The default is False.
    plot : Bool, optional
        Display plot or not. The default is False.

    Returns
    -------
    diff_perc : float
        Percentage of similarity

    """
    S = img1.shape

    diff_perc = np.sum(abs(img1 - img2))/(np.prod(S))*100
    
    if display:
        print('There is a difference of', diff_perc, '%')
        
    if plot:
        plt.matshow( 2*np.abs(img1-img2) + img1)
        plt.title("Plot of the error on img1")
        
    return diff_perc


def draw_segment(links_coor, width, dim, display = True, plot = False):
    """
    Plot and give the matrix of the simplify skeleton
    
    S - number of segments
    
    D - number of dimensions

    L - Length of the image
    
    H - Height of the image
    
    Parameters
    ----------
    links_coor : np.ndarray of float
        [Sx(2*D)] - Array containing the coordinates of each element end 
    width : np.ndarray of float
        [S] - Width of each element 
    dim : tuple
        [2] - dimension(s) of the studied domain.
    display : Bool, optional
        Display message or not. The default is True.
    plot : Bool, optional
        Display plot or not. The default is False.

    Returns
    -------
    output : np.ndarray of float
        [LxH] - obtained image with the features

    """
    L, H = dim
    S = len(links_coor)
    
    output = np.zeros((L,H))
    process = 0
    
    #put a minimum width
    
    if display:
        print("Drawing start:")
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
            if display:
                print("|", end = '')
    
    if display:
        print("\n \n")
    
    if plot:
        plt.matshow(output.transpose()[::-1])
        plt.title("Result of the draw_segment\n")
    
    return output

if __name__ == '__main__':
    print( 'presentation executed')