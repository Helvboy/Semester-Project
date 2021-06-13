# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:25:40 2021

@author: admin
"""

import numpy as np
from matplotlib import pyplot as plt


def binarize(data, inverse = False, plot = False):
    """
    Binarize an array with a half threshold

    L - Long of the image
    
    H - Height of the image
    
    Parameters
    ----------
    data : np.ndarray of float
        [LxH] - Data to binarize or inverse.
    inverse : Bool, optional
        Indicate if the intensity imported data must be inverted. The default is False.
    plot: Bool, optionnal
        Plot the state of the process or not. The default is False.

    Returns
    -------
    results : np.ndarray of float
        [LxH] - Data binarized with values 0 or 1.

    """

    # plt.matshow(data.transpose()[::-1])
    # plt.title("Raw data\n")
                                                              #print

    if inverse == True:
        results = np.where(data > np.max(data)/2, 0, 1 )
    else:
        results = np.where(data > np.max(data)/2, 1, 0 )
    
    if plot:
        plt.matshow(results.transpose()[::-1])
        plt.title("Result of binarization\n")
    
    return results

if __name__ == '__main__':
    print( 'binarization executed')