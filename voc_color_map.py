# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:24:40 2016

@author: yi
"""

import numpy as np
from matplotlib import colors

def labelcolormap(N=256):
    """
    VOCLABELCOLORMAP Creates a label color map such that adjacent indices have different
    colors.  Useful for reading and writing index images which contain large indices,
    by encoding them as RGB images.

    CMAP = VOCLABELCOLORMAP(N) creates a label color map with N entries.
    """
    cmap = np.zeros((N,3))
    for i in xrange(N):
        id = i
        r, g, b = 0, 0, 0
        for j in xrange(8):
            r = r | (id >> 0 & 1) << (7 - j)
            g = g | (id >> 1 & 1) << (7 - j)
            b = b | (id >> 2 & 1) << (7 - j)
            id = id >> 3
        cmap[i, 0], cmap[i, 1], cmap[i, 2] = r, g, b
    cmap = cmap / 255.0;
    cmap = array2cmap(cmap)

    return cmap

def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in xrange(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in xrange(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in xrange(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in xrange(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in xrange(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in xrange(2 * N)])

    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)

if __name__ == '__main__':
    cmap = labelcolormap(21)
    print cmap
