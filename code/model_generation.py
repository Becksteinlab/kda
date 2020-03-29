# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 03/27/2020
# Kinetic Model Generation

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#===============================================================================
#== Functions ==================================================================
#===============================================================================

def pos_3(center=[0, 0], radius=10, N=3):
    """Generates positions for nodes
    """
    angle = np.pi*np.array([1/2, 7/6, 11/6])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def edges_3(G, rates, key='k'):
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (0, 2, rates[4]),
                               (2, 0, rates[5])], weight=key)

def pos_4(center=[0, 0], radius=10, N=4):
    """Generates positions for nodes
    """
    angle = np.pi*np.array([1/4, 3/4, 5/4, 7/4])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def edges_4(G, rates, key='k'):
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (2, 3, rates[4]),
                               (3, 2, rates[5]),
                               (3, 0, rates[6]),
                               (0, 3, rates[7])], weight=key)

def pos_4wl(center=[0, 0], radius=10, N=4):
    """Generates positions for nodes
    """
    angle = np.pi*np.array([1/4, 3/4, 5/4, 7/4])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def edges_4wl(G, rates, key='k'):
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (2, 3, rates[4]),
                               (3, 2, rates[5]),
                               (3, 0, rates[6]),
                               (0, 3, rates[7]),
                               (1, 3, rates[8]),
                               (3, 1, rates[9])], weight=key)

def pos_5wl(center=[0, 0], radius=10):
    """Generates positions for nodes
    """
    h = radius*np.sqrt(3)/2 # height of equilateral triangle
    pos = {0 : [0, h],
           1 : [-radius/2, 0],
           2 : [radius/2, 0],
           3 : [-radius/2, -radius],
           4 : [radius/2, -radius]}
    return pos

def edges_5wl(G, rates, key='k'):
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (0, 2, rates[4]),
                               (2, 0, rates[5]),
                               (1, 3, rates[6]),
                               (3, 1, rates[7]),
                               (2, 4, rates[8]),
                               (4, 2, rates[9]),
                               (3, 4, rates[10]),
                               (4, 3, rates[11])], weight=key)

def pos_6(center=[0, 0], radius=10):
    """
    Generates positions of nodes for a hexagon in the xy-plane

    Parameters
    ----------
    center : list
        Defines the center of the hexagon in the xy-plane
    radius : int
        The radius of the hexagon, from center to node
    """
    N = 6                                                       # number of states/nodes
    angle = np.pi*np.array([1/2, 13/6, 11/6, 3/2, 7/6, 5/6])    # angles go CCW
    array = np.zeros((N, 2))                                    # Empty 2D array of shape (6x2)
    for i in range(N):                                          # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}                                                    # empty dict for positions to go in
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def edges_6(G, rates, key='k'):
    """
    Generates edges 6 state model
    """
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (2, 3, rates[4]),
                               (3, 2, rates[5]),
                               (3, 4, rates[6]),
                               (4, 3, rates[7]),
                               (4, 5, rates[8]),
                               (5, 4, rates[9]),
                               (5, 0, rates[10]),
                               (0, 5, rates[11])], weight=key)
