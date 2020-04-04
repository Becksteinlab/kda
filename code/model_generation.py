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

def generate_edges(G, names, vals, name_key='name', val_key='val'):
    """
    Generate edges with attributes 'name' and 'val'.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    names : array
        'NxN' array where 'N' is the number of nodes in the diagram G. Contains
        the names of all of the attributes corresponding to the values in
        'vals' as strings, i.e. [[0, "k12"], ["k21", 0]].
    vals : array
        'NxN' array where 'N' is the number of nodes in the diagram G. Contains
        the values associated with the attribute names in 'names'. For example,
        assuming k12 and k21 had already been assigned values, for a 2 state
        diagram 'vals' = [[0, k12], [k21, 0]].
    name_key : str (optional)
        Key used to retrieve variable names in 'names'. Default is 'name'.
    val_key : str (optional)
        Key used to retrieve variable values in 'vals'. Default is 'val'.
    """
    for i, row in enumerate(vals):
        for j, elem in enumerate(row):
            if not elem == 0:
                attrs = {name_key : names[i, j], val_key : elem}
                G.add_edge(i, j, **attrs)

def pos_3(center=[0, 0], radius=10):
    """
    Generate node positions for 3 state model

    Parameters
    ----------
    center : list (optional)
        Defines the center of the triangle in the xy-plane. Default is
        (x, y) = (0, 0)
    radius : int (optional)
        The radius of the triangle, from center to node. Default is 10.

    Returns
    -------
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    N = 3
    angle = np.pi*np.array([1/2, 7/6, 11/6])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def pos_4(center=[0, 0], radius=10):
    """
    Generate node positions for 4 state model

    Parameters
    ----------
    center : list (optional)
        Defines the center of the square in the xy-plane. Default is
        (x, y) = (0, 0)
    radius : int (optional)
        The radius of the square, from center to node. Default is 10.

    Returns
    -------
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    N = 4
    angle = np.pi*np.array([1/4, 3/4, 5/4, 7/4])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def pos_4wl(center=[0, 0], radius=10):
    """
    Generate node positions for 4 state model with leakage

    Parameters
    ----------
    center : list (optional)
        Defines the center of the square in the xy-plane. Default is
        (x, y) = (0, 0)
    radius : int (optional)
        The radius of the square, from center to node. Default is 10.

    Returns
    -------
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    N = 4
    angle = np.pi*np.array([1/4, 3/4, 5/4, 7/4])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def pos_5wl(center=[0, 0], radius=10):
    """
    Generate node positions for 5 state model with leakage

    Parameters
    ----------
    center : list (optional)
        Defines the center of the pentagon in the xy-plane. Default is
        (x, y) = (0, 0)
    radius : int (optional)
        The radius of the pentagon, from center to node. Default is 10.

    Returns
    -------
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    h = radius*np.sqrt(3)/2 # height of equilateral triangle
    pos = {0 : [0, h],
           1 : [-radius/2, 0],
           2 : [radius/2, 0],
           3 : [-radius/2, -radius],
           4 : [radius/2, -radius]}
    return pos

def pos_6(center=[0, 0], radius=10):
    """
    Generates positions of nodes for a hexagon in the xy-plane

    Parameters
    ----------
    center : list (optional)
        Defines the center of the hexagon in the xy-plane. Default is
        (x, y) = (0, 0)
    radius : int (optional)
        The radius of the hexagon, from center to node. Default is 10.

    Returns
    -------
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
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
