# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Plotting Diagrams
=========================================================================
This file contains a host of functions used for plotting various diagrams, such
as input, partial, flux, and cycle diagrams. Also contains a function to plot
results from ODE solver.

.. autofunction:: draw_diagrams
.. autofunction:: draw_cycles
.. autofunction:: draw_ODE_results

"""

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from .core import construct_cycle_edges, append_reverse_edges


def draw_diagrams(diagrams, pos=None, panel=False, panel_scale=1, font_size=12, cbt=False, path=None, label=None):
    """
    Plots array of diagrams.

    Parameters
    ----------
    diagrams : list cycles or NetworkX MultiDiGraph
        List of diagrams or single diagram to be plotted.
    pos : dict (optional)
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node. Default
        is None, nx.spring_layout() is used.
    panel : bool (optional)
        Tells the function to output diagrams as an 'NxM' matrix of subplots,
        where 'N' and 'M' are determined by the function. True will output panel
        figure, False will output each figure individually. Default is False.
    panel_scale : float (optional)
        Parameter used to scale figure if panel=True. Linearly scales figure
        height and width. Default is 1.
    font_size : int (optional)
        Sets the font size for the figure. Default is 12.
    cbt : bool (optional)
        'Color by target' option that paints target nodes with a coral red.
        Typically used for plotting directional partial and flux diagrams.
        Default is False.
    path : str (optional)
        String of save path for figure. If path is given figure will be saved
        at the specified location. Default is None.
    label : str (optional)
        Figure label, used to create unique figure label if a save path is
        given. Default is None.

    Notes
    -----
    When using panel=True, if number of diagrams is not a perfect square, extra
    plots will be generated as empty coordinate axes.
    """
    if not isinstance(diagrams, list):  # single diagram case
        G = diagrams
        if pos == None:
            pos = nx.spring_layout(G)
        fig = plt.figure(figsize=(4, 4), tight_layout=True)
        fig.add_subplot(111)
        node_list = list(G.nodes)
        nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=node_list, node_color='0.8')
        nx.draw_networkx_edges(G, pos, node_size=500, width=4, arrowstyle='->', arrowsize=15)
        labels = {}
        for i in node_list:
            labels[i] = r"${}$".format(i+1)
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
        plt.axis('off')
        if not path == None:
            fig.savefig(path + "/{}_diagram.png".format(label))
    else:   # array of diagrams case
        if pos == None:
            pos = nx.spring_layout(diagrams[0])
        node_list = list(diagrams[0].nodes)
        labels = {}
        for j in node_list:
            labels[j] = r"${}$".format(j+1)
        if panel == True:
            N = len(diagrams)
            Nrows = int(np.sqrt(N))
            Ncols = int(np.ceil(N/Nrows))
            excess_plots = Nrows*Ncols - N
            fig, ax = plt.subplots(nrows=Nrows, ncols=Ncols, tight_layout=True)
            fig.set_figheight(Nrows*panel_scale)
            fig.set_figwidth(1.2*Ncols*panel_scale)
            for i, partial in enumerate(diagrams):
                if cbt == True:
                    node_colors = []
                    for n in list(partial.nodes):
                        if partial.nodes[n]['is_target'] == True:
                            node_colors.append('#FF8080')
                        else:
                            node_colors.append('0.8')
                else:
                    node_colors = ['0.8' for n in range(len(partial))]
                ix = np.unravel_index(i, ax.shape)
                plt.sca(ax[ix])
                ax[ix].set_axis_off()
                nx.draw_networkx_nodes(partial, pos, ax=ax[ix], node_size=150*panel_scale, nodelist=node_list, node_color=node_colors)
                nx.draw_networkx_edges(partial, pos, ax=ax[ix], node_size=150*panel_scale, arrowstyle='->')
                nx.draw_networkx_labels(partial, pos, labels, font_size=font_size, ax=ax[ix])
            for i in range(excess_plots):
                ax.flat[-i-1].set_visible(False)
            if not path == None:
                fig.savefig(path + "/{}_diagram_panel.png".format(label))
        else:
            for i, partial in enumerate(diagrams):
                if cbt == True:
                    node_colors = []
                    for n in list(partial.nodes):
                        if partial.nodes[n]['is_target'] == True:
                            node_colors.append('#FF8080')
                        else:
                            node_colors.append('0.8')
                else:
                    node_colors = ['0.8' for n in range(len(partial))]
                fig = plt.figure(figsize=(3, 3), tight_layout=True)
                fig.add_subplot(111)
                nx.draw_networkx_nodes(partial, pos, nodelist=node_list, node_color=node_colors)
                nx.draw_networkx_edges(partial, pos, arrowstyle='->')
                nx.draw_networkx_labels(partial, pos, labels, font_size=font_size)
                plt.axis('off')
                if not path == None:
                    fig.savefig(path + "/{}_diagram_{}.png".format(label, i+1))

def draw_cycles(G, cycles, pos=None, panel=False, panel_scale=1, font_size=12, cbt=False, path=None, label=None):
    """
    Plots a diagram with a cycle labeled.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    cycles : list of lists of int
        List of cycles or individual cycle to be plotted, index zero. Order
        of node indices does not matter.
    pos : dict (optional)
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node. Default
        is None, nx.spring_layout() is used.
    panel : bool (optional)
        Tells the function to output diagrams as an 'NxM' matrix of subplots,
        where 'N' and 'M' are determined by the function. True will output panel
        figure, False will output each figure individually. Default is False.
    panel_scale : float (optional)
        Parameter used to scale figure if panel=True. Linearly scales figure
        height and width. Default is 1.
    font_size : int (optional)
        Sets the font size for the figure. Default is 12.
    cbt : bool (optional)
        'Color by target' option that paints target nodes with a coral red to
        make them easier to spot. Default is False.
    path : str (optional)
        String of save path for figure. If path is given figure will be saved
        at the specified location. Default is None.
    label : str (optional)
        Figure label, used to create unique figure label if a save path is
        given. Default is None.

    Notes
    -----
    When using panel=True, if number of diagrams is not a perfect square, extra
    plots will be generated as empty coordinate axes.
    """
    if pos == None:
        pos = nx.spring_layout(G)
    if isinstance(cycles[0], int):    # single cycle case
        cycle = cycles
        nodes = list(G.nodes)
        if len(cycle) == len(nodes):    # if cycle contains all nodes
            labels = {}
            for i in nodes:
                labels[i] = r"${}$".format(i+1)
            if cbt == True:
                node_colors = ['#FF8080' for n in nodes]
            else:
                node_colors = ['0.8' for n in nodes]
            node_list = nodes
        else:                           # if cycle doesn't contain all nodes
            labels = {}
            for i in cycle:
                labels[i] = r"${}$".format(i+1)
            if cbt == True:
                node_colors = ['#FF8080' for n in cycle]
            else:
                node_colors = ['0.8' for n in cycle]
            pos_copy = pos.copy()
            pos = {}
            for n, position in pos_copy.items():
                if n in cycle:
                    pos[n] = position
            node_list = cycle
        cycle_edges = construct_cycle_edges(cycle)
        edge_list = append_reverse_edges(cycle_edges)
        fig = plt.figure(figsize=(4, 4), tight_layout=True)
        fig.add_subplot(111)
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=500, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, node_size=500, width=4, arrowstyle='->', arrowsize=15)
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
        plt.axis('off')
        if not path == None:
            fig.savefig(path + "/{}_cycle.png".format(label))
    else:   # list of cycles case
        if panel == True: # draw panel case
            N = len(cycles)
            Nrows = int(np.sqrt(N))
            Ncols = int(np.ceil(N/Nrows))
            excess_plots = Nrows*Ncols - N
            fig, ax = plt.subplots(nrows=Nrows, ncols=Ncols, tight_layout=True)
            fig.set_figheight(Nrows*panel_scale)
            fig.set_figwidth(1.2*Ncols*panel_scale)
            nodes = list(G.nodes)
            for i, cycle in enumerate(cycles):
                if len(cycle) == len(nodes):    # if cycle contains all nodes
                    labels = {}
                    for n in nodes:
                        labels[n] = r"${}$".format(n+1)
                    if cbt == True:
                        node_colors = ['#FF8080' for n in nodes]
                    else:
                        node_colors = ['0.8' for n in nodes]
                    node_list = nodes
                    pos_new = pos
                else:       # if cycle doesn't contain all nodes
                    labels = {}
                    for c in cycle:
                        labels[c] = r"${}$".format(c+1)
                    if cbt == True:
                        node_colors = ['#FF8080' for n in cycle]
                    else:
                        node_colors = ['0.8' for n in cycle]
                    pos_copy = pos.copy()
                    pos_new = {}
                    for n, position in pos_copy.items():
                        if n in cycle:
                            pos_new[n] = position
                    node_list = cycle
                cycle_edges = construct_cycle_edges(cycle)
                edge_list = append_reverse_edges(cycle_edges)
                ix = np.unravel_index(i, ax.shape)
                plt.sca(ax[ix])
                ax[ix].set_axis_off()
                nx.draw_networkx_nodes(G, pos_new, ax=ax[ix], node_size=150*panel_scale, nodelist=node_list, node_color=node_colors)
                nx.draw_networkx_edges(G, pos_new, ax=ax[ix], node_size=150*panel_scale, edgelist=edge_list, arrowstyle='->')
                nx.draw_networkx_labels(G, pos_new, labels, font_size=font_size, ax=ax[ix])
            for j in range(excess_plots):
                ax.flat[-j-1].set_visible(False)
            if not path == None:
                fig.savefig(path + "/{}_cycle_panel.png".format(label))
        else:   # draw individual plots case
            nodes = list(G.nodes)
            for i, cycle in enumerate(cycles):
                if len(cycle) == len(nodes):    # if cycle contains all nodes
                    labels = {}
                    for n in nodes:
                        labels[n] = r"${}$".format(n+1)
                    if cbt == True:
                        node_colors = ['#FF8080' for n in nodes]
                    else:
                        node_colors = ['0.8' for n in nodes]
                    node_list = nodes
                    pos_new = pos
                else:       # if cycle doesn't contain all nodes
                    labels = {}
                    for c in cycle:
                        labels[c] = r"${}$".format(c+1)
                    if cbt == True:
                        node_colors = ['#FF8080' for n in cycle]
                    else:
                        node_colors = ['0.8' for n in cycle]
                    pos_copy = pos.copy()
                    pos_new = {}
                    for n, position in pos_copy.items():
                        if n in cycle:
                            pos_new[n] = position
                    node_list = cycle
                cycle_edges = construct_cycle_edges(cycle)
                edge_list = append_reverse_edges(cycle_edges)
                fig = plt.figure(figsize=(4, 4), tight_layout=True)
                fig.add_subplot(111)
                nx.draw_networkx_nodes(G, pos_new, nodelist=node_list, node_size=500, node_color=node_colors)
                nx.draw_networkx_edges(G, pos_new, edgelist=edge_list, node_size=500, width=4, arrowstyle='->', arrowsize=15)
                nx.draw_networkx_labels(G, pos_new, labels, font_size=font_size)
                plt.axis('off')
                if not path == None:
                    fig.savefig(path + "/{}_cycle_{}.png".format(label, i+1))

def draw_ODE_results(results, path=None, label=None):
    """
    Plots probability time series for all states.

    Parameters
    ----------
    results : bunch object
        Contains time information (results.t) and function information at time
        t (results.y), as well as various other fields.
    path : str (optional)
        String of save path for figure. If path is given figure will be saved
        at the specified location. Default is None.
    label : str (optional)
        Figure label, used to create unique figure label if a save path is
        given. Default is None.
    """
    N = int(len(results.y))
    time = results.t
    p_time_series = results.y[:N]
    p_tot = p_time_series.sum(axis=0)
    fig = plt.figure(figsize = (8, 7), tight_layout=True)
    ax = fig.add_subplot(111)
    for i in range(N):
        ax.plot(time, p_time_series[i], '-', lw=2, label='p{}, final = {}'.format(i+1, p_time_series[i][-1]))
    ax.plot(time, p_tot, '--', lw=2, color="black", label="p_tot, final = {}".format(p_tot[-1]))
    ax.set_title("State Probabilities for {} State Model".format(N))
    ax.set_ylabel(r"Probability")
    ax.set_xlabel(r"Time (s)")
    ax.legend(loc='best')
    if not path == None:
        fig.savefig(path + "/ODE_probs_{}.png".format(label))
