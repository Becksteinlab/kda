# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Biochemical Kinetic Diagram Analyzer Plotting

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_input_diagram(G, pos, path=None):
    """
    Plots the input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    fig = plt.figure(figsize=(4, 4), tight_layout=True)
    fig.add_subplot(111)
    node_list = [i for i in range(G.number_of_nodes())]
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=node_list, node_color='0.8')
    nx.draw_networkx_edges(G, pos, width=4, arrow_style='->', arrowsize=15)
    labels = {}
    for i in range(G.number_of_nodes()):
        labels[i] = r"${}$".format(i+1)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    plt.axis('off')
    if not path == None:
        fig.savefig(path + "/input_diagram.png")

def fancy_plot(G, pos, node_sizes):
    node_list = [i for i in range(G.number_of_nodes())]
    labels = {}
    for i in range(G.number_of_nodes()):
        labels[i] = r"${}$".format(i+1)
    fig = plt.figure(figsize=(4, 4), tight_layout=True)
    fig.add_subplot(111)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, nodelist=node_list, node_color='0.8')
    nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrow_style='->')
    nx.draw_networkx_labels(G, pos, labels)
    plt.axis('off')

def plot_partials(partials, pos, panel=False, panel_scale=2, name='partial', path=None):
    """
    Plots all partial diagrams.

    Parameters
    ----------
    partials : list
        List of NetworkX MultiDiGraphs where each graph is a unique partial
        diagram with no loops.
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    labels = {}
    for j in range(partials[0].number_of_nodes()):
        labels[j] = r"${}$".format(j+1)
    node_list = [i for i in range(partials[0].number_of_nodes())]
    if panel == True:
        N = len(partials)
        Nrows = int(np.sqrt(N))
        Ncols = int(np.ceil(N/Nrows))
        fig, ax = plt.subplots(nrows=Nrows, ncols=Ncols, tight_layout=True)
        fig.set_figheight(Nrows*panel_scale)
        fig.set_figwidth(Ncols*panel_scale)
        for i, partial in enumerate(partials):
            ix = np.unravel_index(i, ax.shape)
            plt.sca(ax[ix])
            nx.draw_networkx_nodes(partial, pos, ax=ax[ix], nodelist=node_list, node_color='0.8')
            nx.draw_networkx_edges(partial, pos, ax=ax[ix], arrow_style='->')
            nx.draw_networkx_labels(partial, pos, labels, ax=ax[ix])
            ax[ix].set_axis_off()
        if not path == None:
            fig.savefig(path + "/{}_diagram_panel.png".format(name))
    else:
        for i, partial in enumerate(partials):
            fig = plt.figure(figsize=(3, 3), tight_layout=True)
            fig.add_subplot(111)
            nx.draw_networkx_nodes(partial, pos, nodelist=node_list, node_color='0.8')
            nx.draw_networkx_edges(partial, pos, arrow_style='->')
            nx.draw_networkx_labels(partial, pos, labels)
            plt.axis('off')
            if not path == None:
                fig.savefig(path + "/{}_diagram_{}.png".format(name, i+1))

def plot_ODE_probs(results, save=None, path=None, ident=None):
    """
    Plots probability time series for all states.

    Parameters
    ----------
    results : bunch object
        Contains time information (results.t) and function information at time
        t (results.y), as well as various other fields.
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
    if save == True:
        fig.savefig(path + "/ODE_probs_{}.png".format(ident))
