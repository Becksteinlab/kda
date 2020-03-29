# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Biochemical Kinetic Diagram Analyzer Plotting

import matplotlib.pyplot as plt
import networkx as nx


def plot_input_diagram(G, pos, save=None, path=None, date=None, run=None):
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
    fig1 = plt.figure(figsize=(3, 3), tight_layout=True)
    fig1.add_subplot(111)
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=[i for i in range(G.number_of_nodes())], node_color='grey')
    nx.draw_networkx_edges(G, pos, width=4, arrow_style='->', arrowsize=15)
    labels = {}
    for i in range(G.number_of_nodes()):
        labels[i] = r"${}$".format(i+1)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    if save == True:
        fig1.savefig(path + "/{}/input_{}_diagram.png".format(date, run))


def plot_partials(partials, pos, save=None, path=None, date=None, run=None):
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
    for i in range(len(partials)):
        fig1 = plt.figure(figsize=(3, 3), tight_layout=True)
        fig1.add_subplot(111)
        partial = partials[i]
        nx.draw_networkx_nodes(partial, pos, node_size=500, nodelist=[i for i in range(partials[0].number_of_nodes())], node_color='grey')
        nx.draw_networkx_edges(partial, pos, width=4, arrow_style='->', arrowsize=15)
        labels = {}
        for j in range(partials[0].number_of_nodes()):
            labels[j] = r"${}$".format(j+1)
        nx.draw_networkx_labels(partial, pos, labels, font_size=16)
        if save == True:
            fig1.savefig(path + "/{}/partial_diagram_{}_{}.png".format(date, run, i+1))


def plot_directional_partials(dir_partials, pos, save=None, path=None, date=None, run=None):
    """
    Plots all directional partial diagrams.

    Parameters
    ----------
    dir_partials : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N) and
        the values are NumPy arrays of x, y coordinates for each node.
    """
    for i in range(len(dir_partials)):
        fig1 = plt.figure(figsize=(3, 3), tight_layout=True)
        fig1.add_subplot(111)
        partial = dir_partials[i]
        nx.draw_networkx_nodes(partial, pos, node_size=500, nodelist=[i for i in range(dir_partials[0].number_of_nodes())], node_color='grey')
        nx.draw_networkx_edges(partial, pos, width=4, arrow_style='->', arrowsize=15)
        labels = {}
        for j in range(dir_partials[0].number_of_nodes()):
            labels[j] = r"${}$".format(j+1)
        nx.draw_networkx_labels(partial, pos, labels, font_size=16)
        if save == True:
            fig1.savefig(path + "/{}/directional_partial_diagram_{}_{}.png".format(date, run, i+1))


def plot_ODE_probs(results):
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
    fig1 = plt.figure(figsize = (8, 7), tight_layout=True)
    ax = fig1.add_subplot(111)
    for i in range(N):
        ax.plot(time, p_time_series[i], '-', lw=2, label='p{}, final = {}'.format(i+1, p_time_series[i][-1]))
    ax.plot(time, p_tot, '--', lw=2, color="black", label="p_tot, final = {}".format(p_tot[-1]))
    ax.set_title("State Probabilities for {} State Model".format(N))
    ax.set_ylabel(r"Probability")
    ax.set_xlabel(r"Time (s)")
    ax.legend(loc='best')
    plt.show()
