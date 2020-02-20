# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Biochemical Kinetic Diagram Analyzer Plotting

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_input_diagram(G, pos, save=None, path=None, date=None, run=None):
    fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
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
    for i in range(len(partials)):
        fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
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
    for i in range(len(dir_partials)):
        fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
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
