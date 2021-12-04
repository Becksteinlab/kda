# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: Diagram Plotting
=========================================================================
This file contains a host of functions used for plotting various diagrams, such
as input, partial, flux, and cycle diagrams. Also contains a function to plot
results from ODE solver.

.. autofunction:: draw_diagrams
.. autofunction:: draw_cycles
.. autofunction:: draw_ode_results

"""
import os
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from kda.diagrams import _construct_cycle_edges, _append_reverse_edges


def _get_node_labels(node_list):
    """
    Builds the dictionary of node labels for NetworkX nodes.

    Parameters
    ----------
    node_list : list
        List of integers (i.e. [0, 2, 3, 1]) for which labels should be made.
    """
    labels = {}
    for i in node_list:
        labels[i] = r"${}$".format(i + 1)
    return labels


def _contains_all_nodes(cycle, node_list):
    """
    Determines if the input cycle contains all nodes in the input node list.
    """
    sorted_cycle = np.sort(cycle)
    sorted_nodes = np.sort(node_list)
    if np.array_equal(sorted_cycle, sorted_nodes):
        # if the the arrays are the same length and contain the same
        # node indices, the input cycle must contain all nodes
        return True
    else:
        return False


def _get_node_colors(cbt, obj):
    """
    Returns a list of color values (either grey or coral) depending
    on whether color by target is turned on.
    """
    base_color = "0.8"
    target_color = "#FF8080"
    if isinstance(obj, nx.Graph) or isinstance(obj, nx.MultiDiGraph):
        node_colors = [base_color for i in obj.nodes]
        if cbt:
            node_colors = np.asarray(node_colors, dtype=object)
            target_mask = list(nx.get_node_attributes(obj, "is_target").values())
            node_colors[target_mask] = target_color
    else:
        if cbt:
            color = target_color
        else:
            color = base_color
        node_colors = [color for i in obj]

    return node_colors


def _get_axis_limits(pos, scale_factor=1.4):
    """
    Retrieves the x/y limits based on the node positions. Values are scaled by
    a constant factor to compensate for the size of the nodes.
    """
    x = np.zeros(len(pos))
    y = np.zeros(len(pos))
    for i, positions in pos.items():
        x[i] = positions[0]
        y[i] = positions[1]
    xlims = [scale_factor * x.min(), scale_factor * x.max()]
    ylims = [scale_factor * y.min(), scale_factor * y.max()]
    return xlims, ylims


def _plot_single_diagram(
    diagram,
    pos=None,
    node_labels=None,
    node_list=None,
    edge_list=None,
    font_size=12,
    figsize=(3, 3),
    node_size=300,
    arrow_width=1.5,
    arrow_size=12,
    arrow_style="->",
    connection_style="arc3",
    ax=None,
    cbt=False,
):
    if ax is None:
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None

    if node_labels is None:
        node_list = list(diagram.nodes)
        node_labels = _get_node_labels(node_list)

    if pos is None:
        pos = nx.spring_layout(diagram)

    node_colors = _get_node_colors(cbt=cbt, obj=diagram)
    nx.draw_networkx_nodes(
        diagram,
        pos,
        node_size=node_size,
        nodelist=node_list,
        node_color=node_colors,
        ax=ax,
    )
    nx.draw_networkx_edges(
        diagram,
        pos,
        edgelist=edge_list,
        node_size=node_size,
        width=arrow_width,
        arrowsize=arrow_size,
        arrowstyle=arrow_style,
        connectionstyle=connection_style,
        ax=ax,
    )
    nx.draw_networkx_labels(diagram, pos, node_labels, font_size=font_size, ax=ax)
    ax.set_axis_off()
    return fig


def _plot_panel(
    diagrams,
    rows,
    cols,
    pos=None,
    panel_scale=2,
    font_size=12,
    cbt=False,
    curved_arrows=False,
):
    N = len(diagrams)
    if not rows is None:
        Nrows = rows
    else:
        Nrows = int(np.sqrt(N))
    if not cols is None:
        Ncols = cols
    else:
        Ncols = int(np.ceil(N / Nrows))
    excess_plots = Nrows * Ncols - N
    fig, ax = plt.subplots(nrows=Nrows, ncols=Ncols, tight_layout=True)
    fig.set_figheight(Nrows * panel_scale)
    fig.set_figwidth(1.2 * Ncols * panel_scale)

    node_list = list(diagrams[0].nodes)
    node_labels = _get_node_labels(node_list=node_list)
    node_size = 150 * panel_scale

    if pos is None:
        pos = nx.spring_layout(diagrams[0])

    if curved_arrows:
        connection_style = "arc3, rad = 0.11"
    else:
        connection_style = "arc3"

    for i, diag in enumerate(diagrams):
        ix = np.unravel_index(i, ax.shape)
        plt.sca(ax[ix])
        _plot_single_diagram(
            diagram=diag,
            pos=pos,
            node_list=node_list,
            node_labels=node_labels,
            font_size=font_size,
            node_size=node_size,
            ax=ax[ix],
            cbt=cbt,
            connection_style=connection_style,
        )
    for i in range(excess_plots):
        ax.flat[-i - 1].set_visible(False)
    return fig


def draw_diagrams(
    diagrams,
    pos=None,
    panel=False,
    panel_scale=2,
    font_size=12,
    cbt=False,
    rows=None,
    cols=None,
    path=None,
    label=None,
    curved_arrows=False,
):
    """
    Plots any number of input diagrams. Typically used for plotting input
    diagrams, or arrays of partial, directional partial, or flux diagrams.

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
    rows : int (optional)
        Number of rows to output if `panel=True`. Default is `None`, which
        results in the number of rows being determined based on the number of
        diagrams input.
    cols : int (optional)
        Number of columns to output if  `panel=True`. Default is `None`, which
        results in the number of rows being determined based on the number of
        diagrams input.
    path : str (optional)
        String of save path for figure. If path is given figure will be saved
        at the specified location. Default is None.
    label : str (optional)
        Figure label, used to create unique figure label if a save path is
        given. Includes `.png` file extension. Default is None.

    Notes
    -----
    When using panel=True, if number of diagrams is not a perfect square, extra
    plots will be generated as empty coordinate axes.
    """

    if curved_arrows:
        connection_style = "arc3, rad = 0.11"
    else:
        connection_style = "arc3"

    if isinstance(diagrams, nx.Graph) or isinstance(diagrams, nx.MultiDiGraph):
        # single diagram case
        fig = _plot_single_diagram(
            diagram=diagrams,
            pos=pos,
            font_size=font_size,
            figsize=(4, 4),
            node_size=500,
            arrow_width=2,
            cbt=False,
            connection_style=connection_style,
        )
        if path:
            save_path = os.path.join(path, f"{label}.png")
            fig.savefig(save_path, dpi=300)
            plt.close()

    else:  # array of diagrams case
        if pos is None:
            pos = nx.spring_layout(diagrams[0])
        if panel:
            fig = _plot_panel(
                diagrams=diagrams,
                pos=pos,
                rows=rows,
                cols=cols,
                font_size=font_size,
                panel_scale=panel_scale,
                cbt=cbt,
                curved_arrows=curved_arrows,
            )
            if path:
                save_path = os.path.join(path, f"{label}.png")
                fig.savefig(save_path, dpi=300)
                plt.close()

        else:  # individual plots case
            node_list = list(diagrams[0].nodes)
            node_labels = _get_node_labels(node_list=node_list)

            for i, diag in enumerate(diagrams):
                fig = _plot_single_diagram(
                    diagram=diag,
                    pos=pos,
                    node_list=node_list,
                    node_labels=node_labels,
                    font_size=font_size,
                    cbt=cbt,
                    connection_style=connection_style,
                )
                if path:
                    save_path = os.path.join(path, f"{label}_{i+1}.png")
                    fig.savefig(save_path, dpi=300)
                    plt.close()


def draw_cycles(
    G,
    cycles,
    pos=None,
    panel=False,
    panel_scale=2,
    font_size=12,
    cbt=False,
    path=None,
    label=None,
):
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
        given. Includes `.png` file extension. Default is None.

    Notes
    -----
    When using panel=True, if number of diagrams is not a perfect square, extra
    plots will be generated as empty coordinate axes.
    """
    if pos is None:
        pos = nx.spring_layout(G)
    if isinstance(cycles[0], int):  # single cycle case
        cycle = cycles
        nodes = list(G.nodes)
        if _contains_all_nodes(cycle, nodes):  # if cycle contains all nodes
            labels = _get_node_labels(node_list=nodes)
            node_colors = _get_node_colors(cbt=cbt, obj=nodes)
            node_list = nodes
        else:  # if cycle doesn't contain all nodes
            labels = _get_node_labels(node_list=cycle)
            node_colors = _get_node_colors(cbt=cbt, obj=cycle)
            node_list = cycle

        cycle_edges = _construct_cycle_edges(cycle)
        edge_list = _append_reverse_edges(cycle_edges)

        fig = plt.figure(figsize=(4, 4), tight_layout=True)
        ax = fig.add_subplot(111)

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            node_size=500,
            node_color=node_colors
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_list,
            node_size=500,
            width=1.5,
            arrowsize=12,
            arrowstyle="->",
            connectionstyle="arc3, rad = 0.11",
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
        ax.set_axis_off()

        if path:
            save_path = os.path.join(path, f"{label}.png")
            fig.savefig(save_path, dpi=300)
            plt.close()

    else:  # list of cycles case
        if panel:  # draw panel case
            N = len(cycles)
            Nrows = int(np.sqrt(N))
            Ncols = int(np.ceil(N / Nrows))
            excess_plots = Nrows * Ncols - N
            fig, ax = plt.subplots(nrows=Nrows, ncols=Ncols, tight_layout=True)
            fig.set_figheight(Nrows * panel_scale)
            fig.set_figwidth(1.2 * Ncols * panel_scale)
            nodes = list(G.nodes)
            xlims, ylims = _get_axis_limits(pos, scale_factor=1.4)
            for i, cycle in enumerate(cycles):
                if _contains_all_nodes(cycle, nodes):  # if cycle contains all nodes
                    labels = _get_node_labels(node_list=nodes)
                    node_colors = _get_node_colors(cbt=cbt, obj=nodes)
                    node_list = nodes
                else:  # if cycle doesn't contain all nodes
                    labels = _get_node_labels(node_list=cycle)
                    node_colors = _get_node_colors(cbt=cbt, obj=cycle)
                    node_list = cycle
                cycle_edges = _construct_cycle_edges(cycle)
                edge_list = _append_reverse_edges(cycle_edges)
                ix = np.unravel_index(i, ax.shape)
                plt.sca(ax[ix])
                ax[ix].set_axis_off()
                node_size = 150 * panel_scale
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax[ix],
                    node_size=node_size,
                    nodelist=node_list,
                    node_color=node_colors,
                )
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax[ix],
                    node_size=node_size,
                    edgelist=edge_list,
                    width=1.5,
                    arrowsize=12,
                    arrowstyle="->",
                    connectionstyle="arc3, rad = 0.11",
                )
                ax[ix].set_xlim(xlims)
                ax[ix].set_ylim(ylims)
                nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax[ix])
            for j in range(excess_plots):
                ax.flat[-j - 1].set_visible(False)
            if path:
                save_path = os.path.join(path, f"{label}.png")
                fig.savefig(save_path, dpi=300)
                plt.close()
        else:  # draw individual plots case
            nodes = list(G.nodes)
            for i, cycle in enumerate(cycles):
                if _contains_all_nodes(cycle, nodes):  # if cycle contains all nodes
                    labels = _get_node_labels(node_list=nodes)
                    node_colors = _get_node_colors(cbt=cbt, obj=nodes)
                    node_list = nodes
                else:  # if cycle doesn't contain all nodes
                    labels = _get_node_labels(node_list=cycle)
                    node_colors = _get_node_colors(cbt=cbt, obj=cycle)
                    node_list = cycle
                cycle_edges = _construct_cycle_edges(cycle)
                edge_list = _append_reverse_edges(cycle_edges)

                fig = plt.figure(figsize=(4, 4), tight_layout=True)
                ax = fig.add_subplot(111)
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=node_list,
                    node_size=500,
                    node_color=node_colors,
                )
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=edge_list,
                    node_size=500,
                    width=1.5,
                    arrowsize=12,
                    arrowstyle="->",
                    connectionstyle="arc3, rad = 0.11",
                )
                nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
                ax.set_axis_off()
                if path:
                    save_path = os.path.join(path, f"{label}_{i+1}.png")
                    fig.savefig(save_path, dpi=300)
                    plt.close()


def draw_ode_results(
    results, figsize=(5, 4), legendloc="best", bbox_coords=None, path=None, label=None
):
    """
    Plots probability time series for all states.

    Parameters
    ----------
    results : bunch object
        Contains time information (results.t) and function information at time
        t (results.y), as well as various other fields.
    figsize : tuple (optional)
        Tuple of (x, y) coordinates passed to `plt.figure()` to modify the
        figure size. Default is (5, 4).
    legendloc : str (optional)
        String passed to determine where to place the legend for the figure.
        Default is 'best'.
    bbox_coords : tuple (optional)
        Tuple of (x, y) coordinates to determine where the legend goes in the
        figure. Default is `None`, so default is `loc='best'`.
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
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(111)
    for i in range(N):
        state_label = r"$p_{%d, %s}$" % (i + 1, "final")
        state_val = " = {:.3f}".format(p_time_series[i][-1])
        ax.plot(time, p_time_series[i], "-", lw=2, label=state_label + state_val)
    ptot_label = r"$p_{tot, final}$" + " = {:.2f}".format(p_tot[-1])
    ax.plot(time, p_tot, "--", lw=2, color="black", label=ptot_label)
    ax.set_title("State Probabilities for {} State Model".format(N))
    ax.set_ylabel(r"Probability")
    ax.set_xlabel(r"Time (s)")
    if bbox_coords is None:
        ax.legend(loc=legendloc)
    else:
        ax.legend(loc=legendloc, bbox_to_anchor=bbox_coords)
    if path:
        save_path = os.path.join(path, f"{label}.png")
        fig.savefig(save_path, dpi=300)
        plt.close()
