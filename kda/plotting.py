# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: Diagram Plotting
=========================================================================
This file contains a host of functions used for plotting various diagrams, such
as partial, directional, and flux diagrams. Also contains a function to plot
results from `ode.ode_solver`.

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
        List of node indices (i.e. [0, 2, 3, 1]) indicating
        which node labels should be made.

    Returns
    -------
    labels: dict
        Dictionary where keys are the node index (index-zero) and the
        keys are the node index string (index-one).

    """
    labels = {}
    for i in node_list:
        labels[i] = r"${}$".format(i + 1)
    return labels


def _get_node_colors(cbt, obj):
    """
    Returns a list of color values (either grey or coral) depending
    on whether color by target is turned on.

    Parameters
    ----------
    cbt : bool
        'Color by target' option that paints target nodes with a
        coral red when true. Typically used for plotting partial
        and flux diagrams.
    obj: object
        `NetworkX.Graph`, `NetworkX.MultiDiGraph`, or list of nodes to
        return color values for. If a graph object is input, only nodes
        with attribute `is_target=True` will be colored coral red.

    Returns
    -------
    node_colors: list
        List of strings of color values (i.e. `["0.8", "0.8",...]`).

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
    Retrieves the x/y limits based on the node positions. Values are
    scaled by a constant factor to compensate for the size of the nodes.

    Parameters
    ----------
    pos : dict
        Dictionary where keys are the indexed states (0, 1, 2, ..., N)
        and the values are the x, y coordinates for each node.
    scale_factor: float (optional)
        Factor used to scale the x/y axis limits. Default is `1.4`.

    Returns
    -------
    Tuple of the form ``(xlims, ylims)``, where `xlims` and `ylims` are lists
    containing the scaled minimum and maximum x and y values, respectively.

    """
    x = np.zeros(len(pos))
    y = np.zeros(len(pos))
    for i, positions in pos.items():
        x[i] = positions[0]
        y[i] = positions[1]
    xlims = [scale_factor * x.min(), scale_factor * x.max()]
    ylims = [scale_factor * y.min(), scale_factor * y.max()]
    return xlims, ylims


def _get_panel_dimensions(n_diagrams, rows, cols=None):
    """
    Calculates the number of appropriate rows and columns based on the
    number of diagrams. Generally returns the most square-like shape
    that is feasible for a given number of diagrams. If rows are specified,
    the columns will be adjusted to fit.

    Parameters
    ----------
    n_diagrams: int
        Number of diagrams to plot in panel.
    rows : int
        Number of rows, typically based on the square
        root of the number of diagrams to generate.
    cols : int (optional)
        Number of columns. Default is `None`, which results in the number
        of rows being determined based on the number of diagrams input.

    Returns
    -------
    Tuple of the form ``(rows, cols, excess_plots)``, where `rows` and `cols`
    are the number of rows and columns in the panel, respectively, and
    `excess_plots` is the number of extra graphs available in the panel.

    """
    if rows is None:
        rows = int(np.sqrt(n_diagrams))
    if cols is None:
        cols = int(np.ceil(n_diagrams / rows))
    excess_plots = rows * cols - n_diagrams
    return (rows, cols, excess_plots)


def _plot_single_diagram(
    diagram,
    pos=None,
    node_labels=None,
    node_list=None,
    node_colors=None,
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
    """
    Plots a single diagram.

    Parameters
    ----------
    diagram : `NetworkX.MultiDiGraph` or `NetworkX.Graph()`
        Diagram to be plotted.
    pos : dict (optional)
        Dictionary where keys are the indexed states (0, 1, 2, ..., N)
        and the values are the x, y coordinates for each node. If
        not specified, ``NetworkX.spring_layout()`` is used.
    node_labels: dict (optional)
        Dictionary where keys are the node index (index-zero) and the
        keys are the node index string (index-one). If not specified, labels
        will be created for all nodes in the input diagram.
    node_list : list (optional)
        List of node indices (i.e. [0, 2, 3, 1]) indicating
        which nodes to plot. If not specified, all nodes in the input
        diagram will be plotted.
    node_colors: list (optional)
        List of strings of color values (i.e. `["0.8", "0.8",...]`)
        used to color the nodes. If not specified, node colors will
        be determined using the `cbt` parameter.
    edge_list: list (optional)
        List of edge tuples (i.e. `[(1, 0), (1, 2), ...]`) to plot. If not
        specified, all edges will be plotted.
    font_size : int (optional)
        Sets the font size for the figure. Default is `12`.
    figsize: tuple (optional)
        Tuple of the form ``(x, y)``, where `x` and `y` are the x and y-axis
        figure dimensions in inches. Default is `(3, 3)`.
    node_size: int (optional)
        Size of nodes used for `NetworkX` diagram. Default is `300`.
    arrow_width: float (optional)
        Arrow width used for `NetworkX` diagram. Default is `1.5`.
    arrow_size: int (optional)
        Arrow size used for `NetworkX` diagram. Default is `12`.
    arrow_style: str (optional)
        Style of arrows used for `NetworkX` diagram. Default is "->".
    connection_style: str (optional)
        Style of arrow connections for `NetworkX` diagram. Default is "arc3".
    ax: `matplotlib` axis object (optional)
        Axis to place diagrams on. If not specified, a new figure
        and axis will be created. Default is `None`.
    cbt : bool (optional)
        'Color by target' option that paints target nodes with a coral red.
        Typically used for plotting directional partial and flux diagrams.
        Default is `False`.

    Returns
    -------
    fig: `matplotlib.pyplot.figure` object
        The plotted diagram.

    """
    if ax is None:
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None

    if node_list is None:
        node_list = diagram.nodes()

    if node_labels is None:
        node_labels = _get_node_labels(node_list)

    if pos is None:
        pos = nx.spring_layout(diagram)

    if node_colors is None:
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
    rows=None,
    cols=None,
    pos=None,
    panel_scale=2,
    font_size=12,
    cbt=False,
    curved_arrows=False,
):
    """
    Plots a panel of diagrams of shape `(rows, cols)`.

    Parameters
    ----------
    diagrams : list of cycles or NetworkX graph objects
        List of diagrams or single diagram to be plotted.
    rows : int (optional)
        Number of rows. Default is `None`, which results in the number
        of rows being determined based on the number of diagrams input.
    cols : int (optional)
        Number of columns. Default is `None`, which results in the number
        of columns being determined based on the number of diagrams input.
    pos : dict (optional)
        Dictionary where keys are the indexed states (0, 1, 2, ..., N)
        and the values are the x, y coordinates for each node. If
        not specified, ``NetworkX.spring_layout()`` is used.
    panel_scale : float (optional)
        Parameter used to scale figure if `panel=True`. Linearly
        scales figure height and width. Default is `2`.
    font_size : int (optional)
        Sets the font size for the figure. Default is `12`.
    cbt : bool (optional)
        'Color by target' option that paints target nodes with a
        coral red. Typically used for plotting partial and flux
        diagrams. Default is `False`.
    curved_arrows: bool (optional)
        Switches on arrows with a slight curvature to separate double arrows
        for directional diagrams. Default is `False`.

    Returns
    -------
    fig: `matplotlib.pyplot.figure` object
        A panel of figures, where each figure is a plotted diagram or cycle.

    Notes
    -----
    If number of diagrams is not a perfect square, extra
    plots will be generated as empty coordinate axes.

    """
    nrows, ncols, excess_plots = _get_panel_dimensions(
        n_diagrams=len(diagrams), rows=rows, cols=cols
    )
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, tight_layout=True)
    fig.set_figheight(nrows * panel_scale)
    fig.set_figwidth(1.2 * ncols * panel_scale)

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
    panel=True,
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
    diagrams, or arrays of partial, directional, or flux diagrams.

    Parameters
    ----------
    diagrams : list of cycles or NetworkX graph objects
        List of diagrams or single diagram to be plotted.
    pos : dict (optional)
        Dictionary where keys are the indexed states (0, 1, 2, ..., N)
        and the values are the x, y coordinates for each node. If
        not specified, ``NetworkX.spring_layout()`` is used.
    panel : bool (optional)
        Tells the function to output diagrams as an 'NxM' matrix of subplots,
        where 'N' and 'M' are the number of rows and columns, respectively.
        True will output a panel figure, False will output each figure
        individually. Default is `False`.
    panel_scale : float (optional)
        Parameter used to scale figure if `panel=True`. Linearly
        scales figure height and width. Default is `2`.
    font_size : int (optional)
        Sets the font size for the figure. Default is `12`.
    cbt : bool (optional)
        'Color by target' option that paints target nodes with a
        coral red. Typically used for plotting directional and
        flux diagrams. Default is `False`.
    rows : int (optional)
        Number of rows to output if `panel=True`. Default is `None`, which
        results in the number of rows being determined based on the number of
        diagrams input.
    cols : int (optional)
        Number of columns to output if  `panel=True`. Default is `None`, which
        results in the number of columns being determined based on the number of
        diagrams input.
    path : str (optional)
        String of save path for figure. If a path is specified the figure(s)
        will be saved at the specified location. Default is `None`.
    label : str (optional)
        Figure label used to create unique filename if `path` is
        input. Includes `.png` file extension. Default is `None`.
    curved_arrows: bool (optional)
        Switches on arrows with a slight curvature to separate double arrows
        for directional diagrams. Default is `False`.

    Notes
    -----
    When using `panel=True`, if number of diagrams is not a perfect square,
    extra plots will be generated as empty coordinate axes.

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
            cbt=cbt,
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
    panel=True,
    panel_scale=2,
    rows=None,
    cols=None,
    font_size=12,
    cbt=False,
    curved_arrows=False,
    path=None,
    label=None,
):
    """
    Plots a diagram with a cycle labeled.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram used for plotting the cycles.
    cycles : list of lists of int
        List of cycles or individual cycle to be plotted, index zero. Order
        of node indices does not matter.
    pos : dict (optional)
        Dictionary where keys are the indexed states (0, 1, 2, ..., N)
        and the values are the x, y coordinates for each node. If
        not specified, ``NetworkX.spring_layout()`` is used.
    panel : bool (optional)
        Tells the function to output diagrams as an 'NxM' matrix of subplots,
        where 'N' and 'M' are the number of rows and columns, respectively.
        True will output a panel figure, False will output each figure
        individually. Default is `False`.
    panel_scale : float (optional)
        Parameter used to scale figure if `panel=True`. Linearly scales figure
        height and width. Default is `2`.
    font_size : int (optional)
        Sets the font size for the figure. Default is `12`.
    cbt : bool (optional)
        'Color by target' option that paints target nodes with a
        coral red. Typically used for plotting directional and
        flux diagrams. Default is `False`.
    curved_arrows: bool (optional)
        Switches on arrows with a slight curvature to separate double arrows
        for directional diagrams. Default is `False`.
    path : str (optional)
        String of save path for figure. If a path is specified the figure(s)
        will be saved at the specified location. Default is `None`.
    label : str (optional)
        Figure label used to create unique filename if `path` is
        input. Includes `.png` file extension. Default is `None`.

    Notes
    -----
    When using `panel=True`, if number of diagrams is not a perfect square,
    extra plots will be generated as empty coordinate axes.

    """
    if curved_arrows:
        connection_style = "arc3, rad = 0.11"
    else:
        connection_style = "arc3"

    if isinstance(cycles[0], int):  # single cycle case

        node_list = cycles
        node_labels = _get_node_labels(node_list=node_list)
        node_colors = _get_node_colors(cbt=cbt, obj=node_list)

        cycle_edges = _construct_cycle_edges(node_list)
        edge_list = _append_reverse_edges(cycle_edges)

        fig = _plot_single_diagram(
            diagram=G,
            pos=pos,
            edge_list=edge_list,
            node_list=node_list,
            node_labels=node_labels,
            node_colors=node_colors,
            node_size=500,
            font_size=font_size,
            figsize=(4, 4),
            arrow_width=2,
            cbt=False,
            connection_style=connection_style,
        )

        if path:
            save_path = os.path.join(path, f"{label}.png")
            fig.savefig(save_path, dpi=300)
            plt.close()

    else:  # multiple cycles case
        if pos is None:
            pos = nx.spring_layout(G)

        if panel:  # draw panel case

            nrows, ncols, excess_plots = _get_panel_dimensions(
                n_diagrams=len(cycles), rows=rows, cols=cols
            )

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, tight_layout=True)
            fig.set_figheight(nrows * panel_scale)
            fig.set_figwidth(1.2 * ncols * panel_scale)

            xlims, ylims = _get_axis_limits(pos, scale_factor=1.4)

            for i, cycle in enumerate(cycles):
                node_labels = _get_node_labels(node_list=cycle)
                node_colors = _get_node_colors(cbt=cbt, obj=cycle)
                cycle_edges = _construct_cycle_edges(cycle)
                edge_list = _append_reverse_edges(cycle_edges)

                ix = np.unravel_index(i, ax.shape)
                plt.sca(ax[ix])
                ax[ix].set_xlim(xlims)
                ax[ix].set_ylim(ylims)

                _plot_single_diagram(
                    diagram=G,
                    pos=pos,
                    edge_list=edge_list,
                    node_list=cycle,
                    node_labels=node_labels,
                    node_colors=node_colors,
                    node_size=150 * panel_scale,
                    font_size=font_size,
                    arrow_width=1.5,
                    cbt=False,
                    connection_style=connection_style,
                    ax=ax[ix],
                )

            for j in range(excess_plots):
                ax.flat[-j - 1].set_visible(False)

            if path:
                save_path = os.path.join(path, f"{label}.png")
                fig.savefig(save_path, dpi=300)
                plt.close()

        else:  # draw individual plots case
            for i, cycle in enumerate(cycles):
                node_labels = _get_node_labels(node_list=cycle)
                node_colors = _get_node_colors(cbt=cbt, obj=cycle)
                cycle_edges = _construct_cycle_edges(cycle)
                edge_list = _append_reverse_edges(cycle_edges)
                fig = _plot_single_diagram(
                    diagram=G,
                    pos=pos,
                    edge_list=edge_list,
                    node_list=cycle,
                    node_labels=node_labels,
                    node_colors=node_colors,
                    node_size=500,
                    font_size=font_size,
                    arrow_width=2,
                    cbt=False,
                    connection_style=connection_style,
                )
                if path:
                    save_path = os.path.join(path, f"{label}_{i+1}.png")
                    fig.savefig(save_path, dpi=300)
                    plt.close()


def draw_ode_results(
    results, figsize=(5, 4), legendloc="best", bbox_coords=None, path=None, label=None
):
    """
    Plots probability time series for all states generated by `ode.ode_solver`.

    Parameters
    ----------
    results : ``Bunch`` object
        Contains time information (results.t) and function information
        at time t (results.y), as well as various other fields.
    figsize: tuple (optional)
        Tuple of the form `(x, y)`, where `x` and `y` are the x and y-axis
        figure dimensions in inches. Default is `(5, 4)`.
    legendloc : str (optional)
        String passed to determine where to place the legend for the figure.
        Default is 'best'.
    bbox_coords : tuple (optional)
        Tuple of the form `(x, y)`, where `x` and `y` are the x and y-axis
        coordinates for the legend. Default is `None`.
    path : str (optional)
        String of save path for figure. If a path is specified the figure
        will be saved at the specified location. Default is `None`.
    label : str (optional)
        Figure label used to create unique filename if `path` is
        input. Includes `.png` file extension. Default is `None`.

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
