"""
================================
KDA Verification Timing Analysis
================================
The purpose of this module is to analyze the timing outputs of the KDA
verification script, `verification.py`.
"""
import os
from os.path import join, split
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_avg_degree_data(mat_time, kda_time, nodes, edges):
    avg_degree = np.asarray(edges) / np.asarray(nodes)
    unique_degrees = np.unique(avg_degree)
    mat_data = []
    kda_data = []
    for deg in unique_degrees:
        mask = avg_degree == deg
        matt = mat_time[mask]
        kdat = kda_time[mask]
        mat_avg = np.mean(matt)
        kda_avg = np.mean(kdat)
        mat_std = np.std(matt)
        kda_std = np.std(kdat)
        mat_data.append([mat_avg, mat_std])
        kda_data.append([kda_avg, kda_std])
    mat_data = np.array(mat_data)
    kda_data = np.array(kda_data)
    return (unique_degrees, mat_data, kda_data)


def get_node_data(mat_time, kda_time, nodes):
    unique_nodes = np.unique(nodes)
    mat_data = []
    kda_data = []
    for n_nodes in unique_nodes:
        mask = nodes == n_nodes
        matt = mat_time[mask]
        kdat = kda_time[mask]
        mat_avg = np.mean(matt)
        kda_avg = np.mean(kdat)
        mat_std = np.std(matt)
        kda_std = np.std(kdat)
        mat_data.append([mat_avg, mat_std])
        kda_data.append([kda_avg, kda_std])
    mat_data = np.array(mat_data)
    kda_data = np.array(kda_data)
    return (unique_nodes, mat_data, kda_data)


def get_edge_data(mat_time, kda_time, edges):
    unique_edges = np.unique(edges)
    mat_data = []
    kda_data = []
    for n_edges in unique_edges:
        mask = edges == n_edges
        matt = mat_time[mask]
        kdat = kda_time[mask]
        mat_avg = np.mean(matt)
        kda_avg = np.mean(kdat)
        mat_std = np.std(matt)
        kda_std = np.std(kdat)
        mat_data.append([mat_avg, mat_std])
        kda_data.append([kda_avg, kda_std])
    mat_data = np.array(mat_data)
    kda_data = np.array(kda_data)
    return (unique_edges, mat_data, kda_data)


def get_par_data(mat_time, kda_time, pars):
    unique_pars = np.unique(pars)
    mat_data = []
    kda_data = []
    for n_pars in unique_pars:
        mask = pars == n_pars
        matt = mat_time[mask]
        kdat = kda_time[mask]
        mat_avg = np.mean(matt)
        kda_avg = np.mean(kdat)
        mat_std = np.std(matt)
        kda_std = np.std(kdat)
        mat_data.append([mat_avg, mat_std])
        kda_data.append([kda_avg, kda_std])
    mat_data = np.array(mat_data)
    kda_data = np.array(kda_data)
    return (unique_pars, mat_data, kda_data)


def get_dirpar_data(mat_time, kda_time, dirpars):
    unique_dirpars = np.unique(dirpars)
    mat_data = []
    kda_data = []
    for n_dirpars in unique_dirpars:
        mask = dirpars == n_dirpars
        matt = mat_time[mask]
        kdat = kda_time[mask]
        mat_avg = np.mean(matt)
        kda_avg = np.mean(kdat)
        mat_std = np.std(matt)
        kda_std = np.std(kdat)
        mat_data.append([mat_avg, mat_std])
        kda_data.append([kda_avg, kda_std])
    mat_data = np.array(mat_data)
    kda_data = np.array(kda_data)
    return (unique_dirpars, mat_data, kda_data)


def fit_powerlaw(diagrams, kda_data):
    time = kda_data[:, 0]

    # make data linear by taking the log10
    log_diagrams = np.log10(diagrams)
    log_time = np.log10(time)

    # get a linear fit of the data (Y = mX + b)
    m, b = np.polyfit(x=log_diagrams, y=log_time, deg=1)
    # for y = ax ** k, log y = k log x + log a
    # translated into linear form, this means
    # m = k, b = log a, X = log x, and Y = log y
    k = m
    a = 10 ** b
    fit_func = a * (diagrams ** k)
    return fit_func, a, k


def get_fit_string(a, k):
    return r"$T = %1.1g D ^ {%1.3g} $" % (a, k)


def plot_t_over_avg_degree(mat_time, kda_time, nodes, edges, datapath):
    unique_degrees, mat_data, kda_data = get_avg_degree_data(
        mat_time=mat_time, kda_time=kda_time, nodes=nodes, edges=edges,
    )

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.errorbar(
        unique_degrees,
        mat_data[:, 0],
        yerr=mat_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="MAT",
    )
    ax.errorbar(
        unique_degrees,
        kda_data[:, 0],
        yerr=kda_data[:, 1],
        color="red",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="KDA",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("KDA Verification: Run Time vs. Avg. Degree")
    ax.set_ylabel(r"Time (s)")
    ax.set_xlabel(r"Average. Degree of Graph")
    fig_savepath = join(datapath, f"timing_avg_degree.pdf")
    print(f"Saving average degree plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


def plot_t_over_nodes(mat_time, kda_time, nodes, datapath):
    unique_nodes, mat_data, kda_data = get_node_data(
        mat_time=mat_time, kda_time=kda_time, nodes=nodes,
    )

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.errorbar(
        unique_nodes,
        mat_data[:, 0],
        yerr=mat_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="MAT",
    )
    ax.errorbar(
        unique_nodes,
        kda_data[:, 0],
        yerr=kda_data[:, 1],
        color="red",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="KDA",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("KDA Verification: Run Time vs. Number of Nodes")
    ax.set_ylabel(r"Time (s)")
    ax.set_xlabel(r"Number of Nodes")
    fig_savepath = join(datapath, f"timing_n_nodes.pdf")
    print(f"Saving nodes plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


def plot_t_over_edges(mat_time, kda_time, edges, datapath):
    unique_edges, mat_data, kda_data = get_edge_data(
        mat_time=mat_time, kda_time=kda_time, edges=edges,
    )

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.errorbar(
        unique_edges,
        mat_data[:, 0],
        yerr=mat_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="MAT",
    )
    ax.errorbar(
        unique_edges,
        kda_data[:, 0],
        yerr=kda_data[:, 1],
        color="red",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="KDA",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("KDA Verification: Run Time vs. Number of Edges")
    ax.set_ylabel(r"Time (s)")
    ax.set_xlabel(r"Number of Edges")
    fig_savepath = join(datapath, f"timing_n_edges.pdf")
    print(f"Saving edges plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


def plot_t_over_dirpars(mat_time, kda_time, dirpars, datapath, loglog=True):
    unique_dirpars, mat_data, kda_data = get_dirpar_data(
        mat_time=mat_time, kda_time=kda_time, dirpars=dirpars,
    )

    # get fit for log-log plot
    fit_func, a, k = fit_powerlaw(diagrams=unique_dirpars, kda_data=kda_data)
    fit_func_str = get_fit_string(a, k)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    if loglog == True:
        ax.set_xscale("log", nonpositive="clip")
        ax.set_yscale("log", nonpositive="clip")
        min_y = 0.5 * np.min(np.abs(mat_data[:, 0] - mat_data[:, 1]))
        max_y = 5 * np.max(np.abs(kda_data[:, 0] + kda_data[:, 1]))
        ax.set_ylim(bottom=min_y, top=max_y)
    ax.plot(unique_dirpars, fit_func, color="blue", ls="-", lw=1.5, label=fit_func_str)
    ax.errorbar(
        unique_dirpars,
        mat_data[:, 0],
        yerr=mat_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="MAT",
    )
    ax.errorbar(
        unique_dirpars,
        kda_data[:, 0],
        yerr=kda_data[:, 1],
        color="red",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="KDA",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("KDA Verification: Run Time vs. Directional Partial Diagrams")
    ax.set_ylabel(r"Time (s)")
    ax.set_xlabel(r"Directional Partial Diagrams")
    if loglog == True:
        fig_savepath = join(datapath, f"timing_n_dirpars_loglog.pdf")
    else:
        fig_savepath = join(datapath, f"timing_n_dirpars.pdf")
    print(f"Saving directional partial diagrams plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


def plot_t_over_pars(mat_time, kda_time, pars, datapath, loglog=True):
    unique_pars, mat_data, kda_data = get_par_data(
        mat_time=mat_time, kda_time=kda_time, pars=pars,
    )

    # get fit for log-log plot
    fit_func, a, k = fit_powerlaw(diagrams=unique_pars, kda_data=kda_data)
    fit_func_str = get_fit_string(a, k)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    if loglog == True:
        ax.set_xscale("log", nonpositive="clip")
        ax.set_yscale("log", nonpositive="clip")
        min_y = 0.5 * np.min(np.abs(mat_data[:, 0] - mat_data[:, 1]))
        max_y = 5 * np.max(np.abs(kda_data[:, 0] + kda_data[:, 1]))
        ax.set_ylim(bottom=min_y, top=max_y)
    ax.plot(unique_pars, fit_func, color="blue", ls="-", lw=1.5, label=fit_func_str)
    ax.errorbar(
        unique_pars,
        mat_data[:, 0],
        yerr=mat_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="MAT",
    )
    ax.errorbar(
        unique_pars,
        kda_data[:, 0],
        yerr=kda_data[:, 1],
        color="red",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="KDA",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("KDA Verification: Run Time vs. Partial Diagrams")
    ax.set_ylabel(r"Time (s)")
    ax.set_xlabel(r"Partial Diagrams")
    if loglog == True:
        fig_savepath = join(datapath, f"timing_n_pars_loglog.pdf")
    else:
        fig_savepath = join(datapath, f"timing_n_pars.pdf")
    print(f"Saving partial diagrams plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("all_data_path", type=str, help="Path to all_data.csv.")
    args = parser.parse_args()
    all_data_path = args.all_data_path

    if os.path.isdir(all_data_path):
        raise Exception(
            "Input path is a directory. Please input a path to all_data.csv."
        )
    print(f"Pulling data from {all_data_path}")

    # get the directory to store generated graphs in
    datapath = split(all_data_path)[0]

    # read in `all_data.csv` and collect required data
    df = pd.read_csv(all_data_path)
    graph_indices = df["graph index"]
    nodes = df["n_states"]
    edges = df["n_edges"]
    cycles = df["n_cycles"]
    pars = df["n_pars"]
    dirpars = df["n_dirpars"]
    mat_time = df["mat time (s)"]
    kda_time = df["kda time (s)"]

    plot_t_over_edges(mat_time, kda_time, edges, datapath)
    plot_t_over_avg_degree(mat_time, kda_time, nodes, edges, datapath)
    plot_t_over_pars(mat_time, kda_time, pars, datapath)
    plot_t_over_pars(mat_time, kda_time, pars, datapath, loglog=False)
    plot_t_over_dirpars(mat_time, kda_time, dirpars, datapath)
    plot_t_over_dirpars(mat_time, kda_time, dirpars, datapath, loglog=False)
    plot_t_over_nodes(mat_time, kda_time, nodes, datapath)
