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


def get_avg_degree_data(svd_time, kda_time, nodes, edges):
    avg_degree = np.asarray(edges) / np.asarray(nodes)
    unique_degrees = np.unique(avg_degree)
    svd_data = []
    kda_data = []
    for deg in unique_degrees:
        mask = avg_degree == deg
        svdt = svd_time[mask]
        kdat = kda_time[mask]
        svd_avg = np.mean(svdt)
        kda_avg = np.mean(kdat)
        svd_std = np.std(svdt)
        kda_std = np.std(kdat)
        svd_data.append([svd_avg, svd_std])
        kda_data.append([kda_avg, kda_std])
    svd_data = np.array(svd_data)
    kda_data = np.array(kda_data)
    return (unique_degrees, svd_data, kda_data)


def get_node_data(svd_time, kda_time, nodes):
    unique_nodes = np.unique(nodes)
    svd_data = []
    kda_data = []
    for n_nodes in unique_nodes:
        mask = nodes == n_nodes
        svdt = svd_time[mask]
        kdat = kda_time[mask]
        svd_avg = np.mean(svdt)
        kda_avg = np.mean(kdat)
        svd_std = np.std(svdt)
        kda_std = np.std(kdat)
        svd_data.append([svd_avg, svd_std])
        kda_data.append([kda_avg, kda_std])
    svd_data = np.array(svd_data)
    kda_data = np.array(kda_data)
    return (unique_nodes, svd_data, kda_data)


def get_edge_data(svd_time, kda_time, edges):
    unique_edges = np.unique(edges)
    svd_data = []
    kda_data = []
    for n_edges in unique_edges:
        mask = edges == n_edges
        svdt = svd_time[mask]
        kdat = kda_time[mask]
        svd_avg = np.mean(svdt)
        kda_avg = np.mean(kdat)
        svd_std = np.std(svdt)
        kda_std = np.std(kdat)
        svd_data.append([svd_avg, svd_std])
        kda_data.append([kda_avg, kda_std])
    svd_data = np.array(svd_data)
    kda_data = np.array(kda_data)
    return (unique_edges, svd_data, kda_data)


def get_par_data(svd_time, kda_time, pars):
    unique_pars = np.unique(pars)
    svd_data = []
    kda_data = []
    for n_pars in unique_pars:
        mask = pars == n_pars
        svdt = svd_time[mask]
        kdat = kda_time[mask]
        svd_avg = np.mean(svdt)
        kda_avg = np.mean(kdat)
        svd_std = np.std(svdt)
        kda_std = np.std(kdat)
        svd_data.append([svd_avg, svd_std])
        kda_data.append([kda_avg, kda_std])
    svd_data = np.array(svd_data)
    kda_data = np.array(kda_data)
    return (unique_pars, svd_data, kda_data)


def get_dirpar_data(svd_time, kda_time, dirpars):
    unique_dirpars = np.unique(dirpars)
    svd_data = []
    kda_data = []
    for n_dirpars in unique_dirpars:
        mask = dirpars == n_dirpars
        svdt = svd_time[mask]
        kdat = kda_time[mask]
        svd_avg = np.mean(svdt)
        kda_avg = np.mean(kdat)
        svd_std = np.std(svdt)
        kda_std = np.std(kdat)
        svd_data.append([svd_avg, svd_std])
        kda_data.append([kda_avg, kda_std])
    svd_data = np.array(svd_data)
    kda_data = np.array(kda_data)
    return (unique_dirpars, svd_data, kda_data)


def save_to_csv(data, cols, datapath, filename):
    df = pd.DataFrame(data=data, columns=cols)
    csv_save_string = join(datapath, filename)
    df.to_csv(path_or_buf=csv_save_string, sep=",", columns=cols, index=False)


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


def plot_t_over_avg_degree(svd_time, kda_time, nodes, edges, datapath):
    unique_degrees, svd_data, kda_data = get_avg_degree_data(
        svd_time=svd_time,
        kda_time=kda_time,
        nodes=nodes,
        edges=edges,
    )

    data = [
        unique_degrees,
        svd_data[:, 0],
        svd_data[:, 1],
        kda_data[:, 0],
        kda_data[:, 1],
    ]
    data = np.asarray(data, dtype=np.float64).T
    cols = [
        "Avg. Degree",
        "SVD Avg. (s)",
        "SVD Std. (s)",
        "KDA Avg. (s)",
        "KDA Std. (s)",
    ]
    filename = f"timing_avg_degree.csv"
    save_to_csv(data=data, cols=cols, datapath=datapath, filename=filename)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.errorbar(
        unique_degrees,
        svd_data[:, 0],
        yerr=svd_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="SVD",
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


def plot_t_over_nodes(svd_time, kda_time, nodes, datapath):
    unique_nodes, svd_data, kda_data = get_node_data(
        svd_time=svd_time,
        kda_time=kda_time,
        nodes=nodes,
    )

    data = [
        unique_nodes,
        svd_data[:, 0],
        svd_data[:, 1],
        kda_data[:, 0],
        kda_data[:, 1],
    ]
    data = np.asarray(data, dtype=np.float64).T
    cols = [
        "Nodes",
        "SVD Avg. (s)",
        "SVD Std. (s)",
        "KDA Avg. (s)",
        "KDA Std. (s)",
    ]
    filename = f"timing_n_nodes.csv"
    save_to_csv(data=data, cols=cols, datapath=datapath, filename=filename)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.errorbar(
        unique_nodes,
        svd_data[:, 0],
        yerr=svd_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="SVD",
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


def plot_t_over_edges(svd_time, kda_time, edges, datapath):
    unique_edges, svd_data, kda_data = get_edge_data(
        svd_time=svd_time,
        kda_time=kda_time,
        edges=edges,
    )

    data = [
        unique_edges,
        svd_data[:, 0],
        svd_data[:, 1],
        kda_data[:, 0],
        kda_data[:, 1],
    ]
    data = np.asarray(data, dtype=np.float64).T
    cols = [
        "Edges",
        "SVD Avg. (s)",
        "SVD Std. (s)",
        "KDA Avg. (s)",
        "KDA Std. (s)",
    ]
    filename = f"timing_n_edges.csv"
    save_to_csv(data=data, cols=cols, datapath=datapath, filename=filename)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    ax.errorbar(
        unique_edges,
        svd_data[:, 0],
        yerr=svd_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="SVD",
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


def plot_t_over_dirpars(svd_time, kda_time, dirpars, datapath, loglog=True):
    unique_dirpars, svd_data, kda_data = get_dirpar_data(
        svd_time=svd_time,
        kda_time=kda_time,
        dirpars=dirpars,
    )

    data = [
        unique_dirpars,
        svd_data[:, 0],
        svd_data[:, 1],
        kda_data[:, 0],
        kda_data[:, 1],
    ]
    data = np.asarray(data, dtype=np.float64).T
    cols = [
        "Directional Partial Diagrams",
        "SVD Avg. (s)",
        "SVD Std. (s)",
        "KDA Avg. (s)",
        "KDA Std. (s)",
    ]
    filename = f"timing_n_dirpars.csv"
    save_to_csv(data=data, cols=cols, datapath=datapath, filename=filename)

    # get fit for log-log plot
    fit_func, a, k = fit_powerlaw(diagrams=unique_dirpars, kda_data=kda_data)
    fit_func_str = r"$t = %1.1g N ^ {%1.3g} $" % (a, k)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    if loglog == True:
        ax.set_xscale("log", nonpositive="clip")
        ax.set_yscale("log", nonpositive="clip")
        min_y = 0.5 * np.min(np.abs(svd_data[:, 0] - svd_data[:, 1]))
        max_y = 5 * np.max(np.abs(kda_data[:, 0] + kda_data[:, 1]))
        ax.set_ylim(bottom=min_y, top=max_y)
    ax.plot(unique_dirpars, fit_func, color="blue", ls="-", lw=1.5, label=fit_func_str)
    ax.errorbar(
        unique_dirpars,
        svd_data[:, 0],
        yerr=svd_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="SVD",
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
    print(f"Saving edges plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


def plot_t_over_pars(svd_time, kda_time, pars, datapath, loglog=True):
    unique_pars, svd_data, kda_data = get_par_data(
        svd_time=svd_time,
        kda_time=kda_time,
        pars=pars,
    )

    data = [
        unique_pars,
        svd_data[:, 0],
        svd_data[:, 1],
        kda_data[:, 0],
        kda_data[:, 1],
    ]
    data = np.asarray(data, dtype=np.float64).T
    cols = [
        "Partial Diagrams",
        "SVD Avg. (s)",
        "SVD Std. (s)",
        "KDA Avg. (s)",
        "KDA Std. (s)",
    ]
    filename = f"timing_n_pars.csv"
    save_to_csv(data=data, cols=cols, datapath=datapath, filename=filename)

    # get fit for log-log plot
    fit_func, a, k = fit_powerlaw(diagrams=unique_pars, kda_data=kda_data)
    fit_func_str = r"$t = %1.1g N ^ {%1.3g} $" % (a, k)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    if loglog == True:
        ax.set_xscale("log", nonpositive="clip")
        ax.set_yscale("log", nonpositive="clip")
        min_y = 0.5 * np.min(np.abs(svd_data[:, 0] - svd_data[:, 1]))
        max_y = 5 * np.max(np.abs(kda_data[:, 0] + kda_data[:, 1]))
        ax.set_ylim(bottom=min_y, top=max_y)
    ax.plot(unique_pars, fit_func, color="blue", ls="-", lw=1.5, label=fit_func_str)
    ax.errorbar(
        unique_pars,
        svd_data[:, 0],
        yerr=svd_data[:, 1],
        color="black",
        ecolor="grey",
        fmt=".",
        barsabove=False,
        elinewidth=0.8,
        capsize=4,
        capthick=0.8,
        label="SVD",
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
    print(f"Saving edges plot at location: {fig_savepath}")
    fig.savefig(fig_savepath, dpi=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=str, help="Path to .csv file(s) to analyze.")
    args = parser.parse_args()
    datapath = args.datapath

    if os.path.isdir(datapath):
        # if datapath is a directory, use all data files
        generate_all = True
    else:
        # if not, use only the file specified to generate only the edges
        # and average graph degree plots
        generate_all = False

    cwd = os.getcwd()
    if generate_all:
        datapath = join(cwd, datapath)
        data_files = []
        for file in os.listdir(datapath):
            if file.endswith("data.csv"):
                data_files.append(join(datapath, file))
        if len(data_files) == 0:
            raise FileNotFoundError(f"No files found in {datapath}")
        print(f"Pulling data from {datapath}")
    else:
        datapath, fname = split(datapath)
        datapath = join(cwd, datapath)
        data_files = [join(datapath, fname)]
        print(f"Pulling data from {data_files[0]}")

    nodes = []
    edges = []
    cycles = []
    pars = []
    dirpars = []
    svd_time = []
    kda_time = []
    for file_path in data_files:
        df = pd.read_csv(file_path)
        n_nodes = df["n_states"].values
        n_edges = df["n_edges"].values
        n_cycles = df["n_cycles"].values
        n_pars = df["n_pars"].values
        n_dirpars = df["n_dirpars"].values
        kdat = df.values.T[-1]
        svdt = df.values.T[-2]
        nodes.extend(n_nodes)
        edges.extend(n_edges)
        cycles.extend(n_cycles)
        pars.extend(n_pars)
        dirpars.extend(n_dirpars)
        svd_time.extend(svdt)
        kda_time.extend(kdat)
    svd_time = np.array(svd_time)
    kda_time = np.array(kda_time)

    raw_cols = [
        "n_states",
        "n_edges",
        "n_cycles",
        "n_pars",
        "n_dirpars",
        "svd time (s)",
        "kda time (s)",
    ]

    raw_data = np.column_stack(
        (
            nodes,
            edges,
            cycles,
            pars,
            dirpars,
            svd_time,
            kda_time,
        )
    )

    # consolidate data from all files into single .csv
    filename = f"all_data.csv"
    save_to_csv(data=raw_data, cols=raw_cols, datapath=datapath, filename=filename)

    plot_t_over_edges(svd_time, kda_time, edges, datapath)
    plot_t_over_avg_degree(svd_time, kda_time, nodes, edges, datapath)
    if generate_all:
        plot_t_over_pars(svd_time, kda_time, pars, datapath)
        plot_t_over_pars(svd_time, kda_time, pars, datapath, loglog=False)
        plot_t_over_dirpars(svd_time, kda_time, dirpars, datapath)
        plot_t_over_dirpars(svd_time, kda_time, dirpars, datapath, loglog=False)
        plot_t_over_nodes(svd_time, kda_time, nodes, datapath)
