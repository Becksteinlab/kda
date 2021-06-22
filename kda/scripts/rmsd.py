"""
=============================
KDA Verification RMS Analysis
=============================
The purpose of this module is to analyze the state probability outputs of
the KDA verification script, `verification.py`.
"""
import os
from os.path import join, split
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_rmsd_data(states, rmsd):
    unique_states = np.unique(states)
    avg_rmsd_arr = np.zeros(unique_states.size)
    std_dev_rmsd_arr = np.zeros(unique_states.size)
    for i, n_states in enumerate(unique_states):
        mask = states == n_states
        RMSD_vals = rmsd[mask]
        avg_rmsd = np.mean(RMSD_vals)
        std_dev_rmsd = np.std(RMSD_vals)
        avg_rmsd_arr[i] = avg_rmsd
        std_dev_rmsd_arr[i] = std_dev_rmsd
    return (unique_states, avg_rmsd_arr, std_dev_rmsd_arr)


def generate_rmsd_table(states, rmsd_list, data_path):
    unique_states, avg_rmsd, std_dev_rmsd = get_rmsd_data(states=states, rmsd=rmsd_list)
    data = np.column_stack((unique_states, avg_rmsd, std_dev_rmsd))
    cols = ["Number of States", "Avg. RMSD", "Std. Dev. RMSD"]
    rms_df = pd.DataFrame(data=data, columns=cols)
    table_path = join(data_path, "rmsd_table.tex")
    print(f"Saving RMSD table at location: {table_path}")
    rms_df.to_latex(
        table_path, index=False, float_format="{:0.2e}".format, label="RMSD_table"
    )

def plot_rmsd_over_states(states, rmsd_list, data_path):

    fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
    sns.boxplot(
        x=states, y=rmsd_list, whis=[0, 100], width=0.5, palette="vlag", linewidth=0.8
    )
    sns.stripplot(x=states, y=rmsd_list, size=3, color=".1", linewidth=0)
    ax.set_yscale("log", nonpositive="clip")
    ax.xaxis.grid(True)
    sns.despine(fig=fig, offset=3)

    ax.set_title(r"RMSD vs. Number of States")
    ax.set_xlabel(r"Number of States")
    ax.set_ylabel(r"RMSD")
    fig_savepath = join(datapath, f"rmsd_vs_states.pdf")
    print(f"Saving RMSD vs. states plot at location: {fig_savepath}")
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
    states = df["n_states"]
    partials = df["n_pars"]
    idx = df["graph index"]
    rmsd_list = []
    for i, row in enumerate(df.iterrows()):
        # get the probability values from each row
        p = np.asarray(row[1][8:].values, dtype=np.float64())
        # remove the NaN's from empty cells
        nan_mask = ~np.isnan(p)
        p = p[nan_mask]
        # get the number of states for this row
        n_nodes = row[1]["n_states"]
        svd_vals = p[:n_nodes]
        kda_vals = p[n_nodes:]
        # calculate the RMSD
        RMSD = np.sqrt(np.mean((kda_vals - svd_vals) ** 2))
        rmsd_list.append(RMSD)

    rmsd_list = np.asarray(rmsd_list)

    assert len(rmsd_list) == len(states)

    plot_rmsd_over_states(states=states, rmsd_list=rmsd_list, data_path=datapath)
    generate_rmsd_table(states=states, rmsd_list=rmsd_list, data_path=datapath)
