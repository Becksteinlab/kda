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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to .csv files to analyze.")
    args = parser.parse_args()
    data_path = args.data_path
    cwd = os.getcwd()
    data_path = join(cwd, data_path)
    data_files = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            data_files.append(file)
    if len(data_files) == 0:
        raise FileNotFoundError(f"No files found in {data_path}")
    print(f"Pulling data from {data_path}")

    rms_data = []
    for file in data_files:
        file_path = join(data_path, file)
        df = pd.read_csv(file_path)
        data = df.values.T
        graph_node_count = data[1]
        n_states = graph_node_count[0]
        svd_probs = data[3 : 3 + n_states].T
        kda_probs = data[3 + n_states : 3 + (2 * n_states)].T
        RMS = np.sqrt(np.mean((kda_probs - svd_probs) ** 2))
        rms_data.append([n_states, RMS])

    cols = ["n_states", "RMS"]
    rms_data = np.array(rms_data)
    rms_df = pd.DataFrame(data=rms_data, columns=cols)
    table_path = join(data_path, "rms_table.tex")
    print(f"Saving RMS table at location: {table_path}")
    rms_df.to_latex(
        table_path, index=False, float_format="{:0.2e}".format, label="RMS_table"
    )
