"""
===================================
KDA Verification Data Consolidation
===================================
The purpose of this module is to collect the data in all `_state_data.csv`
files and store them in a single .csv file, `all_data.csv`.
"""
import os
from os.path import join
import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datapath", type=str, help="Path to .csv file(s) to consolidate."
    )
    args = parser.parse_args()
    datapath = args.datapath

    if not os.path.isdir(datapath):
        raise Exception("Input data path must be a directory, not a file.")

    print(f"Pulling data from {datapath}")
    data_files = []
    for file in os.listdir(datapath):
        # make sure not to grab `all_data.csv` if it has already been
        # generated
        if file.endswith("_state_data.csv"):
            data_files.append(join(datapath, file))

    # if no files are found raise an error
    if len(data_files) == 0:
        raise FileNotFoundError(f"No files found at location {datapath}")

    states_list = []
    for f in data_files:
        fname = os.path.basename(f)
        n_states = int("".join(val for val in fname if val.isdigit()))
        states_list.append(n_states)

    file_order = np.argsort(states_list)[::-1]
    data_files = np.asarray(data_files)
    data_files = data_files[file_order]

    consolidated_csv = pd.concat([pd.read_csv(f) for f in data_files])
    save_path = join(datapath, "all_data.csv")
    print(f"Consolidating all .csv files into {save_path}")
    consolidated_csv.to_csv(save_path, index=False)
